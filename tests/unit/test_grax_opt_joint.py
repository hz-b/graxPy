from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grax_opt import joint as joint_module
from grax_opt import objective as objective_module
from grax_opt.joint import (
    AngleMeasurementSpec,
    JointAngleMeasurement,
    JointMeasurementFitConfig,
    optimize_to_joint_measurements,
)
from grax_opt.objective import (
    evaluate_joint_trial_with_metadata,
    reduce_joint_losses,
    simulate_joint_efficiency_curves_with_metadata,
)


class _FakeRunner:
    resolved_max_workers = 2

    batch_sizes: list[int] = []
    result_order: list[int] | None = None
    efficiencies: dict[int, float] | None = None
    status_by_index: dict[int, str] | None = None
    omit_indices: set[int] = set()

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_cases(self, cases, metadata=None):
        type(self).batch_sizes.append(len(cases))
        order = type(self).result_order or list(range(len(cases)))
        for index in order:
            if index in type(self).omit_indices:
                continue
            status = (type(self).status_by_index or {}).get(index, "ok")
            efficiency = (type(self).efficiencies or {}).get(index, 0.25)
            yield SimpleNamespace(
                index=index,
                case_id=cases[index]["case_id"],
                status=status,
                selected_efficiency=efficiency,
            )


class _FakeAxClient:
    def __init__(self, start_index: int = 0) -> None:
        self.next_index = start_index
        self.completed: list[int] = []

    def create_experiment(self, **_kwargs: object) -> None:
        return None

    def get_next_trial(self):
        trial_index = self.next_index
        self.next_index += 1
        return {"depth_nm": 5.0 + trial_index}, trial_index

    def complete_trial(self, trial_index, raw_data=None, data=None) -> None:
        self.completed.append(int(trial_index))

    def save_to_json_file(self, filepath: str) -> None:
        Path(filepath).write_text(json.dumps({"next_index": self.next_index}), encoding="utf-8")


def _reset_fake_runner() -> None:
    _FakeRunner.batch_sizes = []
    _FakeRunner.result_order = None
    _FakeRunner.efficiencies = None
    _FakeRunner.status_by_index = None
    _FakeRunner.omit_indices = set()


def _write_measurement(path: Path, rows: list[tuple[float, float]]) -> Path:
    path.write_text(
        "".join(f"{energy_ev} {efficiency}\n" for energy_ev, efficiency in rows),
        encoding="utf-8",
    )
    return path


def _joint_measurements() -> list[JointAngleMeasurement]:
    return [
        JointAngleMeasurement(
            label="a1",
            grazing_angle_deg=1.0,
            measurement_path=Path("a1.dat"),
            evaluation_energies_ev=np.array([100.0, 200.0]),
            evaluation_efficiency=np.array([0.1, 0.2]),
            weight=1.0,
        ),
        JointAngleMeasurement(
            label="a2",
            grazing_angle_deg=2.0,
            measurement_path=Path("a2.dat"),
            evaluation_energies_ev=np.array([300.0, 400.0]),
            evaluation_efficiency=np.array([0.3, 0.4]),
            weight=1.0,
        ),
    ]


def _joint_eval_config() -> SimpleNamespace:
    return SimpleNamespace(
        diffraction_order=1,
        fourier_orders=5,
        max_workers=2,
        validate_physical_results=True,
        failure_penalty=1.0e6,
        joint_loss_reduction="mean",
    )


def _joint_spec(tmp_path: Path, **overrides: object) -> dict[str, object]:
    first_path = _write_measurement(tmp_path / "m1.dat", [(100.0, 0.2), (200.0, 0.3)])
    second_path = _write_measurement(tmp_path / "m2.dat", [(100.0, 0.4), (200.0, 0.5)])
    spec: dict[str, object] = {
        "build_grating": lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
        "parameter_bounds": {"depth_nm": (1.0, 20.0)},
        "output_dir": tmp_path / "out",
        "measurements": [
            {"grazing_angle_deg": 1.0, "measurement_path": first_path},
            {"grazing_angle_deg": 2.0, "measurement_path": second_path},
        ],
        "total_trials": 3,
        "save_loss_plot": False,
    }
    spec.update(overrides)
    return spec


def test_angle_measurement_spec_rejects_non_positive_angle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grazing_angle_deg must be > 0"):
        AngleMeasurementSpec(grazing_angle_deg=0.0, measurement_path=tmp_path / "m.dat")


def test_angle_measurement_spec_defaults_label_from_angle(tmp_path: Path) -> None:
    spec = AngleMeasurementSpec(grazing_angle_deg=2.5, measurement_path=tmp_path / "m.dat")

    assert spec.label == "alpha2.5deg"


def test_angle_measurement_spec_rejects_mismatched_efficiency_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length as evaluation_energies_ev"):
        AngleMeasurementSpec(
            grazing_angle_deg=1.0,
            measurement_path=tmp_path / "m.dat",
            evaluation_energies_ev=[100.0, 200.0],
            measurement_efficiency=[0.1],
        )


def test_angle_measurement_spec_from_mapping_accepts_numpy_arrays(tmp_path: Path) -> None:
    spec = AngleMeasurementSpec.from_mapping(
        {
            "grazing_angle_deg": 1.0,
            "measurement_path": tmp_path / "m.dat",
            "evaluation_energies_ev": np.array([100.0, 200.0]),
            "measurement_efficiency": np.array([0.1, 0.2]),
        }
    )

    assert spec.evaluation_energies_ev == [100.0, 200.0]
    assert spec.measurement_efficiency == [0.1, 0.2]


def test_angle_measurement_spec_from_mapping_rejects_unexpected_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unexpected joint measurement keys"):
        AngleMeasurementSpec.from_mapping(
            {
                "grazing_angle_deg": 1.0,
                "measurement_path": tmp_path / "m.dat",
                "not_a_real_key": 1,
            }
        )


def test_joint_config_rejects_empty_measurements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="measurements must be provided and non-empty"):
        JointMeasurementFitConfig(
            build_grating=lambda _parameters: None,
            parameter_bounds={"depth_nm": (1.0, 2.0)},
            output_dir=tmp_path / "out",
            measurements=[],
        )


def test_joint_config_rejects_duplicate_measurement_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique labels"):
        JointMeasurementFitConfig(
            build_grating=lambda _parameters: None,
            parameter_bounds={"depth_nm": (1.0, 2.0)},
            output_dir=tmp_path / "out",
            measurements=[
                {"grazing_angle_deg": 1.0, "measurement_path": tmp_path / "m.dat"},
                {"grazing_angle_deg": 1.0, "measurement_path": tmp_path / "other.dat"},
            ],
        )


def test_joint_config_rejects_unknown_reduction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="joint_loss_reduction must be one of"):
        JointMeasurementFitConfig(
            build_grating=lambda _parameters: None,
            parameter_bounds={"depth_nm": (1.0, 2.0)},
            output_dir=tmp_path / "out",
            measurements=[{"grazing_angle_deg": 1.0, "measurement_path": tmp_path / "m.dat"}],
            joint_loss_reduction="median",
        )


def test_joint_config_from_mapping_rejects_unexpected_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unexpected joint measurement-fit spec keys"):
        JointMeasurementFitConfig.from_mapping(
            {
                "build_grating": lambda _parameters: None,
                "parameter_bounds": {"depth_nm": (1.0, 2.0)},
                "output_dir": tmp_path / "out",
                "measurements": [
                    {"grazing_angle_deg": 1.0, "measurement_path": tmp_path / "m.dat"}
                ],
                "not_a_real_key": 1,
            }
        )


def test_reduce_joint_losses_supports_mean_sum_pooled_and_weighted() -> None:
    losses = {"a": 1.0, "b": 3.0}

    assert reduce_joint_losses(losses) == pytest.approx(2.0)
    assert reduce_joint_losses(losses, reduction="sum") == pytest.approx(4.0)
    assert reduce_joint_losses(
        losses,
        reduction="pooled",
        point_counts={"a": 1, "b": 3},
    ) == pytest.approx(2.5)
    assert reduce_joint_losses(
        losses,
        reduction="weighted",
        weights={"a": 3.0, "b": 1.0},
    ) == pytest.approx(1.5)


def test_reduce_joint_losses_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="joint_loss_reduction must be one of"):
        reduce_joint_losses({"a": 1.0}, reduction="median")


def test_simulate_joint_curves_reassembles_out_of_order_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    _FakeRunner.result_order = [3, 1, 0, 2]
    _FakeRunner.efficiencies = {0: 0.11, 1: 0.22, 2: 0.33, 3: 0.44}
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    simulated, resolved_max_workers = simulate_joint_efficiency_curves_with_metadata(
        _joint_eval_config(),
        {"depth_nm": 5.0},
        _joint_measurements(),
        backend="numba",
        build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
        resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
    )

    assert np.allclose(simulated["a1"], np.array([0.11, 0.22]))
    assert np.allclose(simulated["a2"], np.array([0.33, 0.44]))
    assert resolved_max_workers == 2


def test_simulate_joint_curves_builds_one_flat_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    simulate_joint_efficiency_curves_with_metadata(
        _joint_eval_config(),
        {"depth_nm": 5.0},
        _joint_measurements(),
        backend="numba",
        build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
        resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
    )

    assert _FakeRunner.batch_sizes == [4]


def test_simulate_joint_curves_raises_when_results_are_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    _FakeRunner.omit_indices = {2}
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    with pytest.raises(objective_module._BatchCaseFailure):
        simulate_joint_efficiency_curves_with_metadata(
            _joint_eval_config(),
            {"depth_nm": 5.0},
            _joint_measurements(),
            backend="numba",
            build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
        )


def test_evaluate_joint_trial_returns_failure_penalty_when_case_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    _FakeRunner.status_by_index = {1: "error"}
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    joint_loss, per_measurement_losses, simulated, _workers = (
        evaluate_joint_trial_with_metadata(
            _joint_eval_config(),
            {"depth_nm": 5.0},
            _joint_measurements(),
            backend="numba",
            build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
        )
    )

    assert joint_loss == pytest.approx(1.0e6)
    assert per_measurement_losses == {"a1": 1.0e6, "a2": 1.0e6}
    assert simulated == {}


def test_evaluate_joint_trial_averages_per_measurement_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    joint_loss, per_measurement_losses, _simulated, _workers = (
        evaluate_joint_trial_with_metadata(
            _joint_eval_config(),
            {"depth_nm": 5.0},
            _joint_measurements(),
            backend="numba",
            build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
        )
    )

    expected_a1 = float(np.mean((np.array([0.25, 0.25]) - np.array([0.1, 0.2])) ** 2))
    expected_a2 = float(np.mean((np.array([0.25, 0.25]) - np.array([0.3, 0.4])) ** 2))
    assert per_measurement_losses["a1"] == pytest.approx(expected_a1)
    assert per_measurement_losses["a2"] == pytest.approx(expected_a2)
    assert joint_loss == pytest.approx((expected_a1 + expected_a2) / 2.0)


def test_optimize_to_joint_measurements_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        joint_module,
        "_create_ax_client_for_joint_config",
        lambda _config: _FakeAxClient(),
    )

    result = optimize_to_joint_measurements(_joint_spec(tmp_path))

    assert result.completed_trials == 3
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["optimization_mode"] == "joint_measurement_fit"
    assert set(payload["per_measurement_best_losses"]) == {"alpha1deg", "alpha2deg"}
    assert payload["joint_loss_reduction"] == "mean"

    rows = list(csv.reader(result.trial_history_csv_path.open(encoding="utf-8")))
    assert rows[0] == ["trial_index", "loss", "depth_nm", "loss_alpha1deg", "loss_alpha2deg"]
    assert len(rows) - 1 == 3

    comparison_rows = list(csv.reader(result.comparison_csv_path.open(encoding="utf-8")))
    assert len(comparison_rows) - 1 == 4


def test_optimize_to_joint_measurements_builds_one_flat_batch_per_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        joint_module,
        "_create_ax_client_for_joint_config",
        lambda _config: _FakeAxClient(),
    )

    optimize_to_joint_measurements(_joint_spec(tmp_path, total_trials=4))

    assert _FakeRunner.batch_sizes == [4, 4, 4, 4]


def test_optimize_to_joint_measurements_uses_supplied_measurement_efficiency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        joint_module,
        "_create_ax_client_for_joint_config",
        lambda _config: _FakeAxClient(),
    )
    measurement_path = _write_measurement(tmp_path / "m.dat", [(100.0, 0.2), (200.0, 0.3)])

    result = optimize_to_joint_measurements(
        {
            "build_grating": lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            "parameter_bounds": {"depth_nm": (1.0, 20.0)},
            "output_dir": tmp_path / "out",
            "measurements": [
                {
                    "grazing_angle_deg": 1.0,
                    "measurement_path": measurement_path,
                    "evaluation_energies_ev": [100.0, 200.0],
                    "measurement_efficiency": [0.25, 0.25],
                }
            ],
            "total_trials": 1,
            "save_loss_plot": False,
            "save_best_fit_plot": False,
        }
    )

    assert result.best_loss == pytest.approx(0.0)
