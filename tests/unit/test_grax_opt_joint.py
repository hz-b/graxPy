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
    JointMeasurement,
    JointMeasurementFitConfig,
    MeasurementSpec,
    optimize_to_joint_measurements,
    prepare_joint_measurements,
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
    init_kwargs: dict[str, object] = {}
    seen_cases: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs = dict(kwargs)

    def run_cases(self, cases, metadata=None):
        type(self).batch_sizes.append(len(cases))
        type(self).seen_cases = [dict(case) for case in cases]
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
    _FakeRunner.init_kwargs = {}
    _FakeRunner.seen_cases = []


def _write_measurement(path: Path, rows: list[tuple[float, float]]) -> Path:
    path.write_text(
        "".join(f"{energy_ev} {efficiency}\n" for energy_ev, efficiency in rows),
        encoding="utf-8",
    )
    return path


def _joint_measurements() -> list[JointMeasurement]:
    return [
        JointMeasurement(
            label="a1",
            measurement_path=Path("a1.dat"),
            angle_mode="fixed",
            grazing_angle_deg=1.0,
            cff=None,
            diffraction_order=1,
            polarization="s",
            evaluation_energies_ev=np.array([100.0, 200.0]),
            evaluation_efficiency=np.array([0.1, 0.2]),
            weight=1.0,
        ),
        JointMeasurement(
            label="a2",
            measurement_path=Path("a2.dat"),
            angle_mode="fixed",
            grazing_angle_deg=2.0,
            cff=None,
            diffraction_order=1,
            polarization="s",
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


def test_measurement_spec_rejects_non_positive_angle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grazing_angle_deg must be > 0"):
        MeasurementSpec(measurement_path=tmp_path / "m.dat", grazing_angle_deg=0.0)


def test_measurement_spec_defaults_label_from_angle(tmp_path: Path) -> None:
    spec = MeasurementSpec(measurement_path=tmp_path / "m.dat", grazing_angle_deg=2.5)

    assert spec.label == "alpha2.5deg"


def test_measurement_spec_rejects_mismatched_efficiency_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length as evaluation_energies_ev"):
        MeasurementSpec(
            grazing_angle_deg=1.0,
            measurement_path=tmp_path / "m.dat",
            evaluation_energies_ev=[100.0, 200.0],
            measurement_efficiency=[0.1],
        )


def test_measurement_spec_from_mapping_accepts_numpy_arrays(tmp_path: Path) -> None:
    spec = MeasurementSpec.from_mapping(
        {
            "grazing_angle_deg": 1.0,
            "measurement_path": tmp_path / "m.dat",
            "evaluation_energies_ev": np.array([100.0, 200.0]),
            "measurement_efficiency": np.array([0.1, 0.2]),
        }
    )

    assert spec.evaluation_energies_ev == [100.0, 200.0]
    assert spec.measurement_efficiency == [0.1, 0.2]


def test_measurement_spec_from_mapping_rejects_unexpected_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unexpected joint measurement keys"):
        MeasurementSpec.from_mapping(
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


def test_measurement_spec_inherits_run_level_conditions(tmp_path: Path) -> None:
    """A spec that sets no conditions takes all of them from the run."""

    prepared = prepare_joint_measurements(
        [MeasurementSpec(measurement_path=_write_measurement(tmp_path / "m.dat", [(100.0, 0.2)]))],
        angle_mode="fixed",
        grazing_angle_deg=4.0,
        diffraction_order=2,
        polarization="TM",
    )

    assert prepared[0].angle_mode == "fixed"
    assert prepared[0].grazing_angle_deg == 4.0
    assert prepared[0].diffraction_order == 2
    assert prepared[0].polarization == "p"
    assert prepared[0].label == "m"


def test_measurement_spec_overrides_win_over_run_level_conditions(tmp_path: Path) -> None:
    """Each condition is independently overridable per measurement."""

    prepared = prepare_joint_measurements(
        [
            MeasurementSpec(
                measurement_path=_write_measurement(tmp_path / "a.dat", [(100.0, 0.2)]),
                grazing_angle_deg=2.0,
            ),
            MeasurementSpec(
                measurement_path=_write_measurement(tmp_path / "b.dat", [(100.0, 0.3)]),
                angle_mode="cff",
                cff=2.25,
                diffraction_order=3,
                polarization="p",
            ),
        ],
        angle_mode="fixed",
        grazing_angle_deg=4.0,
        diffraction_order=1,
        polarization="s",
    )

    assert (prepared[0].angle_mode, prepared[0].grazing_angle_deg) == ("fixed", 2.0)
    assert prepared[0].diffraction_order == 1
    assert prepared[0].polarization == "s"
    assert (prepared[1].angle_mode, prepared[1].cff) == ("cff", 2.25)
    assert prepared[1].diffraction_order == 3
    assert prepared[1].polarization == "p"


def test_measurement_spec_requires_the_value_its_angle_mode_needs(tmp_path: Path) -> None:
    measurement_path = _write_measurement(tmp_path / "m.dat", [(100.0, 0.2)])

    with pytest.raises(ValueError, match="resolves to angle_mode='cff' but no cff"):
        prepare_joint_measurements(
            [MeasurementSpec(measurement_path=measurement_path, angle_mode="cff")],
            angle_mode="fixed",
            grazing_angle_deg=4.0,
        )

    with pytest.raises(ValueError, match="resolves to angle_mode='fixed' but no grazing_angle_deg"):
        prepare_joint_measurements([MeasurementSpec(measurement_path=measurement_path)])


def test_measurement_spec_labels_describe_the_conditions_they_set(tmp_path: Path) -> None:
    measurement_path = tmp_path / "m.dat"

    assert (
        MeasurementSpec(
            measurement_path=measurement_path, grazing_angle_deg=4.0, diffraction_order=2
        ).label
        == "alpha4deg_order2"
    )
    assert (
        MeasurementSpec(measurement_path=measurement_path, cff=2.25, polarization="TM").label
        == "cff2p25_p"
    )


def test_joint_cases_carry_each_measurement_own_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-measurement order and polarization must reach the batch cases."""

    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)

    measurements = [
        JointMeasurement(
            label="fixed_s",
            measurement_path=Path("a.dat"),
            angle_mode="fixed",
            grazing_angle_deg=4.0,
            cff=None,
            diffraction_order=1,
            polarization="s",
            evaluation_energies_ev=np.array([100.0]),
            evaluation_efficiency=np.array([0.2]),
            weight=1.0,
        ),
        JointMeasurement(
            label="cff_p",
            measurement_path=Path("b.dat"),
            angle_mode="cff",
            grazing_angle_deg=None,
            cff=2.25,
            diffraction_order=2,
            polarization="p",
            evaluation_energies_ev=np.array([100.0]),
            evaluation_efficiency=np.array([0.3]),
            weight=1.0,
        ),
    ]

    simulate_joint_efficiency_curves_with_metadata(
        _joint_eval_config(),
        {"depth_nm": 5.0},
        measurements,
        backend="numba",
        build_grating_fn=lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
        resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
    )

    first, second = _FakeRunner.seen_cases
    assert (first["diffraction_order"], first["polarization"]) == (1, "s")
    assert first["grazing_angle_deg"] == 4.0
    assert (second["diffraction_order"], second["polarization"]) == (2, "p")
    # The cff mode derives its angle from the monochromator relation, not a constant.
    assert second["grazing_angle_deg"] != 4.0


def test_joint_runner_uses_the_configured_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_runner()
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        joint_module, "_create_ax_client_for_joint_config", lambda _config: _FakeAxClient()
    )

    optimize_to_joint_measurements(
        _joint_spec(tmp_path, solver="neviere", solver_options={"step_phase": 0.01})
    )

    assert _FakeRunner.init_kwargs["solver"] == "neviere"
    assert _FakeRunner.init_kwargs["solver_options"] == {"step_phase": 0.01}
