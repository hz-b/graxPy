from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grax_opt import (
    MeasurementFitConfig,
    ParameterBounds,
    build_measurement_fit_ax_parameters,
    build_evaluation_measurement,
    evaluate_trial,
    load_measurement_data,
    optimize_to_measurements,
    resolve_measurement_fit_trial_parameters,
    sample_measurement_data,
)
from grax_opt import dynamic as dynamic_module
from grax_opt import objective as objective_module
from grax_opt import optimize as optimize_module


def test_load_measurement_data_parses_and_filters(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement.dat"
    measurement_path.write_text("100 0.2\n101 --\n102 0.4\n", encoding="utf-8")

    measurement = load_measurement_data(measurement_path)

    assert np.allclose(measurement.energy_ev, np.array([100.0, 102.0]))
    assert np.allclose(measurement.efficiency, np.array([0.2, 0.4]))


def test_sample_measurement_data_interpolates_and_checks_bounds(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement.dat"
    measurement_path.write_text("100 0.2\n200 0.4\n300 0.8\n", encoding="utf-8")
    measurement = load_measurement_data(measurement_path)

    sampled = sample_measurement_data(measurement, [100.0, 150.0, 250.0])
    assert np.allclose(sampled.energy_ev, np.array([100.0, 150.0, 250.0]))
    assert np.allclose(sampled.efficiency, np.array([0.2, 0.3, 0.6]))

    with pytest.raises(ValueError, match="within the measurement energy range"):
        sample_measurement_data(measurement, [50.0])


def test_parameter_bounds_validation() -> None:
    with pytest.raises(ValueError, match="upper > lower"):
        ParameterBounds(1.0, 1.0)


def _build_measurement_fit_config(tmp_path: Path) -> MeasurementFitConfig:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    return MeasurementFitConfig(
        build_grating=lambda parameters: type(
            "DynamicGrating",
            (),
            {"period_lpermm": float(parameters["period_lpermm"])},
        )(),
        parameter_bounds={
            "period_lpermm": ParameterBounds(380.0, 420.0),
            "width_to_period_ratio": ParameterBounds(0.5, 0.85),
            "left_wall_angle_deg": ParameterBounds(1.0, 45.0),
            "right_wall_angle_deg": ParameterBounds(1.0, 45.0),
        },
        equality_constraints={"right_wall_angle_deg": "left_wall_angle_deg"},
        measurement_path=measurement_path,
        output_dir=tmp_path / "out",
        evaluation_energies_ev=[150.0],
    )


def test_measurement_fit_optimizer_resolves_tied_parameters(tmp_path: Path) -> None:
    config = _build_measurement_fit_config(tmp_path)
    parameters = build_measurement_fit_ax_parameters(config)
    assert [parameter["name"] for parameter in parameters] == [
        "period_lpermm",
        "width_to_period_ratio",
        "left_wall_angle_deg",
    ]

    resolved = resolve_measurement_fit_trial_parameters(
        config,
        {
            "period_lpermm": 400.0,
            "width_to_period_ratio": 0.67,
            "left_wall_angle_deg": 15.0,
        },
    )
    assert resolved["right_wall_angle_deg"] == pytest.approx(15.0)


def test_measurement_fit_config_supports_energy_angle_pairs(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    config = MeasurementFitConfig(
        build_grating=lambda parameters: type(
            "DynamicGrating",
            (),
            {"period_lpermm": float(parameters["period_lpermm"])},
        )(),
        parameter_bounds={"period_lpermm": ParameterBounds(380.0, 420.0)},
        measurement_path=measurement_path,
        output_dir=tmp_path / "out",
        evaluation_energies_ev=[150.0],
        evaluation_grazing_angles_deg=[3.0, 5.0],
    )
    evaluation = build_evaluation_measurement(config, load_measurement_data(measurement_path))
    assert np.allclose(evaluation.energy_ev, np.array([150.0, 150.0]))


def test_measurement_fit_config_rejects_many_energy_many_angle(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more than one value"):
        MeasurementFitConfig(
            build_grating=lambda parameters: type(
                "DynamicGrating",
                (),
                {"period_lpermm": float(parameters["period_lpermm"])},
            )(),
            parameter_bounds={"period_lpermm": ParameterBounds(380.0, 420.0)},
            measurement_path=measurement_path,
            output_dir=tmp_path / "out",
            evaluation_energies_ev=[100.0, 150.0],
            evaluation_grazing_angles_deg=[3.0, 5.0],
        )


def test_measurement_fit_config_rejects_legacy_loss_name(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected measurement-fit spec keys"):
        MeasurementFitConfig.from_mapping(
            {
                "build_grating": lambda parameters: type(
                    "DynamicGrating",
                    (),
                    {"period_lpermm": float(parameters["period_lpermm"])},
                )(),
                "parameter_bounds": {"period_lpermm": (380.0, 420.0)},
                "measurement_path": measurement_path,
                "output_dir": tmp_path / "out",
                "evaluation_energies_ev": [150.0],
                "loss_name": "mse",
            }
        )


def test_measurement_fit_config_accepts_trial_level_max_workers(tmp_path: Path) -> None:
    config = _build_measurement_fit_config(tmp_path)
    config = MeasurementFitConfig(
        build_grating=config.build_grating,
        parameter_bounds=config.parameter_bounds,
        equality_constraints=config.equality_constraints,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=config.evaluation_energies_ev,
        max_workers="auto",
    )
    assert config.max_workers == "auto"


def test_measurement_fit_config_rejects_candidate_batching_with_optimizer_workers(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size > 1"):
        MeasurementFitConfig(
            build_grating=lambda parameters: type(
                "DynamicGrating",
                (),
                {"period_lpermm": float(parameters["period_lpermm"])},
            )(),
            parameter_bounds={"period_lpermm": ParameterBounds(380.0, 420.0)},
            measurement_path=tmp_path / "measurement_fit.dat",
            output_dir=tmp_path / "out",
            evaluation_energies_ev=[150.0],
            batch_size=2,
            max_workers=2,
        )


def test_evaluate_trial_requires_measurement_fit_build_hooks(tmp_path: Path) -> None:
    config = _build_measurement_fit_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)
    with pytest.warns(FutureWarning, match="deprecated"):
        loss = evaluate_trial(config, {"period_lpermm": 400.0}, measurement, backend="numpy")
    assert loss == pytest.approx(float(config.failure_penalty))


def test_simulate_efficiency_curve_uses_batch_runner_and_preserves_case_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_measurement_fit_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.resolved_max_workers = 3

        def run_cases(self, cases, metadata=None):
            del metadata
            prepared = list(cases)
            assert [case["grazing_angle_deg"] for case in prepared] == [3.0, 5.0]
            # Return out of order to verify reconstruction by result.index.
            return iter(
                [
                    SimpleNamespace(index=1, case_id="trial_eval_1", status="ok", selected_efficiency=0.5),
                    SimpleNamespace(index=0, case_id="trial_eval_0", status="ok", selected_efficiency=0.25),
                ]
            )

    monkeypatch.setattr(objective_module, "BatchSimulationRunner", FakeRunner)

    efficiencies, resolved_max_workers = objective_module.simulate_efficiency_curve_with_metadata(
        MeasurementFitConfig(
            build_grating=config.build_grating,
            parameter_bounds=config.parameter_bounds,
            equality_constraints=config.equality_constraints,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            evaluation_grazing_angles_deg=[3.0, 5.0],
            max_workers=3,
        ),
        {"period_lpermm": 400.0, "width_to_period_ratio": 0.67, "left_wall_angle_deg": 15.0},
        measurement,
        backend="numba",
        build_grating_fn=lambda parameters: type("DynamicGrating", (), {"period_lpermm": 400.0})(),
        resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": 1.2},
    )

    assert np.allclose(efficiencies, np.array([0.25, 0.5]))
    assert resolved_max_workers == 3


def test_evaluate_trial_with_metadata_returns_failure_penalty_when_batch_case_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_measurement_fit_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.resolved_max_workers = 2

        def run_cases(self, cases, metadata=None):
            del cases, metadata
            return iter([SimpleNamespace(index=0, case_id="trial_eval_0", status="error", selected_efficiency=0.0)])

    monkeypatch.setattr(objective_module, "BatchSimulationRunner", FakeRunner)

    loss, resolved_max_workers = objective_module.evaluate_trial_with_metadata(
        config,
        {"period_lpermm": 400.0, "width_to_period_ratio": 0.67, "left_wall_angle_deg": 15.0},
        measurement,
        backend="numba",
        build_grating_fn=lambda parameters: type("DynamicGrating", (), {"period_lpermm": 400.0})(),
        resolve_solver_parameters_fn=lambda _parameters: {"roughness_sigma_nm": None},
    )

    assert loss == pytest.approx(float(config.failure_penalty))
    assert resolved_max_workers == 2


def test_optimize_to_measurements_smoke_run_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")

    config = MeasurementFitConfig(
        build_grating=lambda parameters: type(
            "DynamicGrating",
            (),
            {"period_lpermm": float(parameters["period_lpermm"])},
        )(),
        parameter_bounds={
            "period_lpermm": ParameterBounds(380.0, 420.0),
            "width_to_period_ratio": ParameterBounds(0.5, 0.85),
            "depth_nm": ParameterBounds(5.0, 30.0),
            "left_wall_angle_deg": ParameterBounds(1.0, 45.0),
            "right_wall_angle_deg": ParameterBounds(1.0, 45.0),
            "top_cap_thickness_nm": ParameterBounds(0.0, 2.7),
        },
        equality_constraints={"right_wall_angle_deg": "left_wall_angle_deg"},
        measurement_path=measurement_path,
        output_dir=tmp_path / "measurement_fit_out",
        total_trials=2,
        batch_size=1,
        evaluation_energies_ev=[150.0],
    )

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {
                    "period_lpermm": 400.0,
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 14.9,
                    "left_wall_angle_deg": 15.0,
                    "top_cap_thickness_nm": 0.3,
                },
                {
                    "period_lpermm": 401.0,
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 15.1,
                    "left_wall_angle_deg": 16.0,
                    "top_cap_thickness_nm": 0.3,
                },
            ]
            self.completed: list[tuple[int, object]] = []

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            trial_index = len(self.completed)
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append((trial_index, raw_data if raw_data is not None else data))

    observed_energies: list[float] = []

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.resolved_max_workers = 1

        def run_cases(self, cases, metadata=None):
            prepared = list(cases)
            del metadata
            observed_energies.extend(float(case["energy_ev"]) for case in prepared)
            return iter(
                [
                    SimpleNamespace(
                        index=index,
                        case_id=f"trial_eval_{index}",
                        status="ok",
                        selected_efficiency=0.25,
                    )
                    for index, _case in enumerate(prepared)
                ]
            )

    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", FakeRunner)
    monkeypatch.setattr(dynamic_module, "_save_best_fit_plot", lambda **_kwargs: None)
    monkeypatch.setattr(dynamic_module, "_save_loss_history_plot", lambda **_kwargs: None)

    result = optimize_to_measurements(config)

    assert result.result_json_path.exists()
    assert result.trial_history_csv_path.exists()
    assert result.completed_trials == 2
    assert observed_energies
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["optimization_mode"] == "measurement_fit"
    assert payload["evaluation_mode"] == "energy_only"
    assert payload["optimizer_execution_strategy"] == "trial_batch_runner"
    assert payload["optimizer_requested_max_workers"] is None
    assert payload["optimizer_resolved_max_workers"] == 1


def test_optimize_to_measurements_pair_mode_uses_explicit_angles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")

    config = MeasurementFitConfig(
        build_grating=lambda parameters: type(
            "DynamicGrating",
            (),
            {"period_lpermm": float(parameters["period_lpermm"])},
        )(),
        parameter_bounds={"period_lpermm": ParameterBounds(380.0, 420.0)},
        measurement_path=measurement_path,
        output_dir=tmp_path / "measurement_fit_out",
        total_trials=1,
        batch_size=1,
        evaluation_energies_ev=[150.0],
        evaluation_grazing_angles_deg=[3.0, 5.0],
    )

    class FakeAxClient:
        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            return {"period_lpermm": 400.0}, 0

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            return None

    observed_angles: list[float] = []

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.resolved_max_workers = 1

        def run_cases(self, cases, metadata=None):
            prepared = list(cases)
            del metadata
            observed_angles.extend(float(case["grazing_angle_deg"]) for case in prepared)
            return iter(
                [
                    SimpleNamespace(
                        index=index,
                        case_id=f"trial_eval_{index}",
                        status="ok",
                        selected_efficiency=0.25,
                    )
                    for index, _case in enumerate(prepared)
                ]
            )

    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", FakeRunner)
    monkeypatch.setattr(dynamic_module, "_save_best_fit_plot", lambda **_kwargs: None)
    monkeypatch.setattr(dynamic_module, "_save_loss_history_plot", lambda **_kwargs: None)

    result = optimize_to_measurements(config)

    assert result.completed_trials == 1
    assert len(observed_angles) >= 2
    assert all(
        observed_angles[index : index + 2] == [3.0, 5.0]
        for index in range(0, len(observed_angles), 2)
    )
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["evaluation_mode"] == "energy_angle_pairs"


def test_optimize_to_measurements_passes_optimizer_max_workers_to_batch_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_path = tmp_path / "measurement_fit.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")

    config = MeasurementFitConfig(
        build_grating=lambda parameters: type(
            "DynamicGrating",
            (),
            {"period_lpermm": float(parameters["period_lpermm"])},
        )(),
        parameter_bounds={"period_lpermm": ParameterBounds(380.0, 420.0)},
        measurement_path=measurement_path,
        output_dir=tmp_path / "measurement_fit_out",
        total_trials=1,
        batch_size=1,
        evaluation_energies_ev=[150.0],
        max_workers="auto",
    )

    class FakeAxClient:
        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            return {"period_lpermm": 400.0}, 0

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            return None

    captured_kwargs: list[dict[str, object]] = []

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.append(dict(kwargs))
            self.resolved_max_workers = 4

        def run_cases(self, cases, metadata=None):
            prepared = list(cases)
            del metadata
            return iter(
                [
                    SimpleNamespace(
                        index=index,
                        case_id=f"trial_eval_{index}",
                        status="ok",
                        selected_efficiency=0.25,
                    )
                    for index, _case in enumerate(prepared)
                ]
            )

    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", FakeRunner)
    monkeypatch.setattr(dynamic_module, "_save_best_fit_plot", lambda **_kwargs: None)
    monkeypatch.setattr(dynamic_module, "_save_loss_history_plot", lambda **_kwargs: None)

    result = optimize_to_measurements(config)

    assert result.completed_trials == 1
    assert captured_kwargs
    assert captured_kwargs[0]["max_workers"] == "auto"
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["optimizer_requested_max_workers"] == "auto"
    assert payload["optimizer_resolved_max_workers"] == 4


def test_resolve_optimizer_backend_numba_first_and_numpy_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: True)
    assert optimize_module._resolve_optimizer_backend("auto") == "numba"
    assert optimize_module._resolve_optimizer_backend("numba") == "numba"
    with pytest.warns(FutureWarning, match="deprecated"):
        assert optimize_module._resolve_optimizer_backend("numpy") == "numpy"

    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: False)
    with pytest.raises(RuntimeError, match="required numba dependency"):
        optimize_module._resolve_optimizer_backend("auto")
    with pytest.raises(RuntimeError, match="required numba dependency"):
        optimize_module._resolve_optimizer_backend("numba")


def test_example_configs_split_optimizer_and_simulation_backends() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    blazed_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_blazed" / "example_config.py")
    )
    laminar_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_laminar" / "example_config.py")
    )

    assert blazed_config["optimizer_backend"] == "auto"
    assert blazed_config["simulation_backend"] == "numba"
    assert laminar_config["optimizer_backend"] == "auto"
    assert laminar_config["simulation_backend"] == "numba"
    assert "optimizer_max_workers" in laminar_config
    assert blazed_config["simulation_backend"] == laminar_config["simulation_backend"]
