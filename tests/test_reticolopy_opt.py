from __future__ import annotations

import json
import runpy
from pathlib import Path

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


def test_evaluate_trial_requires_measurement_fit_build_hooks(tmp_path: Path) -> None:
    config = _build_measurement_fit_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)
    loss = evaluate_trial(config, {"period_lpermm": 400.0}, measurement, backend="numpy")
    assert loss == pytest.approx(float(config.failure_penalty))


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

    def fake_run_simulation(**kwargs):
        observed_energies.append(float(kwargs["energy_ev"]))
        return type("Result", (), {"selected_efficiency": 0.25})()

    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr("grax_opt.objective.run_simulation", fake_run_simulation)
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

    def fake_run_simulation(**kwargs):
        observed_angles.append(float(kwargs["grazing_angle_deg"]))
        return type("Result", (), {"selected_efficiency": 0.25})()

    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr("grax_opt.objective.run_simulation", fake_run_simulation)
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


def test_resolve_optimizer_backend_auto_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: True)
    assert optimize_module._resolve_optimizer_backend("auto") == "numba"
    assert optimize_module._resolve_optimizer_backend("numba") == "numba"
    assert optimize_module._resolve_optimizer_backend("numpy") == "numpy"

    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: False)
    assert optimize_module._resolve_optimizer_backend("auto") == "numpy"
    assert optimize_module._resolve_optimizer_backend("numba") == "numpy"


def test_example_configs_split_optimizer_and_simulation_backends() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    blazed_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_blazed" / "example_config.py")
    )
    laminar_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_laminar" / "example_config.py")
    )
    measurement_fit_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "dynamic_optimizer" / "example_config.py")
    )

    assert blazed_config["optimizer_backend"] == "auto"
    assert blazed_config["simulation_backend"] in {"numba", "numpy"}
    assert laminar_config["optimizer_backend"] == "auto"
    assert laminar_config["simulation_backend"] in {"numba", "numpy"}
    assert measurement_fit_config["optimizer_backend"] == "auto"
    assert measurement_fit_config["simulation_backend"] in {"numba", "numpy"}
