from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grax import LaminarGrating
from grax_opt.config import (
    BlazedAxConfig,
    InitialBlazedGrating,
    InitialLaminarGrating,
    LaminarAxConfig,
    ParameterBounds,
)
from grax_opt.data import load_measurement_data, sample_measurement_data
from grax_opt.model import (
    build_ax_parameters,
    build_blazed_grating,
    build_laminar_grating,
    resolve_grating_parameters,
    resolve_solver_parameters,
)
from grax_opt.objective import (
    build_evaluation_measurement,
    evaluate_trial,
    simulate_efficiency_curve,
)
from grax_opt.optimize import (
    _build_ax_optimize_kwargs,
    json_safe_grating_parameters,
    optimize_blazed,
    optimize_laminar,
)
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[1] / "examples" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
AU = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Au_cxro.txt", "Au")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")
C = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_C_cxro.txt", "C")


def build_test_config(tmp_path: Path) -> BlazedAxConfig:
    """Return a reusable optimizer config."""

    measurement_path = tmp_path / "measurement.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    return BlazedAxConfig(
        initial_grating=InitialBlazedGrating(
            period_lpermm=600.0,
            blaze_angle_deg=0.729,
            anti_blaze_angle_deg=5.597,
            substrate_material=SI,
            layer_material=AU,
            top_cap_material=C,
            top_cap_thickness_nm=1.0,
        ),
        measurement_path=measurement_path,
        output_dir=tmp_path / "out",
        total_trials=3,
        optimize_blaze_angle_deg=True,
        evaluation_energies_ev=[150.0],
    )


def build_laminar_test_config(tmp_path: Path, *, angle_mode: str = "fixed") -> LaminarAxConfig:
    """Return a reusable laminar optimizer config."""

    measurement_path = tmp_path / "laminar_measurement.dat"
    measurement_path.write_text("100 0.2\n200 0.3\n", encoding="utf-8")
    return LaminarAxConfig(
        initial_grating=InitialLaminarGrating(
            period_lpermm=400.0,
            width_to_period_ratio=0.67,
            depth_nm=14.9,
            left_wall_angle_deg=15.0,
            right_wall_angle_deg=15.0,
            substrate_material=SI,
            layer_material=PT,
            layer_thickness_nm=28.77,
            top_cap_material=C,
            top_cap_thickness_nm=0.3,
        ),
        measurement_path=measurement_path,
        output_dir=tmp_path / "laminar_out",
        angle_mode=angle_mode,
        grazing_angle_deg=4.0,
        cff=2.5,
        total_trials=3,
        period_lpermm_bounds=ParameterBounds(380.0, 420.0),
        width_to_period_ratio_bounds=ParameterBounds(0.5, 0.85),
        depth_nm_bounds=ParameterBounds(5.0, 30.0),
        left_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
        right_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
        top_cap_thickness_nm_bounds=ParameterBounds(0.0, 2.7),
        evaluation_energies_ev=[150.0],
    )


def test_load_measurement_data_drops_placeholder_rows(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement.dat"
    measurement_path.write_text("100 0.2\n101 --\n102 0.4\n", encoding="utf-8")

    measurement = load_measurement_data(measurement_path)

    assert np.allclose(measurement.energy_ev, np.array([100.0, 102.0]))
    assert np.allclose(measurement.efficiency, np.array([0.2, 0.4]))


def test_load_measurement_data_parses_semicolon_decimal_comma_rows(tmp_path: Path) -> None:
    measurement_path = tmp_path / "laminar_measurement.csv"
    measurement_path.write_text(
        "Energy;alpha = 4 deg\neV;\n;1 order\n51,031;0,035283\n52,039;--\n53,048;0,036869\n",
        encoding="utf-8",
    )

    measurement = load_measurement_data(measurement_path)

    assert np.allclose(measurement.energy_ev, np.array([51.031, 53.048]))
    assert np.allclose(measurement.efficiency, np.array([0.035283, 0.036869]))


def test_sample_measurement_data_interpolates_and_checks_bounds(tmp_path: Path) -> None:
    measurement_path = tmp_path / "measurement.dat"
    measurement_path.write_text("100 0.2\n200 0.4\n300 0.8\n", encoding="utf-8")
    measurement = load_measurement_data(measurement_path)

    sampled = sample_measurement_data(measurement, [100.0, 150.0, 250.0])
    assert np.allclose(sampled.energy_ev, np.array([100.0, 150.0, 250.0]))
    assert np.allclose(sampled.efficiency, np.array([0.2, 0.3, 0.6]))

    with pytest.raises(ValueError, match="within the measurement energy range"):
        sample_measurement_data(measurement, [50.0])


def test_build_ax_parameters_uses_default_and_override_bounds(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    parameters = build_ax_parameters(config)

    assert parameters[0]["name"] == "period_lpermm"
    assert parameters[0]["bounds"] == pytest.approx([594.0, 606.0])
    assert parameters[1]["name"] == "blaze_angle_deg"
    assert parameters[1]["bounds"] == pytest.approx([0.5832, 0.8748])

    override_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        optimize_blaze_angle_deg=True,
        period_lpermm_bounds=ParameterBounds(590.0, 610.0),
        blaze_angle_deg_bounds=ParameterBounds(0.6, 0.8),
        evaluation_energies_ev=[150.0],
    )
    override_parameters = build_ax_parameters(override_config)
    assert override_parameters[0]["bounds"] == [590.0, 610.0]
    assert override_parameters[1]["bounds"] == [0.6, 0.8]


def test_laminar_ax_parameters_require_explicit_bounds_and_include_all_fit_variables(
    tmp_path: Path,
) -> None:
    config = build_laminar_test_config(tmp_path)

    parameters = build_ax_parameters(config)

    assert [parameter["name"] for parameter in parameters] == [
        "period_lpermm",
        "width_to_period_ratio",
        "depth_nm",
        "left_wall_angle_deg",
        "right_wall_angle_deg",
        "top_cap_thickness_nm",
    ]
    assert parameters[0]["bounds"] == [380.0, 420.0]
    with pytest.raises(ValueError, match="period_lpermm_bounds must be provided"):
        LaminarAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            width_to_period_ratio_bounds=ParameterBounds(0.5, 0.85),
            depth_nm_bounds=ParameterBounds(5.0, 30.0),
            left_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
            right_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
            top_cap_thickness_nm_bounds=ParameterBounds(0.0, 2.7),
            evaluation_energies_ev=[150.0],
        )


def test_laminar_ax_parameters_include_roughness_when_optimized(tmp_path: Path) -> None:
    config = build_laminar_test_config(tmp_path)
    roughness_config = LaminarAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        period_lpermm_bounds=ParameterBounds(380.0, 420.0),
        width_to_period_ratio_bounds=ParameterBounds(0.5, 0.85),
        depth_nm_bounds=ParameterBounds(5.0, 30.0),
        left_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
        right_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
        top_cap_thickness_nm_bounds=ParameterBounds(0.0, 2.7),
        optimize_roughness_sigma_nm=True,
        roughness_sigma_nm_bounds=ParameterBounds(0.0, 5.0),
        evaluation_energies_ev=[150.0],
    )

    parameters = build_ax_parameters(roughness_config)

    assert parameters[-1]["name"] == "roughness_sigma_nm"
    assert parameters[-1]["bounds"] == [0.0, 5.0]
    with pytest.raises(ValueError, match="roughness_sigma_nm_bounds must be provided"):
        LaminarAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            period_lpermm_bounds=ParameterBounds(380.0, 420.0),
            width_to_period_ratio_bounds=ParameterBounds(0.5, 0.85),
            depth_nm_bounds=ParameterBounds(5.0, 30.0),
            left_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
            right_wall_angle_deg_bounds=ParameterBounds(1.0, 45.0),
            top_cap_thickness_nm_bounds=ParameterBounds(0.0, 2.7),
            optimize_roughness_sigma_nm=True,
            evaluation_energies_ev=[150.0],
        )


def test_config_validates_discrete_energy_selection(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    discrete_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[250.0, 100.0, 250.0],
    )
    assert discrete_config.evaluation_energies_ev == [100.0, 250.0]

    with pytest.raises(ValueError, match="must be provided and non-empty"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[],
        )
    with pytest.raises(ValueError, match="must be > 0"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[0.0],
        )


def test_config_validates_objective_sem(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    default_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
    )
    assert default_config.objective_sem == pytest.approx(1.0e-6)

    for invalid_sem in [0.0, -1.0, float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="objective_sem must be finite and > 0"):
            BlazedAxConfig(
                initial_grating=config.initial_grating,
                measurement_path=config.measurement_path,
                output_dir=config.output_dir,
                evaluation_energies_ev=[150.0],
                objective_sem=invalid_sem,
            )


def test_initial_laminar_grating_validates_geometry() -> None:
    with pytest.raises(ValueError, match="width_to_period_ratio must be in"):
        InitialLaminarGrating(
            period_lpermm=400.0,
            width_to_period_ratio=1.2,
            depth_nm=14.9,
            left_wall_angle_deg=15.0,
            right_wall_angle_deg=15.0,
        )
    with pytest.raises(ValueError, match="both wall angles must be zero"):
        InitialLaminarGrating(
            period_lpermm=400.0,
            width_to_period_ratio=0.67,
            depth_nm=14.9,
            left_wall_angle_deg=0.0,
            right_wall_angle_deg=15.0,
        )


def test_resolve_grating_parameters_maps_trial_values(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)

    resolved = resolve_grating_parameters(
        config,
        {
            "period_lpermm": 603.0,
            "blaze_angle_deg": 0.8,
            "anti_blaze_angle_deg": 5.9,
            "top_cap_thickness_nm": 1.5,
        },
    )

    assert resolved["period_lpermm"] == pytest.approx(603.0)
    assert resolved["blaze_angle_deg"] == pytest.approx(0.8)
    assert resolved["anti_blaze_angle_deg"] == pytest.approx(5.9)
    assert resolved["top_cap_thickness_nm"] == pytest.approx(1.5)
    grating = build_blazed_grating(config, {"period_lpermm": 603.0})
    assert grating.period_lpermm == pytest.approx(603.0)


def test_resolve_laminar_grating_parameters_maps_trial_values(tmp_path: Path) -> None:
    config = build_laminar_test_config(tmp_path)

    resolved = resolve_grating_parameters(
        config,
        {
            "period_lpermm": 405.0,
            "width_to_period_ratio": 0.7,
            "depth_nm": 16.0,
            "left_wall_angle_deg": 12.0,
            "right_wall_angle_deg": 18.0,
            "top_cap_thickness_nm": 0.8,
        },
    )
    grating = build_laminar_grating(config, resolved)

    assert resolved["period_lpermm"] == pytest.approx(405.0)
    assert resolved["width_to_period_ratio"] == pytest.approx(0.7)
    assert resolved["depth_nm"] == pytest.approx(16.0)
    assert resolved["left_wall_angle_deg"] == pytest.approx(12.0)
    assert resolved["right_wall_angle_deg"] == pytest.approx(18.0)
    assert resolved["top_cap_thickness_nm"] == pytest.approx(0.8)
    assert isinstance(grating, LaminarGrating)


def test_json_safe_grating_parameters_converts_materials_to_labels(tmp_path: Path) -> None:
    config = build_laminar_test_config(tmp_path)
    resolved = resolve_grating_parameters(config, {"period_lpermm": 405.0})

    serializable = json_safe_grating_parameters(resolved)

    assert serializable["substrate_material"] == "Si"
    assert serializable["layer_material"] == "Pt"
    assert serializable["top_cap_material"] == "C"
    assert serializable["period_lpermm"] == pytest.approx(405.0)


def test_resolve_solver_parameters_maps_trial_roughness(tmp_path: Path) -> None:
    config = build_laminar_test_config(tmp_path)

    resolved = resolve_solver_parameters(config, {"roughness_sigma_nm": 0.75})

    assert resolved["roughness_sigma_nm"] == pytest.approx(0.75)


def test_evaluate_trial_is_deterministic_with_mocked_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    def fake_simulate_efficiency_curve(_config, trial_parameters, _measurement):
        base = float(trial_parameters["period_lpermm"]) / 1000.0
        return np.full(measurement.energy_ev.shape, base, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )

    loss_one = evaluate_trial(config, {"period_lpermm": 600.0}, measurement)
    loss_two = evaluate_trial(config, {"period_lpermm": 600.0}, measurement)

    assert loss_one == pytest.approx(loss_two)
    assert isinstance(loss_one, float)


def test_build_evaluation_measurement_uses_discrete_energies(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    discrete_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
    )
    measurement = load_measurement_data(config.measurement_path)
    evaluation = build_evaluation_measurement(discrete_config, measurement)
    assert np.allclose(evaluation.energy_ev, np.array([150.0]))
    assert np.allclose(evaluation.efficiency, np.array([0.25]))


def test_laminar_fixed_angle_objective_uses_constant_grazing_angle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_laminar_test_config(tmp_path, angle_mode="fixed")
    measurement = load_measurement_data(config.measurement_path)
    observed_angles: list[float] = []

    def fake_run_simulation(**kwargs):
        observed_angles.append(float(kwargs["grazing_angle_deg"]))
        return type("Result", (), {"selected_efficiency": 0.25})()

    monkeypatch.setattr("grax_opt.objective.run_simulation", fake_run_simulation)

    simulate_efficiency_curve(config, {"period_lpermm": 400.0}, measurement)

    assert observed_angles == [4.0, 4.0]


def test_laminar_cff_objective_uses_monochromator_grazing_angles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_laminar_test_config(tmp_path, angle_mode="cff")
    measurement = load_measurement_data(config.measurement_path)
    observed_angles: list[float] = []

    def fake_run_simulation(**kwargs):
        observed_angles.append(float(kwargs["grazing_angle_deg"]))
        return type("Result", (), {"selected_efficiency": 0.25})()

    monkeypatch.setattr("grax_opt.objective.run_simulation", fake_run_simulation)
    monkeypatch.setattr(
        "grax_opt.objective.monochromator_grazing_angles_deg",
        lambda *_args, **_kwargs: np.array([1.0, 2.0]),
    )

    simulate_efficiency_curve(config, {"period_lpermm": 400.0}, measurement)

    assert observed_angles == [1.0, 2.0]


def test_objective_passes_trial_roughness_to_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_laminar_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)
    observed_roughness: list[float | None] = []

    def fake_run_simulation(**kwargs):
        observed_roughness.append(kwargs["roughness_sigma_nm"])
        return type("Result", (), {"selected_efficiency": 0.25})()

    monkeypatch.setattr("grax_opt.objective.run_simulation", fake_run_simulation)

    simulate_efficiency_curve(
        config,
        {"period_lpermm": 400.0, "roughness_sigma_nm": 0.8},
        measurement,
    )

    assert observed_roughness == [0.8, 0.8]


def test_evaluate_trial_uses_discrete_energy_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_config(tmp_path)
    discrete_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
    )
    measurement = load_measurement_data(config.measurement_path)
    observed_energy_grid = {"values": None}

    def fake_simulate_efficiency_curve(_config, _trial_parameters, sampled_measurement):
        observed_energy_grid["values"] = np.asarray(sampled_measurement.energy_ev, dtype=float)
        return np.asarray(sampled_measurement.efficiency, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )

    loss = evaluate_trial(discrete_config, {"period_lpermm": 600.0}, measurement)
    assert loss == pytest.approx(0.0)
    assert observed_energy_grid["values"] is not None
    assert np.allclose(observed_energy_grid["values"], np.array([150.0]))


def test_build_ax_optimize_kwargs_uses_configured_objective_sem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)
    sem_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
        objective_sem=2.5e-6,
    )

    monkeypatch.setattr(
        "grax_opt.optimize._import_ax_optimize",
        lambda: lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.evaluate_trial",
        lambda _config, _trial_parameters, _measurement: 0.125,
    )

    kwargs = _build_ax_optimize_kwargs(sem_config, measurement)
    evaluation_result = kwargs["evaluation_function"]({"period_lpermm": 600.0})

    assert evaluation_result == {sem_config.objective_name: (0.125, 2.5e-6)}


def test_optimize_blazed_smoke_run_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ax")
    config = build_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    def fake_simulate_efficiency_curve(_config, trial_parameters, _measurement):
        period = float(trial_parameters["period_lpermm"])
        blaze = float(
            trial_parameters.get("blaze_angle_deg", config.initial_grating.blaze_angle_deg)
        )
        target = 0.25 + 0.001 * (period - 600.0) - 0.01 * (blaze - 0.729)
        return np.full(measurement.energy_ev.shape, target, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )

    result = optimize_blazed(config)

    assert result.result_json_path.exists()
    assert result.trial_history_csv_path.exists()
    assert result.best_fit_plot_path is not None and result.best_fit_plot_path.exists()
    assert len(result.trial_records) == config.total_trials


def test_optimize_laminar_smoke_run_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ax")
    config = build_laminar_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    def fake_simulate_efficiency_curve(_config, trial_parameters, _measurement):
        period = float(trial_parameters["period_lpermm"])
        depth = float(trial_parameters["depth_nm"])
        target = 0.25 + 0.001 * (period - 400.0) - 0.001 * (depth - 14.9)
        return np.full(measurement.energy_ev.shape, target, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )

    result = optimize_laminar(config)

    assert result.result_json_path.exists()
    assert result.trial_history_csv_path.exists()
    assert result.best_fit_plot_path is not None and result.best_fit_plot_path.exists()
    assert len(result.trial_records) == config.total_trials
