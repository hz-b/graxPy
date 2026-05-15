from __future__ import annotations

from pathlib import Path
import json
from contextlib import contextmanager
import runpy

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
from grax_opt import optimize as optimize_module
from grax_opt.cli import build_argument_parser
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


def test_blazed_ax_parameters_can_disable_period_optimization(tmp_path: Path) -> None:
    """Ensure period is omitted from Ax parameters when optimization is disabled."""

    config = build_test_config(tmp_path)
    fixed_period_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        optimize_period_lpermm=False,
        optimize_blaze_angle_deg=True,
        evaluation_energies_ev=[150.0],
    )

    parameter_names = [parameter["name"] for parameter in build_ax_parameters(fixed_period_config)]
    assert "period_lpermm" not in parameter_names
    assert "blaze_angle_deg" in parameter_names


def test_blazed_config_requires_at_least_one_optimized_parameter(tmp_path: Path) -> None:
    """Reject blazed configs that disable every optimization parameter."""

    config = build_test_config(tmp_path)
    with pytest.raises(ValueError, match="At least one blazed optimization parameter"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            optimize_period_lpermm=False,
            optimize_blaze_angle_deg=False,
            optimize_anti_blaze_angle_deg=False,
            optimize_top_cap_thickness_nm=False,
            evaluation_energies_ev=[150.0],
        )


def test_cli_can_disable_period_optimization_flag() -> None:
    """Parse --no-optimize-period-lpermm to disable period optimization."""

    parser = build_argument_parser()
    arguments = parser.parse_args(
        [
            "--measurement-path",
            "m.dat",
            "--output-dir",
            "out",
            "--period-lpermm",
            "600",
            "--blaze-angle-deg",
            "0.7",
            "--substrate-optical-constants",
            "si.dat",
            "--layer-optical-constants",
            "au.dat",
            "--no-optimize-period-lpermm",
        ]
    )
    assert arguments.optimize_period_lpermm is False


def test_cli_backend_defaults_to_auto() -> None:
    """Parse backend option with expected default."""

    parser = build_argument_parser()
    arguments = parser.parse_args(
        [
            "--measurement-path",
            "m.dat",
            "--output-dir",
            "out",
            "--period-lpermm",
            "600",
            "--blaze-angle-deg",
            "0.7",
            "--substrate-optical-constants",
            "si.dat",
            "--layer-optical-constants",
            "au.dat",
        ]
    )
    assert arguments.backend == "auto"
    assert arguments.batch_size == 1


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


def test_config_validates_early_stopping_controls(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)

    valid_config = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
        enable_early_stopping=True,
        early_stopping_patience=4,
        early_stopping_min_relative_improvement=1.0e-2,
        early_stopping_warmup_trials=3,
    )
    assert valid_config.save_loss_plot is True

    with pytest.raises(ValueError, match="early_stopping_patience must be > 0"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            early_stopping_patience=0,
        )
    with pytest.raises(
        ValueError,
        match="early_stopping_min_relative_improvement must be finite and >= 0",
    ):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            early_stopping_min_relative_improvement=-1.0,
        )
    with pytest.raises(ValueError, match="early_stopping_warmup_trials must be >= 0"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            early_stopping_warmup_trials=-1,
        )


def test_config_validates_backend_choice(tmp_path: Path) -> None:
    """Accept known backend values and reject unknown ones."""

    config = build_test_config(tmp_path)
    for backend in ["auto", "numba", "numpy"]:
        validated = BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            backend=backend,
        )
        assert validated.backend == backend

    with pytest.raises(ValueError, match="backend must be one of"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            backend="jax",
        )


def test_config_validates_batch_size(tmp_path: Path) -> None:
    """Accept valid batch size and reject non-positive values."""

    config = build_test_config(tmp_path)
    validated = BlazedAxConfig(
        initial_grating=config.initial_grating,
        measurement_path=config.measurement_path,
        output_dir=config.output_dir,
        evaluation_energies_ev=[150.0],
        batch_size=3,
    )
    assert validated.batch_size == 3

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        BlazedAxConfig(
            initial_grating=config.initial_grating,
            measurement_path=config.measurement_path,
            output_dir=config.output_dir,
            evaluation_energies_ev=[150.0],
            batch_size=0,
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

    loss_one = evaluate_trial(config, {"period_lpermm": 600.0}, measurement, backend="numpy")
    loss_two = evaluate_trial(config, {"period_lpermm": 600.0}, measurement, backend="numpy")

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

    simulate_efficiency_curve(config, {"period_lpermm": 400.0}, measurement, backend="numpy")

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

    simulate_efficiency_curve(config, {"period_lpermm": 400.0}, measurement, backend="numpy")

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
        backend="numpy",
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

    def fake_simulate_efficiency_curve(_config, _trial_parameters, sampled_measurement, *, backend):
        observed_energy_grid["values"] = np.asarray(sampled_measurement.energy_ev, dtype=float)
        return np.asarray(sampled_measurement.efficiency, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )

    loss = evaluate_trial(discrete_config, {"period_lpermm": 600.0}, measurement, backend="numpy")
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
        lambda _config, _trial_parameters, _measurement, *, backend: 0.125,
    )

    kwargs = _build_ax_optimize_kwargs(sem_config, measurement)
    evaluation_result = kwargs["evaluation_function"]({"period_lpermm": 600.0})

    assert evaluation_result == {sem_config.objective_name: (0.125, 2.5e-6)}


def test_optimize_blazed_smoke_run_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {"period_lpermm": 600.0, "blaze_angle_deg": 0.729},
                {"period_lpermm": 601.0, "blaze_angle_deg": 0.729},
                {"period_lpermm": 602.0, "blaze_angle_deg": 0.729},
            ]
            self.completed: list[tuple[int, object]] = []

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            trial_index = len(self.completed)
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append((trial_index, raw_data if raw_data is not None else data))

    def fake_simulate_efficiency_curve(_config, trial_parameters, sampled_measurement, *, backend):
        period = float(trial_parameters["period_lpermm"])
        blaze = float(
            trial_parameters.get("blaze_angle_deg", config.initial_grating.blaze_angle_deg)
        )
        target = 0.25 + 0.001 * (period - 600.0) - 0.01 * (blaze - 0.729)
        return np.full(sampled_measurement.energy_ev.shape, target, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize._create_ax_client_for_config",
        lambda _config: FakeAxClient(),
    )

    persist_call_count = {"value": 0}
    original_persist = optimize_module._persist_optimizer_artifacts

    def counting_persist(**kwargs):
        persist_call_count["value"] += 1
        return original_persist(**kwargs)

    monkeypatch.setattr("grax_opt.optimize._persist_optimizer_artifacts", counting_persist)

    result = optimize_blazed(config)

    assert result.result_json_path.exists()
    assert result.trial_history_csv_path.exists()
    assert result.best_fit_plot_path is not None and result.best_fit_plot_path.exists()
    assert result.loss_history_plot_path is not None and result.loss_history_plot_path.exists()
    assert len(result.trial_records) == config.total_trials
    assert result.completed_trials == config.total_trials
    assert result.stopped_early is False
    assert persist_call_count["value"] == config.total_trials

    result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert result_payload["completed_trials"] == config.total_trials
    assert result_payload["stopped_early"] is False


def test_optimize_laminar_smoke_run_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_laminar_test_config(tmp_path)
    measurement = load_measurement_data(config.measurement_path)

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {
                    "period_lpermm": 400.0,
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 14.9,
                    "left_wall_angle_deg": 15.0,
                    "right_wall_angle_deg": 15.0,
                    "top_cap_thickness_nm": 0.3,
                },
                {
                    "period_lpermm": 401.0,
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 15.1,
                    "left_wall_angle_deg": 15.0,
                    "right_wall_angle_deg": 15.0,
                    "top_cap_thickness_nm": 0.3,
                },
                {
                    "period_lpermm": 402.0,
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 15.2,
                    "left_wall_angle_deg": 15.0,
                    "right_wall_angle_deg": 15.0,
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

    def fake_simulate_efficiency_curve(_config, trial_parameters, sampled_measurement, *, backend):
        period = float(trial_parameters["period_lpermm"])
        depth = float(trial_parameters["depth_nm"])
        target = 0.25 + 0.001 * (period - 400.0) - 0.001 * (depth - 14.9)
        return np.full(sampled_measurement.energy_ev.shape, target, dtype=float)

    monkeypatch.setattr(
        "grax_opt.objective.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        fake_simulate_efficiency_curve,
    )
    monkeypatch.setattr(
        "grax_opt.optimize._create_ax_client_for_config",
        lambda _config: FakeAxClient(),
    )

    result = optimize_laminar(config)

    assert result.result_json_path.exists()
    assert result.trial_history_csv_path.exists()
    assert result.best_fit_plot_path is not None and result.best_fit_plot_path.exists()
    assert result.loss_history_plot_path is not None and result.loss_history_plot_path.exists()
    assert len(result.trial_records) == config.total_trials
    assert result.completed_trials == config.total_trials
    assert result.stopped_early is False


def test_optimize_laminar_early_stopping_stops_on_plateau(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = build_laminar_test_config(tmp_path)
    config = LaminarAxConfig(
        initial_grating=base_config.initial_grating,
        measurement_path=base_config.measurement_path,
        output_dir=base_config.output_dir,
        angle_mode=base_config.angle_mode,
        grazing_angle_deg=base_config.grazing_angle_deg,
        cff=base_config.cff,
        total_trials=6,
        period_lpermm_bounds=base_config.period_lpermm_bounds,
        width_to_period_ratio_bounds=base_config.width_to_period_ratio_bounds,
        depth_nm_bounds=base_config.depth_nm_bounds,
        left_wall_angle_deg_bounds=base_config.left_wall_angle_deg_bounds,
        right_wall_angle_deg_bounds=base_config.right_wall_angle_deg_bounds,
        top_cap_thickness_nm_bounds=base_config.top_cap_thickness_nm_bounds,
        evaluation_energies_ev=base_config.evaluation_energies_ev,
        enable_early_stopping=True,
        early_stopping_warmup_trials=1,
        early_stopping_patience=2,
        early_stopping_min_relative_improvement=1.0e-2,
    )

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {
                    "period_lpermm": 400.0 + float(index),
                    "width_to_period_ratio": 0.67,
                    "depth_nm": 14.9,
                    "left_wall_angle_deg": 15.0,
                    "right_wall_angle_deg": 15.0,
                    "top_cap_thickness_nm": 0.3,
                }
                for index in range(6)
            ]
            self.completed: list[tuple[int, object]] = []

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            trial_index = len(self.completed)
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append((trial_index, raw_data if raw_data is not None else data))

    loss_values = iter([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])

    monkeypatch.setattr(
        "grax_opt.optimize._create_ax_client_for_config",
        lambda _config: FakeAxClient(),
    )
    monkeypatch.setattr(
        "grax_opt.optimize.evaluate_trial",
        lambda _config, _trial_parameters, _measurement, *, backend: next(loss_values),
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        lambda _config, _trial_parameters, sampled_measurement, *, backend: np.full(
            sampled_measurement.energy_ev.shape,
            0.2,
            dtype=float,
        ),
    )

    result = optimize_laminar(config)

    assert result.stopped_early is True
    assert result.completed_trials == 3
    assert len(result.trial_records) == 3
    assert result.early_stop_reason is not None
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["stopped_early"] is True
    assert payload["completed_trials"] == 3


def test_cpu_only_fork_rng_patch_avoids_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_fork_rng = torch.random.fork_rng

    @contextmanager
    def sentinel_fork_rng(*args, **kwargs):
        assert kwargs.get("devices") == []
        yield

    monkeypatch.setattr(optimize_module, "_is_cuda_usable", lambda: False)
    monkeypatch.setattr(torch.random, "fork_rng", sentinel_fork_rng)

    try:
        optimize_module._patch_torch_fork_rng_for_cpu_only()

        with torch.random.fork_rng():
            pass
    finally:
        torch.random.fork_rng = original_fork_rng


def test_build_optimizer_compute_banner_formats_output() -> None:
    gpu_banner = optimize_module._build_optimizer_compute_banner(
        mode="GPU",
        model="NVIDIA RTX A4000",
        torch_version="2.12.0+cu130",
        torch_cuda_version="13.0",
    )
    cpu_banner = optimize_module._build_optimizer_compute_banner(
        mode="CPU",
        model="Intel(R) Xeon(R)",
        torch_version="2.12.0+cu130",
        torch_cuda_version="13.0",
    )

    assert gpu_banner == (
        "Optimizer compute: GPU | model=NVIDIA RTX A4000 | torch=2.12.0+cu130 | cuda=13.0"
    )
    assert cpu_banner == (
        "Optimizer compute: CPU | model=Intel(R) Xeon(R) | torch=2.12.0+cu130 | cuda=13.0"
    )


def test_create_ax_client_prints_compute_context_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = build_laminar_test_config(tmp_path)

    class FakeAxClient:
        def __init__(self, **_kwargs) -> None:
            self.created = False

        def create_experiment(self, objective_name, **_kwargs) -> None:
            self.created = True

    monkeypatch.setattr(optimize_module, "_patch_torch_fork_rng_for_cpu_only", lambda: None)
    monkeypatch.setattr(optimize_module, "_describe_optimizer_compute_context", lambda: "BANNER")
    monkeypatch.setattr(optimize_module, "_import_ax_client", lambda: FakeAxClient)

    ax_client = optimize_module._create_ax_client_for_config(config)

    captured = capsys.readouterr()
    assert captured.out.count("BANNER") == 1
    assert isinstance(ax_client, FakeAxClient)


def test_resolve_optimizer_backend_auto_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve optimizer backend from policy and numba availability."""

    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: True)
    assert optimize_module._resolve_optimizer_backend("auto") == "numba"
    assert optimize_module._resolve_optimizer_backend("numba") == "numba"
    assert optimize_module._resolve_optimizer_backend("numpy") == "numpy"

    monkeypatch.setattr(optimize_module, "_is_numba_available", lambda: False)
    assert optimize_module._resolve_optimizer_backend("auto") == "numpy"
    assert optimize_module._resolve_optimizer_backend("numba") == "numpy"


def test_example_configs_split_optimizer_and_simulation_backends() -> None:
    """Ensure examples use auto only for optimizer config, not simulation runners."""

    repo_root = Path(__file__).resolve().parents[1]
    blazed_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_blazed" / "example_config.py")
    )
    laminar_config = runpy.run_path(
        str(repo_root / "examples" / "optimizer" / "optimizer_grating" / "example_config.py")
    )

    assert blazed_config["optimizer_backend"] == "auto"
    assert blazed_config["simulation_backend"] in {"numba", "numpy"}
    assert blazed_config["simulation_backend"] != "auto"

    assert laminar_config["optimizer_backend"] == "auto"
    assert laminar_config["simulation_backend"] in {"numba", "numpy"}
    assert laminar_config["simulation_backend"] != "auto"


def test_optimize_blazed_batch_mode_groups_trials_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run batched optimizer loop and verify ordering/completion semantics."""

    base = build_test_config(tmp_path)
    config = BlazedAxConfig(
        initial_grating=base.initial_grating,
        measurement_path=base.measurement_path,
        output_dir=base.output_dir,
        total_trials=5,
        batch_size=3,
        optimize_blaze_angle_deg=True,
        evaluation_energies_ev=[150.0],
    )

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {"period_lpermm": 600.0, "blaze_angle_deg": 0.70},
                {"period_lpermm": 601.0, "blaze_angle_deg": 0.71},
                {"period_lpermm": 602.0, "blaze_angle_deg": 0.72},
                {"period_lpermm": 603.0, "blaze_angle_deg": 0.73},
                {"period_lpermm": 604.0, "blaze_angle_deg": 0.74},
            ]
            self.completed: list[int] = []
            self.issued = 0

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            trial_index = self.issued
            self.issued += 1
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append(int(trial_index))

    issued_candidates: list[list[int]] = []

    def fake_evaluate_batch(candidates, *, config, measurement, backend_effective):
        issued_candidates.append([int(index) for index, _ in candidates])
        evaluated = []
        for trial_index, parameters in candidates:
            evaluated.append((int(trial_index), dict(parameters), float(10 - trial_index)))
        return sorted(evaluated, key=lambda item: int(item[0]))

    monkeypatch.setattr("grax_opt.optimize._create_ax_client_for_config", lambda _config: FakeAxClient())
    monkeypatch.setattr("grax_opt.optimize._evaluate_candidate_batch", fake_evaluate_batch)
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        lambda _config, _trial_parameters, sampled_measurement, *, backend: np.full(
            sampled_measurement.energy_ev.shape,
            0.2,
            dtype=float,
        ),
    )

    result = optimize_blazed(config)

    assert result.completed_trials == 5
    assert [record.trial_index for record in result.trial_records] == [0, 1, 2, 3, 4]
    assert issued_candidates == [[0, 1, 2], [3, 4]]


def test_optimize_blazed_batch_mode_clamps_on_ax_max_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure optimizer continues with partial batches when Ax blocks generation."""

    base = build_test_config(tmp_path)
    config = BlazedAxConfig(
        initial_grating=base.initial_grating,
        measurement_path=base.measurement_path,
        output_dir=base.output_dir,
        total_trials=5,
        batch_size=3,
        optimize_blaze_angle_deg=True,
        evaluation_energies_ev=[150.0],
    )

    class FakeMaxParallelismReachedException(Exception):
        """Fake Ax max parallelism exception."""

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {"period_lpermm": 600.0, "blaze_angle_deg": 0.70},
                {"period_lpermm": 601.0, "blaze_angle_deg": 0.71},
                {"period_lpermm": 602.0, "blaze_angle_deg": 0.72},
                {"period_lpermm": 603.0, "blaze_angle_deg": 0.73},
                {"period_lpermm": 604.0, "blaze_angle_deg": 0.74},
            ]
            self.issued = 0
            self.completed: list[int] = []
            self.max_running = 2

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            if (self.issued - len(self.completed)) >= self.max_running:
                raise FakeMaxParallelismReachedException("parallelism cap")
            trial_index = self.issued
            self.issued += 1
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append(int(trial_index))

    monkeypatch.setattr("grax_opt.optimize._create_ax_client_for_config", lambda _config: FakeAxClient())
    monkeypatch.setattr(
        "grax_opt.optimize._import_max_parallelism_exception",
        lambda: FakeMaxParallelismReachedException,
    )
    monkeypatch.setattr(
        "grax_opt.optimize.evaluate_trial",
        lambda _config, trial_parameters, _measurement, *, backend: float(
            trial_parameters["period_lpermm"] - 590.0
        ),
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        lambda _config, _trial_parameters, sampled_measurement, *, backend: np.full(
            sampled_measurement.energy_ev.shape,
            0.2,
            dtype=float,
        ),
    )

    result = optimize_blazed(config)

    assert result.completed_trials == 5
    assert [record.trial_index for record in result.trial_records] == [0, 1, 2, 3, 4]
    assert result.stopped_early is False


def test_optimize_blazed_batch_mode_clamps_on_ax_data_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure optimizer continues with partial batches on Ax data-required blocks."""

    base = build_test_config(tmp_path)
    config = BlazedAxConfig(
        initial_grating=base.initial_grating,
        measurement_path=base.measurement_path,
        output_dir=base.output_dir,
        total_trials=5,
        batch_size=3,
        optimize_blaze_angle_deg=True,
        evaluation_energies_ev=[150.0],
    )

    class FakeDataRequiredError(Exception):
        """Fake Ax data-required exception."""

    class FakeAxClient:
        def __init__(self) -> None:
            self._parameter_sets = [
                {"period_lpermm": 600.0, "blaze_angle_deg": 0.70},
                {"period_lpermm": 601.0, "blaze_angle_deg": 0.71},
                {"period_lpermm": 602.0, "blaze_angle_deg": 0.72},
                {"period_lpermm": 603.0, "blaze_angle_deg": 0.73},
                {"period_lpermm": 604.0, "blaze_angle_deg": 0.74},
            ]
            self.issued = 0
            self.completed: list[int] = []
            self.max_running = 2

        def create_experiment(self, **_kwargs) -> None:
            return None

        def get_next_trial(self):
            if (self.issued - len(self.completed)) >= self.max_running:
                raise FakeDataRequiredError("data required for next node")
            trial_index = self.issued
            self.issued += 1
            return self._parameter_sets[trial_index], trial_index

        def complete_trial(self, trial_index: int, raw_data=None, data=None) -> None:
            self.completed.append(int(trial_index))

    monkeypatch.setattr("grax_opt.optimize._create_ax_client_for_config", lambda _config: FakeAxClient())
    monkeypatch.setattr("grax_opt.optimize._import_data_required_exception", lambda: FakeDataRequiredError)
    monkeypatch.setattr(
        "grax_opt.optimize.evaluate_trial",
        lambda _config, trial_parameters, _measurement, *, backend: float(
            trial_parameters["period_lpermm"] - 590.0
        ),
    )
    monkeypatch.setattr(
        "grax_opt.optimize.simulate_efficiency_curve",
        lambda _config, _trial_parameters, sampled_measurement, *, backend: np.full(
            sampled_measurement.energy_ev.shape,
            0.2,
            dtype=float,
        ),
    )

    result = optimize_blazed(config)

    assert result.completed_trials == 5
    assert [record.trial_index for record in result.trial_records] == [0, 1, 2, 3, 4]
    assert result.stopped_early is False
