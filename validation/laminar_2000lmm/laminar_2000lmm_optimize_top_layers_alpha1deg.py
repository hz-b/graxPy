"""Optimize the SiO2 and C4O top-layer thicknesses against alpha = 1 deg measurements."""

from __future__ import annotations

import json
from pathlib import Path

import grax
from grax_opt import load_measurement_data, optimize_to_measurements

from fixed_angle_parameters import (
    DEPTH_NM,
    FOURIER_ORDERS,
    LEFT_WALL_ANGLE_DEG,
    PERIOD_LPERMM,
    RIGHT_WALL_ANGLE_DEG,
    SI_DENSITY_G_CM3,
    SIO2_DENSITY_G_CM3,
    C4O_DENSITY_G_CM3,
    WIDTH_TO_PERIOD_RATIO,
    X_RESOLUTION_NM,
    Z_RESOLUTION_NM,
    create_layered_stack,
    resolve_material,
)

MIN_LAYER_THICKNESS_NM = 0.5
MAX_LAYER_THICKNESS_NM = 7.0
TOTAL_TRIALS = 30
RANDOM_SEED = 7
OPTIMIZER_BACKEND = "auto"

grax.setup_logging(level="INFO", run_id="laminar_2000lmm_optimize_top_layers_alpha1deg")

example_root = Path(__file__).resolve().parent
measurement_path = (
    example_root / "simulation" / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-1deg.dat"
)
output_dir = example_root / "results" / "laminar_2000lmm_optimize_top_layers_alpha1deg"

measurement = load_measurement_data(measurement_path)
output_dir.mkdir(parents=True, exist_ok=True)

silicon = resolve_material(material_name="Si", density_g_cm3=SI_DENSITY_G_CM3)
silicon_o2 = resolve_material(material_name="SiO2", density_g_cm3=SIO2_DENSITY_G_CM3)
c4o = resolve_material(material_name="C4O", density_g_cm3=C4O_DENSITY_G_CM3)


def build_grating(parameters: dict[str, float]) -> grax.LaminarGrating:
    """Build the layered laminar grating for one optimizer trial."""

    layered_stack = create_layered_stack(
        substrate_material=silicon,
        sio2_material=silicon_o2,
        c4o_material=c4o,
        sio2_thickness_nm=float(parameters["sio2_thickness_nm"]),
        c4o_thickness_nm=float(parameters["c4o_thickness_nm"]),
    )
    return grax.LaminarGrating(
        period_lpermm=PERIOD_LPERMM,
        width_to_period_ratio=WIDTH_TO_PERIOD_RATIO,
        depth_nm=DEPTH_NM,
        left_wall_angle_deg=LEFT_WALL_ANGLE_DEG,
        right_wall_angle_deg=RIGHT_WALL_ANGLE_DEG,
        substrate_material=silicon,
        layer_material=silicon,
        layer_thickness_nm=0.0,
        top_cap_material=None,
        top_cap_thickness_nm=0.0,
        coating_stack=layered_stack,
        x_resolution_nm=X_RESOLUTION_NM,
        z_resolution_nm=Z_RESOLUTION_NM,
    )


spec = {
    "build_grating": build_grating,
    "parameter_bounds": {
        "sio2_thickness_nm": (MIN_LAYER_THICKNESS_NM, MAX_LAYER_THICKNESS_NM),
        "c4o_thickness_nm": (MIN_LAYER_THICKNESS_NM, MAX_LAYER_THICKNESS_NM),
    },
    "measurement_path": measurement_path,
    "output_dir": output_dir,
    "angle_mode": "fixed",
    "grazing_angle_deg": 1.0,
    "diffraction_order": 1,
    "fourier_orders": FOURIER_ORDERS,
    "validate_physical_results": True,
    "total_trials": TOTAL_TRIALS,
    "batch_size": 1,
    "random_seed": RANDOM_SEED,
    "experiment_name": "laminar_2000lmm_top_layers_alpha1deg",
    "save_best_fit_plot": True,
    "evaluation_energies_ev": measurement.energy_ev.tolist(),
    "backend": OPTIMIZER_BACKEND,
}

try:
    result = optimize_to_measurements(spec)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `uv pip install --python .venv/bin/python -e '.[opt]'`.")
    raise SystemExit(1) from error

fitted_parameters_path = output_dir / "fitted_parameters.json"
payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
payload["result_json_path"] = str(result.result_json_path)
payload["trial_history_csv_path"] = str(result.trial_history_csv_path)
payload["best_fit_plot_path"] = (
    None if result.best_fit_plot_path is None else str(result.best_fit_plot_path)
)
payload["loss_history_plot_path"] = (
    None if result.loss_history_plot_path is None else str(result.loss_history_plot_path)
)
fitted_parameters_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"Measurement: {result.measurement_path}")
print(f"Optimizer backend request: {OPTIMIZER_BACKEND}")
print(f"Best loss: {result.best_loss:.6g}")
print(f"Best parameters: {result.best_parameters}")
print(f"Completed trials: {result.completed_trials}")
print(f"Stopped early: {result.stopped_early}")
if result.early_stop_reason is not None:
    print(f"Early-stop reason: {result.early_stop_reason}")
print(f"Fitted parameters JSON: {fitted_parameters_path}")
print(f"Best result JSON: {result.result_json_path}")
print(f"Trial history CSV: {result.trial_history_csv_path}")
if result.best_fit_plot_path is not None:
    print(f"Best-fit plot: {result.best_fit_plot_path}")
if result.loss_history_plot_path is not None:
    print(f"Loss-history plot: {result.loss_history_plot_path}")
