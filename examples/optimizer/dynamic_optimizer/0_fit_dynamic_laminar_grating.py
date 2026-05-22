"""Fit a custom laminar grating with measurement-fit parameter bounds and ties."""

from __future__ import annotations

import json

import pandas as pd

from grax import LaminarGrating
from grax_opt import optimize_to_measurements
from example_config import (
    batch_size,
    design_depth_nm,
    design_layer_thickness_nm,
    design_period_lpermm,
    design_top_cap_thickness_nm,
    design_width_to_period_ratio,
    design_x_resolution_nm,
    design_z_resolution_nm,
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    grazing_angle_deg,
    measurement_path,
    optical_constants_dir,
    optimizer_backend,
    random_seed,
    results_dir,
    total_trials,
)

results_dir.mkdir(parents=True, exist_ok=True)

silicon = pd.read_csv(
    optical_constants_dir / "n_Si_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

platinum = pd.read_csv(
    optical_constants_dir / "n_Pt_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

if not measurement_path.exists():
    raise FileNotFoundError(
        f"Missing measurement file: {measurement_path}. "
        "Copy the laminar measurement file into the measurement-fit example directory."
    )


def build_custom_grating(parameters: dict[str, float]) -> LaminarGrating:
    """Build a laminar grating from the resolved measurement-fit parameter set."""

    # Parameters are optimized only if present in parameter_bounds.
    return LaminarGrating(
        period_lpermm=design_period_lpermm,
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=design_layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
        x_resolution_nm=design_x_resolution_nm,
        z_resolution_nm=design_z_resolution_nm,
    )


spec = {
    "build_grating": build_custom_grating,
    "parameter_bounds": {
        "width_to_period_ratio": (0.5, 0.8),
        "depth_nm": (13.9, 15.9),
        "left_wall_angle_deg": (5.0, 20.0),
        "right_wall_angle_deg": (5.0, 20.0),
        "top_cap_thickness_nm": (0.3, 2.0),
    },
    "equality_constraints": {
        "right_wall_angle_deg": "left_wall_angle_deg",
    },
    "measurement_path": measurement_path,
    "output_dir": results_dir,
    "angle_mode": "fixed",
    "grazing_angle_deg": grazing_angle_deg,
    "diffraction_order": diffraction_order,
    "fourier_orders": fourier_orders,
    "total_trials": total_trials,
    "batch_size": batch_size,
    "random_seed": random_seed,
    "evaluation_energies_ev": list(evaluation_energies_ev),
    "evaluation_grazing_angles_deg": [],
    "experiment_name": "measurement_fit_laminar_tied_walls",
    "backend": optimizer_backend,
    "save_best_fit_plot": True,
    "save_loss_plot": True,
}

try:
    result = optimize_to_measurements(spec)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = results_dir / "fitted_parameters.json"
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

print(f"Measurement: {measurement_path}")
print(f"Best loss: {result.best_loss:.6g}")
print(f"Best parameters: {result.best_parameters}")
print(f"Completed trials: {result.completed_trials}")
print(f"Stopped early: {result.stopped_early}")
print(f"Fitted parameters JSON: {fitted_parameters_path}")
print(f"Best result JSON: {result.result_json_path}")
print(f"Trial history CSV: {result.trial_history_csv_path}")
if result.best_fit_plot_path is not None:
    print(f"Best-fit plot: {result.best_fit_plot_path}")
if result.loss_history_plot_path is not None:
    print(f"Loss-history plot: {result.loss_history_plot_path}")
