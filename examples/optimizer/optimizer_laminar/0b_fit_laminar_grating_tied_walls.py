"""Optimize a laminar grating against measured data with tied wall angles."""

from __future__ import annotations

import json

import pandas as pd
from grax import LaminarGrating

from grax_opt import optimize_to_measurements
from example_config import (
    angle_mode,
    batch_size,
    cff,
    depth_nm,
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    grazing_angle_deg,
    layer_thickness_nm,
    left_wall_angle_deg,
    measurement_path,
    optical_constants_dir,
    optimizer_backend,
    optimizer_max_workers,
    period_lpermm,
    tied_wall_results_dir,
    right_wall_angle_deg,
    random_seed,
    top_cap_thickness_nm,
    total_trials,
    width_to_period_ratio,
    x_resolution_nm,
    z_resolution_nm,
    tied_wall_equality_constraints,
    tied_wall_experiment_name,
)

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
tied_wall_results_dir.mkdir(parents=True, exist_ok=True)


def build_grating(parameters: dict[str, float]) -> LaminarGrating:
    """Build laminar grating from the tied-wall measurement-fit parameters."""

    return LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


spec = {
    "build_grating": build_grating,
    "parameter_bounds": {
        "width_to_period_ratio": (0.5, 0.8),
        "depth_nm": (13.9, 15.9),
        "left_wall_angle_deg": (5.0, 20.0),
        "right_wall_angle_deg": (5.0, 20.0),
        "top_cap_thickness_nm": (0.3, 2.0),
    },
    "equality_constraints": tied_wall_equality_constraints,
    "measurement_path": measurement_path,
    "output_dir": tied_wall_results_dir,
    "angle_mode": angle_mode,
    "grazing_angle_deg": grazing_angle_deg,
    "cff": cff,
    "diffraction_order": diffraction_order,
    "fourier_orders": fourier_orders,
    "validate_physical_results": True,
    "total_trials": total_trials,
    "batch_size": batch_size,
    "random_seed": random_seed,
    "experiment_name": tied_wall_experiment_name,
    "save_best_fit_plot": True,
    "evaluation_energies_ev": list(evaluation_energies_ev),
    "backend": optimizer_backend,
    "max_workers": optimizer_max_workers,
}

try:
    result = optimize_to_measurements(spec)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = tied_wall_results_dir / "fitted_parameters.json"
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
print(f"Optimizer backend request: {optimizer_backend}")
print(f"Optimizer max_workers request: {optimizer_max_workers}")
print(f"Batch size: {batch_size}")
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
