"""Optimize a laminar grating against measured fixed-angle data."""

from __future__ import annotations

import json

import pandas as pd

from grax_opt import (
    InitialLaminarGrating,
    LaminarAxConfig,
    ParameterBounds,
    json_safe_grating_parameters,
    optimize_laminar,
)
from example_config import (
    angle_mode,
    batch_size,
    optimizer_backend,
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
    period_lpermm,
    random_seed,
    results_dir,
    right_wall_angle_deg,
    top_cap_thickness_nm,
    total_trials,
    width_to_period_ratio,
    x_resolution_nm,
    z_resolution_nm,
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
results_dir.mkdir(parents=True, exist_ok=True)

config = LaminarAxConfig(
    initial_grating=InitialLaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=width_to_period_ratio,
        depth_nm=depth_nm,
        left_wall_angle_deg=left_wall_angle_deg,
        right_wall_angle_deg=right_wall_angle_deg,
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=top_cap_thickness_nm,
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    ),
    measurement_path=measurement_path,
    output_dir=results_dir,
    angle_mode=angle_mode,
    grazing_angle_deg=grazing_angle_deg,
    cff=cff,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    validate_physical_results=True,
    total_trials=60,
    random_seed=7,
    optimize_period_lpermm=False,
    optimize_width_to_period_ratio=True,
    optimize_depth_nm=True,
    optimize_left_wall_angle_deg=True,
    optimize_right_wall_angle_deg=True,
    optimize_top_cap_thickness_nm=True,
    optimize_roughness_sigma_nm=False,
    width_to_period_ratio_bounds=ParameterBounds(0.5, 0.8),
    depth_nm_bounds=ParameterBounds(14.5, 15.5),
    left_wall_angle_deg_bounds=ParameterBounds(10.0, 25.0),
    right_wall_angle_deg_bounds=ParameterBounds(10.0, 25.0),
    top_cap_thickness_nm_bounds=ParameterBounds(0.3, 1.3),
    experiment_name="laminar_fit",
    loss_name="mse",
    save_best_fit_plot=True,
    evaluation_energies_ev=list(evaluation_energies_ev),
)

try:
    result = optimize_laminar(config)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = results_dir / "fitted_parameters.json"
best_result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
payload = {
    "measurement_path": str(config.measurement_path),
    "angle_mode": config.angle_mode,
    "grazing_angle_deg": config.grazing_angle_deg,
    "cff": config.cff,
    "diffraction_order": config.diffraction_order,
    "fourier_orders": config.fourier_orders,
    "backend": best_result_payload.get("backend_effective", "numba"),
    "backend_requested": config.backend,
    "backend_effective": best_result_payload.get("backend_effective", "numba"),
    "evaluation_energies_ev": config.evaluation_energies_ev,
    "best_loss": result.best_loss,
    "best_parameters": result.best_parameters,
    "best_grating_parameters": json_safe_grating_parameters(result.best_grating_parameters),
    "stopped_early": result.stopped_early,
    "completed_trials": result.completed_trials,
    "early_stop_reason": result.early_stop_reason,
}
fitted_parameters_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"Measurement: {result.measurement_path}")
print(f"Optimizer backend request: {optimizer_backend}")
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
