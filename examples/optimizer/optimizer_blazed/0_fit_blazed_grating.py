"""Optimize a blazed grating against measured monochromator data."""

from __future__ import annotations

import json

import pandas as pd

from grax_opt import (
    BlazedAxConfig,
    InitialBlazedGrating,
    ParameterBounds,
    json_safe_grating_parameters,
    optimize_blazed,
)
from example_config import (
    anti_blaze_angle_deg,
    optimizer_backend,
    batch_size,
    blaze_angle_deg,
    cff,
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    layer_thickness_nm,
    measurement_path,
    optical_constants_dir,
    period_lpermm,
    random_seed,
    results_dir,
    top_cap_thickness_nm,
    top_cap_material_name,
    total_trials,
    use_top_cap,
    x_resolution_nm,
    z_resolution_nm,
)

silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

gold = pd.read_csv(
    optical_constants_dir / "OC_Au_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"
results_dir.mkdir(parents=True, exist_ok=True)
top_cap_material = carbon if use_top_cap else None

config = BlazedAxConfig(
    initial_grating=InitialBlazedGrating(
        period_lpermm=period_lpermm,
        blaze_angle_deg=blaze_angle_deg,
        anti_blaze_angle_deg=anti_blaze_angle_deg,
        substrate_material=silicon,
        layer_material=gold,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=top_cap_material,
        top_cap_thickness_nm=top_cap_thickness_nm,
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    ),
    measurement_path=measurement_path,
    output_dir=results_dir,
    cff=cff,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    validate_physical_results=True,
    total_trials=total_trials,
    batch_size=batch_size,
    random_seed=random_seed,
    backend=optimizer_backend,
    optimize_period_lpermm=False,
    optimize_blaze_angle_deg=True,
    optimize_anti_blaze_angle_deg=True,
    optimize_top_cap_thickness_nm=False,
    # period_lpermm_bounds=ParameterBounds(600.0, 610.0),
    blaze_angle_deg_bounds=ParameterBounds(0.5, 1.2),
    anti_blaze_angle_deg_bounds=ParameterBounds(5.0, 6.0),
    top_cap_thickness_nm_bounds=ParameterBounds(0.0, 1.2),
    experiment_name="blazed_fit",
    loss_name="mse",
    save_best_fit_plot=True,
    evaluation_energies_ev=list(evaluation_energies_ev),
)

try:
    result = optimize_blazed(config)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = results_dir / "fitted_parameters.json"
best_result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
payload = {
    "measurement_path": str(config.measurement_path),
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
print(
    f"Baseline top-cap setting: use_top_cap={use_top_cap}, "
    f"material={top_cap_material_name if use_top_cap else 'None'}, "
    f"thickness_nm={top_cap_thickness_nm}"
)
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
