"""Optimize a blazed grating against measured monochromator data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grax_opt import (
    BlazedAxConfig,
    InitialBlazedGrating,
    ParameterBounds,
    json_safe_grating_parameters,
    optimize_blazed,
)

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"

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

measurement_path = example_root / "GR600-BEIChem_energy-Cff2.5.dat"
measurement = pd.read_csv(
    measurement_path,
    sep=r"\s+",
    header=None,
    names=["energy_ev", "efficiency"],
).apply(pd.to_numeric, errors="coerce").dropna()

output_dir = example_root / "results" / "blazed_fit"

config = BlazedAxConfig(
    initial_grating=InitialBlazedGrating(
        period_lpermm=600.0,
        blaze_angle_deg=0.729,
        anti_blaze_angle_deg=5.597,
        substrate_material=silicon,
        layer_material=gold,
        layer_thickness_nm=30.0,
        top_cap_material=carbon,
        top_cap_thickness_nm=0.7,
        z_resolution_nm=.3,
        x_resolution_nm=.3,
    ),
    measurement_path=measurement_path,
    output_dir=output_dir,
    cff=2.25,
    diffraction_order=1,
    fourier_orders=5,
    validate_physical_results=True,
    total_trials=50,
    batch_size=15,
    random_seed=7,
    backend="auto",
    optimize_period_lpermm=False,
    optimize_blaze_angle_deg=True,
    optimize_anti_blaze_angle_deg=True,
    optimize_top_cap_thickness_nm=True,
    # period_lpermm_bounds=ParameterBounds(600.0, 610.0),
    blaze_angle_deg_bounds=ParameterBounds(0.5, 1.2),
    anti_blaze_angle_deg_bounds=ParameterBounds(5.0, 6.0),
    top_cap_thickness_nm_bounds=ParameterBounds(0.5, 1.2),
    experiment_name="blazed_fit",
    loss_name="mse",
    save_best_fit_plot=True,
    evaluation_energies_ev=np.arange(51,1800.1, 10),
)

try:
    result = optimize_blazed(config)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = output_dir / "fitted_parameters.json"
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
