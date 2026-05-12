"""Ax optimization for the laminar fixed-angle comparison grating."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from grax_opt import (
    InitialLaminarGrating,
    LaminarAxConfig,
    ParameterBounds,
    json_safe_grating_parameters,
    optimize_laminar,
)

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
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

config = LaminarAxConfig(
    initial_grating=InitialLaminarGrating(
        period_lpermm=400.0,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=28.77,
        top_cap_material=carbon,
        top_cap_thickness_nm=0.3,
        z_resolution_nm=0.1,
        x_resolution_nm=0.1,
    ),
    measurement_path=example_root / "measured_alpha4deg_order1.csv",
    output_dir=example_root / "results" / "laminar_discrete_fit_with_top_layer",
    angle_mode="fixed",
    grazing_angle_deg=4.0,
    cff=2.5,
    diffraction_order=1,
    fourier_orders=15,
    solver_backend="s_matrix",
    validate_physical_results=True,
    total_trials=20,
    random_seed=7,
    optimize_period_lpermm=False,
    optimize_width_to_period_ratio=True,
    optimize_depth_nm=True,
    optimize_left_wall_angle_deg=True,
    optimize_right_wall_angle_deg=True,
    optimize_top_cap_thickness_nm=True,
    # period_lpermm_bounds=ParameterBounds(380.0, 420.0),
    width_to_period_ratio_bounds=ParameterBounds(0.0, 1),
    depth_nm_bounds=ParameterBounds(13, 18.0),
    left_wall_angle_deg_bounds=ParameterBounds(1.0, 90.0),
    right_wall_angle_deg_bounds=ParameterBounds(1.0, 90.0),
    top_cap_thickness_nm_bounds=ParameterBounds(0.0, 1),
    experiment_name="laminar_discrete_fit_with_top_layer",
    loss_name="mse",
    save_best_fit_plot=True,
    # evaluation_energies_ev=[100.0, 145.0, 240.0, 300.0],
    evaluation_energies_ev=np.arange(100, 501, 10),
)

try:
    result = optimize_laminar(config)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = config.output_dir / "fitted_parameters.json"
fitted_parameters_payload = {
    "measurement_path": str(config.measurement_path),
    "angle_mode": config.angle_mode,
    "grazing_angle_deg": config.grazing_angle_deg,
    "cff": config.cff,
    "diffraction_order": config.diffraction_order,
    "fourier_orders": config.fourier_orders,
    "solver_backend": config.solver_backend,
    "evaluation_mode": "discrete",
    "evaluation_energies_ev": config.evaluation_energies_ev,
    "best_loss": result.best_loss,
    "best_parameters": result.best_parameters,
    "best_grating_parameters": json_safe_grating_parameters(result.best_grating_parameters),
}
fitted_parameters_path.write_text(
    json.dumps(fitted_parameters_payload, indent=2),
    encoding="utf-8",
)

print(f"Measurement: {result.measurement_path}")
print(f"Best loss: {result.best_loss:.6g}")
print(f"Best parameters: {result.best_parameters}")
print(f"Fitted parameters JSON: {fitted_parameters_path}")
print(f"Best result JSON: {result.result_json_path}")
print(f"Trial history CSV: {result.trial_history_csv_path}")
if result.best_fit_plot_path is not None:
    print(f"Best-fit plot: {result.best_fit_plot_path}")
