"""Fit only roughness using the previously fitted laminar geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd  # noqa: E402

from grax_opt import (  # noqa: E402
    InitialLaminarGrating,
    LaminarAxConfig,
    ParameterBounds,
    json_safe_grating_parameters,
    optimize_laminar,
)
from grax_opt.model import resolve_solver_parameters  # noqa: E402

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
base_fit_dir = example_root / "results" / "laminar_discrete_fit_with_top_layer"
base_fitted_parameters_path = base_fit_dir / "fitted_parameters.json"

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

if not base_fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted geometry file: {base_fitted_parameters_path}. "
        "Run fit_laminar_grating.py first."
    )

base_payload = json.loads(base_fitted_parameters_path.read_text(encoding="utf-8"))
base_grating_parameters = dict(base_payload["best_grating_parameters"])

config = LaminarAxConfig(
    initial_grating=InitialLaminarGrating(
        period_lpermm=float(base_grating_parameters["period_lpermm"]),
        width_to_period_ratio=float(base_grating_parameters["width_to_period_ratio"]),
        depth_nm=float(base_grating_parameters["depth_nm"]),
        left_wall_angle_deg=float(base_grating_parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(base_grating_parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=float(base_grating_parameters["layer_thickness_nm"]),
        top_cap_material=carbon,
        top_cap_thickness_nm=float(base_grating_parameters["top_cap_thickness_nm"]),
        z_resolution_nm=float(base_grating_parameters["z_resolution_nm"]),
        x_resolution_nm=float(base_grating_parameters["x_resolution_nm"]),
    ),
    measurement_path=example_root / "measured_alpha4deg_order1.csv",
    output_dir=example_root / "results" / "laminar_discrete_fit_roughness_only",
    angle_mode=base_payload.get("angle_mode", "fixed"),
    grazing_angle_deg=float(base_payload["grazing_angle_deg"]),
    cff=float(base_payload["cff"]),
    diffraction_order=int(base_payload["diffraction_order"]),
    fourier_orders=int(base_payload["fourier_orders"]),
    backend=base_payload["backend"],
    roughness_sigma_nm=0.5,
    validate_physical_results=True,
    total_trials=20,
    random_seed=7,
    optimize_period_lpermm=False,
    optimize_width_to_period_ratio=False,
    optimize_depth_nm=False,
    optimize_left_wall_angle_deg=False,
    optimize_right_wall_angle_deg=False,
    optimize_top_cap_thickness_nm=False,
    optimize_roughness_sigma_nm=True,
    roughness_sigma_nm_bounds=ParameterBounds(0.0, 5.0),
    experiment_name="laminar_discrete_fit_roughness_only",
    loss_name="mse",
    save_best_fit_plot=True,
    evaluation_energies_ev=base_payload["evaluation_energies_ev"],
)

try:
    result = optimize_laminar(config)
except ImportError as error:
    print(error)
    print("Install the optional optimizer dependency first: `pip install .[opt]`.")
    raise SystemExit(1) from error

fitted_parameters_path = config.output_dir / "fitted_parameters.json"
fitted_parameters_payload = {
    "base_fitted_parameters_path": str(base_fitted_parameters_path),
    "measurement_path": str(config.measurement_path),
    "angle_mode": config.angle_mode,
    "grazing_angle_deg": config.grazing_angle_deg,
    "cff": config.cff,
    "diffraction_order": config.diffraction_order,
    "fourier_orders": config.fourier_orders,
    "backend": config.backend,
    "evaluation_mode": "discrete",
    "evaluation_energies_ev": config.evaluation_energies_ev,
    "best_loss": result.best_loss,
    "best_parameters": result.best_parameters,
    "best_grating_parameters": json_safe_grating_parameters(result.best_grating_parameters),
    "best_solver_parameters": resolve_solver_parameters(config, result.best_parameters),
}
fitted_parameters_path.write_text(
    json.dumps(fitted_parameters_payload, indent=2),
    encoding="utf-8",
)

print(f"Base fitted geometry: {base_fitted_parameters_path}")
print(f"Measurement: {result.measurement_path}")
print(f"Best loss: {result.best_loss:.6g}")
print(f"Best parameters: {result.best_parameters}")
print(f"Fitted parameters JSON: {fitted_parameters_path}")
print(f"Best result JSON: {result.result_json_path}")
print(f"Trial history CSV: {result.trial_history_csv_path}")
if result.best_fit_plot_path is not None:
    print(f"Best-fit plot: {result.best_fit_plot_path}")
