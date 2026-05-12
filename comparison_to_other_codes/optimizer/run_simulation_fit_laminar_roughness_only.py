"""Laminar fixed-angle sweep using geometry fit plus roughness-only fit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
quick_mode = "--quick" in sys.argv
LIVE_PLOT = False
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "grax-matplotlib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import grax as rp  # noqa: E402

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
fit_output_dir = example_root / "results" / "laminar_discrete_fit_roughness_only"
fitted_parameters_path = fit_output_dir / "fitted_parameters.json"
simulation_csv_path = fit_output_dir / "simulated_curve.csv"
simulation_plot_path = fit_output_dir / "simulated_curve.png"
profile_plot_path = fit_output_dir / "simulated_profile.png"

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

if not fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted parameters file: {fitted_parameters_path}. "
        "Run fit_laminar_roughness_only.py first."
    )

fitted_payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
grating_parameters = dict(fitted_payload["best_grating_parameters"])
solver_parameters = dict(fitted_payload.get("best_solver_parameters", {}))
roughness_sigma_nm = solver_parameters.get(
    "roughness_sigma_nm",
    fitted_payload["best_parameters"].get("roughness_sigma_nm"),
)
grating_parameters["substrate_material"] = silicon
grating_parameters["layer_material"] = platinum
grating_parameters["top_cap_material"] = carbon
if quick_mode:
    grating_parameters["z_resolution_nm"] = 2.0
    grating_parameters["x_resolution_nm"] = 10.0
grating = rp.LaminarGrating(**grating_parameters)

angle_mode = fitted_payload.get("angle_mode", "fixed")
diffraction_order = int(fitted_payload["diffraction_order"])
fourier_orders = 5 if quick_mode else int(fitted_payload["fourier_orders"])
solver_backend = fitted_payload["solver_backend"]
energies = (
    np.asarray([100.0, 300.0, 600.0], dtype=float)
    if quick_mode
    else np.arange(50.0, 2000.1, 10.0)
)
if angle_mode == "fixed":
    cases = rp.fixed_angle_cases(
        grating=grating,
        energies_ev=energies,
        grazing_angle_deg=float(fitted_payload["grazing_angle_deg"]),
        case_id_prefix="case",
        label="fixed-angle",
        roughness_sigma_nm=roughness_sigma_nm,
        solver_backend=solver_backend,
    )
else:
    cases = rp.monochromator_cases(
        grating=grating,
        energies_ev=energies,
        diffraction_order=diffraction_order,
        cff=float(fitted_payload["cff"]),
        case_id_prefix="case",
        label="monochromator",
        roughness_sigma_nm=roughness_sigma_nm,
        solver_backend=solver_backend,
    )

runner = rp.BatchSimulationRunner(
    default_diffraction_order=diffraction_order,
    default_fourier_orders=fourier_orders,
    show_progress=True,
    live_plot=LIVE_PLOT,
    live_plot_x_key="energy_ev",
    live_plot_order_count=1,
    on_error="fail_fast",
    resume=not quick_mode,
)
grating.plot_profile(profile_plot_path)
batch_result = list(runner.run_cases(cases))
rp.write_all_orders_csv(batch_result, simulation_csv_path)
rp.plot_order_subset(
    batch_result,
    simulation_plot_path,
    diffraction_orders=[diffraction_order],
    title="Laminar Simulation from Fitted Parameters With Roughness",
)

print(f"Loaded fitted parameters from: {fitted_parameters_path}")
print(f"Roughness sigma: {roughness_sigma_nm} nm")
print(f"Computed {sum(case.status == 'ok' for case in batch_result)} energy points.")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Simulation CSV saved to: {simulation_csv_path}")
print(f"Simulation plot saved to: {simulation_plot_path}")
