"""Plot measurement, initial-design simulation, and fitted simulation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax as rp

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants" / "old"
results_dir = example_root / "results" / "laminar_fit"
results_dir.mkdir(parents=True, exist_ok=True)

fitted_parameters_path = results_dir / "fitted_parameters.json"
if not fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted parameters file: {fitted_parameters_path}. Run fit_laminar_grating.py first."
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

measurement = pd.read_csv(
    example_root / "measured_alpha4deg_order1.csv",
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

initial_grating = rp.LaminarGrating(
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
)

payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
fitted_grating_parameters = dict(payload["best_grating_parameters"])
fitted_grating_parameters["substrate_material"] = silicon
fitted_grating_parameters["layer_material"] = platinum
fitted_grating_parameters["top_cap_material"] = carbon
fitted_grating = rp.LaminarGrating(**fitted_grating_parameters)

cases_initial = rp.fixed_angle_cases(
    grating=initial_grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
)
cases_fitted = rp.fixed_angle_cases(
    grating=fitted_grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=int(payload.get("fourier_orders", 15)),
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
)

initial_results = list(runner.run_cases(cases_initial))
fitted_results = list(runner.run_cases(cases_fitted))

initial_csv_path = results_dir / "simulated_curve_initial.csv"
fitted_csv_path = results_dir / "simulated_curve_fitted.csv"
comparison_plot_path = results_dir / "laminar_fit_measurement_comparison.png"

rp.write_all_orders_csv(initial_results, initial_csv_path)
rp.write_all_orders_csv(fitted_results, fitted_csv_path)

initial_frame = pd.read_csv(initial_csv_path)
fitted_frame = pd.read_csv(fitted_csv_path)
initial_m1 = initial_frame[initial_frame["order"] == -1].sort_values("energy_ev")
fitted_m1 = fitted_frame[fitted_frame["order"] == -1].sort_values("energy_ev")

figure, axis = plt.subplots(figsize=(11, 7))
axis.plot(measurement["energy_ev"], measurement["efficiency"], "o", label="Measurement", markersize=3.0)
axis.plot(initial_m1["energy_ev"], initial_m1["efficiency"], "-", label="Initial design")
axis.plot(fitted_m1["energy_ev"], fitted_m1["efficiency"], "-", label="Fitted")
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency")
axis.set_title("Laminar fit: measurement vs initial vs fitted")
axis.grid(alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Initial simulation CSV: {initial_csv_path}")
print(f"Fitted simulation CSV: {fitted_csv_path}")
print(f"Comparison plot: {comparison_plot_path}")
