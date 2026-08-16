"""Monochromator sweep using explicit p polarization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import grax

grating = grax.BlazedGrating(
    period_lpermm=600,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    blaze_angle_deg=0.75,
    anti_blaze_angle_deg=5.597,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

energies_ev = np.arange(50.0, 2000.1, 10)

cases = grax.monochromator_cases(
    grating=grating,
    energies_ev=energies_ev,
    diffraction_order=1,
    cff=2.25,
    polarization="p",
)

parser = argparse.ArgumentParser(description="Monochromator energy sweep")
parser.add_argument(
    "--solver",
    choices=("rcwa", "neviere"),
    default="rcwa",
    help="Electromagnetic solver to run. Both compute every diffraction order; "
    "they differ only in how each layer is crossed in z.",
)
args = parser.parse_args()

runner = grax.BatchSimulationRunner(
    solver=args.solver,
    fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
    backend="numba",
)

results = list(runner.run_cases(cases))

csv_path = output_dir / f"monochromator_all_orders_{args.solver}.csv"
grax.write_all_orders_csv(results, csv_path)

orders_plot_path = output_dir / f"monochromator_orders_1_3_{args.solver}.png"
grax.plot_order_subset(
    results,
    orders_plot_path,
    diffraction_orders=[1, 2, 3],
    title="Monochromator Sweep: Orders 1-3 Efficiency vs Energy",
)

profile_path = output_dir / "monochromator_profile.png"
grating.plot_profile(profile_path)

print(f"Results saved to: {csv_path}")
print(f"Orders 1-3 plot saved to: {orders_plot_path}")
print(f"Profile plot saved to: {profile_path}")
