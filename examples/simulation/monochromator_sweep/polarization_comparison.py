"""Polarization comparison for monochromator sweep (s vs p)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
diffraction_orders = [1, 2, 3]

parser = argparse.ArgumentParser(description="Polarization comparison (s vs p)")
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
    live_plot=False,
    on_error="continue",
    backend="numba",
)

results_s = list(runner.run_cases(
    grax.monochromator_cases(
        grating=grating,
        energies_ev=energies_ev,
        diffraction_order=1,
        cff=2.25,
        polarization="s",
    )
))
results_p = list(runner.run_cases(
    grax.monochromator_cases(
        grating=grating,
        energies_ev=energies_ev,
        diffraction_order=1,
        cff=2.25,
        polarization="p",
    )
))


def sorted_ok(results):
    return sorted([r for r in results if r.status == "ok"], key=lambda r: float(r.energy_ev))


collected_s = sorted_ok(results_s)
collected_p = sorted_ok(results_p)
energies = np.asarray([r.energy_ev for r in collected_s], dtype=float)

comparison_plot_path = output_dir / f"monochromator_pol_comparison_{args.solver}.png"
colors = ["tab:blue", "tab:orange", "tab:green"]

figure, axis = plt.subplots(figsize=(10, 6))
for index, order in enumerate(diffraction_orders):
    eff_s = np.asarray(
        [grax.efficiency_for_order(r.orders, r.efficiency_all, diffraction_order=order) for r in collected_s],
        dtype=float,
    )
    eff_p = np.asarray(
        [grax.efficiency_for_order(r.orders, r.efficiency_all, diffraction_order=order) for r in collected_p],
        dtype=float,
    )
    axis.plot(energies, eff_s, "-", color=colors[index], linewidth=1.5, markersize=2, label=f"Order {order} (s)")
    axis.plot(energies, eff_p, "--", color=colors[index], linewidth=1.5, markersize=2, label=f"Order {order} (p)")

axis.set_xlabel("Photon Energy (eV)")
axis.set_ylabel("Diffraction Efficiency")
axis.set_title("Monochromator Sweep: s vs p Polarization, Orders 1–3")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Polarization comparison plot saved to: {comparison_plot_path}")
