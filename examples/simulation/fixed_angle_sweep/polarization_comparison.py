"""Polarization comparison for fixed-angle energy sweep (s vs p)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import grax

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    x_resolution_nm=0.5,
    z_resolution_nm=0.5,
)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

grazing_angle_deg = 4.0
energies_ev = np.arange(50.0, 650.0, 10)
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

if __name__ == "__main__":
    runner = grax.BatchSimulationRunner(
        solver=args.solver,
        diffraction_order=1,
        fourier_orders=15,
        show_progress=True,
        live_plot=True,
        live_plot_x_key="energy_ev",
        max_workers="auto",
        on_error="continue",
        backend="numba",
    )

    results_s = list(runner.run_cases(
        grax.fixed_angle_cases(
            grating=grating,
            energies_ev=energies_ev,
            grazing_angle_deg=grazing_angle_deg,
            polarization="s",
        )
    ))
    results_p = list(runner.run_cases(
        grax.fixed_angle_cases(
            grating=grating,
            energies_ev=energies_ev,
            grazing_angle_deg=grazing_angle_deg,
            polarization="p",
        )
    ))


    def sorted_ok(results):
        return sorted([r for r in results if r.status == "ok"], key=lambda r: float(r.energy_ev))


    collected_s = sorted_ok(results_s)
    collected_p = sorted_ok(results_p)
    energies = np.asarray([r.energy_ev for r in collected_s], dtype=float)

    comparison_plot_path = output_dir / f"fixed_angle_pol_comparison_{args.solver}.png"
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
    axis.set_title("Fixed-Angle Sweep: s vs p Polarization, Orders 1–3")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Polarization comparison plot saved to: {comparison_plot_path}")
