"""Polarization comparison for a single RCWA simulation (s vs p)."""

from __future__ import annotations

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
    x_resolution_nm=1.0,
    z_resolution_nm=0.1,
)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

energy_ev = 200.0
grazing_angle_deg = 4.0
diffraction_order = 1
fourier_orders = 5

result_s = grax.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    polarization="s",
    backend="numba",
)

result_p = grax.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    polarization="p",
    backend="numba",
)

comparison_plot_path = output_dir / "single_simulation_pol_comparison.png"

orders = result_s.orders
efficiency_s = np.asarray(result_s.efficiency_all, dtype=float)
efficiency_p = np.asarray(result_p.efficiency_all, dtype=float)

x = np.arange(len(orders))
width = 0.35
figure, axis = plt.subplots(figsize=(10, 5))
axis.bar(x - width / 2, efficiency_s, width, label="s (TE)", color="tab:blue", alpha=0.8)
axis.bar(x + width / 2, efficiency_p, width, label="p (TM)", color="tab:orange", alpha=0.8)
axis.set_xlabel("Diffraction Order")
axis.set_ylabel("Diffraction Efficiency")
axis.set_title(f"Polarization Comparison — {energy_ev:.0f} eV, {grazing_angle_deg:.1f}° grazing")
axis.set_xticks(x)
axis.set_xticklabels([str(o) for o in orders])
axis.grid(True, alpha=0.3, axis="y")
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Polarization comparison plot saved to: {comparison_plot_path}")
print(f"Order {diffraction_order} efficiency — s: {result_s.selected_efficiency:.6g}, p: {result_p.selected_efficiency:.6g}")
