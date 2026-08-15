"""Polarization comparison for batch depth-sweep (s vs p)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import grax

base_grating_kwargs = dict(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energy_ev = 1000.0
diffraction_order = 1
cff = 2.25
depths_nm = np.arange(10.0, 31.0, 1.0)
output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

grazing_angle_deg = float(
    grax.monochromator_grazing_angles_deg(
        [energy_ev],
        period_lpermm=base_grating_kwargs["period_lpermm"],
        diffraction_order=diffraction_order,
        cff=cff,
    )[0]
)

cases_s = []
cases_p = []
for depth_nm in depths_nm:
    grating = grax.LaminarGrating(depth_nm=float(depth_nm), **base_grating_kwargs)
    base = {
        "grating": grating,
        "energy_ev": energy_ev,
        "grazing_angle_deg": grazing_angle_deg,
        "diffraction_order": diffraction_order,
    }
    cases_s.append({**base, "case_id": f"pol-s-depth-{int(depth_nm):03d}", "polarization": "s"})
    cases_p.append({**base, "case_id": f"pol-p-depth-{int(depth_nm):03d}", "polarization": "p"})

runner = grax.BatchSimulationRunner(
    diffraction_order=diffraction_order,
    fourier_orders=25,
    show_progress=True,
    live_plot=False,
    on_error="continue",
    backend="numba",
)

results_s_by_id = {r.case_id: r for r in runner.run_cases(cases_s)}
results_p_by_id = {r.case_id: r for r in runner.run_cases(cases_p)}

ordered_s = [results_s_by_id[f"pol-s-depth-{int(d):03d}"] for d in depths_nm]
ordered_p = [results_p_by_id[f"pol-p-depth-{int(d):03d}"] for d in depths_nm]

diffraction_orders = [1, 2, 3]
colors = ["tab:blue", "tab:orange", "tab:green"]
markers = ["o", "s", "^"]

comparison_plot_path = output_dir / "batch_user_cases_pol_comparison.png"
figure, axis = plt.subplots(figsize=(10, 6))
for index, order in enumerate(diffraction_orders):
    eff_s = np.asarray(
        [grax.efficiency_for_order(r.orders, r.efficiency_all, diffraction_order=order) for r in ordered_s],
        dtype=float,
    )
    eff_p = np.asarray(
        [grax.efficiency_for_order(r.orders, r.efficiency_all, diffraction_order=order) for r in ordered_p],
        dtype=float,
    )
    axis.plot(depths_nm, eff_s, f"{markers[index]}-", color=colors[index], linewidth=1.5, markersize=4, label=f"Order {order} (s)")
    axis.plot(depths_nm, eff_p, f"{markers[index]}--", color=colors[index], linewidth=1.5, markersize=4, label=f"Order {order} (p)")

axis.set_xlabel("Grating Depth (nm)")
axis.set_ylabel("Diffraction Efficiency")
axis.set_title(f"Depth Sweep: s vs p Polarization, Orders 1–3 at {energy_ev:.0f} eV")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Polarization comparison plot saved to: {comparison_plot_path}")
print(f"Energy: {energy_ev:.1f} eV, grazing angle (cff={cff}): {grazing_angle_deg:.6f} deg")
