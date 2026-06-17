"""Fixed-angle energy sweep using fixed_angle_cases helper."""

from __future__ import annotations

from pathlib import Path

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
    z_resolution_nm=0.1,
)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

grazing_angle_deg = 4.0
energies_ev = np.arange(50.0, 650.0, 10)

cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies_ev,
    grazing_angle_deg=grazing_angle_deg,
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
    backend="numba",
)

results = list(runner.run_cases(cases))

csv_path = output_dir / "fixed_angle_all_orders.csv"
grax.write_all_orders_csv(results, csv_path)

orders_plot_path = output_dir / "fixed_angle_orders_1_3.png"
grax.plot_order_subset(
    results,
    orders_plot_path,
    diffraction_orders=[1, 2, 3],
    title="Fixed-Angle Sweep: Orders 1-3 Efficiency vs Energy",
)

profile_path = output_dir / "fixed_angle_profile.png"
grating.plot_profile(profile_path)

print(f"Results saved to: {csv_path}")
print(f"Orders 1-3 plot saved to: {orders_plot_path}")
print(f"Profile plot saved to: {profile_path}")
