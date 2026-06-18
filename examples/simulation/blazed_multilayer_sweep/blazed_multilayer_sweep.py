"""Blazed monochromator sweep with a Cr/C multilayer coating stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import grax

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=6,
    gamma=0.4,
    n_bilayers=50,
    top_material="C",
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=0.1,
    z_resolution_nm=0.1,
)

energies_ev = np.arange(500.0, 4000.0, 10.0)
cases = grax.monochromator_cases(
    grating=grating,
    energies_ev=energies_ev,
    diffraction_order=1,
    cff=2.25,
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    backend="numba",
)

csv_path = output_dir / "blazed_multilayer_all_orders.csv"
orders_plot_path = output_dir / "blazed_multilayer_orders_1_3.png"
profile_plot_path = output_dir / "blazed_multilayer_profile.png"
stack_plot_path = output_dir / "multilayer_stack_schematic.png"

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, csv_path)
grax.plot_order_subset(
    results,
    orders_plot_path,
    diffraction_orders=[1, 2, 3],
    title="Blazed Multilayer Monochromator Sweep: Orders 1-3",
)
grating.plot_profile(profile_plot_path)
multilayer_stack.plot_schematic(stack_plot_path)

print(f"Results saved to: {csv_path}")
print(f"Orders 1-3 plot saved to: {orders_plot_path}")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Stack schematic saved to: {stack_plot_path}")
