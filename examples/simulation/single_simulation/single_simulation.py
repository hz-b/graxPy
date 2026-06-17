"""Single simulation example using run_simulation API."""

from __future__ import annotations

from pathlib import Path
import argparse

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

parser = argparse.ArgumentParser(description="Run single RCWA simulation")
output_dir = Path(__file__).resolve().parent / "results"

args = parser.parse_args()
output_dir.mkdir(parents=True, exist_ok=True)

energy_ev = 200.0
grazing_angle_deg = 4.0
diffraction_order = 1
fourier_orders = 5

result = grax.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    backend="numba",
)

csv_path = output_dir / "single_simulation.csv"
grax.write_all_orders_csv(result, csv_path)

profile_path = output_dir / "single_simulation_profile.png"
grating.plot_profile(profile_path)

print(f"Results saved to: {csv_path}")
print(f"Profile plot saved to: {profile_path}")
print(f"Selected efficiency (order {diffraction_order}): {result.selected_efficiency:.6g}")
