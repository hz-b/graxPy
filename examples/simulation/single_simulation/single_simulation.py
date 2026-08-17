"""Single simulation example using explicit p polarization."""

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
    z_resolution_nm=0.5,
)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

energy_ev = 200.0
grazing_angle_deg = 4.0
diffraction_order = 1
fourier_orders = 5

parser = argparse.ArgumentParser(description="Run a single grating simulation")
parser.add_argument(
    "--solver",
    choices=("rcwa", "neviere"),
    default="rcwa",
    help="Electromagnetic solver to run. Both compute every diffraction order; "
    "they differ only in how each layer is crossed in z.",
)
args = parser.parse_args()

result = grax.run_simulation(
    solver=args.solver,
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    polarization="p",
    backend="numba",
)

csv_path = output_dir / f"single_simulation_{args.solver}.csv"
grax.write_all_orders_csv(result, csv_path)

profile_path = output_dir / "single_simulation_profile.png"
grating.plot_profile(profile_path)

print(f"Results saved to: {csv_path}")
print(f"Profile plot saved to: {profile_path}")
print(f"Selected efficiency (order {diffraction_order}): {result.selected_efficiency:.6g}")
