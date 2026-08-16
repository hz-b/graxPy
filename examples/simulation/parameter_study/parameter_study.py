"""Blazed-grating parameter study across energies with explicit p polarization."""

from __future__ import annotations

import argparse
from pathlib import Path

import grax

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

grating = grax.BlazedGrating(
    period_lpermm=600,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    blaze_angle_deg=0.75,
    anti_blaze_angle_deg=None,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energies_ev = [100.0, 600.0, 2000.0]
grazing_angle_deg = 1.5
fourier_orders_values = list(range(5, 26, 2))
x_resolution_values = grax.get_default_parameter_study_ranges()[1]
z_resolution_values = grax.get_default_parameter_study_ranges()[2]

parser = argparse.ArgumentParser(description="Convergence parameter study")
parser.add_argument(
    "--solver",
    choices=("rcwa", "neviere"),
    default="rcwa",
    help="Electromagnetic solver to run. Both compute every diffraction order; "
    "they differ only in how each layer is crossed in z.",
)
args = parser.parse_args()

study = grax.run_parameter_study(
    solver=args.solver,
    grating=grating,
    energies_ev=energies_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=1,
    polarization="p",
    fourier_orders_values=fourier_orders_values,
    x_resolution_values=x_resolution_values,
    z_resolution_values=z_resolution_values,
    output_dir=output_dir,
    save_csv=True,
    show_progress=True,
)

plot_path = output_dir / f"parameter_study_grid_{args.solver}.png"
grax.plot_parameter_study(
    study,
    output_filename=plot_path,
    title="Blazed Parameter Study: Orders vs Fourier/x/z Resolution",
)

profile_path = output_dir / "parameter_study_profile.png"
grating.plot_profile(profile_path)

print(f"Parameter-study plot saved to: {plot_path}")
print(f"Profile plot saved to: {profile_path}")
