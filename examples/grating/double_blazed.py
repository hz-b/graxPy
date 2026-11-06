"""Generate a built-in double blazed grating profile plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import grax

grating = grax.DoubleBlazedGrating(
    period_lpermm=600,
    blaze_angle_1_deg=1.5,
    blaze_angle_2_deg=0.75,
    anti_blaze_angle_deg=30.0,
    second_facet_fraction=0.5,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    x_resolution_nm=1.0,
    z_resolution_nm=0.5,
)

parser = argparse.ArgumentParser(description="Generate a double blazed grating profile plot")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Override output directory (default: examples/grating/results/)",
)
args = parser.parse_args()

output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "double_blazed.png"

grating.plot_profile(output_path)
print(f"Saved: {output_path}")
