"""Generate the laminar grating tutorial image without top cap."""

from pathlib import Path
import argparse

import grax

laminar_grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    coating_stack=None,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    z_resolution_nm=0.5,
    x_resolution_nm=1.0,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
)

parser = argparse.ArgumentParser(description="Generate laminar grating profile plot")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Override output directory (default: examples/grating/results/)",
)
args = parser.parse_args()

if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(__file__).resolve().parent / "results"

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "laminar_no_top_cap.png"

laminar_grating.plot_profile(output_path)
print(f"Saved: {output_path}")
