"""Generate the laminar grating tutorial image with top cap."""

from pathlib import Path
import argparse

import grax
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.329, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")
carbon = xrt_materials.Material("C", rho=2.2, table="Henke", name="C")

laminar_grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    coating_stack=None,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    top_cap_material=carbon,
    top_cap_thickness_nm=2.0,
    z_resolution_nm=0.1,
    x_resolution_nm=1.0,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
)

parser = argparse.ArgumentParser(description="Generate laminar grating profile plot with top cap")
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
output_path = output_dir / "laminar_with_top_cap.png"

laminar_grating.plot_profile(output_path)
print(f"Saved: {output_path}")
