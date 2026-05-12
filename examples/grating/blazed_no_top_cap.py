"""Generate the blazed grating tutorial image without top cap."""

from pathlib import Path
import argparse

import grax as rp
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.329, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")

blazed_grating = rp.BlazedGrating(
    period_lpermm=400,
    coating_stack=None,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    z_resolution_nm=0.1,
    x_resolution_nm=1.0,
    blaze_angle_deg=0.9,
    anti_blaze_angle_deg=None,
)

parser = argparse.ArgumentParser(description="Generate blazed grating profile plot")
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
output_path = output_dir / "blazed_no_top_cap.png"

blazed_grating.plot_profile(output_path)
print(f"Saved: {output_path}")
