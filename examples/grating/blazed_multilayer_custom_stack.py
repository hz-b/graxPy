"""Generate a blazed multilayer profile with custom stack assembly.

This example uses ``MultilayerStack`` as the authoritative Cr/C repeated block,
then composes a custom stack around it:
- Si substrate
- Pt 2 nm layer
- Cr/C multilayer block
- Te 2 nm layer
- O 2 nm top cap
"""

from pathlib import Path
import argparse

import grax

# Periodic multilayer definition (Cr/C) from the built-in multilayer stack API.
d_period_nm = 6.0
gamma = 0.4
n_bilayers = 50

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=d_period_nm,
    gamma=gamma,
    n_bilayers=n_bilayers,
    top_material="C",
)

layers_bottom_up: list[grax.LayerSpec] = [grax.LayerSpec(material="Pt", thickness_nm=2.0)]
layers_bottom_up.extend(multilayer_stack.layer_specs_bottom_up())
layers_bottom_up.append(grax.LayerSpec(material="Te", thickness_nm=2.0))

custom_stack = grax.assemble_custom_stack(
    substrate_material="Si",
    layers_bottom_up=layers_bottom_up,
    top_cap_material="O",
    top_cap_thickness_nm=2.0,  # O top layer
)

blazed_grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=custom_stack,
    x_resolution_nm=0.5,
    z_resolution_nm=0.5,
)

parser = argparse.ArgumentParser(description="Generate custom-stack blazed multilayer profile plot")
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
output_path = output_dir / "blazed_multilayer_custom_stack.png"
schematic_output_path = output_dir / "blazed_multilayer_custom_stack_schematic.png"

blazed_grating.plot_profile(output_path)
custom_stack.plot_schematic(schematic_output_path)
print(f"Saved: {output_path}")
print(f"Saved: {schematic_output_path}")
