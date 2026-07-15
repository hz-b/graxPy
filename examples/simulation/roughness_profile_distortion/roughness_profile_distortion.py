"""Export a picture of a grating profile distorted by very large roughness.

This is a geometry-only example: it builds a single laminar grating whose
coating layers carry an extreme ``random-interface`` roughness (sigma = 100 nm)
and saves a close-up image of the resulting distorted material profile. No
simulation is run.

The roughness is placed on the *internal* interface between two thick coating
layers so the full +/- excursion is visible. The topmost interface is left flat
because the solver z-grid ends just above the nominal top surface, which would
otherwise clip large upward excursions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grax  # noqa: E402
from grax.stacks import LayerSpec, assemble_custom_stack  # noqa: E402

ROUGHNESS_SIGMA_NM = 100.0
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def build_distorted_grating() -> grax.LaminarGrating:
    """Return a laminar grating with an extreme internal-interface roughness."""
    # Two thick coating layers. The Cr layer top is an internal interface, so its
    # 100 nm roughness is fully visible with Cr below and C above. The C top is
    # left flat (sigma 0) so it does not clip against the top of the z-grid.
    stack = assemble_custom_stack(
        substrate_material="Si",
        layers_bottom_up=[
            LayerSpec(material="Cr", thickness_nm=250.0, roughness_sigma_nm=ROUGHNESS_SIGMA_NM),
            LayerSpec(material="C", thickness_nm=250.0, roughness_sigma_nm=0.0),
        ],
    )
    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.5,
        depth_nm=40.0,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material="Si",
        coating_stack=stack,
        x_resolution_nm=2.0,
        z_resolution_nm=2.0,
        roughness=grax.RoughnessSpec(kind="random-interface", sigma_nm=0.0, seed=0),
    )


def save_profile_image(grating: grax.LaminarGrating, output_path: Path) -> Path:
    """Save a full-period image of the distorted grating material profile."""
    x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=1)
    material_colors = grating._plot_material_colors(
        coating_stack=grating.resolved_stack(),
        material_labels=material_labels,
    )
    color_map = plt.matplotlib.colors.ListedColormap(material_colors)
    color_map.set_bad(color="#ffffff", alpha=1.0)
    masked_material_map = np.ma.masked_less(material_map, 0)

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.imshow(
        masked_material_map,
        origin="lower",
        aspect="auto",
        extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
        cmap=color_map,
        interpolation="nearest",
        vmin=0,
        vmax=max(len(material_labels) - 1, 1),
    )
    legend_handles = [
        plt.matplotlib.patches.Patch(color=color_map.colors[index], label=label)
        for index, label in enumerate(material_labels)
    ]
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("z (nm)")
    axis.set_title(f"random-interface roughness sigma = {ROUGHNESS_SIGMA_NM:.0f} nm")
    axis.legend(handles=legend_handles, loc="upper right")
    axis.grid(False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    """Build the grating and export the distorted-profile image."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grating = build_distorted_grating()
    output_path = OUTPUT_DIR / "roughness_profile_distortion.png"
    save_profile_image(grating, output_path)
    print(f"Saved distorted profile image to: {output_path}")


if __name__ == "__main__":
    main()
