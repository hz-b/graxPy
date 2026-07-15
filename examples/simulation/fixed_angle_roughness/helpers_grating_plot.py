"""Shared whole-grating geometry plotting for the roughness examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import grax


def save_grating_plot(grating: grax.LaminarGrating, output_path: Path, *, title: str) -> Path:
    """Save a whole-grating PDF of the generated material geometry."""
    x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=1)
    material_colors = grating._plot_material_colors(
        coating_stack=grating.resolved_stack(),
        material_labels=material_labels,
    )
    color_map = plt.matplotlib.colors.ListedColormap(material_colors)
    color_map.set_bad(color="#ffffff", alpha=1.0)
    masked_material_map = np.ma.masked_less(material_map, 0)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
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
    axis.set_xlim(float(x_grid[0]), float(x_grid[-1]))
    axis.set_ylim(float(np.min(z_grid)), float(np.max(z_grid)))
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("z (nm)")
    axis.set_title(f"{title} grating")
    axis.legend(handles=legend_handles, loc="upper right")
    axis.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path
