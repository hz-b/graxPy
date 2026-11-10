"""Cached material-stack images used by saved run pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .files import replace_file
from .persistence import build_grating_from_spec


def ensure_run_material_stack_preview(data_dir: Path, run: dict[str, Any]) -> Path | None:
    """Create a run's stack preview once, returning its cached path."""

    spec = run.get("grating_spec")
    run_id = str(run.get("id", ""))
    if not isinstance(spec, dict) or not run_id:
        return None

    preview_path = data_dir / "previews" / "run-stacks" / f"{run_id}.png"
    if preview_path.is_file():
        return preview_path

    temporary_path = preview_path.with_name(f".{run_id}.{uuid4().hex}.tmp.png")
    try:
        grating = build_grating_from_spec(spec)
        x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=3)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        figure, axes = plt.subplots(figsize=(8, 3.5))
        try:
            colors = grating._plot_material_colors(
                coating_stack=grating.resolved_stack(),
                material_labels=material_labels,
            )
            color_map = plt.matplotlib.colors.ListedColormap(colors)
            color_map.set_bad(color="#ffffff", alpha=1.0)
            masked_material_map = np.ma.masked_less(material_map, 0)
            axes.imshow(
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
            axes.set_xlabel("x (nm)")
            axes.set_ylabel("z (nm)")
            axes.set_title("Material Stack")
            axes.legend(handles=legend_handles, loc="upper right")
            figure.tight_layout()
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(temporary_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(figure)
        replace_file(temporary_path, preview_path)
        return preview_path
    except Exception:
        return None
    finally:
        temporary_path.unlink(missing_ok=True)
