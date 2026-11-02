"""Top and side view of the comb electrodes and the deformation they produce."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

_PLUS = "#c0392b"
_MINUS = "#2c6fbb"
_VISUAL_DISPLACEMENT_UM = 0.35


def idt_preview_data_uri(grating: Any, amplitude_nm: float) -> str:
    """PNG data URI of the top view and the side view for an ``IdtGrating``.

    The side view draws lattice planes moved by the deformation of zero-to-peak
    amplitude ``amplitude_nm``, exaggerated so that it is visible; its factor is
    printed on the figure. Everything else is to scale along the surface.
    """

    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from ..bragg_grating import surface_displacement

    f, g = grating.finger_um, grating.gap_um
    period = grating.period_um
    n_periods = 2
    x_min, x_max = -g / 2.0, n_periods * period + g / 2.0
    aperture, thickness = grating.aperture_um, grating.al_thickness_um
    depth = max(3.0 * grating.penetration_um, period)

    fig = Figure(figsize=(8.0, 6.4), dpi=110)
    top, side = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1.0, 1.7]})

    # Top view: fingers of both polarities between the two bus bars.
    end, bar = 0.06 * aperture, 0.14 * aperture
    for k in range(n_periods + 1):
        plus_x, minus_x = k * period, k * period + f + g
        top.add_patch(Rectangle((plus_x, 0.0), f, aperture + end, color=_PLUS, lw=0))
        top.add_patch(Rectangle((minus_x, -end), f, aperture + end, color=_MINUS, lw=0))
    top.add_patch(Rectangle((x_min, aperture + end), x_max - x_min, bar, color=_PLUS, alpha=0.55, lw=0))
    top.add_patch(Rectangle((x_min, -end - bar), x_max - x_min, bar, color=_MINUS, alpha=0.55, lw=0))
    top.set_xlim(x_min, x_max)
    top.set_ylim(-end - bar, aperture + end + bar)
    top.set_ylabel("along the fingers (µm)")
    top.set_title(
        f"Top view: aperture {aperture:g} µm, period {period:g} µm "
        f"(finger {f:g} + gap {g:g}, twice); red and blue are the two polarities",
        fontsize=9,
    )
    top.tick_params(labelbottom=False)

    # Side view: lattice planes moved by the deformation, Al fingers on the surface.
    amplitude_um = amplitude_nm * 1e-3
    n_x = 512
    plane_depths = np.linspace(0.0, depth, 15)
    displacement = surface_displacement(grating, amplitude_um, plane_depths, n_x)
    x_one = (np.arange(n_x) + 0.5) / n_x * period
    x = np.concatenate([x_one + k * period for k in range(n_periods + 1)])
    scale = _VISUAL_DISPLACEMENT_UM / amplitude_um if amplitude_um > 0.0 else 1.0
    side.add_patch(Rectangle((x_min, 0.0), x_max - x_min, depth, color="#eef1f5", lw=0, zorder=0))
    for index, plane_depth in enumerate(plane_depths):
        z = plane_depth + scale * np.tile(displacement[index], n_periods + 1)
        side.plot(x, z, color="#44546a" if index else "#111111", lw=1.6 if index == 0 else 0.8)
    surface = scale * np.tile(displacement[0], n_periods + 1)
    for k in range(n_periods + 1):
        for start, color in ((k * period, _PLUS), (k * period + f + g, _MINUS)):
            centre = start + f / 2.0
            lift = float(np.interp(centre, x, surface))
            side.add_patch(
                Rectangle((start, lift - thickness), f, thickness, color=color, lw=0, zorder=3)
            )
    side.set_xlim(x_min, x_max)
    side.set_ylim(depth, -0.12 * depth)
    side.set_xlabel("along the grating vector (µm)")
    side.set_ylabel("depth (µm)")
    side.set_title(
        f"Side view: Al {thickness:g} µm; lattice planes moved by the deformation, "
        f"amplitude {amplitude_nm:g} nm drawn ×{scale:,.0f}",
        fontsize=9,
    )
    side.annotate(
        "",
        xy=(x_min + 0.06 * (x_max - x_min), grating.penetration_um),
        xytext=(x_min + 0.06 * (x_max - x_min), 0.0),
        arrowprops={"arrowstyle": "<->", "color": "#111111"},
    )
    side.text(
        x_min + 0.08 * (x_max - x_min),
        grating.penetration_um / 2.0,
        f"penetration {grating.penetration_um:g} µm",
        fontsize=8,
        va="center",
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def coating_cross_section_data_uri(grating: Any) -> str:
    """PNG data URI of the materials over one period of an ``IdtSurfaceGrating``.

    Heights are to scale (nm); the displacement is the grating's own amplitude.
    """

    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    x, z, codes, labels = grating._material_plot_data(num_periods=1)
    colors = grating._plot_material_colors(coating_stack=grating.stack(), material_labels=labels)
    color_map = ListedColormap(colors).with_extremes(bad="#ffffff")
    fig = Figure(figsize=(8.0, 3.4), dpi=110)
    axes = fig.subplots()
    axes.imshow(
        np.ma.masked_less(codes, 0),
        origin="lower",
        aspect="auto",
        extent=[x[0] * 1e-3, x[-1] * 1e-3, z[0], z[-1]],
        cmap=color_map,
        interpolation="nearest",
        vmin=0,
        vmax=max(len(labels) - 1, 1),
    )
    axes.legend(
        handles=[Patch(color=colors[index], label=label) for index, label in enumerate(labels)],
        loc="upper right",
        fontsize=8,
    )
    axes.set_xlabel("along the grating vector (µm)")
    axes.set_ylabel("height (nm)")
    axes.set_title(
        f"Cross-section of one period: crystal, fingers and coating; displacement "
        f"{grating.amplitude_nm:g} nm to scale; {grating.solver_slice_count()} solver slices",
        fontsize=9,
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
