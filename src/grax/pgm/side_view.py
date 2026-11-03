# ruff: noqa: N803, N806
# Match the colleague's scientific notation.
"""Side-view drawing coordinates of the SX700 optics and rays for one state."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import _mirror_geometry_from_theta, geometry_for_state
from .parameters import SX700Parameters


def side_view_geometry(energy_eV, cff, p: SX700Parameters) -> dict[str, Any] | None:
    """Return optics, ray segments and markers in mm, or None without an optical solution."""
    g = geometry_for_state(energy_eV, cff, p)
    if g is None:
        return None

    beta = np.deg2rad(g["beta_deg"])
    theta = np.deg2rad(g["theta_deg"])
    Dx, Dy, umx, umy = _mirror_geometry_from_theta(theta, p)
    M_down = np.array([Dx, Dy], dtype=float)
    M_up = M_down + p.mirror_length_mm * np.array([umx, umy], dtype=float)
    ug = np.array([-np.sin(beta), np.cos(beta)])
    G1 = -0.5 * p.grating_length_mm * ug
    G2 = +0.5 * p.grating_length_mm * ug

    xm = float(g["mirror_x_rays_mm"][1])
    ym = -float(p.beam_offset_mm)
    xg = float(g["grating_x_rays_mm"][1])
    yg = float(g["grating_y_rays_mm"][1])
    x_in = float(min(M_up[0], xm)) - 80.0
    x_out = 150.0

    def segments(mirror_x, mirror_y, grating_x, grating_y):
        return [
            {"x": [x_in, mirror_x], "y": [mirror_y, mirror_y]},
            {"x": [mirror_x, grating_x], "y": [mirror_y, grating_y]},
            {"x": [grating_x, x_out], "y": [grating_y, grating_y]},
        ]

    edge_rays = []
    half_beam = 0.5 * float(p.beam_height_mm)
    for index, y0 in ((0, ym - half_beam), (2, ym + half_beam)):
        edge_rays += segments(
            float(g["mirror_x_rays_mm"][index]),
            y0,
            float(g["grating_x_rays_mm"][index]),
            float(g["grating_y_rays_mm"][index]),
        )

    y_values = [
        M_up[1], M_down[1], G1[1], G2[1], ym, yg,
        ym - half_beam, ym + half_beam, p.mirror_grating_axis_dy_mm, 0.0,
    ]
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    y_span = max(y_max - y_min, 1.0)
    return {
        "mirror": {
            "x": [float(M_up[0]), float(M_down[0])],
            "y": [float(M_up[1]), float(M_down[1])],
        },
        "grating": {"x": [float(G1[0]), float(G2[0])], "y": [float(G1[1]), float(G2[1])]},
        "central_ray": segments(xm, ym, xg, yg),
        "edge_rays": edge_rays,
        "grating_axis": {"x": 0.0, "y": 0.0},
        "mirror_axis": {
            "x": float(p.mirror_axis_dx_mm),
            "y": float(p.mirror_grating_axis_dy_mm),
        },
        "hit_points": {"x": [xm, xg], "y": [ym, yg]},
        "y_range": [y_min - 0.15 * y_span, y_max + 0.15 * y_span],
    }
