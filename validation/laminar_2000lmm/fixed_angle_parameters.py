"""Shared parameters for the 2000 l/mm fixed-angle validation sweeps."""

from __future__ import annotations

import grax

NUM_ENERGY_POINTS = 20
FOURIER_ORDERS = 10
X_RESOLUTION_NM = 0.3
Z_RESOLUTION_NM = 0.3

PERIOD_LPERMM = 2000
WIDTH_TO_PERIOD_RATIO = 0.6
DEPTH_NM = 5.0
LEFT_WALL_ANGLE_DEG = 90.0
RIGHT_WALL_ANGLE_DEG = 90.0


def create_grating(*, substrate_material: object) -> grax.LaminarGrating:
    """Build the shared laminar grating used by all fixed-angle validation scripts.

    Args:
        substrate_material: Substrate material table for the grating.

    Returns:
        Configured laminar grating instance.
    """

    return grax.LaminarGrating(
        period_lpermm=PERIOD_LPERMM,
        width_to_period_ratio=WIDTH_TO_PERIOD_RATIO,
        depth_nm=DEPTH_NM,
        left_wall_angle_deg=LEFT_WALL_ANGLE_DEG,
        right_wall_angle_deg=RIGHT_WALL_ANGLE_DEG,
        substrate_material=substrate_material,
        layer_material=substrate_material,
        layer_thickness_nm=0.0,
        top_cap_material=None,
        top_cap_thickness_nm=0.0,
        x_resolution_nm=X_RESOLUTION_NM,
        z_resolution_nm=Z_RESOLUTION_NM,
    )
