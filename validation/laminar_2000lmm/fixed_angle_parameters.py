"""Shared parameters for the 2000 l/mm fixed-angle validation sweeps."""

from __future__ import annotations

import grax
import numpy as np

NUM_ENERGY_POINTS = 1039
FOURIER_ORDERS = 20
X_RESOLUTION_NM = 0.1
Z_RESOLUTION_NM = 0.1

PERIOD_LPERMM = 2000
WIDTH_TO_PERIOD_RATIO = 0.6
DEPTH_NM = 5.0
LEFT_WALL_ANGLE_DEG = 10.0
RIGHT_WALL_ANGLE_DEG = 10.0
SI_DENSITY_G_CM3 = 2.330
SIO2_DENSITY_G_CM3 = 2.530
C4O_DENSITY_G_CM3 = 1.000
SIO2_THICKNESS_NM = 0.87
C4O_THICKNESS_NM = 0.68
EDGE_EXCLUDED_ENERGY_WINDOWS_EV = (
    (280.0, 300.0),
    (520.0, 560.0),
)


def resolve_material(
    *,
    material_name: str,
    density_g_cm3: float | None = None,
) -> object:
    """Return one explicit material definition for elemental or formula inputs."""

    return grax.MaterialSpec(name=material_name, density_g_cm3=density_g_cm3)


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


def create_layered_stack(
    *,
    substrate_material: object,
    sio2_material: object,
    c4o_material: object,
    sio2_thickness_nm: float = SIO2_THICKNESS_NM,
    c4o_thickness_nm: float = C4O_THICKNESS_NM,
) -> grax.CustomStack:
    """Build the Si substrate + SiO2 + C4O stack used in the layered sweeps."""

    return grax.assemble_custom_stack(
        substrate_material=substrate_material,
        layers_bottom_up=[
            grax.LayerSpec(material=sio2_material, thickness_nm=sio2_thickness_nm),
            grax.LayerSpec(material=c4o_material, thickness_nm=c4o_thickness_nm),
        ],
    )


def create_layered_grating(
    *,
    substrate_material: object,
    sio2_material: object,
    c4o_material: object,
    sio2_thickness_nm: float = SIO2_THICKNESS_NM,
    c4o_thickness_nm: float = C4O_THICKNESS_NM,
) -> grax.LaminarGrating:
    """Build the shared laminar grating with the SiO2/C4O top-layer stack."""

    layered_stack = create_layered_stack(
        substrate_material=substrate_material,
        sio2_material=sio2_material,
        c4o_material=c4o_material,
        sio2_thickness_nm=sio2_thickness_nm,
        c4o_thickness_nm=c4o_thickness_nm,
    )
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
        coating_stack=layered_stack,
        x_resolution_nm=X_RESOLUTION_NM,
        z_resolution_nm=Z_RESOLUTION_NM,
    )


def build_edge_excluded_optimization_energies(
    measurement_energies_ev: np.ndarray | list[float] | tuple[float, ...],
) -> list[float]:
    """Return measurement energies excluding the configured edge windows."""

    energies_ev = np.asarray(measurement_energies_ev, dtype=float)
    if energies_ev.ndim != 1:
        raise ValueError("Measurement energies must be one-dimensional.")
    if energies_ev.size == 0:
        raise ValueError("Measurement energies must not be empty.")

    included_mask = np.ones(energies_ev.shape, dtype=bool)
    for lower_ev, upper_ev in EDGE_EXCLUDED_ENERGY_WINDOWS_EV:
        included_mask &= (energies_ev < lower_ev) | (energies_ev > upper_ev)

    filtered_energies_ev = energies_ev[included_mask]
    if filtered_energies_ev.size == 0:
        raise ValueError(
            "No evaluation energies remain after excluding the configured absorption-edge windows."
        )
    return filtered_energies_ev.tolist()
