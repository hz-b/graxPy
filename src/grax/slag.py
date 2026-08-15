"""Legacy SLAG convenience helpers built on the generic grating/simulation API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gratings import LaminarGrating
from .simulation import GratingSimulation


@dataclass
class SlagConfig:
    """Compatibility configuration for the historical SLAG example."""

    grating_period_lpermm: int = 400
    diffraction_order: int = 1
    width_to_period_ratio: float = 0.67
    depth_nm: float = 14.9
    trapezoid_left_deg: float = 15.0
    trapezoid_right_deg: float = 15.0
    substrate_material: Any = "Si"
    layer_material: Any = "Pt"
    layer_thickness_nm: float = 28.77
    top_cap_material: Any | None = None
    top_cap_thickness_nm: float = 0.0
    z_resolution_nm: float = 0.1
    x_resolution_nm: float = 1.0
    fourier_orders: int = 25
    photon_energy_ev: np.ndarray = field(default_factory=lambda: np.arange(50.0, 650.0, 10.0))
    grazing_angle_deg: float = 4.0


def default_example_slag_config() -> SlagConfig:
    """Return legacy SLAG defaults."""

    return SlagConfig()


def _build_legacy_grating(config: SlagConfig) -> LaminarGrating:
    """Create a laminar grating from a legacy SLAG config."""

    return LaminarGrating(
        period_lpermm=config.grating_period_lpermm,
        width_to_period_ratio=config.width_to_period_ratio,
        depth_nm=config.depth_nm,
        left_wall_angle_deg=config.trapezoid_left_deg,
        right_wall_angle_deg=config.trapezoid_right_deg,
        substrate_material=config.substrate_material,
        layer_material=config.layer_material,
        layer_thickness_nm=config.layer_thickness_nm,
        top_cap_material=config.top_cap_material,
        top_cap_thickness_nm=config.top_cap_thickness_nm,
        z_resolution_nm=config.z_resolution_nm,
        x_resolution_nm=config.x_resolution_nm,
    )


def simulate_single_energy(config: SlagConfig, photon_energy_ev: float) -> dict[str, float]:
    """Run the legacy SLAG flow for one energy."""

    simulation = GratingSimulation(
        grating=_build_legacy_grating(config),
        diffraction_order=config.diffraction_order,
        fourier_orders=config.fourier_orders,
        grazing_angle_deg=config.grazing_angle_deg,
    )
    return simulation.run_single(photon_energy_ev)


def run_example_slag(
    config: SlagConfig | None = None,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run the legacy SLAG energy sweep helper."""

    config = config or default_example_slag_config()
    simulation = GratingSimulation(
        grating=_build_legacy_grating(config),
        diffraction_order=config.diffraction_order,
        fourier_orders=config.fourier_orders,
        grazing_angle_deg=config.grazing_angle_deg,
    )

    energies = np.asarray(config.photon_energy_ev, dtype=float)
    if show_progress:
        for index, energy in enumerate(energies, start=1):
            print(f"[SLAG] {index}/{len(energies)}: computing {energy:.3f} eV", flush=True)

    result = simulation.run(energies)
    return {
        "energy_ev": result.energy_ev,
        "efficiency": result.efficiency,
        "diffraction_angle_deg": result.diffraction_angle_deg,
    }
