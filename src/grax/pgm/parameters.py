# ruff: noqa: N815, N818
# Match the colleague's eV field names and cancellation API.
"""Inputs for an SX700 plane grating monochromator working-range calculation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import FALLBACK_GROOVE_DENSITY_L_MM

if TYPE_CHECKING:
    from threading import Event

    from grax.gratings import BaseGrating
    from grax.stacks import BaseStack


class PGMCancelled(RuntimeError):
    """The caller requested that a working-range calculation stop."""


@dataclass
class SX700Parameters:
    """Geometry, scan and solver settings for an SX700 calculation."""

    grating: BaseGrating | None = None
    mirror_stack: BaseStack | None = None
    mirror_roughness_nm: float = 0.0
    mirror_length_mm: float = 600.0
    grating_length_mm: float = 150.0
    optic_dx_mm: float = 350.0
    beam_offset_mm: float = 20.0
    mirror_grating_axis_dy_mm: float = 10.0
    mirror_axis_dx_mm: float = 0.0
    mirror_surface_axis_offset_mm: float | None = None
    mirror_downstream_edge_offset_mm: float | None = None
    theta_min_deg: float = 68.0
    theta_max_deg: float = 90.0
    beta_min_deg: float = 68.0
    beta_max_deg: float = 90.0
    beam_height_mm: float = 1.0
    groove_density_l_mm: float | None = None
    diffraction_order: int = 2
    edge_margin_mm: float = 2.0
    crit_distance: float = 6.35
    polarization: str = "s"
    rcwa_harmonics: int = 9
    rcwa_solver: str = "rcwa"
    rcwa_workers: int | str = "auto"
    rcwa_energy_nodes: int = 20
    rcwa_cff_nodes: int = 10
    rcwa_max_henke_nodes: int = 90
    rcwa_include_henke_nodes: bool = True
    rcwa_use_cache: bool = True
    rcwa_convergence_check: bool = True
    rcwa_progress: bool = False
    rcwa_adaptive_sampling: bool = True
    rcwa_adaptive_initial_energy_nodes: int = 30
    rcwa_adaptive_initial_cff_nodes: int = 20
    rcwa_adaptive_iterations: int = 3
    rcwa_adaptive_refine_fraction: float = 0.35
    cache_dir: Path | None = None
    stop_event: Event | None = None
    progress_callback: Callable[[int, int], None] | None = None
    maps_ready_callback: Callable[..., None] | None = None
    nodes_callback: Callable[..., None] | None = None
    worker_pids_callback: Callable[[set[int]], None] | None = None
    energy_min_eV: float = 1000.0
    energy_max_eV: float = 6000.0
    c_min: float = 1.5
    c_max: float = 10.0
    n_energy: int = 501
    n_c: int = 450

    def __post_init__(self) -> None:
        """Validate inputs and derive groove density from the grating."""
        if self.grating is not None:
            period = float(self.grating.period_lpermm)
            if self.groove_density_l_mm is not None and abs(
                float(self.groove_density_l_mm) - period
            ) > 1e-9 * max(1.0, period):
                raise ValueError("groove_density_l_mm contradicts grating.period_lpermm")
            self.groove_density_l_mm = period
        elif self.groove_density_l_mm is None:
            self.groove_density_l_mm = FALLBACK_GROOVE_DENSITY_L_MM
        positive = (
            "mirror_length_mm",
            "grating_length_mm",
            "optic_dx_mm",
            "beam_height_mm",
            "groove_density_l_mm",
            "crit_distance",
            "energy_min_eV",
            "energy_max_eV",
            "c_min",
            "c_max",
        )
        for key in positive:
            value = float(getattr(self, key))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and greater than zero")
        for key in (
            "beam_offset_mm", "mirror_grating_axis_dy_mm", "mirror_axis_dx_mm",
            "mirror_surface_axis_offset_mm", "mirror_downstream_edge_offset_mm",
        ):
            value = getattr(self, key)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
        if not isfinite(self.edge_margin_mm) or self.edge_margin_mm < 0:
            raise ValueError("edge_margin_mm must be non-negative")
        for key in ("n_energy", "n_c", "rcwa_energy_nodes", "rcwa_cff_nodes", "rcwa_harmonics"):
            if int(getattr(self, key)) < 2:
                raise ValueError(f"{key} must be at least 2")
        if self.energy_min_eV >= self.energy_max_eV or self.c_min >= self.c_max:
            raise ValueError("Energy and CFF maximums must exceed their minimums")
        if not (0 < self.theta_min_deg < self.theta_max_deg <= 90):
            raise ValueError("Mirror angle limits must lie within 0–90 degrees")
        if not (0 < self.beta_min_deg < self.beta_max_deg <= 90):
            raise ValueError("Grating angle limits must lie within 0–90 degrees")
        if int(self.diffraction_order) < 1:
            raise ValueError("diffraction_order must be positive")
        if self.rcwa_solver not in {"rcwa", "neviere"}:
            raise ValueError("rcwa_solver must be 'rcwa' or 'neviere'")
        if self.polarization.lower() not in {"s", "te"}:
            raise ValueError("The PGM model currently supports s polarization only")
        if not isfinite(self.mirror_roughness_nm) or self.mirror_roughness_nm < 0:
            raise ValueError("mirror_roughness_nm must be non-negative")

    @property
    def grating_period_nm(self) -> float:
        """Groove period in nanometres."""
        return 1e6 / float(self.groove_density_l_mm)

    @property
    def c_offset_mm(self) -> float:
        """Mirror surface offset used by the geometry model."""
        return float(
            self.beam_offset_mm
            if self.mirror_surface_axis_offset_mm is None
            else self.mirror_surface_axis_offset_mm
        )

    @property
    def a_offset_mm(self) -> float:
        """Mirror downstream edge offset used by the geometry model."""
        if self.mirror_downstream_edge_offset_mm is not None:
            return float(self.mirror_downstream_edge_offset_mm)
        return float(self.optic_dx_mm - 0.5 * self.mirror_length_mm)

    def check_stopped(self) -> None:
        """Raise when a caller requested cancellation."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise PGMCancelled("PGM run stopped")
