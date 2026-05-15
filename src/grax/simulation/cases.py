"""Lazy case-generation helpers for simulation sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import brentq

from ..gratings import BaseGrating
from ..peak_fitting import PeakSelectionMode

_FIXED_CASE_ID_PREFIX = "fixed"
_MONOCHROMATOR_CASE_ID_PREFIX = "mono"
_ENERGY_ANGLE_CASE_ID_PREFIX = "pair"
_THETA_SEARCH_CASE_ID_PREFIX = "theta-search"


def fixed_angle_cases(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    grazing_angle_deg: float,
) -> Iterator[dict[str, object]]:
    """Yield fixed-angle energy-sweep cases lazily.

    Creates case dictionaries for a fixed-incident-angle energy scan. Each case
    includes the grating, photon energy, and fixed grazing angle. Case IDs are
    generated automatically with a stable internal prefix.

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.

    Yields:
        Case dictionaries suitable for :class:`BatchSimulationRunner`.

    Example:
        >>> grating = grax.LaminarGrating(period_nm=500, duty=0.5, height_nm=100)
        >>> cases = fixed_angle_cases(
        ...     grating=grating,
        ...     energies_ev=[500, 600, 700],
        ...     grazing_angle_deg=5.0
        ... )
        >>> for case in cases:
        ...     print(f"E={case['energy_ev']:.0f} eV, theta={case['grazing_angle_deg']:.1f}°")
    """

    for index, energy_ev in enumerate(energies_ev):
        yield {
            "case_id": f"{_FIXED_CASE_ID_PREFIX}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": float(grazing_angle_deg),
        }


def monochromator_cases(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    period_lpermm: float | None = None,
    diffraction_order: int = 1,
    cff: float = 2.25,
) -> Iterator[dict[str, object]]:
    """Yield fixed-cff monochromator sweep cases lazily.

    Creates case dictionaries for a monochromator configuration with fixed
    constant-focus-factor (CFF). The grazing angle is solved for each energy
    using the monochromator equation, ensuring the optic maintains the same
    optical configuration across the energy sweep.

    The monochromator relation solves for incidence angle alpha and diffraction
    angle beta subject to ``cff = cos(alpha) / cos(beta)`` and
    ``m*lambda = period * (sin(alpha) + sin(beta))``.

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        period_lpermm: Optional grating line density. Defaults to the grating value.
        diffraction_order: Positive diffraction order used by the monochromator relation.
        cff: Constant-focus factor. Higher values relax tolerance but reduce angular range.

    Yields:
        Case dictionaries suitable for :class:`BatchSimulationRunner` with computed
        grazing_angle_deg and diffraction_order fields.

    Example:
        >>> grating = grax.LaminarGrating(period_nm=500, duty=0.5, height_nm=100)
        >>> cases = monochromator_cases(
        ...     grating=grating,
        ...     energies_ev=[500, 600, 700],
        ...     diffraction_order=1,
        ...     cff=2.25
        ... )
    """

    line_density = float(grating.period_lpermm if period_lpermm is None else period_lpermm)
    for index, energy_ev in enumerate(energies_ev):
        grazing_angle = float(
            monochromator_grazing_angles_deg(
                np.asarray([float(energy_ev)], dtype=float),
                period_lpermm=line_density,
                diffraction_order=diffraction_order,
                cff=cff,
            )[0]
        )
        yield {
            "case_id": f"{_MONOCHROMATOR_CASE_ID_PREFIX}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": grazing_angle,
            "diffraction_order": diffraction_order,
        }


def energy_angle_cases(
    *,
    grating: BaseGrating,
    energy_angle_pairs: Iterable[tuple[float, float]],
) -> Iterator[dict[str, object]]:
    """Yield arbitrary energy-angle simulation cases lazily.

    Creates case dictionaries for user-specified energy and angle combinations.
    This provides maximum flexibility for custom sweep configurations that don't
    follow standard patterns like fixed-angle or monochromator scans.

    Args:
        grating: Grating reused by every generated case.
        energy_angle_pairs: Iterable of ``(energy_ev, grazing_angle_deg)`` pairs.

    Yields:
        Case dictionaries suitable for :class:`BatchSimulationRunner`.

    Example:
        >>> grating = grax.LaminarGrating(period_nm=500, duty=0.5, height_nm=100)
        >>> pairs = [(500, 5.0), (600, 5.5), (700, 6.0)]
        >>> cases = energy_angle_cases(
        ...     grating=grating,
        ...     energy_angle_pairs=pairs
        ... )
    """

    for index, (energy_ev, grazing_angle_deg) in enumerate(energy_angle_pairs):
        yield {
            "case_id": f"{_ENERGY_ANGLE_CASE_ID_PREFIX}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": float(grazing_angle_deg),
        }


def multilayer_theta_search_cases(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    diffraction_order: int = 1,
    rough_scan_half_width_deg: float = 0.5,
    rough_scan_points: int = 82,
    rough_fourier_orders: int = 3,
    rough_x_resolution_nm: float = 1.0,
    rough_z_resolution_nm: float = 1.0,
    fine_scan_half_width_deg: float = 0.1,
    fine_scan_points: int = 81,
    fine_fourier_orders: int = 5,
    fine_x_resolution_nm: float = 0.5,
    fine_z_resolution_nm: float = 0.5,
    final_fourier_orders: int = 25,
    final_x_resolution_nm: float = 0.3,
    final_z_resolution_nm: float = 0.3,
    multilayer_bragg_order: int = 1,
    roughness_sigma_nm: float | None = None,
    precise_peak_selection_mode: PeakSelectionMode = "max",
) -> Iterator[dict[str, object]]:
    """Yield energy-only cases for the multilayer theta-search workflow.

    Creates simplified case dictionaries that trigger the multilayer theta-search
    workflow during batch execution. The workflow internally performs a three-stage
    adaptive angular scan: rough scan, precise scan, and final solve.

    This workflow automatically determines the optimal grazing angle for each
    energy point, making it ideal for multilayer gratings where the Bragg angle
    shifts with photon energy.

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        diffraction_order: Positive diffraction order used by the theta-search workflow.
        rough_scan_half_width_deg: Rough scan half-width around the estimate.
        rough_scan_points: Number of rough scan points.
        rough_fourier_orders: Fourier order used during the rough scan.
        rough_x_resolution_nm: X resolution used during the rough scan.
        rough_z_resolution_nm: Z resolution used during the rough scan.
        fine_scan_half_width_deg: Precise scan half-width around the rough maximum.
        fine_scan_points: Number of precise scan points.
        fine_fourier_orders: Fourier order used during the precise scan.
        fine_x_resolution_nm: X resolution used during the precise scan.
        fine_z_resolution_nm: Z resolution used during the precise scan.
        final_fourier_orders: Fourier order used during the final solve.
        final_x_resolution_nm: X resolution used during the final solve.
        final_z_resolution_nm: Z resolution used during the final solve.
        multilayer_bragg_order: Positive Bragg order used for the analytical estimate.
        roughness_sigma_nm: Optional rms roughness in nanometers.
        precise_peak_selection_mode: Mode used to select the final theta from the
            precise scan. ``"max"`` uses the sampled maximum, ``"gauss"`` fits a
            local Gaussian, and ``"voigt"`` fits a local Voigt profile.

    Yields:
        Case dictionaries suitable for :class:`BatchSimulationRunner` with
        workflow-specific fields stored in the case dictionary and stable
        internally generated case IDs.

    Example:
        >>> grating = grax.BlazedGrating(period_nm=500, blaze_angle=10, height_nm=100)
        >>> cases = multilayer_theta_search_cases(
        ...     grating=grating,
        ...     energies_ev=[500, 600, 700],
        ...     precise_peak_selection_mode="gauss"
        ... )
    """

    for index, energy_ev in enumerate(energies_ev):
        yield {
            "workflow": "multilayer_theta_search",
            "rough_scan_half_width_deg": rough_scan_half_width_deg,
            "rough_scan_points": rough_scan_points,
            "rough_fourier_orders": rough_fourier_orders,
            "rough_x_resolution_nm": rough_x_resolution_nm,
            "rough_z_resolution_nm": rough_z_resolution_nm,
            "fine_scan_half_width_deg": fine_scan_half_width_deg,
            "fine_scan_points": fine_scan_points,
            "fine_fourier_orders": fine_fourier_orders,
            "fine_x_resolution_nm": fine_x_resolution_nm,
            "fine_z_resolution_nm": fine_z_resolution_nm,
            "final_fourier_orders": final_fourier_orders,
            "final_x_resolution_nm": final_x_resolution_nm,
            "final_z_resolution_nm": final_z_resolution_nm,
            "multilayer_bragg_order": multilayer_bragg_order,
            "roughness_sigma_nm": roughness_sigma_nm,
            "precise_peak_selection_mode": precise_peak_selection_mode,
            "case_id": f"{_THETA_SEARCH_CASE_ID_PREFIX}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "diffraction_order": int(diffraction_order),
        }


def monochromator_grazing_angles_deg(
    energy_ev: Sequence[float] | np.ndarray,
    *,
    period_lpermm: float,
    diffraction_order: int = 1,
    cff: float = 2.25,
) -> np.ndarray:
    """Return grazing angles for a fixed-cff monochromator sweep.

    Args:
        energy_ev: Photon energies in electronvolts.
        period_lpermm: Grating line density in lines per millimeter.
        diffraction_order: Diffraction order used in monochromator mode.
        cff: Constant-focus factor.

    Returns:
        Grazing angles in degrees.
    """

    period_nm = 1e6 / period_lpermm

    def solve_alpha_deg(wavelength_nm: float) -> float:
        """Solve the incidence angle for one wavelength."""

        rhs = diffraction_order * wavelength_nm / period_nm
        alpha_min_deg = np.rad2deg(np.arccos(min(1.0, 1.0 / cff))) + 1e-6
        alpha_max_deg = 89.999

        def residual(alpha_deg: float) -> float:
            """Return the fixed-cff monochromator residual."""

            alpha_rad = np.deg2rad(alpha_deg)
            cos_beta = cff * np.cos(alpha_rad)
            if cos_beta < -1.0 or cos_beta > 1.0:
                return np.nan
            beta_rad = np.arccos(cos_beta)
            return np.sin(alpha_rad) - np.sin(beta_rad) - rhs

        return brentq(residual, alpha_min_deg, alpha_max_deg)

    wavelengths_nm = 1239.8 / np.asarray(energy_ev, dtype=float)
    alpha_deg = np.asarray([solve_alpha_deg(float(w)) for w in wavelengths_nm], dtype=float)
    return 90.0 - alpha_deg
