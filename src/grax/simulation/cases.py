"""Lazy case-generation helpers for simulation sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import brentq

from ..gratings import BaseGrating
from ..peak_fitting import PeakSelectionMode

def fixed_angle_cases(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    grazing_angle_deg: float,
    case_id_prefix: str = "fixed",
    **case_defaults: object,
) -> Iterator[dict[str, object]]:
    """Yield fixed-angle energy-sweep cases lazily.

    Creates case dictionaries for a fixed-incident-angle energy scan. Each case
    includes the grating, photon energy, and fixed grazing angle. This is the
    simplest sweep configuration where all simulations use the same incidence
    geometry.

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        case_id_prefix: Prefix used for stable generated case IDs.
        **case_defaults: Additional case fields copied into every yielded case.

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
            **case_defaults,
            "case_id": f"{case_id_prefix}-{index:08d}",
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
    case_id_prefix: str = "mono",
    **case_defaults: object,
) -> Iterator[dict[str, object]]:
    """Yield fixed-cff monochromator sweep cases lazily.

    Creates case dictionaries for a monochromator configuration with fixed
    constant-focus-factor (CFF). The grazing angle is solved for each energy
    using the monochromator equation, ensuring the optic maintains the same
    optical configuration across the energy sweep.

    The monochromator relation solves for incidence angle alpha and diffraction
    angle beta subject to:
    - cff = cos(alpha) / cos(beta) (fixed)
    - m*lambda = period * (sin(alpha) + sin(beta)) (diffraction condition)

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        period_lpermm: Optional grating line density. Defaults to the grating value.
        diffraction_order: Positive diffraction order used by the monochromator relation.
        cff: Constant-focus factor. Higher values relax tolerance but reduce angular range.
        case_id_prefix: Prefix used for stable generated case IDs.
        **case_defaults: Additional case fields copied into every yielded case.

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
            **case_defaults,
            "case_id": f"{case_id_prefix}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": grazing_angle,
            "diffraction_order": diffraction_order,
        }


def energy_angle_cases(
    *,
    grating: BaseGrating,
    energy_angle_pairs: Iterable[tuple[float, float]],
    case_id_prefix: str = "pair",
    **case_defaults: object,
) -> Iterator[dict[str, object]]:
    """Yield arbitrary energy-angle simulation cases lazily.

    Creates case dictionaries for user-specified energy and angle combinations.
    This provides maximum flexibility for custom sweep configurations that don't
    follow standard patterns like fixed-angle or monochromator scans.

    Args:
        grating: Grating reused by every generated case.
        energy_angle_pairs: Iterable of ``(energy_ev, grazing_angle_deg)`` pairs.
        case_id_prefix: Prefix used for stable generated case IDs.
        **case_defaults: Additional case fields copied into every yielded case.

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
            **case_defaults,
            "case_id": f"{case_id_prefix}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": float(grazing_angle_deg),
        }


def multilayer_theta_search_cases(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    case_id_prefix: str = "theta-search",
    precise_peak_selection_mode: PeakSelectionMode = "max",
    **case_defaults: object,
) -> Iterator[dict[str, object]]:
    """Yield energy-only cases for the multilayer theta-search workflow.

    Creates simplified case dictionaries that trigger the multilayer theta-search
    workflow during batch execution. The workflow internally performs a three-stage
    adaptive angular scan:
    1. **Rough scan**: Wide angular range with low Fourier orders and coarse resolution
    2. **Precise scan**: Narrow range around rough maximum with moderate settings
    3. **Final solve**: High-resolution calculation at selected angle

    This workflow automatically determines the optimal grazing angle for each
    energy point, making it ideal for multilayer gratings where the Bragg angle
    shifts with photon energy.

    Args:
        grating: Grating reused by every generated case.
        energies_ev: Iterable of photon energies in electronvolts.
        case_id_prefix: Prefix used for stable generated case IDs.
        precise_peak_selection_mode: Mode used to select the final theta from the
            precise scan. ``"max"`` uses the sampled maximum, ``"gauss"`` fits a
            local Gaussian, and ``"voigt"`` fits a local Voigt profile.
        **case_defaults: Additional case fields copied into every yielded case.

    Yields:
        Case dictionaries suitable for :class:`BatchSimulationRunner` with
        workflow-specific fields stored in the case dictionary.

    Example:
        >>> grating = grax.BlazedGrating(period_nm=500, blaze_angle=10, height_nm=100)
        >>> cases = multilayer_theta_search_cases(
        ...     grating=grating,
        ...     energies_ev=[500, 600, 700],
        ...     precise_peak_selection_mode="gauss"
        ... )
    """

    default_fields = {
        "workflow": "multilayer_theta_search",
        "rough_fourier_orders": 3,
        "fine_fourier_orders": 5,
        "final_fourier_orders": 25,
        "rough_x_resolution_nm": 1.0,
        "rough_z_resolution_nm": 1.0,
        "fine_x_resolution_nm": 0.5,
        "fine_z_resolution_nm": 0.5,
        "final_x_resolution_nm": 0.3,
        "final_z_resolution_nm": 0.3,
        "rough_scan_half_width_deg": 0.5,
        "rough_scan_points": 82,
        "precise_scan_half_width_deg": 0.1,
        "precise_scan_points": 81,
        "multilayer_bragg_order": 1,
        "precise_peak_selection_mode": precise_peak_selection_mode,
    }
    default_fields.update(case_defaults)

    for index, energy_ev in enumerate(energies_ev):
        yield {
            **default_fields,
            "case_id": f"{case_id_prefix}-{index:08d}",
            "grating": grating,
            "energy_ev": float(energy_ev),
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

