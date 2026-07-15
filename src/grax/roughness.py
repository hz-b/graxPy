"""Roughness models and diagnostics for diffraction post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RoughnessKind = Literal["debye-waller", "random-interface"]


@dataclass(frozen=True)
class RoughnessSpec:
    """Roughness configuration attached to a grating at construction time.

    Attributes:
        kind: Roughness model, ``"debye-waller"`` or ``"random-interface"``.
        sigma_nm: Rms roughness height in nanometers.
        seed: Deterministic seed for ``"random-interface"`` field generation.
        resolution_factor: Grid-resolution safety factor for the underresolved
            roughness warning.
        correlation_length_nm: Lateral autocorrelation length of the
            ``"random-interface"`` roughness, in nanometers. The interface is a
            Gaussian random field with autocorrelation
            ``C(tau) = sigma**2 * exp(-tau**2 / (2 * xi**2))``. When ``None`` the
            correlation length defaults to one tenth of the grating period, which
            keeps visible per-period structure at any pitch. ``0.0`` reproduces
            the legacy uncorrelated (white-noise) interface. Real metrology shows
            correlation lengths of order ~10 um; on fine-pitch gratings such long
            correlations wash out geometrically and are better modelled by the
            ``"debye-waller"`` kind. Note that this period-relative default does
            **not** scale with ``num_supercells``: to see any effect from a
            larger supercell, set ``correlation_length_nm`` explicitly (often
            close to or above one grating period).
        num_supercells: Number of grating periods over which the
            ``"random-interface"`` roughness field is generated as one
            continuous correlated Gaussian random field, instead of one
            period. Only meaningful for ``kind == "random-interface"``; must
            be ``1`` for ``"debye-waller"``, which has no geometric structure
            to correlate across periods. ``num_supercells > 1`` changes the
            RCWA solver's fundamental Fourier period to
            ``num_supercells * grating.period_nm``, letting the simulation
            resolve diffuse/satellite diffraction orders introduced by the
            disorder. This increases solver cost roughly with the cube of
            ``num_supercells`` (see ``run_simulation``'s Fourier-order
            warning), so start small.
    """

    kind: RoughnessKind
    sigma_nm: float
    seed: int = 0
    resolution_factor: float = 4.0
    correlation_length_nm: float | None = None
    num_supercells: int = 1

    def __post_init__(self) -> None:
        """Validate roughness configuration."""

        if self.kind not in {"debye-waller", "random-interface"}:
            raise ValueError(
                "roughness kind must be 'debye-waller' or 'random-interface'."
            )
        if self.sigma_nm < 0.0:
            raise ValueError("roughness sigma_nm must be >= 0.")
        if self.resolution_factor <= 0.0:
            raise ValueError("roughness resolution_factor must be > 0.")
        if self.correlation_length_nm is not None and self.correlation_length_nm < 0.0:
            raise ValueError("roughness correlation_length_nm must be >= 0 when provided.")
        if isinstance(self.num_supercells, bool) or not isinstance(self.num_supercells, int):
            raise ValueError("roughness num_supercells must be an int.")
        if self.num_supercells < 1:
            raise ValueError("roughness num_supercells must be >= 1.")
        if self.kind == "debye-waller" and self.num_supercells != 1:
            raise ValueError(
                "roughness num_supercells > 1 is only meaningful for kind='random-interface'."
            )


def apply_debye_waller_roughness(
    *,
    reflected: object,
    transmitted: object,
    wavelength_nm: float,
    beta0: float,
    roughness_sigma_nm: float,
) -> tuple[object, object]:
    """Apply scalar Debye-Waller damping to reflected/transmitted efficiencies."""

    incidence_sine = incidence_sine_from_beta0(beta0)
    return (
        _apply_debye_waller_roughness(
            reflected,
            wavelength_nm=wavelength_nm,
            incidence_sine=incidence_sine,
            roughness_sigma_nm=roughness_sigma_nm,
        ),
        _apply_debye_waller_roughness(
            transmitted,
            wavelength_nm=wavelength_nm,
            incidence_sine=incidence_sine,
            roughness_sigma_nm=roughness_sigma_nm,
        ),
    )


def _apply_debye_waller_roughness(
    result: object,
    *,
    wavelength_nm: float,
    incidence_sine: float,
    roughness_sigma_nm: float,
) -> object:
    """Return diffraction efficiencies damped by scalar roughness losses."""

    damping = _debye_waller_roughness_factor(
        wavelength_nm=wavelength_nm,
        incidence_sine=incidence_sine,
        roughness_sigma_nm=roughness_sigma_nm,
    )
    result_type = result.__class__
    return result_type(
        order=result.order,  # type: ignore[attr-defined]
        theta=result.theta,  # type: ignore[attr-defined]
        efficiency=result.efficiency * damping,  # type: ignore[attr-defined]
        amplitude=result.amplitude,  # type: ignore[attr-defined]
    )


def _debye_waller_roughness_factor(
    *,
    wavelength_nm: float,
    incidence_sine: float,
    roughness_sigma_nm: float | None,
) -> float:
    """Return scalar Debye-Waller damping for rms roughness."""

    if roughness_sigma_nm is None:
        return 1.0
    return float(np.exp(-((4.0 * np.pi * roughness_sigma_nm * incidence_sine / wavelength_nm) ** 2)))


def incidence_sine_from_beta0(beta0: float) -> float:
    """Return sin(grazing angle) from the solver ``beta0`` convention."""

    clipped_beta0 = float(np.clip(beta0, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - clipped_beta0**2)))


def debye_waller_roughness_diagnostics(
    *,
    sigma_nm: float,
    wavelength_nm: float,
    beta0: float,
    theta_surface_rad: float | None = None,
) -> dict[str, float]:
    """Return diagnostic quantities for Debye-Waller roughness damping."""

    if sigma_nm < 0.0:
        raise ValueError("sigma_nm must be >= 0.")
    if wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be > 0.")

    clipped_beta0 = float(np.clip(beta0, -1.0, 1.0))
    resolved_theta_surface_rad = (
        float(np.arccos(clipped_beta0))
        if theta_surface_rad is None
        else float(theta_surface_rad)
    )
    theta_normal_rad = (np.pi / 2.0) - resolved_theta_surface_rad
    incidence_sine = incidence_sine_from_beta0(clipped_beta0)
    argument = 4.0 * np.pi * sigma_nm * incidence_sine / wavelength_nm
    argument_squared = argument**2
    damping_factor = float(np.exp(-argument_squared))
    return {
        "sigma_nm": float(sigma_nm),
        "wavelength_nm": float(wavelength_nm),
        "theta_surface_rad": resolved_theta_surface_rad,
        "theta_normal_rad": float(theta_normal_rad),
        "beta0": clipped_beta0,
        "sin_theta_used": incidence_sine,
        "A": float(argument),
        "A_squared": float(argument_squared),
        "damping_factor": damping_factor,
    }
