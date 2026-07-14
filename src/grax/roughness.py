"""Roughness models and diagnostics for diffraction post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RoughnessKind = Literal["debye-waller", "random-interface"]


@dataclass(frozen=True)
class RoughnessSpec:
    """Roughness configuration attached to a grating at construction time."""

    kind: RoughnessKind
    sigma_nm: float
    seed: int = 0
    resolution_factor: float = 4.0

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
