"""Print Debye-Waller roughness diagnostics for laminar comparison settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "grax-matplotlib"))

import numpy as np  # noqa: E402

from grax.rcwa_1d import debye_waller_roughness_diagnostics  # noqa: E402

SIGMA_VALUES_NM = (0.3, 0.5)
ENERGIES_EV = (100.0, 300.0, 600.0)
GRAZING_ANGLE_DEG = 4.0
NEAR_ZERO_THRESHOLD = 1.0e-3


def print_diagnostics(
    *,
    label: str,
    sigma_nm: float,
    energy_ev: float,
    beta0: float,
    theta_surface_rad: float,
) -> None:
    """Print one Debye-Waller diagnostic block."""

    wavelength_nm = 1239.8 / energy_ev
    diagnostics = debye_waller_roughness_diagnostics(
        sigma_nm=sigma_nm,
        wavelength_nm=wavelength_nm,
        beta0=beta0,
        theta_surface_rad=theta_surface_rad,
    )
    near_zero = diagnostics["damping_factor"] < NEAR_ZERO_THRESHOLD
    warning = "  <-- near-zero damping" if near_zero else ""
    print(f"  {label}")
    print(f"    sigma_nm={diagnostics['sigma_nm']:.8g}")
    print(f"    wavelength_nm={diagnostics['wavelength_nm']:.8g}")
    print(f"    theta_surface_rad={diagnostics['theta_surface_rad']:.8g}")
    print(f"    theta_normal_rad={diagnostics['theta_normal_rad']:.8g}")
    print(f"    beta0={diagnostics['beta0']:.8g}")
    print(f"    sin_theta_used={diagnostics['sin_theta_used']:.8g}")
    print(f"    A={diagnostics['A']:.8g}")
    print(f"    A_squared={diagnostics['A_squared']:.8g}")
    print(f"    exp(-A_squared)={diagnostics['damping_factor']:.8g}{warning}")


grazing_rad = np.deg2rad(GRAZING_ANGLE_DEG)
beta0 = np.cos(grazing_rad)
previous_incorrect_factor = beta0
corrected_factor = np.sin(grazing_rad)

print("Debye-Waller roughness diagnostics")
print(f"grazing_angle_deg={GRAZING_ANGLE_DEG}")
print("fixed implementation uses sin(grazing angle) = sqrt(1 - aa.beta0^2)")
print("previous incorrect implementation used aa.beta0 = cos(grazing angle)")
print()

for energy_ev in ENERGIES_EV:
    for sigma_nm in SIGMA_VALUES_NM:
        wavelength_nm = 1239.8 / energy_ev
        previous_argument = (
            4.0 * np.pi * sigma_nm * previous_incorrect_factor / wavelength_nm
        )
        previous_damping = np.exp(-(previous_argument**2))
        print(f"energy_ev={energy_ev:.1f}, sigma_nm={sigma_nm:.3f}")
        previous_warning = (
            "  <-- previous near-zero damping"
            if previous_damping < NEAR_ZERO_THRESHOLD
            else ""
        )
        print("  previous incorrect convention: factor=cos(grazing)")
        print(f"    angular_factor={previous_incorrect_factor:.8g}")
        print(f"    A={previous_argument:.8g}")
        print(f"    A_squared={previous_argument**2:.8g}")
        print(f"    exp(-A_squared)={previous_damping:.8g}{previous_warning}")
        print_diagnostics(
            label="fixed convention: factor=sin(grazing)=sqrt(1-beta0^2)",
            sigma_nm=sigma_nm,
            energy_ev=energy_ev,
            beta0=beta0,
            theta_surface_rad=grazing_rad,
        )
        print(f"  fixed angular factor check={corrected_factor:.8g}")
        print()
