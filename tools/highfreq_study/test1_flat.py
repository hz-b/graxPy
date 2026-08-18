"""Test 1: flat interface, fixed period, energy swept over a 40x range in d/lambda.

A flat interface couples no diffraction orders, so its envelope is *exactly*
constant: the carrier-extracted formulation should need one harmonic per density
at any energy, while the classical formulation has to resolve the physical
boundary field and so must track d/lambda. The reference is analytic Fresnel, so
there is no reference error to hide behind.

The table separates the two requirements deliberately, because they are two
different things and only one of them is removed by carrier extraction:

``modes``
    ``2 M + 1`` envelope harmonics per density -- the actual unknown count, and
    the quantity that should stop scaling with energy.
``Nquad``
    Nodes carrying the kernel quadrature. The projection is formed from the
    classical nodal blocks, so this still has to resolve the *kernel*, whose
    Fourier content spans the diffraction orders.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import grax
from grax.solvers._carrier import build_carrier_basis, res2_hf
from grax.solvers.integral import IntegralOptions, build_stack, res2_im

PERIOD_NM = 2500.0
GRAZING_DEG = 2.0
POLARIZATION = -1
ENERGIES = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def flat_grating():
    """Return a flat Si surface with the benchmark period."""

    return grax.ProfileGrating(
        period_lpermm=1e6 / PERIOD_NM,
        x_points_nm=np.array([0.0, PERIOD_NM]),
        z_points_nm=np.array([0.0, 0.0]),
        substrate_material="Si",
        layer_material="Si",
        layer_thickness_nm=0.0,
        z_resolution_nm=0.5,
        x_resolution_nm=5.0,
    )


def fresnel(energy_ev: float, polarization: int) -> float:
    """Return the analytic specular reflectance of the flat interface."""

    index = complex(grax.materials.resolve_refractive_index("Si", energy_ev))
    cos_i = float(np.sin(np.deg2rad(GRAZING_DEG)))
    sin_i = float(np.cos(np.deg2rad(GRAZING_DEG)))
    cos_t = np.sqrt(1.0 - (sin_i / index) ** 2 + 0j)
    if polarization == 1:
        reflection = (cos_i - index * cos_t) / (cos_i + index * cos_t)
    else:
        reflection = (index * cos_i - cos_t) / (index * cos_i + cos_t)
    return float(np.abs(reflection) ** 2)


def main() -> None:
    """Sweep energy and report the two requirements separately."""

    parser = argparse.ArgumentParser(description="Flat-interface carrier-extraction ladder")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--energies", type=float, nargs="*", default=list(ENERGIES))
    args = parser.parse_args()

    grating = flat_grating()
    beta0 = float(np.sin(np.deg2rad(90.0 - GRAZING_DEG)))
    orders = np.arange(-2, 3, dtype=float)

    print(
        f"flat Si, d = {PERIOD_NM:.0f} nm, {GRAZING_DEG} deg grazing, TM, "
        f"tolerance {args.tolerance:g} absolute on specular efficiency"
    )
    print(
        f"\n{'E (eV)':>7} {'d/lam':>6} {'Fresnel':>9} | "
        f"{'classical N':>11} {'dev':>8} {'secs':>6} | "
        f"{'hf modes':>8} {'hf Nquad':>8} {'dev':>8} {'secs':>6} {'gram':>8}"
    )
    print("-" * 104)

    for energy in args.energies:
        wavelength = 1239.8 / energy
        reference = fresnel(energy, POLARIZATION)
        common = dict(
            grating=grating,
            wavelength_nm=wavelength,
            period_nm=PERIOD_NM,
            orders=orders,
            beta0=beta0,
            polarization=POLARIZATION,
            photon_energy_ev=energy,
        )

        def deviation(result) -> float:
            """Return the absolute error of the specular order."""

            return abs(float(np.real(result.inc_top_reflected.efficiency[2])) - reference)

        classical = [None, float("nan"), float("nan")]
        nodes = 64
        while nodes <= args.max_nodes:
            started = time.perf_counter()
            try:
                found = deviation(
                    res2_im(
                        **common,
                        options=IntegralOptions(
                            boundary_points=nodes,
                            discretization="nystrom",
                            corner_grading=1.0,
                        ),
                    )
                )
            except Exception:
                found = float("inf")
            elapsed = time.perf_counter() - started
            if found <= args.tolerance:
                classical = [nodes, found, elapsed]
                break
            nodes *= 2

        # The unknown count is swept at the node count the *quadrature* needs,
        # which is the point: they are independent requirements.
        quad = classical[0] or args.max_nodes
        high = [None, float("nan"), float("nan")]
        for modes in (0, 1, 2, 4, 8):
            started = time.perf_counter()
            try:
                found = deviation(
                    res2_hf(**common, envelope_modes=modes, quadrature_nodes=quad)
                )
            except Exception:
                found = float("inf")
            elapsed = time.perf_counter() - started
            if found <= args.tolerance:
                high = [2 * modes + 1, found, elapsed]
                break

        stack = build_stack(
            grating,
            photon_energy_ev=energy,
            wavelength_nm=wavelength,
            n_inc=1.0 + 0.0j,
            orders=2,
            options=IntegralOptions(
                boundary_points=quad, discretization="nystrom", corner_grading=1.0
            ),
        )
        basis = build_carrier_basis(
            stack.interfaces[0], alpha0=2.0 * np.pi / wavelength * beta0, modes=4
        )

        print(
            f"{energy:>7.0f} {PERIOD_NM / wavelength:>6.0f} {reference:>9.6f} | "
            f"{str(classical[0]):>11} {classical[1]:>8.1e} {classical[2]:>6.1f} | "
            f"{str(high[0]):>8} {quad:>8} {high[1]:>8.1e} {high[2]:>6.1f} "
            f"{basis.orthogonality_defect():>8.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
