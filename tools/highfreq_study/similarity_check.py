"""Is u = exp(i alpha0 x) u_tilde, on the same node set, actually a new scheme?

Substituting the carrier out of the unknowns rescales column ``j`` of every
operator block by ``exp(i alpha0 x_j)`` and row ``i`` by ``exp(-i alpha0 x_i)``.
That is a diagonal similarity transform of the assembled system, so it cannot
change the solution, the conditioning, or the quadrature error at a given node
count. This script checks that claim numerically before anything is built on it,
because if it holds then carrier extraction has to change the *representation* of
the unknown, not just its phase.
"""

from __future__ import annotations

import numpy as np

from grax.solvers._green import PeriodicGreen, default_ewald_splitting
from grax.solvers._nystrom import build_trig_boundary, nystrom_operators

PERIOD = 2500.0
GRAZING_DEG = 2.0


def main() -> None:
    """Compare the classical block against its carrier-extracted counterpart."""

    for energy in (100.0, 500.0):
        wavelength = 1239.8 / energy
        k0 = 2.0 * np.pi / wavelength
        alpha0 = k0 * float(np.sin(np.deg2rad(90.0 - GRAZING_DEG)))
        count = 128
        x = np.linspace(0.0, PERIOD, 129)
        boundary = build_trig_boundary(
            x, 0.02 * PERIOD * np.sin(2 * np.pi * x / PERIOD), period=PERIOD, count=count
        )
        green = PeriodicGreen(
            period=PERIOD,
            wavenumber=k0,
            alpha0=alpha0,
            method="ewald",
            splitting=default_ewald_splitting(PERIOD, k0),
        )
        single, _ = nystrom_operators(
            green, target=boundary, source=boundary, same_boundary=True
        )

        # The carrier-extracted block, formed exactly as the transformed
        # integral equation prescribes: kernel times exp(-i alpha0 (x_t - x_s)).
        xs = boundary.position[:, 0]
        carrier = np.exp(1j * alpha0 * xs)
        extracted = single * np.conj(carrier)[:, None] * carrier[None, :]
        rebuilt = np.diag(1.0 / carrier) @ single @ np.diag(carrier)

        print(f"E = {energy:.0f} eV, d/lambda = {PERIOD / wavelength:.0f}")
        print(
            "  extracted block equals D^-1 A D exactly: "
            f"{np.allclose(extracted, rebuilt, rtol=0, atol=1e-300)}"
        )
        print(
            f"  cond(classical) = {np.linalg.cond(single):.6e}\n"
            f"  cond(extracted) = {np.linalg.cond(extracted):.6e}"
        )
        eig_a = np.sort_complex(np.linalg.eigvals(single))
        eig_b = np.sort_complex(np.linalg.eigvals(extracted))
        print(f"  max |eigenvalue difference| = {np.max(np.abs(eig_a - eig_b)):.3e}")


if __name__ == "__main__":
    main()
