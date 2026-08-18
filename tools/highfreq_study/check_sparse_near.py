"""Does the pair-list near band reproduce the dense one, and is it faster?

near_band_operators evaluates the same Kress entries as the classical scheme but
only on the pairs the plane-wave series cannot cover. This checks it against the
dense block entry for entry, then times the projected single-layer block built
both ways.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from grax.solvers._carrier import build_carrier_basis
from grax.solvers._green import PeriodicGreen, default_ewald_splitting
from grax.solvers._nystrom import build_trig_boundary, nystrom_operators
from grax.solvers._spectral_galerkin import (
    near_band_mask,
    near_band_operators,
    projected_blocks,
)

PERIOD = 2500.0
GRAZING_DEG = 2.0


def main() -> None:
    """Verify entry-for-entry agreement, then compare runtime."""

    parser = argparse.ArgumentParser(description="Sparse near-band check")
    parser.add_argument("--energy", type=float, default=50.0)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--depth-nm", type=float, default=20.0)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--delta-nm", type=float, default=1.0)
    args = parser.parse_args()

    wavelength = 1239.8 / args.energy
    k0 = 2.0 * np.pi / wavelength
    alpha0 = k0 * float(np.sin(np.deg2rad(90.0 - GRAZING_DEG)))

    sample = np.linspace(0.0, PERIOD, 1025)
    profile = 0.5 * args.depth_nm * (1.0 - np.cos(2.0 * np.pi * sample / PERIOD))
    boundary = build_trig_boundary(sample, profile, period=PERIOD, count=args.nodes)
    green = PeriodicGreen(
        period=PERIOD,
        wavenumber=k0,
        alpha0=alpha0,
        method="ewald",
        splitting=default_ewald_splitting(PERIOD, k0),
    )
    basis = build_carrier_basis(boundary, alpha0=alpha0, modes=args.modes)
    heights = boundary.position[:, 1]
    fraction = float(np.mean(near_band_mask(heights, min_separation=args.delta_nm)))

    print(
        f"sinusoid depth {args.depth_nm:g} nm, d = {PERIOD:.0f} nm, {args.energy:.0f} eV, "
        f"d/lambda = {PERIOD / wavelength:.0f}, {args.nodes} nodes, "
        f"{2 * args.modes + 1} modes, delta = {args.delta_nm:g} nm"
    )
    print(f"near band is {fraction:.1%} of pairs\n")

    started = time.perf_counter()
    dense, dense_double = nystrom_operators(
        green, target=boundary, source=boundary, same_boundary=True
    )
    dense_seconds = time.perf_counter() - started

    started = time.perf_counter()
    rows, columns, single, double = near_band_operators(
        green, boundary=boundary, min_separation=args.delta_nm
    )
    sparse_seconds = time.perf_counter() - started

    scale = np.max(np.abs(dense))
    print(f"{'entry agreement, single layer':<34}"
          f"{np.max(np.abs(single - dense[rows, columns])) / scale:.3e}")
    print(f"{'entry agreement, double layer':<34}"
          f"{np.max(np.abs(double - dense_double[rows, columns])) / np.max(np.abs(dense_double)):.3e}")

    print(f"\n{'dense nystrom_operators':<34}{dense_seconds:>8.2f} s")
    print(f"{'near-band pair list':<34}{sparse_seconds:>8.2f} s"
          f"   ({dense_seconds / max(sparse_seconds, 1e-9):.1f}x)")

    started = time.perf_counter()
    fast_single, _ = projected_blocks(
        green, boundary=boundary, basis=basis, min_separation=args.delta_nm
    )
    fast_seconds = time.perf_counter() - started
    reference = basis.analysis @ dense @ basis.synthesis
    print(f"\n{'projected block, spliced':<34}{fast_seconds:>8.2f} s")
    print(f"{'rel dev vs full projection':<34}"
          f"{np.max(np.abs(fast_single - reference)) / np.max(np.abs(reference)):.3e}")


if __name__ == "__main__":
    main()
