"""Does near band plus factorized far field reproduce the full projected block?

The two sets partition the node pairs by construction, so agreement is a
statement about quadrature, not bookkeeping: the near band is integrated by the
classical Kress rule and the far field by the plane-wave series, and the question
is whether the series is summed far enough for the smallest gap it is asked to
resolve.

Run on a corrugated profile at a modest d/lambda so it is seconds, not minutes.
"""

from __future__ import annotations

import argparse

import numpy as np

from grax.solvers._carrier import build_carrier_basis
from grax.solvers._green import PeriodicGreen, default_ewald_splitting
from grax.solvers._nystrom import build_trig_boundary, nystrom_operators
from grax.solvers._spectral_galerkin import (
    max_stable_reach,
    near_band_mask,
    projected_single_layer,
    spectral_reach_for_separation,
)

PERIOD = 2500.0
GRAZING_DEG = 2.0


def main() -> None:
    """Compare the spliced block against the full nodal projection."""

    parser = argparse.ArgumentParser(description="Near/far splice check")
    parser.add_argument("--energy", type=float, default=50.0)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--depth-nm", type=float, default=120.0)
    parser.add_argument("--modes", type=int, default=4)
    args = parser.parse_args()

    wavelength = 1239.8 / args.energy
    k0 = 2.0 * np.pi / wavelength
    alpha0 = k0 * float(np.sin(np.deg2rad(90.0 - GRAZING_DEG)))

    sample = np.linspace(0.0, PERIOD, 513)
    profile = 0.5 * args.depth_nm * (1.0 - np.cos(2.0 * np.pi * sample / PERIOD))
    boundary = build_trig_boundary(sample, profile, period=PERIOD, count=args.nodes)

    green = PeriodicGreen(
        period=PERIOD,
        wavenumber=k0,
        alpha0=alpha0,
        method="ewald",
        splitting=default_ewald_splitting(PERIOD, k0),
    )
    nodal, _ = nystrom_operators(green, target=boundary, source=boundary, same_boundary=True)
    basis = build_carrier_basis(boundary, alpha0=alpha0, modes=args.modes)

    x = boundary.position[:, 0]
    heights = boundary.position[:, 1]
    jacobian = boundary.speed / (PERIOD / (2.0 * np.pi))

    full = basis.analysis @ nodal @ basis.synthesis
    scale = np.max(np.abs(full))

    print(
        f"sinusoid depth {args.depth_nm:g} nm, d = {PERIOD:.0f} nm, "
        f"{args.energy:.0f} eV, d/lambda = {PERIOD / wavelength:.0f}, "
        f"{args.nodes} nodes, {2 * args.modes + 1} modes"
    )
    print(f"\n{'delta (nm)':>11} {'near pairs':>11} {'far reach':>10} {'rel dev vs full':>16}")
    print("-" * 52)

    for delta in (2.0, 5.0, 10.0, 20.0, 40.0):
        reach = spectral_reach_for_separation(period=PERIOD, separation=delta, decades=14.0)
        orders = np.arange(-reach, reach + 1)
        alpha = alpha0 + 2.0 * np.pi * orders / PERIOD
        beta = np.sqrt(k0**2 - alpha**2 + 0j)
        beta = np.where(np.imag(beta) < 0, -beta, beta)

        limit = max_stable_reach(period=PERIOD, height_range=float(np.ptp(heights)))
        if reach > limit:
            print(
                f"{delta:>11.1f} {float(np.mean(near_band_mask(heights, min_separation=delta))):>10.1%} "
                f"{2 * reach + 1:>10d} {'reach > ' + str(limit) + ' (unstable)':>16}"
            )
            continue

        spliced = projected_single_layer(
            nodal_block=nodal,
            basis=basis,
            x=x,
            heights=heights,
            jacobian=jacobian,
            beta=beta,
            orders=orders,
            period=PERIOD,
            min_separation=delta,
        )
        fraction = float(np.mean(near_band_mask(heights, min_separation=delta)))
        print(
            f"{delta:>11.1f} {fraction:>10.1%} {2 * reach + 1:>10d} "
            f"{np.max(np.abs(spliced - full)) / scale:>16.3e}",
            flush=True,
        )

    print(
        "\nSmaller delta shrinks the near band but lengthens the series that has to\n"
        "cover it; the deviation is dominated by whichever of the two is starved."
    )


if __name__ == "__main__":
    main()
