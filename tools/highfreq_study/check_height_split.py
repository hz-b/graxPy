"""Verify the height-split prefix-sum algebra against a brute-force double sum.

The factorization is exact, so this is a pure bookkeeping check and it is cheap:
it compares :func:`height_split_projection` against the same spectral sum
evaluated as an explicit ``O(N^2)`` double loop over node pairs, on the same
orders and the same quadrature. Any discrepancy is a masking or prefix-sum bug,
not a physics question.
"""

from __future__ import annotations

import numpy as np

from grax.solvers._spectral_galerkin import height_split_projection

PERIOD = 2500.0


def brute_force(*, x, heights, jacobian, modes, beta, orders, period):
    """Return the same projected block by explicit summation over node pairs."""

    nodes = x.size
    weight = period / nodes
    block = np.zeros((modes.size, modes.size), dtype=complex)
    gap = np.subtract.outer(heights, heights)
    distinct = gap != 0.0
    for index, spectral_order in enumerate(orders):
        kernel = np.where(distinct, np.exp(1j * beta[index] * np.abs(gap)), 0.0)
        for a, m_target in enumerate(modes):
            phase_t = np.exp(2j * np.pi * (spectral_order - m_target) * x / period) / nodes
            for b, m_source in enumerate(modes):
                phase_s = (
                    np.exp(-2j * np.pi * (spectral_order - m_source) * x / period)
                    * jacobian
                    * weight
                )
                block[a, b] += phase_t @ kernel @ phase_s / beta[index]
    return 0.5j / period * block


def main() -> None:
    """Compare the two on a corrugated profile with generic heights."""

    nodes = 96
    x = np.arange(nodes) * PERIOD / nodes
    # Deliberately asymmetric so heights are distinct and ties are rare, which is
    # the generic case; ties are checked separately below.
    heights = 12.0 * np.sin(2 * np.pi * x / PERIOD) + 4.0 * np.sin(6 * np.pi * x / PERIOD + 0.7)
    slope = (
        12.0 * (2 * np.pi / PERIOD) * np.cos(2 * np.pi * x / PERIOD)
        + 4.0 * (6 * np.pi / PERIOD) * np.cos(6 * np.pi * x / PERIOD + 0.7)
    )
    jacobian = np.sqrt(1.0 + slope**2)
    modes = np.arange(-2, 3)
    orders = np.arange(-6, 7)
    k = 2 * np.pi / 3.0
    alpha0 = k * float(np.sin(np.deg2rad(88.0)))
    alpha = alpha0 + 2 * np.pi * orders / PERIOD
    beta = np.sqrt(k**2 - alpha**2 + 0j)
    beta = np.where(np.imag(beta) < 0, -beta, beta)

    kwargs = dict(
        x=x, heights=heights, jacobian=jacobian, modes=modes,
        beta=beta, orders=orders, period=PERIOD,
    )
    fast = height_split_projection(**kwargs)
    slow = brute_force(**kwargs)
    scale = np.max(np.abs(slow))
    print(f"block {fast.shape}, {orders.size} orders, {nodes} nodes")
    print(f"  max |fast - brute| / |brute| = {np.max(np.abs(fast - slow)) / scale:.3e}")

    # A flat profile is all ties: every pair carries |Y| = 0, so the series part
    # must vanish identically and the whole block belongs to the near band.
    flat = height_split_projection(**{**kwargs, "heights": np.zeros(nodes)})
    print(f"  flat profile, far-field block max = {np.max(np.abs(flat)):.3e} (must be 0)")


if __name__ == "__main__":
    main()
