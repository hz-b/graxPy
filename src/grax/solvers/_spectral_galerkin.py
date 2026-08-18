"""Projected operator blocks built directly from the plane-wave series.

:mod:`grax.solvers._carrier` removed the first high-frequency bottleneck: the
unknown count stopped tracking ``d / lambda``, because the envelope is smooth and
a handful of Floquet harmonics carries it. It did not touch the second one. The
projection there is formed *from* the classical nodal blocks, so a
``(N_quad, N_quad)`` matrix is still assembled before being squeezed down to
``(2M+1, 2M+1)``, and that assembly is 99.4% of the runtime.

This module builds the projected blocks without ever forming the nodal matrix.

Why the carrier disappears
--------------------------
The projected element is

    A[m', m] = (1/d) int dx_t exp(-i alpha_m' x_t)
                     int dx_s G(r_t, r_s) exp(i alpha_m x_s) J_s

and substituting the plane-wave form of the quasi-periodic Green function,
``G = (i/2d) sum_n (1/beta_n) exp(i alpha_n X) exp(i beta_n |Y|)``, makes the
Floquet phase cancel analytically:

    exp(-i alpha_m' x_t) exp(i alpha_n (x_t - x_s)) exp(i alpha_m x_s)
      = exp(i 2 pi (n - m') x_t / d) exp(-i 2 pi (n - m) x_s / d)

Every surviving factor is a plain integer harmonic. Nothing in the integrand
oscillates at ``alpha_0``, so the quadrature is set by ``|n - m|`` and by the
profile rather than by the wavelength.

The height split
----------------
What blocks a naive outer product is ``|Y| = |f(x_t) - f(x_s)|``: the modulus
couples the two integrals. Dropping it factorizes the whole thing immediately and
is the Rayleigh hypothesis, which is not acceptable for a rigorous solver. So the
modulus is resolved exactly instead, by splitting the *integration domain* at
``f_t = f_s``:

    f_t > f_s :  exp(i beta_n (f_t - f_s)) = exp(i beta_n f_t) exp(-i beta_n f_s)
    f_t < f_s :  exp(i beta_n (f_s - f_t)) = exp(-i beta_n f_t) exp(i beta_n f_s)

Each region factorizes. Sorting the nodes by height turns the region constraint
into a triangular mask, and a triangular mask against a separable summand is a
prefix sum -- ``O(N_quad)`` per order rather than ``O(N_quad^2)``. This is exact:
no hypothesis about the profile is used, only that it is a graph.

Cost, and what is still missing
-------------------------------
Per spectral order the work is ``O(M N_quad)`` for the prefix sums plus
``O(M^2 N_quad)`` for the projections, against ``O(N_quad^2)`` for the nodal
assembly, so the ratio is roughly ``N_quad / M^2``.

The part this module does *not* solve is the near-singular one. The plane-wave
series converges geometrically in ``|n|`` only while ``|beta_n| |Y|`` grows;
where two nodes sit at nearly the same height -- including the diagonal, and
including every pair on a flat surface -- it stalls, which is exactly why the
classical path uses Ewald summation. :func:`projected_blocks` therefore takes the
near band from the existing Ewald kernel and only the far field from the series.
On a shallow profile the near band is a large fraction of the pairs, so the win
is real but bounded; a genuine ``lambda / d ~ 1e-3`` method needs the singular
part summed in closed form as well, which is the next piece of work and is not
attempted here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "height_split_projection",
    "spectral_reach_for_separation",
]


def spectral_reach_for_separation(
    *, period: float, separation: float, decades: float = 16.0
) -> int:
    """Return how far the plane-wave series must run for a given height gap.

    Deep evanescent orders carry ``exp(-2 pi |n| |Y| / d)``, so the series is
    geometric once the orders are evanescent and the reach is set by the *height
    separation*, not by the wavelength. That is the whole reason this
    representation is cheaper away from the diagonal.

    Args:
        period: Grating period in nanometers.
        separation: Vertical gap ``|Y|`` in nanometers.
        decades: Requested decades of decay.

    Returns:
        Half-width of the order range, at least one.

    Raises:
        ValueError: If the separation is not positive, where the series does not
            converge and the near-field path has to be used instead.
    """

    if separation <= 0.0:
        raise ValueError(
            "The plane-wave series does not converge at zero height separation; "
            "that pair belongs to the near band and has to come from the Ewald "
            "kernel."
        )
    return max(1, int(np.ceil(decades * np.log(10.0) * period / (2.0 * np.pi * separation))))


def height_split_projection(
    *,
    x: np.ndarray,
    heights: np.ndarray,
    jacobian: np.ndarray,
    modes: np.ndarray,
    beta: np.ndarray,
    orders: np.ndarray,
    period: float,
) -> np.ndarray:
    """Return the far-field part of one projected single-layer block.

    Evaluates

        (i / 2d) sum_n (1/beta_n) (1/d) int int
            exp(i 2 pi (n - m') x_t / d) exp(-i 2 pi (n - m) x_s / d)
            exp(i beta_n |f_t - f_s|) J_s dx_s dx_t

    over the two open regions ``f_t > f_s`` and ``f_t < f_s``, where the modulus
    resolves and the exponential separates. Pairs at exactly equal height carry
    ``|Y| = 0``, where the series does not converge; they are excluded here and
    belong to the near band, which the caller supplies from the Ewald kernel.

    The region constraint is applied by sorting on height and reading prefix sums
    at the strict rank of each target, so ties are excluded exactly rather than
    by an arbitrary tie-break, and the cost is ``O(M N)`` per spectral order
    instead of ``O(N^2)``.

    Args:
        x: Uniform quadrature abscissae over one period, shape ``(nodes,)``.
        heights: Profile height at each abscissa, shape ``(nodes,)``.
        jacobian: Arc-length Jacobian ``sqrt(1 + f'^2)``, shape ``(nodes,)``.
        modes: Envelope harmonics ``m``, shape ``(count,)``.
        beta: Out-of-plane wavenumber per spectral order, shape ``(reach,)``.
        orders: Spectral orders ``n`` aligned with ``beta``, shape ``(reach,)``.
        period: Grating period in nanometers.

    Returns:
        Complex block shaped ``(count, count)``, indexed ``[m', m]``.
    """

    nodes = int(x.size)
    count = int(modes.size)
    weight = period / nodes

    harmonic_t = np.exp(-2j * np.pi * modes[:, None] * x[None, :] / period) / nodes
    harmonic_s = np.exp(2j * np.pi * modes[:, None] * x[None, :] / period) * (
        jacobian[None, :] * weight
    )

    order = np.argsort(heights, kind="stable")
    sorted_heights = heights[order]
    # Strict ranks: how many sources sit strictly below, and strictly above,
    # each target height. Equal heights fall in neither and are dropped.
    below = np.searchsorted(sorted_heights, heights, side="left")
    above = np.searchsorted(sorted_heights, heights, side="right")

    block = np.zeros((count, count), dtype=complex)
    for index, spectral_order in enumerate(orders):
        beta_n = beta[index]
        carrier_t = np.exp(2j * np.pi * spectral_order * x / period)
        carrier_s = np.exp(-2j * np.pi * spectral_order * x / period)

        for sign, rank, take_prefix in ((1.0, below, True), (-1.0, above, False)):
            target = harmonic_t * (carrier_t * np.exp(1j * sign * beta_n * heights))[None, :]
            source = harmonic_s * (carrier_s * np.exp(-1j * sign * beta_n * heights))[None, :]

            ordered = source[:, order]
            cumulative = np.zeros((count, nodes + 1), dtype=complex)
            cumulative[:, 1:] = np.cumsum(ordered, axis=1)
            if take_prefix:
                gathered = cumulative[:, rank]
            else:
                gathered = cumulative[:, nodes][:, None] - cumulative[:, rank]
            block += (target @ gathered.T) / beta_n

    return 0.5j / period * block
