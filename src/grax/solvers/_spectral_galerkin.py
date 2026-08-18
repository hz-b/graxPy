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
    "max_stable_reach",
    "near_band_mask",
    "projected_single_layer",
    "spectral_reach_for_separation",
]


def max_stable_reach(*, period: float, height_range: float) -> int:
    """Return the largest spectral order the height split can carry stably.

    The factorization ``exp(i beta_n (f_t - f_s)) = exp(i beta_n f_t)
    exp(-i beta_n f_s)`` is exact, but only the *product* is bounded. For a deep
    evanescent order ``beta_n`` is nearly imaginary, so one factor decays like
    ``exp(-|beta_n| f)`` while the other grows like ``exp(+|beta_n| f)``, and the
    individual factors reach ``exp(|beta_n| H)`` across a profile of height range
    ``H`` however small the pair's own gap is.

    This is the same failure that the Kress split hits in an absorbing medium: an
    exact identity whose two halves leave the range float64 can carry. Here it
    caps the reach rather than the material, and the cap is what ties the near
    band threshold to the profile depth -- a deeper profile forces a wider band,
    because the series covering that band cannot be summed far enough.

    Args:
        period: Grating period in nanometers.
        height_range: Peak-to-trough height of the profile in nanometers.

    Returns:
        Half-width of the largest safely summable order range.

    Raises:
        ValueError: If the height range is not positive.
    """

    if height_range <= 0.0:
        raise ValueError("height_range must be positive; a flat profile has no far field.")
    # exp overflows past ~709 in float64; leave an order of magnitude of margin
    # so the products, not just the factors, stay comfortably in range.
    return max(1, int(650.0 * period / (2.0 * np.pi * height_range)))


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
    min_separation: float = 0.0,
) -> np.ndarray:
    """Return the far-field part of one projected single-layer block.

    Evaluates

        (i / 2d) sum_n (1/beta_n) (1/d) int int
            exp(i 2 pi (n - m') x_t / d) exp(-i 2 pi (n - m) x_s / d)
            exp(i beta_n |f_t - f_s|) J_s dx_s dx_t

    over the two regions ``f_t - f_s > delta`` and ``f_s - f_t > delta``, where
    the modulus resolves and the exponential separates. Pairs closer than
    ``delta`` in height are excluded: the series needs ``|beta_n| |Y|`` to grow to
    converge, so they belong to the near band and the caller supplies them from
    the Ewald kernel.

    Excluding the band costs nothing. ``f_s < f_t - delta`` is still a *prefix*
    in the height-sorted order, so the threshold only moves where the rank is
    read, and the prefix-sum structure is untouched.

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
        min_separation: Height gap ``delta`` below which a pair is left to the
            near band. ``0`` keeps every distinct-height pair, which is only
            usable when the series is summed far enough for the smallest gap on
            the grid.

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
    gap = float(min_separation)
    below = np.searchsorted(sorted_heights, heights - gap, side="left")
    above = np.searchsorted(sorted_heights, heights + gap, side="right")

    span = float(sorted_heights[-1] - sorted_heights[0])
    if span > 0.0:
        limit = max_stable_reach(period=period, height_range=span)
        if int(np.max(np.abs(orders))) > limit:
            raise ValueError(
                f"The height split was asked for order {int(np.max(np.abs(orders)))}, but a "
                f"profile spanning {span:.3g} nm over a {period:.3g} nm period can only "
                f"carry {limit} before the two halves of the factorization overflow. "
                "Raise min_separation, which shortens the series the far field needs, at "
                "the cost of a wider near band."
            )

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


def near_band_mask(heights: np.ndarray, *, min_separation: float) -> np.ndarray:
    """Return the pairs the plane-wave series cannot resolve.

    Exactly the complement of what :func:`height_split_projection` sums, so the
    two partition the operator with no pair counted twice or dropped.

    Args:
        heights: Profile height at each node, shape ``(nodes,)``.
        min_separation: Height gap below which a pair is near.

    Returns:
        Boolean ``(nodes, nodes)`` array, true where the pair is near.
    """

    return np.abs(np.subtract.outer(heights, heights)) <= float(min_separation)


def projected_single_layer(
    *,
    nodal_block: np.ndarray,
    basis,
    x: np.ndarray,
    heights: np.ndarray,
    jacobian: np.ndarray,
    beta: np.ndarray,
    orders: np.ndarray,
    period: float,
    min_separation: float,
) -> np.ndarray:
    """Return one projected block as near band plus factorized far field.

    The near band comes from ``nodal_block``, which is the classical Nystrom
    operator and carries the Kress product quadrature that the logarithmic
    singularity needs. The far field comes from the factorized spectral sum. The
    two sets partition the node pairs, so the result is the same operator, built
    the cheap way wherever the series converges.

    This is the correctness harness for the splice, not yet the speedup: the near
    band is read out of a full nodal block, so the assembly cost is unchanged
    until the near evaluation is itself restricted to the band. Getting the
    partition right has to come first, because a splice that quietly double
    counts or drops a region produces a plausible wrong number rather than an
    obvious failure.

    Args:
        nodal_block: Classical Nystrom single-layer block, ``(nodes, nodes)``.
        basis: :class:`~grax.solvers._carrier.CarrierBasis` for this boundary.
        x: Uniform quadrature abscissae over one period.
        heights: Profile height at each abscissa.
        jacobian: Arc-length Jacobian at each abscissa.
        beta: Out-of-plane wavenumber per spectral order.
        orders: Spectral orders aligned with ``beta``.
        period: Grating period in nanometers.
        min_separation: Height gap separating near from far.

    Returns:
        Projected block shaped ``(modes, modes)``.
    """

    near = near_band_mask(heights, min_separation=min_separation)
    near_part = basis.analysis @ (nodal_block * near) @ basis.synthesis
    far_part = height_split_projection(
        x=x,
        heights=heights,
        jacobian=jacobian,
        modes=basis.orders,
        beta=beta,
        orders=orders,
        period=period,
        min_separation=min_separation,
    )
    return near_part + far_part
