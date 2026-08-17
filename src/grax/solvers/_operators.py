"""Boundary-integral operator assembly on flat panels.

Builds the single-layer and double-layer matrices

    S[i, j] = int_{panel j} G(c_i, q) ds(q)
    D[i, j] = int_{panel j} dG(c_i, q) / dn(q) ds(q)

for collocation points ``c_i`` on one interface and source panels on another (or
the same) interface, using the quasi-periodic Green function from
:mod:`grax.solvers._green`.

Three regimes are handled separately:

*far*
    Plain Gauss-Legendre on the source panel.

*near*
    The kernel has a logarithmic singularity just off the panel, so the panel is
    recursively bisected until every piece is small compared with its distance to
    the collocation point.

*self*
    The singularity sits on the panel. Because every panel is exactly straight:

    - the single layer is split as ``G = [G + ln R / 2pi] + [-ln R / 2pi]``. The
      bracket is continuous at ``R = 0`` and goes to Gauss; the second term has
      the closed form ``-(L/2pi)(ln(L/2) - 1)`` for a midpoint collocation point.
    - the double layer's free-space part vanishes identically, since
      ``(c_i - q) . n`` is zero for two points on the same straight panel. Only
      the periodic remainder survives, and it is evaluated by subtracting the
      free-space gradient from the periodic one.
"""

from __future__ import annotations

import numpy as np

from ._boundary import BoundaryPanels
from ._green import PeriodicGreen

__all__ = ["layer_operators"]

# A source panel is refined while its length exceeds this fraction of its
# distance to the collocation point.
_NEAR_RATIO = 1.5
_MAX_REFINEMENT_DEPTH = 8
_TINY = 1e-300


def layer_operators(
    green: PeriodicGreen,
    *,
    target: BoundaryPanels,
    source: BoundaryPanels,
    same_interface: bool,
    quadrature_order: int = 8,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single- and double-layer matrices between two interfaces.

    Args:
        green: Green function of the medium bounded by these interfaces.
        target: Interface carrying the collocation points.
        source: Interface carrying the source panels.
        same_interface: Whether ``target`` and ``source`` are the same interface,
            which is what puts the singularity on the panel rather than near it.
        quadrature_order: Gauss-Legendre nodes per panel.
        chunk_size: Number of collocation points evaluated per pass, which caps
            peak memory without changing the result.

    Returns:
        ``(S, D)``, each shaped ``(target.count, source.count)``.
    """

    n_target = target.count
    n_source = source.count
    single = np.zeros((n_target, n_source), dtype=complex)
    double = np.zeros((n_target, n_source), dtype=complex)

    nodes, weights = source.quadrature(quadrature_order)

    for begin in range(0, n_target, chunk_size):
        end = min(begin + chunk_size, n_target)
        collocation = target.midpoint[begin:end]
        block_single, block_double = _far_block(
            green,
            collocation=collocation,
            nodes=nodes,
            weights=weights,
            source_normal=source.normal,
        )
        single[begin:end] = block_single
        double[begin:end] = block_double

    _refine_near_panels(
        green,
        single=single,
        double=double,
        target=target,
        source=source,
        quadrature_order=quadrature_order,
        skip_self=same_interface,
    )

    if same_interface:
        _fill_self_panels(
            green,
            single=single,
            double=double,
            panels=source,
            quadrature_order=max(quadrature_order, 10),
        )
    return single, double


def _far_block(
    green: PeriodicGreen,
    *,
    collocation: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    source_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one chunk of the two matrices by plain Gauss quadrature.

    Args:
        green: Green function of the medium.
        collocation: Collocation points, shape ``(chunk, 2)``.
        nodes: Quadrature nodes, shape ``(n_source, order, 2)``.
        weights: Arc-length weights, shape ``(n_source, order)``.
        source_normal: Source panel normals, shape ``(n_source, 2)``.

    Returns:
        ``(S_chunk, D_chunk)``, each shaped ``(chunk, n_source)``.
    """

    dx = collocation[:, None, None, 0] - nodes[None, :, :, 0]
    dy = collocation[:, None, None, 1] - nodes[None, :, :, 1]

    value, grad_x, grad_y = green.value_and_gradient(dx, dy)
    # The gradient is with respect to the field point, so the source-normal
    # derivative flips sign.
    normal_derivative = -(
        grad_x * source_normal[None, :, None, 0] + grad_y * source_normal[None, :, None, 1]
    )

    single = np.sum(value * weights[None, :, :], axis=-1)
    double = np.sum(normal_derivative * weights[None, :, :], axis=-1)
    return single, double


def _refine_near_panels(
    green: PeriodicGreen,
    *,
    single: np.ndarray,
    double: np.ndarray,
    target: BoundaryPanels,
    source: BoundaryPanels,
    quadrature_order: int,
    skip_self: bool,
) -> None:
    """Recompute matrix entries whose source panel is too close to be resolved.

    Args:
        green: Green function of the medium.
        single: Single-layer matrix, updated in place.
        double: Double-layer matrix, updated in place.
        target: Interface carrying the collocation points.
        source: Interface carrying the source panels.
        quadrature_order: Gauss-Legendre nodes per sub-panel.
        skip_self: Whether to leave the diagonal alone, because the self-panel
            routine handles it.
    """

    separation = _minimum_image_distance(
        target.midpoint[:, None, :], source.midpoint[None, :, :], source.period
    )
    too_close = separation < _NEAR_RATIO * source.length[None, :]
    if skip_self:
        np.fill_diagonal(too_close, False)

    rows, columns = np.nonzero(too_close)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        value, derivative = _refined_panel_integral(
            green,
            collocation=target.midpoint[row],
            start=source.start[column],
            end=source.end[column],
            normal=source.normal[column],
            quadrature_order=quadrature_order,
            period=source.period,
        )
        single[row, column] = value
        double[row, column] = derivative


def _refined_panel_integral(
    green: PeriodicGreen,
    *,
    collocation: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
    quadrature_order: int,
    period: float,
    depth: int = 0,
) -> tuple[complex, complex]:
    """Return one panel integral, bisecting until the panel is well resolved.

    Args:
        green: Green function of the medium.
        collocation: Collocation point.
        start: Panel start point.
        end: Panel end point.
        normal: Panel unit normal.
        quadrature_order: Gauss-Legendre nodes per piece.
        period: Grating period, used for the minimum-image distance test.
        depth: Current recursion depth.

    Returns:
        ``(S_entry, D_entry)`` for this panel.
    """

    length = float(np.hypot(*(end - start)))
    midpoint = 0.5 * (start + end)
    distance = _minimum_image_distance(collocation, midpoint, period)
    if depth < _MAX_REFINEMENT_DEPTH and length > _NEAR_RATIO * distance:
        middle = midpoint
        left = _refined_panel_integral(
            green,
            collocation=collocation,
            start=start,
            end=middle,
            normal=normal,
            quadrature_order=quadrature_order,
            period=period,
            depth=depth + 1,
        )
        right = _refined_panel_integral(
            green,
            collocation=collocation,
            start=middle,
            end=end,
            normal=normal,
            quadrature_order=quadrature_order,
            period=period,
            depth=depth + 1,
        )
        return left[0] + right[0], left[1] + right[1]

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    parameter = 0.5 * (nodes + 1.0)
    positions = start[None, :] + (end - start)[None, :] * parameter[:, None]
    scale = 0.5 * length * weights

    dx = collocation[0] - positions[:, 0]
    dy = collocation[1] - positions[:, 1]
    value, grad_x, grad_y = green.value_and_gradient(dx, dy)
    normal_derivative = -(grad_x * normal[0] + grad_y * normal[1])
    return complex(np.sum(value * scale)), complex(np.sum(normal_derivative * scale))


def _fill_self_panels(
    green: PeriodicGreen,
    *,
    single: np.ndarray,
    double: np.ndarray,
    panels: BoundaryPanels,
    quadrature_order: int,
) -> None:
    """Fill the diagonal, where the singularity sits on the panel itself.

    Args:
        green: Green function of the medium.
        single: Single-layer matrix, updated in place.
        double: Double-layer matrix, updated in place.
        panels: The interface, used as both target and source.
        quadrature_order: Gauss-Legendre nodes on the panel.
    """

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    parameter = 0.5 * (nodes + 1.0)

    for index in range(panels.count):
        start = panels.start[index]
        end = panels.end[index]
        length = float(panels.length[index])
        collocation = panels.midpoint[index]
        normal = panels.normal[index]

        positions = start[None, :] + (end - start)[None, :] * parameter[:, None]
        scale = 0.5 * length * weights
        dx = collocation[0] - positions[:, 0]
        dy = collocation[1] - positions[:, 1]
        distance = np.hypot(dx, dy)

        # Single layer: peel the logarithm, integrate it exactly, and send the
        # continuous remainder to Gauss.
        value, grad_x, grad_y = green.value_and_gradient(dx, dy)
        regular = value + np.log(np.maximum(distance, _TINY)) / (2.0 * np.pi)
        analytic_log = -(length / (2.0 * np.pi)) * (np.log(length / 2.0) - 1.0)
        single[index, index] = complex(np.sum(regular * scale)) + analytic_log

        # Double layer: the free-space contribution is identically zero on a flat
        # panel, so only the periodic remainder is integrated. Both gradients
        # diverge like 1/R and their difference does not, which is why they are
        # subtracted before contracting with the normal.
        free_x, free_y = _free_space_gradient(green.wavenumber, dx, dy, distance)
        remainder_x = grad_x - free_x
        remainder_y = grad_y - free_y
        normal_derivative = -(remainder_x * normal[0] + remainder_y * normal[1])
        double[index, index] = complex(np.sum(normal_derivative * scale))


def _free_space_gradient(
    wavenumber: complex,
    dx: np.ndarray,
    dy: np.ndarray,
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient of ``(i/4) H_0^(1)(k R)`` with respect to the field point.

    Args:
        wavenumber: Medium wavenumber.
        dx: In-plane separation.
        dy: Out-of-plane separation.
        distance: ``sqrt(dx^2 + dy^2)``.

    Returns:
        Pair of complex arrays, the two gradient components.
    """

    from scipy.special import hankel1

    safe = np.maximum(distance, _TINY)
    radial = -0.25j * complex(wavenumber) * hankel1(1, complex(wavenumber) * safe)
    return radial * dx / safe, radial * dy / safe


def _minimum_image_distance(
    left: np.ndarray, right: np.ndarray, period: float
) -> np.ndarray:
    """Return the distance to the nearest periodic image of ``right``.

    Panels near ``x = 0`` and ``x = period`` are neighbours across the period
    boundary, so a plain distance would call them far apart and skip the
    refinement they need.

    Args:
        left: Field points, last axis of length two.
        right: Source points, last axis of length two.
        period: Grating period.

    Returns:
        Distances, shaped like the broadcast of the inputs minus the last axis.
    """

    dx = left[..., 0] - right[..., 0]
    dy = left[..., 1] - right[..., 1]
    dx = dx - period * np.round(dx / period)
    return np.hypot(dx, dy)
