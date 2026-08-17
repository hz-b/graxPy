"""Panel discretization of a grating boundary for the integral method.

The boundary-integral solver works on the grating *surface*, not on a z-sliced
volume, so this module turns the polyline that
:meth:`grax.gratings.BaseGrating.profile_points` already returns into a set of
flat panels with collocation points, normals and quadrature nodes.

Why flat panels
---------------
Every stock profile is piecewise linear, and some of them are not graphs of a
function at all: ``LaminarGrating`` with 90 degree sidewalls has exactly vertical
segments, and ``BlazedGrating`` without an anti-blaze angle has a near-vertical
reset edge. A parametrization by arc length along the polyline handles all of
them uniformly, and because each panel is exactly straight:

- the geometry carries no discretization error of its own, so ``z_resolution_nm``
  and ``x_resolution_nm`` never enter the integral solver;
- the free-space single-layer self-panel integral has a closed form;
- the free-space double-layer self-panel integrand vanishes identically, since
  the normal is orthogonal to the panel;
- corners are just panel endpoints and need no special parametrization.

Panels are graded toward corners, where the surface densities develop
singularities, using a one-sided power-law clustering controlled by
``corner_grading``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoundaryPanels", "build_panels"]

# Below this fraction of the period a polyline segment is treated as a corner
# artefact rather than a facet worth its own panels.
_MIN_SEGMENT_FRACTION = 1e-9
# A polyline vertex turning by more than this is a corner that panels must land
# on exactly. Below it the profile is treated as smooth and panels become chords.
_CORNER_ANGLE_DEG = 5.0


@dataclass(frozen=True)
class BoundaryPanels:
    """Flat-panel discretization of one periodic boundary.

    All arrays are ordered along the boundary, which runs left to right across
    one period. The normal points *up*, into the medium above the interface,
    matching the sign convention of the integral equations in
    :mod:`grax.solvers.integral`.

    Attributes:
        start: Panel start points, shape ``(n_panels, 2)``.
        end: Panel end points, shape ``(n_panels, 2)``.
        midpoint: Collocation points, shape ``(n_panels, 2)``.
        tangent: Unit tangents, shape ``(n_panels, 2)``.
        normal: Unit normals pointing up, shape ``(n_panels, 2)``.
        length: Panel lengths in nanometers, shape ``(n_panels,)``.
        period: Grating period in nanometers.
    """

    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    period: float

    @property
    def count(self) -> int:
        """Return the number of panels."""

        return int(self.length.size)

    @property
    def total_length(self) -> float:
        """Return the developed length of one period of the boundary."""

        return float(np.sum(self.length))

    @property
    def collocation(self) -> np.ndarray:
        """Return the collocation points, shape ``(n_panels, 2)``."""

        return self.midpoint

    @property
    def weight(self) -> np.ndarray:
        """Return the arc-length quadrature weight of each node."""

        return self.length

    def shifted(self, offset: float) -> BoundaryPanels:
        """Return the same panels translated vertically.

        Coatings in :mod:`grax.gratings` are conformal: every interface is the
        substrate profile plus a constant thickness. Shifting therefore produces
        the exact geometry of the next interface up, with the tangents, normals
        and lengths all unchanged.

        Args:
            offset: Vertical displacement in nanometers.

        Returns:
            A new panel set displaced by ``offset`` in ``y``.
        """

        displacement = np.asarray([0.0, float(offset)], dtype=float)
        return BoundaryPanels(
            start=self.start + displacement,
            end=self.end + displacement,
            midpoint=self.midpoint + displacement,
            tangent=self.tangent,
            normal=self.normal,
            length=self.length,
            period=self.period,
        )

    def quadrature(self, order: int) -> tuple[np.ndarray, np.ndarray]:
        """Return Gauss-Legendre nodes and weights on every panel.

        Args:
            order: Number of nodes per panel.

        Returns:
            Node positions shaped ``(n_panels, order, 2)`` and arc-length
            weights shaped ``(n_panels, order)``.
        """

        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        # Map [-1, 1] onto each panel.
        half = 0.5 * self.length[:, None]
        parameter = 0.5 * (nodes[None, :] + 1.0)
        positions = (
            self.start[:, None, :]
            + (self.end - self.start)[:, None, :] * parameter[:, :, None]
        )
        return positions, weights[None, :] * half


def build_panels(
    positions: np.ndarray,
    heights: np.ndarray,
    *,
    period: float,
    panel_count: int,
    corner_grading: float = 2.0,
) -> BoundaryPanels:
    """Return a graded flat-panel discretization of one profile period.

    Args:
        positions: Profile x coordinates over one period, ascending, starting at
            ``0`` and ending at ``period``.
        heights: Profile heights at ``positions``, with equal first and last
            entries so the profile is periodic.
        period: Grating period in nanometers.
        panel_count: Target total number of panels over one period.
        corner_grading: Power-law clustering exponent toward the two ends of
            every straight facet. ``1.0`` gives uniform panels; larger values
            cluster more strongly at the corners, where the surface densities
            are singular.

    Returns:
        The panel discretization.

    Raises:
        ValueError: If the profile is not a usable periodic polyline or the
            requested panel count is too small to cover every facet.
    """

    positions = np.asarray(positions, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if positions.ndim != 1 or heights.shape != positions.shape:
        raise ValueError("positions and heights must be matching 1D arrays.")
    if positions.size < 2:
        raise ValueError("The profile needs at least two points.")
    if not np.all(np.diff(positions) >= 0.0):
        raise ValueError("Profile positions must be non-decreasing.")
    if not np.isclose(heights[0], heights[-1], rtol=0.0, atol=1e-9 * max(period, 1.0)):
        raise ValueError(
            "The profile must be periodic: the first and last heights differ by "
            f"{abs(float(heights[-1] - heights[0])):.6g} nm."
        )
    if corner_grading < 1.0:
        raise ValueError("corner_grading must be >= 1.")

    vertices = np.column_stack((positions, heights))
    segments = vertices[1:] - vertices[:-1]
    segment_length = np.hypot(segments[:, 0], segments[:, 1])
    keep = segment_length > _MIN_SEGMENT_FRACTION * period
    if not np.any(keep):
        raise ValueError("The profile has no segment of usable length.")
    vertices = np.vstack((vertices[:-1][keep], vertices[-1]))
    segment_length = segment_length[keep]
    arclength = np.concatenate(([0.0], np.cumsum(segment_length)))

    # Facets are the stretches between genuine corners. A stock profile turns
    # sharply at every vertex, so each facet is one straight segment and the
    # panels are exact. A densely sampled smooth profile (AFM data, a sinusoid)
    # has no corners at all, and its panels become chords, which is the right
    # behaviour: the panel count then controls the geometric error instead of the
    # sampling of the input profile.
    corners = _corner_indices(vertices, period)
    facet_bounds = arclength[corners]
    facet_length = np.diff(facet_bounds)

    counts = _allocate_panels(facet_length, int(panel_count))

    breakpoint_arclengths: list[np.ndarray] = []
    for index, count in enumerate(counts):
        graded = _graded_breakpoints(int(count), corner_grading)
        span = facet_bounds[index + 1] - facet_bounds[index]
        breakpoint_arclengths.append(facet_bounds[index] + span * graded[:-1])
    breakpoints = np.concatenate(breakpoint_arclengths + [facet_bounds[-1:]])

    sampled = _point_at_arclength(vertices, arclength, breakpoints)
    start = sampled[:-1]
    end = sampled[1:]
    delta = end - start
    length = np.hypot(delta[:, 0], delta[:, 1])
    tangent = delta / length[:, None]
    # Rotate the tangent by +90 degrees. The boundary runs left to right, so this
    # points up, into the medium above.
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    return BoundaryPanels(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        tangent=tangent,
        normal=normal,
        length=length,
        period=float(period),
    )


def _corner_indices(vertices: np.ndarray, period: float) -> np.ndarray:
    """Return the vertex indices that are genuine corners, including both ends.

    A vertex counts as a corner when the polyline turns there by more than
    :data:`_CORNER_ANGLE_DEG`. The two profile endpoints are always included,
    because they bound the period; whether the profile is smooth *across* the
    period boundary is decided by the same turn test, using the wrap-around
    tangent.

    Args:
        vertices: Polyline vertices, shape ``(n, 2)``.
        period: Grating period, used for the wrap-around tangent.

    Returns:
        Ascending vertex indices, starting at ``0`` and ending at ``n - 1``.
    """

    tangents = np.diff(vertices, axis=0)
    tangents = tangents / np.hypot(tangents[:, 0], tangents[:, 1])[:, None]
    cosine = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
    turn = np.degrees(np.arccos(cosine))
    interior = np.nonzero(turn > _CORNER_ANGLE_DEG)[0] + 1
    return np.concatenate(([0], interior, [vertices.shape[0] - 1]))


def _point_at_arclength(
    vertices: np.ndarray, arclength: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Return points on the polyline at the requested arc lengths.

    Args:
        vertices: Polyline vertices, shape ``(n, 2)``.
        arclength: Cumulative arc length at each vertex, shape ``(n,)``.
        targets: Arc lengths to sample, ascending.

    Returns:
        Sampled points, shape ``(targets.size, 2)``.
    """

    x = np.interp(targets, arclength, vertices[:, 0])
    y = np.interp(targets, arclength, vertices[:, 1])
    return np.column_stack((x, y))


def _allocate_panels(segment_length: np.ndarray, panel_count: int) -> np.ndarray:
    """Return how many panels each facet gets.

    Panels are handed out in proportion to facet length, but every facet gets at
    least one, because a facet the discretization skipped would be a hole in the
    boundary rather than a small error. Vertical sidewalls are short and would
    otherwise be dropped.

    Args:
        segment_length: Length of each facet.
        panel_count: Requested total panel count.

    Returns:
        Integer panel count per facet, summing to at least ``panel_count``.

    Raises:
        ValueError: If fewer panels are requested than there are facets.
    """

    facets = segment_length.size
    if panel_count < facets:
        raise ValueError(
            f"boundary_points={panel_count} is below the {facets} straight facets of this "
            "profile. Each facet needs at least one panel, so raise boundary_points."
        )
    share = segment_length / np.sum(segment_length)
    counts = np.maximum(1, np.round(share * panel_count).astype(int))
    # Give any rounding shortfall to the longest facets.
    deficit = panel_count - int(np.sum(counts))
    if deficit > 0:
        order = np.argsort(-segment_length)
        for index in range(deficit):
            counts[order[index % facets]] += 1
    return counts


def _graded_breakpoints(count: int, grading: float) -> np.ndarray:
    """Return ``count + 1`` breakpoints on ``[0, 1]`` clustered at both ends.

    The map ``s -> s^g / (s^g + (1 - s)^g)`` is smooth, symmetric, and pushes
    breakpoints toward ``0`` and ``1`` as ``g`` grows, which is where the facet
    meets its neighbours at a corner.

    Args:
        count: Number of panels on the facet.
        grading: Clustering exponent, ``>= 1``.

    Returns:
        Ascending breakpoints from ``0`` to ``1``.
    """

    uniform = np.linspace(0.0, 1.0, int(count) + 1)
    if grading == 1.0 or count < 2:
        return uniform
    lower = uniform**grading
    upper = (1.0 - uniform) ** grading
    return lower / (lower + upper)
