"""Trigonometric Nystrom discretization of the boundary integral operators.

This is the high-order replacement for the flat-panel scheme in
:mod:`grax.solvers._operators`. The panel scheme converges as ``O(h^2)``, and the
Stage 1 study measured what that costs: ``N_required ~ (d/lambda)^0.85``, reaching
about 12000 unknowns at ``d/lambda = 100``. The same study showed why. The
boundary densities are strongly band-limited -- at ``d/lambda = 50`` the field
density carries 8 significant harmonics and its normal derivative 31 -- while the
panel count needed was 6266. The panels were being spent resolving the
oscillatory *kernel*, not the unknowns, because a collocation scheme uses one
grid for both.

A Nystrom scheme on a trigonometric grid separates them. The densities are
represented by their values on a uniform parameter grid, which for a smooth
periodic boundary is a spectrally accurate representation of a band-limited
function, while the kernel's singularity is handled analytically by product
quadrature rather than by refining the grid.

Method
------
Parametrize one period by ``t`` in ``[0, 2pi)``. Both the kernel and the density
are pseudo-periodic, and their product is genuinely ``2pi``-periodic: shifting
the source by one period multiplies the Green function by ``exp(-i alpha_0 d)``
and the density by ``exp(+i alpha_0 d)``. The trapezoidal rule is therefore
spectrally accurate on the smooth part.

The logarithmic singularity is split off in the classical Kress form

    G(t, s) = M1(t, s) ln(4 sin^2((t - s) / 2)) + M2(t, s)

with both factors analytic. Only the *free-space* part of ``G`` is singular, so

    M1 = -J_0(k R) / 4pi
    M2 = G + J_0(k R) ln(4 sin^2((t-s)/2)) / 4pi

and the periodic remainder ``G - G_free``, which is analytic through the origin,
never needs special treatment away from the diagonal. On the diagonal it is
supplied by :meth:`grax.solvers._green.PeriodicGreen.regular_at_zero`.

The ``ln(4 sin^2)`` factor is integrated by the Martensen-Kussmaul weights, which
are exact for trigonometric polynomials up to the grid order, so the whole scheme
inherits the spectral convergence of the trapezoidal rule on smooth boundaries.

Corners
-------
A corner breaks the analyticity the spectral accuracy rests on, twice over: the
parametrization loses its derivative, and the density develops an algebraic
singularity. :func:`build_graded_boundary` reparametrizes so the Jacobian
vanishes at every corner, which clusters nodes there and multiplies the singular
density down. :func:`build_trig_boundary` stays the better choice for genuinely
smooth profiles, because it reads the geometry spectrally rather than off the
polyline; :func:`has_corners` picks between them.

Scope
-----
Profiles that are graphs ``y = f(x)``. A profile with exactly vertical walls is
not a graph and is still out of reach here; the flat-panel scheme in
:mod:`grax.solvers._operators` handles those through an arc-length
parametrization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from ._green import PeriodicGreen

__all__ = [
    "TrigBoundary",
    "build_graded_boundary",
    "build_trig_boundary",
    "has_corners",
    "kress_log_weights",
    "nystrom_operators",
]

#: Floor for the parametrization speed inside logarithms. Only reached when a
#: node lands essentially on a corner of a graded parametrization.
_MIN_SPEED = 1e-300
#: A facet whose horizontal run is below this fraction of the period is treated
#: as vertical, which this parametrization cannot represent.
_VERTICAL_RUN_FRACTION = 1e-9
#: Rough peak-memory budget for one block of the Ewald spectral intermediate.
_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
_COMPLEX_BYTES = 16
#: The spectral half forms several arrays of the same shape at once.
_EWALD_TEMPORARIES = 6
#: Largest ``ln`` amplification the Bessel factor of the Kress split is allowed
#: to reach inside the window. See :func:`_absorption_window`.
_WINDOW_LOG_BUDGET = 10.0
#: Growth rate ``|Im k| d / 2pi`` below which the Bessel factor never gets large
#: enough over one period to need a window at all.
_WINDOW_OFF_BELOW = 1.0


@dataclass(frozen=True)
class TrigBoundary:
    """One periodic boundary sampled on a uniform parameter grid.

    Attributes:
        position: Points ``r(t_j)``, shape ``(n, 2)``.
        derivative: ``dr/dt``, shape ``(n, 2)``.
        speed: ``|dr/dt|``, shape ``(n,)``.
        normal: Unit normals pointing up, shape ``(n, 2)``.
        curvature: Signed curvature at each node, shape ``(n,)``.
        period: Grating period in nanometers.
        nodes: Parameter values ``t_j``, shape ``(n,)``.
    """

    position: np.ndarray
    derivative: np.ndarray
    speed: np.ndarray
    normal: np.ndarray
    curvature: np.ndarray
    period: float
    nodes: np.ndarray

    @property
    def count(self) -> int:
        """Return the number of collocation nodes."""

        return int(self.speed.size)

    @property
    def collocation(self) -> np.ndarray:
        """Return the collocation points, shape ``(n, 2)``."""

        return self.position

    @property
    def weight(self) -> np.ndarray:
        """Return the arc-length quadrature weight of each node.

        The uniform parameter grid makes this the trapezoidal weight times the
        parametrization speed, which is spectrally accurate for the smooth
        integrands the Rayleigh projection uses.
        """

        return (2.0 * np.pi / self.count) * self.speed

    def shifted(self, offset: float) -> TrigBoundary:
        """Return the same boundary translated vertically.

        Coatings are conformal, so a shifted copy is the exact geometry of the
        next interface up, with tangents, normals, speed and curvature unchanged.

        Args:
            offset: Vertical displacement in nanometers.

        Returns:
            A new boundary displaced by ``offset`` in ``y``.
        """

        displacement = np.asarray([0.0, float(offset)], dtype=float)
        return TrigBoundary(
            position=self.position + displacement,
            derivative=self.derivative,
            speed=self.speed,
            normal=self.normal,
            curvature=self.curvature,
            period=self.period,
            nodes=self.nodes,
        )


def build_trig_boundary(
    positions: np.ndarray,
    heights: np.ndarray,
    *,
    period: float,
    count: int,
) -> TrigBoundary:
    """Return a trigonometric boundary sampled from a profile polyline.

    The profile is treated as a graph ``y = f(x)`` and parametrized by
    ``x = d t / 2pi``, which makes ``dx/dt`` constant and leaves all the geometry
    in ``f``. Derivatives of ``f`` are taken spectrally by FFT, which is exact
    for a band-limited profile.

    The input polyline is resampled onto the uniform grid by linear
    interpolation, so its own sampling caps the accuracy: for a polyline of
    spacing ``h`` the interpolation error is ``O(h^2)``. Supply a profile sampled
    at least as finely as the requested node count, and finer when chasing
    tolerances below about 1e-4.

    Args:
        positions: Profile x coordinates over one period, ascending from ``0``.
        heights: Profile heights at ``positions``, periodic.
        period: Grating period in nanometers.
        count: Number of collocation nodes; must be even.

    Returns:
        The sampled boundary.

    Raises:
        ValueError: If the node count is not a positive even number, or the
            profile is not a usable periodic graph.
    """

    if count <= 0 or count % 2 != 0:
        raise ValueError(f"count must be a positive even number, got {count}.")
    positions = np.asarray(positions, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if positions.ndim != 1 or heights.shape != positions.shape:
        raise ValueError("positions and heights must be matching 1D arrays.")
    if not np.all(np.diff(positions) >= 0.0):
        raise ValueError("Profile positions must be non-decreasing.")
    if not np.isclose(heights[0], heights[-1], rtol=0.0, atol=1e-9 * max(period, 1.0)):
        raise ValueError("The profile must be periodic in height.")

    nodes = 2.0 * np.pi * np.arange(count, dtype=float) / count
    x = period * nodes / (2.0 * np.pi)
    y = np.interp(x, positions, heights, period=period)

    dy_dt, d2y_dt2 = _spectral_derivatives(y)
    dx_dt = np.full(count, period / (2.0 * np.pi))

    derivative = np.column_stack((dx_dt, dy_dt))
    speed = np.hypot(dx_dt, dy_dt)
    # Rotate the tangent by +90 degrees; the boundary runs left to right, so this
    # points up into the medium above.
    normal = np.column_stack((-dy_dt, dx_dt)) / speed[:, None]
    # x(t) is linear, so d2x/dt2 vanishes and the signed curvature reduces to
    # x' y'' / speed^3.
    curvature = dx_dt * d2y_dt2 / speed**3

    return TrigBoundary(
        position=np.column_stack((x, y)),
        derivative=derivative,
        speed=speed,
        normal=normal,
        curvature=curvature,
        period=float(period),
        nodes=nodes,
    )


def build_graded_boundary(
    positions: np.ndarray,
    heights: np.ndarray,
    *,
    period: float,
    count: int,
    grading: float = 3.0,
    corner_angle_deg: float = 5.0,
) -> TrigBoundary:
    """Return a corner-graded boundary for a piecewise-linear profile.

    Spectral accuracy rests on the integrand being analytic, and a corner breaks
    that twice over: the parametrization loses its derivative, and the surface
    densities themselves develop an algebraic singularity there. The classical
    cure is to reparametrize, ``t = w(tau)``, with ``w'`` vanishing at every
    corner. That does two jobs at once -- it clusters nodes where the solution is
    hardest, and the vanishing Jacobian multiplies the singular density down to
    something smooth.

    The map used here is the multi-corner generalisation of Kress's single-corner
    substitution::

        w'(tau) proportional to prod_j |sin((tau - c_j) / 2)|^grading

    which is periodic, non-negative, and vanishes to order ``grading`` at each
    corner parameter ``c_j``. It is integrated numerically and normalised so that
    ``w`` maps one period onto one period.

    Geometry comes from the polyline directly rather than from FFT
    differentiation: a piecewise-linear profile is not band-limited, so spectral
    differentiation would ring at every corner. Within a facet the tangent is
    exact and the curvature is zero.

    Args:
        positions: Profile x coordinates over one period, ascending from ``0``.
        heights: Profile heights at ``positions``, periodic.
        period: Grating period in nanometers.
        count: Number of collocation nodes; must be even.
        grading: Order to which ``w'`` vanishes at each corner. Higher clusters
            more aggressively; ``0`` disables grading entirely.
        corner_angle_deg: A polyline vertex turning by more than this counts as a
            corner.

    Returns:
        The graded boundary, with ``nodes`` on the uniform ``tau`` grid the Kress
        weights require and ``speed`` carrying the reparametrization Jacobian.

    Raises:
        ValueError: If the node count is not a positive even number or the
            profile is not a usable periodic graph.
    """

    if count <= 0 or count % 2 != 0:
        raise ValueError(f"count must be a positive even number, got {count}.")
    positions = np.asarray(positions, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if positions.ndim != 1 or heights.shape != positions.shape:
        raise ValueError("positions and heights must be matching 1D arrays.")
    if not np.all(np.diff(positions) >= 0.0):
        raise ValueError("Profile positions must be non-decreasing.")
    if not np.isclose(heights[0], heights[-1], rtol=0.0, atol=1e-9 * max(period, 1.0)):
        raise ValueError("The profile must be periodic in height.")
    _reject_vertical_facets(positions, heights, period)

    corners = _corner_parameters(positions, heights, period, corner_angle_deg)
    # Offset the grid by half a step. The grading Jacobian vanishes exactly at a
    # corner, so a node landing on one produces an all-zero row and a singular
    # system. Corners at x = 0 make that the default rather than a coincidence:
    # a blazed profile has its reset edge exactly there. Shifting the whole grid
    # is free for the Kress weights, which depend only on node differences.
    nodes = 2.0 * np.pi * (np.arange(count, dtype=float) + 0.5) / count
    parameter, jacobian = _grading_map(
        nodes, corners=corners, grading=float(grading)
    )

    x = period * parameter / (2.0 * np.pi)
    y = np.interp(x, positions, heights, period=period)
    slope = _polyline_slope(positions, heights, x, period)

    # dr/dtau = dr/dt * w'(tau), and x(t) is linear in t.
    scale = period / (2.0 * np.pi)
    dx_dtau = scale * jacobian
    dy_dtau = scale * slope * jacobian

    derivative = np.column_stack((dx_dtau, dy_dtau))
    speed = np.hypot(dx_dtau, dy_dtau)
    # The normal follows the facet direction and is unaffected by the
    # reparametrization, so it is taken from the slope rather than from a
    # derivative that vanishes at corners.
    tangent_norm = np.hypot(1.0, slope)
    normal = np.column_stack((-slope, np.ones_like(slope))) / tangent_norm[:, None]

    return TrigBoundary(
        position=np.column_stack((x, y)),
        derivative=derivative,
        speed=speed,
        # Piecewise linear within every facet.
        curvature=np.zeros(count),
        normal=normal,
        period=float(period),
        nodes=nodes,
    )


def has_corners(
    positions: np.ndarray,
    heights: np.ndarray,
    *,
    corner_angle_deg: float = 5.0,
) -> bool:
    """Return whether the profile turns sharply anywhere.

    Used to choose between the two boundary builders: a smooth profile is better
    served by spectral differentiation, a corner profile by a graded
    parametrization reading the polyline directly.

    Args:
        positions: Profile x coordinates over one period, ascending.
        heights: Profile heights at ``positions``.
        corner_angle_deg: Turn angle above which a vertex counts as a corner.

    Returns:
        Whether at least one corner was found.
    """

    positions = np.asarray(positions, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if positions.size < 3:
        return False
    period = float(positions[-1] - positions[0])
    if period <= 0.0:
        return False
    return _corner_parameters(positions, heights, period, corner_angle_deg).size > 0


def _reject_vertical_facets(
    positions: np.ndarray, heights: np.ndarray, period: float
) -> None:
    """Raise when the profile is not a graph of ``y = f(x)``.

    This module parametrizes by ``x``, which a vertical facet makes impossible:
    the slope is infinite and the profile is multi-valued there. A laminar
    grating with exactly 90 degree sidewalls is the case that hits this. Left
    unchecked the slope would silently come out as zero and the wall would be
    modelled as flat, which is a wrong answer rather than a failure.

    The flat-panel scheme in :mod:`grax.solvers._operators` parametrizes by arc
    length and handles these profiles.

    Args:
        positions: Profile x coordinates over one period.
        heights: Profile heights.
        period: Grating period in nanometers.

    Raises:
        ValueError: If any facet is vertical or near-vertical.
    """

    run = np.diff(positions)
    rise = np.diff(heights)
    steep = (np.abs(run) <= _VERTICAL_RUN_FRACTION * period) & (np.abs(rise) > 0.0)
    if not np.any(steep):
        return
    index = int(np.nonzero(steep)[0][0])
    raise ValueError(
        "The trigonometric Nystrom discretization parametrizes the profile by x, so it "
        "needs a single-valued graph, but this profile has a vertical facet: the segment "
        f"from x={positions[index]:.6g} to x={positions[index + 1]:.6g} nm rises "
        f"{rise[index]:.6g} nm over a run of {run[index]:.6g} nm. A laminar grating with "
        "90 degree sidewalls is the usual source. Use discretization='panel', which "
        "parametrizes by arc length, or set the wall angles slightly below 90 degrees."
    )


def _corner_parameters(
    positions: np.ndarray,
    heights: np.ndarray,
    period: float,
    corner_angle_deg: float,
) -> np.ndarray:
    """Return the parameter values where the profile turns sharply.

    The wrap-around vertex is tested with the tangent from the last facet against
    the first, so a profile that is smooth across the period boundary is not
    given a spurious corner there.

    Args:
        positions: Profile x coordinates over one period.
        heights: Profile heights.
        period: Grating period in nanometers.
        corner_angle_deg: Turn angle above which a vertex counts as a corner.

    Returns:
        Corner parameters in ``[0, 2pi)``, ascending.
    """

    vertices = np.column_stack((positions, heights))
    tangents = np.diff(vertices, axis=0)
    length = np.hypot(tangents[:, 0], tangents[:, 1])
    keep = length > 0.0
    tangents = tangents[keep] / length[keep, None]
    kept_positions = positions[:-1][keep]
    if tangents.shape[0] < 2:
        return np.zeros(0, dtype=float)

    # Compare each facet with the previous one, wrapping the last onto the first.
    previous = np.vstack((tangents[-1:], tangents[:-1]))
    cosine = np.clip(np.sum(previous * tangents, axis=1), -1.0, 1.0)
    turn = np.degrees(np.arccos(cosine))
    corner_x = kept_positions[turn > corner_angle_deg]
    return np.sort(2.0 * np.pi * np.mod(corner_x, period) / period)


def _grading_map(
    nodes: np.ndarray,
    *,
    corners: np.ndarray,
    grading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``t = w(tau)`` and ``w'(tau)`` for a corner-graded parametrization.

    The corners split one period into segments. Each segment is given a share of
    the uniform parameter range in proportion to its length, and within a segment
    the map is Kress's sigmoid

        v(s) = s^g / (s^g + (1 - s)^g),
        v'(s) = g [s (1 - s)]^(g-1) / (s^g + (1 - s)^g)^2

    which carries ``[0, 1]`` onto itself with derivative vanishing to order
    ``g - 1`` at both ends. Composing it segment by segment gives a map whose
    derivative vanishes at every corner and nowhere else.

    Building it this way is self-consistent by construction: the corners are the
    images of prescribed break points, so no nonlinear solve is needed to place
    them. Integrating a density that vanishes at the corners would instead
    require inverting the map to find where those corners land, and getting that
    inversion wrong silently corrupts the Jacobian -- which shows up immediately
    as a quadrature that no longer reproduces the boundary's arc length.

    Args:
        nodes: Uniform parameter grid over ``[0, 2pi)``.
        corners: Corner parameters in ``[0, 2pi)``, ascending.
        grading: Vanishing order at each corner; ``<= 1`` disables grading.

    Returns:
        The mapped parameters and the Jacobian at each node.
    """

    if corners.size == 0 or grading <= 1.0:
        return nodes.copy(), np.ones_like(nodes)

    # Segment boundaries in t, wrapping the last corner round the period.
    breaks = np.concatenate((corners, [corners[0] + 2.0 * np.pi]))
    spans = np.diff(breaks)
    # Each segment gets a share of tau proportional to its share of t, so the
    # node density stays even away from the corners.
    shares = spans / spans.sum()
    tau_breaks = corners[0] + 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(shares)))

    shifted = corners[0] + np.mod(nodes - corners[0], 2.0 * np.pi)
    index = np.clip(np.searchsorted(tau_breaks, shifted, side="right") - 1, 0, spans.size - 1)

    width = tau_breaks[index + 1] - tau_breaks[index]
    local = (shifted - tau_breaks[index]) / width
    value, slope = _kress_sigmoid(local, grading)

    parameter = np.mod(breaks[index] + spans[index] * value, 2.0 * np.pi)
    jacobian = spans[index] * slope / width
    return parameter, jacobian


def _kress_sigmoid(local: np.ndarray, grading: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Kress's grading sigmoid and its derivative on ``[0, 1]``.

    Args:
        local: Positions within a segment, in ``[0, 1]``.
        grading: Vanishing order at both ends.

    Returns:
        ``(v, v')`` at each position.
    """

    local = np.clip(local, 0.0, 1.0)
    lower = local**grading
    upper = (1.0 - local) ** grading
    denominator = lower + upper
    safe = np.where(denominator > 0.0, denominator, 1.0)
    value = lower / safe
    slope = (
        grading
        * (local * (1.0 - local)) ** (grading - 1.0)
        / safe**2
    )
    return value, slope


def _polyline_slope(
    positions: np.ndarray, heights: np.ndarray, x: np.ndarray, period: float
) -> np.ndarray:
    """Return the exact facet slope of the polyline at each sample.

    Args:
        positions: Profile x coordinates over one period, ascending.
        heights: Profile heights.
        x: Sample positions.
        period: Grating period in nanometers.

    Returns:
        ``dy/dx`` on the facet containing each sample.
    """

    wrapped = np.mod(x, period)
    index = np.clip(np.searchsorted(positions, wrapped, side="right") - 1, 0, positions.size - 2)
    run = positions[index + 1] - positions[index]
    rise = heights[index + 1] - heights[index]
    return np.where(run > 0.0, rise / np.where(run > 0.0, run, 1.0), 0.0)


def _spectral_derivatives(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the first two derivatives of a periodic sequence with respect to ``t``.

    Args:
        values: Samples on a uniform grid over one period.

    Returns:
        First and second derivative samples.
    """

    count = values.size
    spectrum = np.fft.fft(values)
    frequency = np.fft.fftfreq(count, d=1.0 / count)
    # The Nyquist mode has no well-defined odd derivative on a real grid.
    if count % 2 == 0:
        spectrum[count // 2] = 0.0
    first = np.real(np.fft.ifft(1j * frequency * spectrum))
    second = np.real(np.fft.ifft(-(frequency**2) * spectrum))
    return first, second


def kress_log_weights(count: int) -> np.ndarray:
    """Return the Martensen-Kussmaul weights for the periodic log kernel.

    These integrate ``ln(4 sin^2(t / 2))`` against a trigonometric polynomial
    exactly, which is what lets the singular part of the kernel be handled
    without refining the grid::

        R_j = -(2 pi / n) sum_{m=1}^{n-1} cos(m t_j) / m - (pi / n^2) cos(n t_j)

    with ``n = count / 2`` and ``t_j = j pi / n``.

    Args:
        count: Number of nodes; must be even.

    Returns:
        Weights indexed by node separation, shape ``(count,)``.

    Raises:
        ValueError: If ``count`` is not a positive even number.
    """

    if count <= 0 or count % 2 != 0:
        raise ValueError(f"count must be a positive even number, got {count}.")
    half = count // 2
    separation = np.arange(count, dtype=float) * np.pi / half
    harmonics = np.arange(1, half, dtype=float)
    series = np.sum(
        np.cos(harmonics[None, :] * separation[:, None]) / harmonics[None, :], axis=1
    )
    return -(2.0 * np.pi / half) * series - (np.pi / half**2) * np.cos(half * separation)


def nystrom_operators(
    green: PeriodicGreen,
    *,
    target: TrigBoundary,
    source: TrigBoundary,
    same_boundary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single- and double-layer matrices by trigonometric Nystrom.

    Args:
        green: Green function of the medium between the two boundaries.
        target: Boundary carrying the collocation points.
        source: Boundary carrying the densities.
        same_boundary: Whether the two are the same boundary, which is what puts
            the singularity on the diagonal and brings in the Kress weights.

    Returns:
        ``(S, D)``, each shaped ``(target.count, source.count)``.

    Raises:
        ValueError: If the two boundaries have different node counts.
    """

    if target.count != source.count:
        raise ValueError("Both boundaries must share the same node count.")
    count = source.count
    trapezoid = 2.0 * np.pi / count

    dx = target.position[:, None, 0] - source.position[None, :, 0]
    dy = target.position[:, None, 1] - source.position[None, :, 1]
    # Evaluated in row blocks. The Ewald spectral sum broadcasts to
    # (rows, count, orders), and the order count grows with d/lambda, so the
    # whole matrix at once reaches gigabytes for the very cases this scheme
    # exists to make affordable -- at N = 512 and d/lambda = 100 it is 5.9 GB per
    # array, which thrashes long before it runs out.
    value = np.empty((count, count), dtype=complex)
    normal_derivative = np.empty((count, count), dtype=complex)
    rows = _chunk_rows(count=count, orders=green.spectral_reach())
    for begin in range(0, count, rows):
        end = min(begin + rows, count)
        block, grad_x, grad_y = green.value_and_gradient(dx[begin:end], dy[begin:end])
        value[begin:end] = block
        # The gradient is with respect to the field point, so the source-normal
        # derivative flips sign.
        normal_derivative[begin:end] = -(
            grad_x * source.normal[None, :, 0] + grad_y * source.normal[None, :, 1]
        )

    if not same_boundary:
        # Distinct boundaries never touch, so the kernel is smooth and plain
        # trapezoidal quadrature is already spectrally accurate.
        weight = trapezoid * source.speed[None, :]
        return value * weight, normal_derivative * weight

    # The singular image is the *nearest* one, which is not always the direct
    # pair. Nodes at opposite ends of the parameter range are neighbours across
    # the period wrap: ``ln(4 sin^2)`` already sees them as close, so the
    # free-space part subtracted from them has to be the near image too, or the
    # two halves of the splitting describe different singularities and M2 stops
    # being smooth. The near image also carries the quasi-periodic phase.
    image = np.round(dx / source.period)
    dx_image = dx - image * source.period
    phase = np.exp(1j * green.alpha0 * image * source.period)
    distance = np.hypot(dx_image, dy)

    separation = target.nodes[:, None] - source.nodes[None, :]
    log_factor = _log_sine_factor(separation)

    # The Bessel factors must carry the *complex* wavenumber. Using only its
    # real part leaves a residual logarithmic singularity in M2 of size
    # proportional to Im k, which caps convergence at roughly third order for
    # absorbing materials instead of spectral. The price is that they then grow
    # exponentially away from the diagonal, which the window bounds; see
    # _absorption_window for why an unwindowed split cannot be evaluated at all
    # for an absorbing coating on an X-ray period.
    window = _absorption_window(
        separation, wavenumber=green.wavenumber, period=source.period
    )
    bessel0 = phase * jv(0, green.wavenumber * distance)
    if window is not None:
        bessel0 = bessel0 * window
    m1_single = -bessel0 / (4.0 * np.pi)
    m2_single = value + bessel0 * log_factor / (4.0 * np.pi)

    # The double layer's singular coefficient carries q = (r_t - r_s) . n_s,
    # which vanishes quadratically on the diagonal, so M1 goes to zero there.
    q = dx_image * source.normal[None, :, 0] + dy * source.normal[None, :, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = np.where(
            distance > 0.0,
            jv(1, green.wavenumber * distance) * q / np.where(distance > 0.0, distance, 1.0),
            0.0,
        )
    m1_double = -phase * green.wavenumber * radial / (4.0 * np.pi)
    if window is not None:
        m1_double = m1_double * window
    m2_double = normal_derivative - m1_double * log_factor

    regular_value, regular_dx, regular_dy = green.regular_at_zero()
    diagonal = np.arange(count)
    free_regular = 0.25j - (
        np.log(green.wavenumber / 2.0) + np.euler_gamma
    ) / (2.0 * np.pi)

    m1_single[diagonal, diagonal] = -1.0 / (4.0 * np.pi)
    # A corner-graded parametrization drives the speed to zero at the corners, so
    # the diagonal's log diverges there. The final weighting multiplies by the
    # same speed, and s log(s) tends to zero, so the entry is well behaved -- but
    # it has to be formed without ever evaluating log(0).
    safe_speed = np.maximum(source.speed, _MIN_SPEED)
    m2_single[diagonal, diagonal] = (
        free_regular + regular_value - np.log(safe_speed) / (2.0 * np.pi)
    )
    m1_double[diagonal, diagonal] = 0.0
    # The free-space double layer tends to curvature / 4pi on the diagonal; the
    # periodic remainder contributes its own normal derivative there.
    m2_double[diagonal, diagonal] = source.curvature / (4.0 * np.pi) - (
        regular_dx * source.normal[:, 0] + regular_dy * source.normal[:, 1]
    )

    log_weights = kress_log_weights(count)
    index = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    kress = log_weights[index]

    speed = source.speed[None, :]
    single = (kress * m1_single + trapezoid * m2_single) * speed
    double = (kress * m1_double + trapezoid * m2_double) * speed
    return single, double


def _chunk_rows(*, count: int, orders: int) -> int:
    """Return how many matrix rows to evaluate per pass.

    Sized so one block of the Ewald spectral intermediate stays within a memory
    budget. Purely a memory-versus-call-overhead trade; the result is identical
    either way.

    Args:
        count: Number of source nodes.
        orders: Number of Ewald spectral orders retained.

    Returns:
        Row count, at least one.
    """

    per_row = max(count * max(orders, 1) * _COMPLEX_BYTES * _EWALD_TEMPORARIES, 1)
    return int(max(1, min(count, _MEMORY_BUDGET_BYTES // per_row)))


def _absorption_window(
    separation: np.ndarray, *, wavenumber: complex, period: float
) -> np.ndarray | None:
    """Return the cutoff that keeps the Kress split usable in an absorbing medium.

    The split subtracts ``J_0(k R)``, and ``J_0 = (H_0^(1) + H_0^(2)) / 2``. In a
    lossy medium ``H_0^(2)`` grows as ``exp(|Im k| R)`` while the Green function
    it is desingularizing *decays*, so over one period the two halves of the
    split reach ``exp(pi |Im k| d / 2pi)`` and cancel back to a kernel of order
    one. For gold on a 600 l/mm grating at 50 eV that factor is 7e18: the split
    is an exact identity that float64 cannot evaluate, and the assembled system
    comes out with entries of 1e13 where the kernel is 1e-2.

    Restricting the subtraction to a neighbourhood of the singularity fixes it.
    The split stays an identity because the window is one at ``t = s``, where the
    singularity is, and ``M2`` stays analytic because the window is smooth, so
    nothing about the scheme's order changes -- only the range over which the
    Bessel factor is allowed to grow.

    The width follows from the budget. With ``g = |Im k| d / 2pi`` the growth
    over a parameter separation ``u`` is at most ``g |u|``, and the window
    suppresses by ``(sin(u/2) / s)^2 >= (u / pi s)^2``, so the exponent peaks at
    ``g^2 pi^2 s^2 / 4``. Setting that to the budget gives the width below. It
    stays many grid steps wide for every node count this solver can afford --
    about nineteen at ``N = 256`` on the case above -- so the Kress weights,
    which are exact only up to the grid order, still see a resolved factor.

    Args:
        separation: Parameter differences ``t_i - t_j``.
        wavenumber: Medium wavenumber ``k0 * n`` in inverse nanometers.
        period: Grating period in nanometers.

    Returns:
        The cutoff, or ``None`` when the medium absorbs too weakly over one
        period for the Bessel factor to need one.
    """

    growth = abs(float(np.imag(wavenumber))) * float(period) / (2.0 * np.pi)
    if growth <= _WINDOW_OFF_BELOW:
        return None
    width = 2.0 * np.sqrt(_WINDOW_LOG_BUDGET) / (np.pi * growth)
    return np.exp(-((np.sin(0.5 * separation) / width) ** 2))


def _log_sine_factor(separation: np.ndarray) -> np.ndarray:
    """Return ``ln(4 sin^2(separation / 2))``, with the diagonal set aside.

    Args:
        separation: Parameter differences ``t_i - t_j``.

    Returns:
        The logarithm, with zeros where the separation vanishes; those entries
        are overwritten by the analytic diagonal.
    """

    sine = np.sin(0.5 * separation)
    squared = 4.0 * sine * sine
    return np.where(squared > 0.0, np.log(np.where(squared > 0.0, squared, 1.0)), 0.0)
