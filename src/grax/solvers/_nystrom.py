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

Scope
-----
Smooth (corner-free) boundaries, parametrized as graphs ``y = f(x)``. Profiles
with corners -- laminar sidewalls, blazed apexes -- lose the analyticity the
spectral accuracy rests on and need graded reparametrization on top of this;
that is the second half of Stage 2 and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from ._green import PeriodicGreen

__all__ = ["TrigBoundary", "kress_log_weights", "nystrom_operators"]


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
    value, grad_x, grad_y = green.value_and_gradient(dx, dy)
    # The gradient is with respect to the field point, so the source-normal
    # derivative flips sign.
    normal_derivative = -(
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
    # absorbing materials instead of spectral.
    bessel0 = phase * jv(0, green.wavenumber * distance)
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
    m2_double = normal_derivative - m1_double * log_factor

    regular_value, regular_dx, regular_dy = green.regular_at_zero()
    diagonal = np.arange(count)
    free_regular = 0.25j - (
        np.log(green.wavenumber / 2.0) + np.euler_gamma
    ) / (2.0 * np.pi)

    m1_single[diagonal, diagonal] = -1.0 / (4.0 * np.pi)
    m2_single[diagonal, diagonal] = (
        free_regular
        + regular_value
        - np.log(source.speed) / (2.0 * np.pi)
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
