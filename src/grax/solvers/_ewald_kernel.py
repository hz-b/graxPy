"""Compiled inner kernel for the Ewald spectral sum.

Profiling the Nystrom solver put 76 percent of its runtime in one place: the
complex ``erfcx`` calls inside the Ewald spectral half, with another 15 percent
in the array arithmetic wrapped around them. Both come from the same shape. The
spectral sum is evaluated over ``(points, points, orders)``, so the vectorised
NumPy form materialises several arrays of that size, and the order count grows
with the period-to-wavelength ratio -- exactly the regime this solver exists to
serve.

This module fuses the whole sum into one compiled loop that accumulates over
orders in registers. Nothing of size ``(points, points, orders)`` is ever
created; the output is three ``(points, points)`` matrices. Three things make it
faster than the vectorised form:

- no intermediates, so the traffic is one pass over the output rather than
  several over the full tensor;
- ``exp(i alpha_n X)`` becomes a recurrence. The ``alpha_n`` are an arithmetic
  progression, so the phase advances by one complex multiply per order instead
  of a transcendental call;
- ``erfcx`` is evaluated by Weideman's rational approximation to the Faddeeva
  function, inlined. Against SciPy it agrees to 2.6e-13 at 32 terms over the
  argument range this solver produces, which is far below the discretization
  error.

The NumPy implementation in :mod:`grax.solvers._green` stays as the reference
and is used when Numba is unavailable. The two agree to round-off, which is
asserted by the test suite rather than assumed.

Reference:
    Weideman, *SIAM J. Numer. Anal.* **31**, 1497 (1994) -- computation of the
    complex error function.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """Return a no-op decorator when Numba is missing."""

        def wrap(function):
            return function

        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range  # type: ignore[assignment]

__all__ = [
    "NUMBA_AVAILABLE",
    "lattice_terms",
    "spectral_terms",
    "weideman_coefficients",
]

#: Terms in the Weideman expansion. 32 gives 2.6e-13 worst case over the
#: arguments this solver produces; 24 would give 4.2e-10, too close to the
#: tolerances the solver targets.
_WEIDEMAN_TERMS = 32


def weideman_coefficients(terms: int = _WEIDEMAN_TERMS) -> tuple[float, np.ndarray]:
    """Return the scale and polynomial coefficients of Weideman's approximation.

    These depend only on the term count, so they are computed once and reused.

    Args:
        terms: Number of terms in the expansion.

    Returns:
        ``(L, a)``, the scale parameter and the coefficients in descending
        powers, ready for Horner evaluation.
    """

    count = int(terms)
    half = 2 * count
    index = np.arange(-half + 1, half, dtype=float)
    scale = np.sqrt(count / np.sqrt(2.0))
    tangent = scale * np.tan(index * np.pi / (2.0 * half) * 2.0 / 2.0)
    weight = np.exp(-(tangent**2)) * (scale**2 + tangent**2)
    padded = np.concatenate(([0.0], weight))
    spectrum = np.real(np.fft.fft(np.fft.fftshift(padded))) / (2.0 * half)
    return float(scale), np.ascontiguousarray(spectrum[1 : count + 1][::-1])


@njit(cache=True, inline="always")
def _erfcx(argument: complex, scale: float, coefficients: np.ndarray) -> complex:
    """Return ``erfcx(argument)`` for ``Re(argument) >= 0``.

    Uses ``erfcx(z) = w(i z)`` with Weideman's rational form of the Faddeeva
    function, which is accurate throughout the upper half plane and therefore for
    every argument with a non-negative real part.

    Args:
        argument: Where to evaluate; must have a non-negative real part.
        scale: Weideman's ``L``.
        coefficients: Polynomial coefficients in descending powers.

    Returns:
        The scaled complementary error function.
    """

    denominator = scale + argument
    ratio = (scale - argument) / denominator
    polynomial = 0.0 + 0.0j
    for index in range(coefficients.size):
        polynomial = polynomial * ratio + coefficients[index]
    return 2.0 * polynomial / (denominator * denominator) + 0.5641895835477563 / denominator


@njit(cache=True, parallel=True, fastmath=True)
def spectral_terms(
    dx: np.ndarray,
    dy: np.ndarray,
    alpha_first: float,
    alpha_step: float,
    beta: np.ndarray,
    splitting: float,
    period: float,
    scale: float,
    coefficients: np.ndarray,
    value: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
) -> None:
    """Accumulate the Ewald spectral sum and its gradient into three matrices.

    Args:
        dx: In-plane separations, shape ``(rows, columns)``.
        dy: Out-of-plane separations, same shape.
        alpha_first: In-plane wavenumber of the lowest retained order.
        alpha_step: Spacing between consecutive ``alpha_n``.
        beta: Out-of-plane wavenumbers on the ``Im beta >= 0`` branch.
        splitting: Ewald splitting parameter.
        period: Grating period in nanometers.
        scale: Weideman's ``L``.
        coefficients: Weideman polynomial coefficients.
        value: Output for ``G``, shape ``(rows, columns)``.
        derivative_x: Output for ``dG/dx``.
        derivative_y: Output for ``dG/dy``.
    """

    orders = beta.size
    prefactor = 0.25j / period
    inverse_splitting = 1.0 / (2.0 * splitting)

    # Per-order factors that do not depend on the evaluation point.
    shifted = np.empty(orders, dtype=np.complex128)
    growth = np.empty(orders, dtype=np.complex128)
    for n in range(orders):
        shifted[n] = -1j * beta[n] * inverse_splitting
        growth[n] = np.exp(beta[n] * beta[n] * inverse_splitting * inverse_splitting)

    for row in prange(dx.shape[0]):
        for column in range(dx.shape[1]):
            separation_x = dx[row, column]
            separation_y = dy[row, column]
            height = abs(separation_y)
            sign = 0.0
            if separation_y > 0.0:
                sign = 1.0
            elif separation_y < 0.0:
                sign = -1.0

            shift = splitting * height
            decay = np.exp(-shift * shift)

            # exp(i alpha_n X) by recurrence: the alpha_n are equally spaced.
            phase = np.exp(1j * alpha_first * separation_x)
            phase_step = np.exp(1j * alpha_step * separation_x)

            total = 0.0 + 0.0j
            total_x = 0.0 + 0.0j
            total_y = 0.0 + 0.0j
            for n in range(orders):
                combined = growth[n] * decay
                lower = combined * _erfcx(shifted[n] + shift, scale, coefficients)
                mirrored = shifted[n] - shift
                if mirrored.real >= 0.0:
                    upper = combined * _erfcx(mirrored, scale, coefficients)
                else:
                    upper = 2.0 * np.exp(1j * beta[n] * height) - combined * _erfcx(
                        -mirrored, scale, coefficients
                    )
                bracket = lower + upper
                common = phase / beta[n] * prefactor
                total += common * bracket
                total_x += 1j * (alpha_first + alpha_step * n) * common * bracket
                total_y += common * 1j * beta[n] * (upper - lower)
                phase = phase * phase_step

            value[row, column] = total
            derivative_x[row, column] = total_x
            derivative_y[row, column] = total_y * sign


@njit(cache=True, parallel=True, fastmath=True)
def lattice_terms(
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    first_integral: np.ndarray,
    decay: np.ndarray,
    argument: np.ndarray,
    phases: np.ndarray,
    coefficients: np.ndarray,
    splitting_squared: float,
    value: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
) -> None:
    """Accumulate the Ewald lattice sum and its gradient into three matrices.

    The radial series is ``sum_p c_p E_{p+1}(z)`` with ``z = R^2 E^2``, evaluated
    by the upward recurrence ``E_{p+1} = (exp(-z) - z E_p) / p`` from ``E_1``.
    Because ``dE_{p+1}/dz = -E_p``, the derivative reuses the same ladder one
    rung down.

    ``E_1`` itself is supplied by the caller from SciPy rather than approximated
    here. It is evaluated once per lattice image, while the recurrence runs for
    every term, so the recurrence is what costs and the special function is not
    worth trading accuracy for: the polynomial approximations for ``E_1`` reach
    about ``1e-7`` relative, which would cap the whole solver three orders above
    where the rest of it works.

    Args:
        offsets_x: ``dx - m d`` per image, shape ``(rows, columns, images)``.
        offsets_y: ``dy`` broadcast to the same shape.
        first_integral: ``E_1(z)`` at each entry.
        decay: ``exp(-z)`` at each entry.
        argument: ``z = R^2 E^2`` at each entry.
        phases: ``exp(i alpha_0 m d)`` for each image.
        coefficients: ``c_p = (k^2 / 4 E^2)^p / p!``, ascending in ``p``.
        splitting_squared: ``E^2``.
        value: Output for ``G``, shape ``(rows, columns)``.
        derivative_x: Output for ``dG/dx``.
        derivative_y: Output for ``dG/dy``.
    """

    quarter_pi = 1.0 / (4.0 * np.pi)
    images = phases.size
    for row in prange(offsets_x.shape[0]):
        for column in range(offsets_x.shape[1]):
            total = 0.0 + 0.0j
            total_x = 0.0 + 0.0j
            total_y = 0.0 + 0.0j
            for image in range(images):
                z = argument[row, column, image]
                exponential = decay[row, column, image]
                current = first_integral[row, column, image] + 0.0j
                series = current
                series_derivative = -splitting_squared * (exponential / z) + 0.0j
                for p in range(1, coefficients.size):
                    following = (exponential - z * current) / p
                    series = series + coefficients[p] * following
                    series_derivative = (
                        series_derivative - coefficients[p] * splitting_squared * current
                    )
                    current = following

                weight = phases[image] * quarter_pi
                total += weight * series
                total_x += weight * series_derivative * 2.0 * offsets_x[row, column, image]
                total_y += weight * series_derivative * 2.0 * offsets_y[row, column, image]
            value[row, column] = total
            derivative_x[row, column] = total_x
            derivative_y[row, column] = total_y
