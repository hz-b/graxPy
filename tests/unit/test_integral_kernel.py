"""Unit tests for the boundary-integral solver's numerical kernels.

The solver is not reachable through ``grax.run_simulation`` yet, so these test
the pieces directly. They exist to pin the invariants that are easy to break and
expensive to notice: the compiled kernel agreeing with its reference, the
special-function approximations, and the guards on geometries the solver cannot
represent.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfcx, hankel1

from grax.solvers._ewald_kernel import NUMBA_AVAILABLE, weideman_coefficients
from grax.solvers._green import PeriodicGreen, default_ewald_splitting
from grax.solvers._nystrom import (
    build_trig_boundary,
    kress_log_weights,
    nystrom_operators,
)

pytestmark = pytest.mark.unit

PERIOD = 1239.8
WAVENUMBER = 2.0 * np.pi / 12.398
ALPHA0 = 0.8123 * WAVENUMBER


def _separations(count: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible grid of separations spanning one period."""

    generator = np.random.default_rng(20260818)
    return (
        generator.uniform(-PERIOD, PERIOD, (count, count)),
        generator.uniform(-25.0, 25.0, (count, count)),
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="requires numba")
@pytest.mark.parametrize("index", [1.0 + 0.0j, 0.95 + 0.02j])
def test_compiled_kernel_matches_the_reference_implementation(index: complex) -> None:
    """The compiled Ewald kernel reproduces the NumPy one it replaced.

    The compiled path exists purely for speed -- it is a factor of thirteen on a
    production-sized solve -- so any disagreement is a defect rather than a
    trade-off.
    """

    reference = PeriodicGreen(
        period=PERIOD,
        wavenumber=WAVENUMBER * index,
        alpha0=ALPHA0,
        force_reference_kernel=True,
    )
    compiled = PeriodicGreen(period=PERIOD, wavenumber=WAVENUMBER * index, alpha0=ALPHA0)
    dx, dy = _separations()

    expected = reference.value_and_gradient(dx, dy)
    actual = compiled.value_and_gradient(dx, dy)
    for left, right in zip(actual, expected, strict=True):
        assert np.max(np.abs(left - right)) / np.max(np.abs(right)) < 1e-8


def test_weideman_approximation_matches_scipy_erfcx() -> None:
    """The inlined Faddeeva form is accurate over the arguments actually used.

    The solver only ever evaluates ``erfcx`` where the real part is non-negative,
    which is the half plane this approximation is built for.
    """

    scale, coefficients = weideman_coefficients()
    real, imaginary = np.meshgrid(np.linspace(0.0, 6.0, 60), np.linspace(-4.0, 4.0, 60))
    argument = (real + 1j * imaginary).ravel()

    denominator = scale + argument
    ratio = (scale - argument) / denominator
    polynomial = np.zeros_like(argument)
    for coefficient in coefficients:
        polynomial = polynomial * ratio + coefficient
    approximation = 2.0 * polynomial / denominator**2 + (1.0 / np.sqrt(np.pi)) / denominator

    exact = erfcx(argument)
    assert np.max(np.abs(approximation - exact) / np.abs(exact)) < 1e-11


def test_regular_part_at_zero_matches_the_numerical_limit() -> None:
    """``regular_at_zero`` agrees with what evaluating near zero converges to.

    It cannot be obtained by evaluating close to the origin in production -- two
    diverging terms cancel -- so the analytic form is what the Nystrom diagonal
    uses, and this pins it.
    """

    green = PeriodicGreen(period=PERIOD, wavenumber=WAVENUMBER, alpha0=ALPHA0)
    analytic = green.regular_at_zero()[0]

    previous = None
    for radius in (1e-3, 1e-4):
        numerical = complex(green.value(np.array(radius), np.array(0.0))) - complex(
            0.25j * hankel1(0, WAVENUMBER * radius)
        )
        error = abs(numerical - analytic) / abs(analytic)
        if previous is not None:
            # The numerical estimate approaches the limit linearly in the radius.
            assert error < previous
        previous = error
    assert previous is not None and previous < 1e-3


def test_larger_splitting_only_improves_the_kernel() -> None:
    """The splitting parameter is a conditioning knob, not a free speed knob.

    In exact arithmetic the Ewald split is independent of it. In double precision
    the two halves cancel by roughly ``exp(r) / sqrt(r)`` with
    ``r = k^2 / 4 E^2``, so accuracy improves monotonically as the splitting
    grows. The default sits at ``r = 1``, which costs about five digits against
    the plane-wave series; this pins that it is not worse than that, and that
    raising the splitting closes the gap rather than opening it.

    Deliberately checked on the kernel and not on a solve. A solve whose
    discretization error dominates is insensitive to a kernel that is wrong on
    its near-field entries, which is how a far too small splitting once survived
    a solve-level sweep.
    """

    dx = np.array(3.0)
    dy = np.array(2.0)
    oracle = complex(
        PeriodicGreen(
            period=PERIOD,
            wavenumber=WAVENUMBER,
            alpha0=ALPHA0,
            method="spectral",
            spectral_orders=6000,
        ).value(dx, dy)
    )

    default = default_ewald_splitting(PERIOD, WAVENUMBER)
    errors = []
    for scale in (1.0, 2.0, 4.0):
        value = complex(
            PeriodicGreen(
                period=PERIOD,
                wavenumber=WAVENUMBER,
                alpha0=ALPHA0,
                splitting=scale * default,
            ).value(dx, dy)
        )
        errors.append(abs(value - oracle) / abs(oracle))

    assert errors[0] < 1e-4
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-8


def test_too_small_a_splitting_is_rejected() -> None:
    """A splitting parameter that would destroy the kernel must not be accepted."""

    with pytest.raises(ValueError, match="too small for this medium"):
        PeriodicGreen(
            period=PERIOD,
            wavenumber=WAVENUMBER,
            alpha0=ALPHA0,
            splitting=0.175 * WAVENUMBER,
        )


def test_rayleigh_anomaly_is_rejected() -> None:
    """An order exactly at grazing emergence is a pole, and must not pass silently."""

    # Order 13 lands exactly on beta = 0 for this geometry.
    with pytest.raises(ValueError, match="Rayleigh anomaly"):
        PeriodicGreen(period=PERIOD, wavenumber=WAVENUMBER, alpha0=0.87 * WAVENUMBER)


def test_vertical_facets_are_rejected_by_the_graph_parametrization() -> None:
    """A profile with a vertical wall is not a graph and must raise, not flatten it."""

    from grax.solvers._nystrom import build_graded_boundary

    positions = np.array([0.0, 40.0, 40.0, 60.0, 60.0, 100.0])
    heights = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="vertical facet"):
        build_graded_boundary(positions, heights, period=100.0, count=32)


def test_kress_weights_integrate_the_periodic_logarithm_exactly() -> None:
    """The Martensen-Kussmaul weights are exact on trigonometric polynomials.

    ``int ln(4 sin^2((t - s)/2)) exp(i m s) ds`` is ``-2 pi / |m|`` times the same
    exponential, and zero for the constant.
    """

    count = 64
    weights = kress_log_weights(count)
    nodes = 2.0 * np.pi * np.arange(count) / count
    index = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    operator = weights[index]

    assert abs(np.sum(weights)) < 1e-12
    for harmonic in (1, 2, 5):
        exact = (-2.0 * np.pi / harmonic) * np.exp(1j * harmonic * nodes)
        assert np.max(np.abs(operator @ np.exp(1j * harmonic * nodes) - exact)) < 1e-12


def test_trig_boundary_reproduces_a_known_arc_length() -> None:
    """The quadrature weights integrate to the boundary's developed length."""

    period = 100.0
    x = np.linspace(0.0, period, 2049)
    depth = 6.0
    y = 0.5 * depth * (1.0 - np.cos(2.0 * np.pi * x / period))
    boundary = build_trig_boundary(x, y, period=period, count=256)

    exact = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    assert abs(float(np.sum(boundary.weight)) - exact) / exact < 1e-6


def test_absorbing_medium_does_not_blow_up_the_kress_split() -> None:
    """The operator blocks stay bounded when the medium absorbs over one period.

    The Kress split subtracts ``J_0(k R)``, which grows as ``exp(|Im k| R)`` in a
    lossy medium while the Green function it is desingularizing decays. On an
    X-ray period that reaches ``exp(43)`` before the window introduced with this
    test bounded it, and the assembled system came out with entries of 1e13
    where the kernel itself is 1e-2. The regime is not exotic: it is gold at
    50 eV on a 600 l/mm grating, which is a graxPy validation case.
    """

    period = 1666.667
    wavelength = 24.796
    k0 = 2.0 * np.pi / wavelength
    wavenumber = k0 * (0.828696 + 0.205835j)
    alpha0 = k0 * float(np.sin(np.deg2rad(90.0 - 4.882)))

    positions = np.array([0.0, 1475.134, period])
    heights = np.array([0.0, 18.77, 0.0])
    boundary = build_trig_boundary(
        np.linspace(0.0, period, 65),
        np.interp(np.linspace(0.0, period, 65), positions, heights),
        period=period,
        count=64,
    )
    green = PeriodicGreen(
        period=period,
        wavenumber=wavenumber,
        alpha0=alpha0,
        method="ewald",
        splitting=default_ewald_splitting(period, wavenumber),
    )

    single, double = nystrom_operators(
        green, target=boundary, source=boundary, same_boundary=True
    )

    # The Green function itself is order 10 on this geometry, and the quadrature
    # weights are below the period, so anything past a few thousand means the
    # split has been evaluated outside the range float64 can carry.
    assert np.max(np.abs(single)) < 1.0e4
    assert np.max(np.abs(double)) < 1.0e4
    assert np.all(np.isfinite(single))
    assert np.all(np.isfinite(double))
