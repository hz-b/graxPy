"""End-to-end checks for the Nevière differential-method solver.

These use the published RETICOLO ``exemple1_1D.m`` configuration already
referenced by ``test_simulation_reticolo_parity``: a deep, high-contrast
lamellar grating at normal incidence. It is a much harder convergence case than
the shallow X-ray gratings the project normally runs, and it exercises both the
TM fast-Fourier-factorization rules and the stability of the interface-response
cascade across many wavelengths of optical thickness.
"""

from __future__ import annotations

import numpy as np
import pytest

from grax.solvers import res0, res1, res2, res2_dm
from grax.solvers.neviere import NeviereOptions

# lambda = 6 um, period = 10 um, 20 um deep, n = 1.5 ridges over [0, 1] and
# [9, 10] um with n = 1 grooves between, normal incidence, 25 Fourier orders.
_WAVELENGTH_NM = 6000.0
_PERIOD_NM = 10000.0
_DEPTH_NM = 20000.0
_FOURIER_ORDERS = 25
_GRATING_TEXTURE = [
    np.array([1000.0, 9000.0]),
    np.array([1.5, 1.0], dtype=complex),
]

# Published RETICOLO exemple1_1D.m zeroth-order transmission.
_RETICOLO_TRANSMISSION = {1: 0.5050632035, -1: 0.9111790287}


def _solve(polarization: int, *, depth_nm: float = _DEPTH_NM, solver: str = "neviere",
           options: NeviereOptions | None = None):
    """Return the zeroth-order transmission and the total energy for one solve."""

    parm = res0(polarization)
    aa = res1(
        _WAVELENGTH_NM,
        _PERIOD_NM,
        [1.0, 1.5, _GRATING_TEXTURE],
        _FOURIER_ORDERS,
        0.0,
        parm,
        _fourier_backend="numba",
    )
    profile = ([0.0, depth_nm, 0.0], [0, 2, 1])
    if solver == "rcwa":
        result = res2(aa, profile, parm)
    else:
        result = res2_dm(aa, profile, parm, options=options)

    zeroth_index = int(np.where(result.inc_top_reflected.order == 0)[0][0])
    transmission = float(result.inc_top_transmitted.efficiency[zeroth_index])
    total = float(
        np.sum(result.inc_top_reflected.efficiency)
        + np.sum(result.inc_top_transmitted.efficiency)
    )
    return transmission, total, result


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("polarization", "label"),
    [(1, "TE"), (-1, "TM")],
)
def test_neviere_matches_published_reticolo_lamellar_reference(
    polarization: int,
    label: str,
) -> None:
    """Verify the differential method reproduces the published RETICOLO values.

    TM is the load-bearing case: without the Li/fast-Fourier-factorization
    inverse rule a differential-method solve of this high-contrast profile
    converges far too slowly to reach this agreement at 25 orders.
    """

    transmission, total, _ = _solve(polarization)

    assert transmission == pytest.approx(_RETICOLO_TRANSMISSION[polarization], abs=1e-6), label
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.parametrize("polarization", [1, -1])
def test_neviere_agrees_with_rcwa_on_the_lamellar_reference(polarization: int) -> None:
    """Verify both solvers land on the same answer for the reference geometry."""

    neviere_transmission, _, neviere = _solve(polarization)
    rcwa_transmission, _, rcwa = _solve(polarization, solver="rcwa")

    assert neviere_transmission == pytest.approx(rcwa_transmission, abs=1e-9)
    assert np.allclose(
        np.real(neviere.inc_top_reflected.efficiency),
        np.real(rcwa.inc_top_reflected.efficiency),
        atol=1e-9,
    )
    assert np.allclose(
        np.real(neviere.inc_top_transmitted.efficiency),
        np.real(rcwa.inc_top_transmitted.efficiency),
        atol=1e-9,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("polarization", [1, -1])
def test_neviere_agrees_with_rcwa_at_the_deepest_shared_depth(polarization: int) -> None:
    """Verify the solvers still agree just below the modal solver's depth ceiling.

    The modal solver overflows above roughly 7 wavelengths of depth for this
    high-contrast profile (TE first, TM shortly after), so this is close to the
    deepest structure the two can be compared on directly.
    """

    depth_nm = 40000.0
    neviere_transmission, neviere_total, _ = _solve(polarization, depth_nm=depth_nm)
    rcwa_transmission, _, _ = _solve(polarization, depth_nm=depth_nm, solver="rcwa")

    assert neviere_transmission == pytest.approx(rcwa_transmission, abs=1e-8)
    assert neviere_total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.parametrize("polarization", [1, -1])
def test_neviere_stays_stable_far_beyond_the_rcwa_depth_ceiling(polarization: int) -> None:
    """Verify very deep gratings stay finite and energy-conserving.

    At 167 wavelengths the evanescent orders span hundreds of e-foldings. The
    modal solver forms ``q / sinh(q d)`` across the whole layer in one go and
    overflows well before this depth; the differential method only ever forms a
    transfer matrix across a sub-block bounded by ``block_phase`` and combines
    those with the interface-response cascade, which is an R-matrix propagation
    and never builds a growing exponential.
    """

    depth_nm = 1.0e6
    transmission, total, result = _solve(polarization, depth_nm=depth_nm)

    assert np.all(np.isfinite(np.real(result.inc_top_reflected.efficiency)))
    assert np.all(np.isfinite(np.real(result.inc_top_transmitted.efficiency)))
    assert 0.0 <= transmission <= 1.0
    assert total == pytest.approx(1.0, abs=1e-6)

    # Guards the premise: the modal solver genuinely cannot reach this depth, so
    # if it ever gains the range this test should be revisited rather than kept
    # as a claimed differential-method advantage.
    with pytest.raises(ValueError, match="NaN/Inf"):
        _solve(polarization, depth_nm=depth_nm, solver="rcwa")


@pytest.mark.smoke
def test_neviere_reference_is_insensitive_to_the_sub_block_size() -> None:
    """Verify block_phase only trades conditioning, not the converged answer."""

    transmissions = [
        _solve(-1, options=NeviereOptions(block_phase=block_phase))[0]
        for block_phase in (4.0, 2.0, 1.0, 0.5)
    ]

    assert np.allclose(transmissions, transmissions[0], atol=1e-9)
