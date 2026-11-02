# ruff: noqa: D103

"""Checks of the coupled-wave model of a comb-electrode Bragg grating."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from grax.bragg_grating import (
    IdtGrating,
    active_fraction,
    bragg_crystal,
    gaussian_resolution,
    rocking_curve,
    simulate_rocking_curves,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def crystal():
    return bragg_crystal("LiNbO3", (1, 0, 4), 8000.0)


def _fwhm(x, y):
    above = np.where(y >= y.max() / 2.0)[0]
    return x[above[-1]] - x[above[0]]


def test_lithium_niobate_104_bragg_angle_matches_the_paper(crystal):
    assert np.degrees(crystal.theta_b_rad) == pytest.approx(16.45, abs=0.01)
    assert crystal.d_spacing_um * 1e4 == pytest.approx(2.736, abs=0.002)


def test_other_crystals_and_reflections_are_available():
    silicon = bragg_crystal("Si", (1, 1, 1), 8000.0)

    assert np.degrees(silicon.theta_b_rad) == pytest.approx(14.31, abs=0.01)


@pytest.mark.parametrize(
    ("material", "hkl", "energy", "message"),
    [
        ("Si", (2, 0, 0), 8000.0, "forbidden"),
        ("Si", (8, 8, 8), 5000.0, "cannot be reached"),
        ("quartz", (1, 0, 0), 8000.0, "Unknown crystal"),
        ("Si", (0, 0, 0), 8000.0, "zero"),
    ],
)
def test_impossible_reflections_are_explained(material, hkl, energy, message):
    with pytest.raises(ValueError, match=message):
        bragg_crystal(material, hkl, energy)


def test_unstrained_curve_is_a_darwin_curve(crystal):
    angles = np.radians(np.linspace(-0.01, 0.015, 251))
    curve = rocking_curve(crystal, IdtGrating(), 0.0, angles, n_orders=1, include_al_mask=False)
    darwin_width = 2.0 * abs(crystal.chih) / np.sin(2.0 * crystal.theta_b_rad)
    centre = -crystal.chi0.real / np.sin(2.0 * crystal.theta_b_rad)
    above = np.where(curve.total >= curve.total.max() / 2.0)[0]
    assert curve.total.max() > 0.9
    assert _fwhm(angles, curve.total) == pytest.approx(darwin_width, rel=0.06)
    assert 0.5 * (angles[above[0]] + angles[above[-1]]) == pytest.approx(centre, rel=0.1)


def test_orders_stay_decoupled_without_deformation(crystal):
    angles = np.radians(np.linspace(-0.01, 0.015, 41))
    single = rocking_curve(crystal, IdtGrating(), 0.0, angles, n_orders=1, include_al_mask=False)
    many = rocking_curve(crystal, IdtGrating(), 0.0, angles, n_orders=9, include_al_mask=False)
    np.testing.assert_allclose(many.total, single.total, rtol=1e-6, atol=1e-9)


def test_without_absorption_the_peak_reflects_everything_with_or_without_deformation(crystal):
    lossless = replace(
        crystal,
        chi0=complex(crystal.chi0.real, 0.0),
        chih=complex(crystal.chih.real, 0.0),
        chihbar=complex(crystal.chihbar.real, 0.0),
    )
    angles = np.radians(np.linspace(0.0, 0.006, 25))
    for amplitude in (0.0, 0.05):
        curve = rocking_curve(
            lossless, IdtGrating(), amplitude, angles, n_orders=11, include_al_mask=False
        )
        assert curve.total.max() == pytest.approx(1.0, abs=0.01)


def test_deformation_creates_satellites_at_the_grating_spacing(crystal):
    grating = IdtGrating()
    angles = np.radians(np.linspace(-0.012, 0.012, 241))
    curve = rocking_curve(crystal, grating, 0.108, angles, n_orders=13, include_al_mask=False)
    expected = crystal.wavelength_um / (2.0 * grating.period_um * np.sin(crystal.theta_b_rad))
    centre = -crystal.chi0.real / np.sin(2.0 * crystal.theta_b_rad)
    order_one = curve.order_efficiency[:, list(curve.orders).index(1)]
    peak = angles[np.argmax(order_one)]
    assert peak - centre == pytest.approx(expected, rel=0.2)
    assert curve.total[np.argmin(np.abs(angles - (centre + expected)))] > 0.2


def test_a_larger_amplitude_lowers_the_central_peak_and_raises_the_satellites(crystal):
    angles = np.radians(np.linspace(-0.012, 0.012, 121))
    result = simulate_rocking_curves(
        crystal, IdtGrating(), [0.0, 0.03, 0.1], angles, resolution_rad=np.radians(8.7 / 3600)
    )
    curves = result["curves"]
    centre = int(np.argmin(np.abs(angles - result["peak_angle_rad"])))
    assert curves[0.0].max() == pytest.approx(1.0)
    assert curves[0.1][centre] < curves[0.03][centre] < curves[0.0][centre]
    assert curves[0.1][angles < result["peak_angle_rad"] - np.radians(0.003)].max() > 0.3


def test_beam_outside_the_electrodes_leaves_the_undeformed_peak(crystal):
    grating = IdtGrating(electrode_length_um=100.0, aperture_um=50.0)
    angles = np.radians(np.linspace(-0.012, 0.012, 61))
    small = active_fraction(grating, crystal, 10.0, 10.0)
    large = active_fraction(grating, crystal, 2000.0, 2000.0)
    assert small > 0.95
    assert large < 0.05
    outside = simulate_rocking_curves(
        crystal, grating, [0.1], angles, beam_x_um=2000.0, beam_y_um=2000.0
    )
    inside = simulate_rocking_curves(crystal, grating, [0.1], angles, beam_x_um=10.0, beam_y_um=10.0)
    assert outside["curves"][0.1].max() > inside["curves"][0.1].max()


def test_resolution_smooths_without_changing_the_area():
    angles = np.linspace(-1e-3, 1e-3, 401)
    curve = np.exp(-0.5 * (angles / 5e-5) ** 2)
    smooth = gaussian_resolution(angles, curve, 1e-4)
    assert smooth.max() < curve.max()
    assert smooth.sum() == pytest.approx(curve.sum(), rel=1e-3)


def test_even_number_of_orders_is_rejected(crystal):
    with pytest.raises(ValueError, match="odd"):
        rocking_curve(crystal, IdtGrating(), 0.0, np.array([0.0]), n_orders=4)
