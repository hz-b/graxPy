import pickle
from dataclasses import replace

import numpy as np
import pytest

from grax import run_simulation
from grax.bragg_grating import IdtGrating, bragg_crystal, coating_transmission, rocking_curve
from grax.idt_surface_grating import (
    CoatingLayer,
    IdtSurfaceGrating,
    default_theta_range_deg,
    grazing_active_fraction,
    multilayer_period_nm,
    surface_rocking_scan,
    theta_window,
)
from grax.materials import MaterialSpec
from grax.planar_mirror import parratt_reflectivity

ENERGY = 8000.0
PT = (CoatingLayer("Pt", 30.0),)


def _orders(result):
    return {int(order): float(value) for order, value in zip(result.orders, result.efficiency_all)}


def _solve(grating, angle=0.4, **options):
    options.setdefault("fourier_orders", 15)
    return _orders(
        run_simulation(
            grating=grating, energy_ev=ENERGY, grazing_angle_deg=angle, validate_physical_results=False, **options
        )
    )


def test_period_follows_the_electrodes():
    grating = IdtSurfaceGrating(idt=IdtGrating(finger_um=1.1, gap_um=0.9))
    assert grating.period_nm == pytest.approx(4000.0)


def test_flat_coated_crystal_matches_parratt():
    flat = IdtSurfaceGrating(idt=IdtGrating(al_thickness_um=0.0), layers=PT)
    for angle in (0.2, 0.4, 0.7):
        expected = parratt_reflectivity(flat.stack(), ENERGY, angle)[0, 0]
        assert _solve(flat, angle)[0] == pytest.approx(expected, rel=1e-4)  # solver uses hc = 1239.8 eV nm


def test_fingers_alone_give_only_even_orders():
    orders = _solve(IdtSurfaceGrating(amplitude_nm=0.0, layers=PT))
    assert orders[2] > 1e-3
    assert max(value for order, value in orders.items() if order % 2) < 1e-12


def test_odd_orders_grow_as_amplitude_squared():
    first = _solve(IdtSurfaceGrating(amplitude_nm=0.02, layers=PT))[1]
    second = _solve(IdtSurfaceGrating(amplitude_nm=0.04, layers=PT))[1]
    assert first > 0.0
    assert second / first == pytest.approx(4.0, rel=0.02)


def test_rcwa_and_neviere_agree():
    grating = IdtSurfaceGrating(amplitude_nm=0.1, layers=PT)
    rcwa = _solve(grating)
    neviere = _solve(grating, solver="neviere")
    for order in (0, 1, 2):
        assert neviere[order] == pytest.approx(rcwa[order], rel=1e-3)


def test_ripple_slicing_is_converged():
    coarse = _solve(IdtSurfaceGrating(amplitude_nm=0.1, layers=PT, ripple_slices=8))[1]
    fine = _solve(IdtSurfaceGrating(amplitude_nm=0.1, layers=PT, ripple_slices=32))[1]
    assert coarse == pytest.approx(fine, rel=1e-2)


def test_slices_sit_at_the_interfaces():
    grating = IdtSurfaceGrating(amplitude_nm=0.0, layers=PT)
    _, (thicknesses, indices) = grating.build_textures(ENERGY)
    assert thicknesses.sum() == pytest.approx(130.0)
    assert sorted(np.round(thicknesses[1:-1], 6)) == [30.0, 30.0, 70.0]
    assert indices[0] != indices[-1]


def test_planarization_smooths_the_steps_without_crossing():
    layers = tuple(CoatingLayer("Pt", 20.0, planarization_nm=5.0) for _ in range(3))
    _, profiles = IdtSurfaceGrating(layers=layers).interface_profiles()
    envelope, *upper = profiles[1:]
    slopes = [np.abs(np.diff(profile)).max() for profile in (envelope, *upper)]
    assert all(later < earlier for earlier, later in zip(slopes, slopes[1:]))
    for lower, profile in zip((envelope, *upper), upper):
        assert np.all(profile >= lower)
    for index, profile in enumerate(upper, start=1):
        # Layers thin at the finger corners, never below zero, so the mean rises slightly.
        assert np.mean(profile[:-1]) == pytest.approx(np.mean(envelope[:-1]) + 20.0 * index, abs=0.5)


def test_zero_planarization_is_a_vertical_shift():
    layers = (CoatingLayer("Pt", 20.0), CoatingLayer("C", 5.0))
    _, profiles = IdtSurfaceGrating(amplitude_nm=0.1, layers=layers).interface_profiles()
    np.testing.assert_allclose(profiles[2], profiles[1] + 20.0)
    np.testing.assert_allclose(profiles[3], profiles[1] + 25.0)


def test_grating_pickles():
    grating = IdtSurfaceGrating(amplitude_nm=0.05, layers=PT)
    restored = pickle.loads(pickle.dumps(grating))
    assert restored.layers == grating.layers and restored.period_nm == grating.period_nm


def test_material_map_labels_every_material():
    grating = IdtSurfaceGrating(layers=(CoatingLayer("Pt", 10.0), CoatingLayer(MaterialSpec("B4C", 2.52), 5.0)))
    _, _, codes, labels = grating._material_plot_data()
    assert len(labels) == 4
    assert set(np.unique(codes)) == {-1, 0, 1, 2, 3}


def test_multilayer_window_contains_the_satellite_peaks():
    layers = tuple(
        layer for _ in range(20) for layer in (CoatingLayer("W", 1.2), CoatingLayer(MaterialSpec("B4C", 2.52), 1.8))
    )
    low, high = default_theta_range_deg(IdtSurfaceGrating(layers=layers), ENERGY)
    wavelength = 1239.84198 / ENERGY
    bragg = np.degrees(np.arcsin(wavelength / (2 * 3.0)))
    assert low < bragg < high
    assert high - low > np.degrees(wavelength / (4000.0 * np.sin(np.radians(bragg))))


def test_footprint_shrinks_the_active_fraction_at_grazing_incidence():
    fraction = grazing_active_fraction(IdtGrating(), np.radians([0.2, 2.0]), 100.0, 0.0)
    assert fraction[0] < fraction[1] < 1.0


def test_scan_mixes_the_flat_footprint_into_order_zero():
    grating = IdtSurfaceGrating(layers=PT)
    theta = np.array([0.3, 0.4, 0.5])
    reported = []
    result = surface_rocking_scan(
        grating, [0.0, 0.1], theta, energy_ev=ENERGY, fourier_orders=10, beam_x_um=100.0,
        progress=lambda amplitude, curves, total, exit_angles, computed: reported.append((amplitude, computed)),
        progress_interval_s=0.0, max_workers=1,
    )
    assert reported == [(0.0, 1), (0.0, 2), (0.0, 3), (0.1, 1), (0.1, 2), (0.1, 3)]
    assert 1 in result["orders"] and 2 in result["orders"]
    fraction = result["active_fraction"]
    expected = fraction * _solve(grating, 0.4, fourier_orders=10)[0] + (1 - fraction[1]) * result["flat_reference"][1]
    assert result["curves"][0.0][0][1] == pytest.approx(expected[1], rel=1e-6)
    assert result["curves"][0.0][1].max() < 1e-12 < result["curves"][0.1][1].max()
    assert result["exit_angles_deg"][0] == pytest.approx(theta)


def test_takagi_coating_only_attenuates():
    crystal = bragg_crystal("Si", (1, 1, 1), ENERGY)
    grating = IdtGrating(al_thickness_um=0.0)
    angles = np.radians(np.linspace(-0.005, 0.005, 21))
    bare = rocking_curve(crystal, grating, 0.0, angles, n_orders=3)
    coated = rocking_curve(crystal, grating, 0.0, angles, n_orders=3, coating=(("Pt", 0.03),))
    attenuation = abs(coating_transmission((("Pt", 0.03),), crystal)) ** 4
    assert 0.0 < attenuation < 1.0
    np.testing.assert_allclose(coated.total, bare.total * attenuation, rtol=1e-10)


def test_replace_keeps_derived_fields():
    grating = replace(IdtSurfaceGrating(layers=PT), amplitude_nm=0.2)
    assert grating.amplitude_nm == 0.2 and grating.coating_stack.layers_bottom_up[0].material == "Pt"


def test_multilayer_peak_is_found_even_when_weak():
    layers = (CoatingLayer("W", 1.2), CoatingLayer(MaterialSpec("B4C", 2.52), 1.8)) * 5 + (CoatingLayer("Pt", 2.0),)
    assert multilayer_period_nm(layers) == pytest.approx(3.0)
    assert multilayer_period_nm(PT) is None
    grating = IdtSurfaceGrating(layers=layers)
    for energy in (8000.0, 12000.0):
        peak = theta_window(grating, energy)["multilayer_peak_deg"]
        angles = np.linspace(peak - 0.2, peak + 0.2, 801)
        reflectivity = parratt_reflectivity(grating.stack(), energy, angles)[0]
        assert angles[np.argmax(reflectivity)] == pytest.approx(peak, abs=2e-3)
    assert theta_window(IdtSurfaceGrating(layers=PT), ENERGY)["multilayer_peak_deg"] is None


def test_batches_without_checkpoints_skip_the_case_fingerprint(monkeypatch):
    from grax.simulation import BatchSimulationRunner

    def fail(*_args, **_kwargs):
        raise AssertionError("fingerprinted the cases without a checkpoint folder")

    monkeypatch.setattr(BatchSimulationRunner, "_checkpoint_input_metadata", fail)
    result = surface_rocking_scan(
        IdtSurfaceGrating(layers=PT), [0.0], np.array([0.4]), energy_ev=ENERGY, fourier_orders=5, max_workers=1
    )
    assert result["curves"][0.0][0][0] > 0.0
