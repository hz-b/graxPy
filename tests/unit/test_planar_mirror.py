"""Unit tests for :mod:`grax.planar_mirror`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from grax import (
    CustomStack,
    LateralGrading,
    LayerSpec,
    SingleLayerStack,
    footprint_reflectivity,
    parratt_reflectivity,
)
from grax.materials import resolve_refractive_index
from grax.planar_mirror import HC_EV_NM


def _bragg_stack(n_pairs: int = 10, d_nm: float = 7.0, gamma: float = 0.4) -> CustomStack:
    layers: list[LayerSpec] = []
    for _ in range(n_pairs):
        layers += [LayerSpec("Ru", gamma * d_nm), LayerSpec("C", (1 - gamma) * d_nm)]
    return CustomStack(substrate_material="Si", layers_bottom_up=layers)


def test_bare_substrate_matches_closed_form_fresnel() -> None:
    stack = SingleLayerStack(substrate_material="Si", layer_material="Si", layer_thickness_nm=1e-9)
    energy, angles = 200.0, np.array([2.0, 5.0, 12.0])
    n = resolve_refractive_index("Si", energy)
    kz0 = np.sin(np.deg2rad(angles))
    kz1 = np.sqrt(n**2 - np.cos(np.deg2rad(angles)) ** 2)
    expected = np.abs((kz0 - kz1) / (kz0 + kz1)) ** 2

    result = parratt_reflectivity(stack, energy, angles)

    assert result.shape == (1, 3)
    np.testing.assert_allclose(result[0], expected, rtol=1e-9)


def test_single_layer_matches_airy_formula() -> None:
    stack = SingleLayerStack(substrate_material="Si", layer_material="Pt", layer_thickness_nm=15.0)
    energy, angle = 150.0, np.array([4.0, 8.0])
    n1, n2 = resolve_refractive_index("Pt", energy), resolve_refractive_index("Si", energy)
    c2 = np.cos(np.deg2rad(angle)) ** 2
    kz0 = np.sin(np.deg2rad(angle))
    kz1, kz2 = np.sqrt(n1**2 - c2), np.sqrt(n2**2 - c2)
    r01 = (kz0 - kz1) / (kz0 + kz1)
    r12 = (kz1 - kz2) / (kz1 + kz2)
    phase = np.exp(2j * (2 * np.pi * energy / HC_EV_NM) * kz1 * 15.0)
    expected = np.abs((r01 + r12 * phase) / (1 + r01 * r12 * phase)) ** 2

    np.testing.assert_allclose(parratt_reflectivity(stack, energy, angle)[0], expected, rtol=1e-9)


def test_matches_xrt_multilayer_within_xrt_first_order_approximation() -> None:
    from xrt.backends.raycing import materials as xrt_materials

    ru = xrt_materials.Material("Ru", rho=12.45)
    b4c = xrt_materials.Material(["B", "C"], quantities=[4, 1], rho=2.52)
    si = xrt_materials.Material("Si", rho=2.33)
    # xrt thicknesses are in angstrom.
    xrt_ml = xrt_materials.Multilayer(
        tLayer=b4c, tThickness=40, bLayer=ru, bThickness=30, nPairs=5, substrate=si
    )
    layers: list[LayerSpec] = []
    for _ in range(5):
        layers += [LayerSpec(ru, 3.0), LayerSpec(b4c, 4.0)]
    stack = CustomStack(substrate_material=si, layers_bottom_up=layers)
    energies, angles = np.array([150.0, 200.0, 250.0]), np.array([3.0, 10.0, 20.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for polarization, index in (("s", 0), ("p", 1)):
            reference = np.array(
                [
                    [
                        abs(xrt_ml.get_amplitude(e, np.sin(np.deg2rad(a)))[index]) ** 2
                        for a in angles
                    ]
                    for e in energies
                ]
            )
            result = parratt_reflectivity(stack, energies, angles, polarization=polarization)
            # xrt linearises n**2 - 1 ~ 2(n - 1); grax keeps it exact.
            np.testing.assert_allclose(result, reference, atol=5e-3)


def test_polarizations_agree_at_normal_incidence_and_average_is_mean() -> None:
    stack = _bragg_stack(n_pairs=3)
    normal = parratt_reflectivity(stack, 100.0, 90.0, polarization="s")
    np.testing.assert_allclose(normal, parratt_reflectivity(stack, 100.0, 90.0, polarization="p"))

    s = parratt_reflectivity(stack, 100.0, 15.0, polarization="s")
    p = parratt_reflectivity(stack, 100.0, 15.0, polarization="TM")
    np.testing.assert_allclose(parratt_reflectivity(stack, 100.0, 15.0, polarization="average"), 0.5 * (s + p))


def test_reflectivity_is_bounded_and_roughness_reduces_it() -> None:
    stack = _bragg_stack()
    energies = np.linspace(150.0, 250.0, 40)
    smooth = parratt_reflectivity(stack, energies, 20.0)
    rough = parratt_reflectivity(stack, energies, 20.0, roughness_sigma_nm=0.5)

    assert np.all((smooth >= 0.0) & (smooth <= 1.0))
    assert rough.max() < smooth.max()


def test_per_layer_roughness_overrides_default() -> None:
    layers = [LayerSpec("Ru", 3.0, roughness_sigma_nm=0.0), LayerSpec("C", 4.0, roughness_sigma_nm=0.0)] * 5
    explicit = CustomStack(substrate_material="Si", layers_bottom_up=layers, substrate_roughness_sigma_nm=0.0)
    smooth = parratt_reflectivity(explicit, 200.0, 15.0)
    # Layer overrides of 0 win over a large default sigma.
    np.testing.assert_allclose(parratt_reflectivity(explicit, 200.0, 15.0, roughness_sigma_nm=2.0), smooth)


def test_thickness_scale_one_equals_ungraded_and_adds_axis() -> None:
    stack = _bragg_stack(n_pairs=4)
    base = parratt_reflectivity(stack, [150.0, 200.0], [10.0, 20.0])
    scaled = parratt_reflectivity(stack, [150.0, 200.0], [10.0, 20.0], thickness_scale=[1.0, 1.1])

    assert scaled.shape == (2, 2, 2)
    np.testing.assert_allclose(scaled[:, :, 0], base, atol=1e-12)


def test_grading_broadens_and_lowers_the_bragg_peak() -> None:
    stack = _bragg_stack(n_pairs=20)
    energies = np.linspace(100.0, 300.0, 600)
    angle = 25.0
    flat = parratt_reflectivity(stack, energies, angle)[:, 0]
    graded = footprint_reflectivity(
        stack,
        energies,
        angle,
        grading=LateralGrading.linear(0.01),
        positions_mm=np.linspace(-10.0, 10.0, 41),
    )[:, 0]

    assert graded.max() < flat.max()
    assert np.sum(graded > 0.5 * graded.max()) > np.sum(flat > 0.5 * flat.max())


def test_trivial_grading_equals_ungraded_with_weights() -> None:
    stack = _bragg_stack(n_pairs=4)
    ungraded = parratt_reflectivity(stack, 180.0, 15.0)
    averaged = footprint_reflectivity(
        stack, 180.0, 15.0, grading=LateralGrading(), positions_mm=[-1.0, 0.0, 3.0], weights=[1, 2, 3]
    )
    np.testing.assert_allclose(averaged, ungraded, atol=1e-12)


def test_lateral_grading_table_and_callable() -> None:
    table = LateralGrading(table_positions_mm=(-10.0, 10.0), table_scale=(0.9, 1.1))
    np.testing.assert_allclose(table.scale([-20.0, 0.0, 10.0]), [0.9, 1.0, 1.1])
    stack = _bragg_stack(n_pairs=3)
    a = footprint_reflectivity(stack, 150.0, 20.0, grading=lambda s: 1 + 0.01 * s, positions_mm=[0.0, 5.0])
    b = footprint_reflectivity(stack, 150.0, 20.0, grading=LateralGrading.linear(0.01), positions_mm=[0.0, 5.0])
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energies_ev": [-1.0]}, "energies_ev"),
        ({"grazing_angles_deg": [0.0]}, "grazing_angles_deg"),
        ({"grazing_angles_deg": [95.0]}, "grazing_angles_deg"),
        ({"roughness_sigma_nm": -0.1}, "roughness_sigma_nm"),
        ({"thickness_scale": [0.0]}, "thickness_scale"),
        ({"polarization": "circular"}, "polarization"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict, message: str) -> None:
    params = {"energies_ev": [150.0], "grazing_angles_deg": [10.0]}
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        parratt_reflectivity(_bragg_stack(n_pairs=1), **params)


def test_invalid_grading_inputs_raise() -> None:
    with pytest.raises(ValueError, match="together"):
        LateralGrading(table_positions_mm=(0.0, 1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        LateralGrading(table_positions_mm=(1.0, 0.0), table_scale=(1.0, 1.0))
    with pytest.raises(ValueError, match="weights"):
        footprint_reflectivity(
            _bragg_stack(n_pairs=1), 150.0, 10.0, grading=LateralGrading(), positions_mm=[0.0, 1.0], weights=[1.0]
        )
