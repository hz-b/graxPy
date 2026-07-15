from __future__ import annotations

import numpy as np
import pytest

from grax.stacks import (
    CustomStack,
    LayerSpec,
    MultilayerStack,
    SingleLayerStack,
    assemble_custom_stack,
    build_multilayer_stack,
    build_single_layer_stack,
)


def test_layer_spec_validates_roughness_sigma() -> None:
    assert LayerSpec(material="C", thickness_nm=1.0).roughness_sigma_nm is None
    assert LayerSpec(material="C", thickness_nm=1.0, roughness_sigma_nm=0.0).roughness_sigma_nm == 0.0

    with pytest.raises(ValueError, match="roughness_sigma_nm"):
        LayerSpec(material="C", thickness_nm=1.0, roughness_sigma_nm=-0.1)


def test_base_stack_defaults_to_uniform_interface_sigmas() -> None:
    stack = SingleLayerStack(substrate_material="Si", layer_material="Pt", layer_thickness_nm=10.0)

    assert stack.has_per_layer_roughness() is False
    # One layer -> two interfaces, both the default.
    assert stack.interface_roughness_sigmas_bottom_up(0.7) == [0.7, 0.7]


def test_substrate_roughness_overrides_interface_zero() -> None:
    stack = build_single_layer_stack(
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=10.0,
        substrate_roughness_sigma_nm=0.8,
        layer_roughness_sigma_nm=1.2,
    )

    assert stack.has_per_layer_roughness() is True
    # interfaces: [substrate boundary (override), top of layer]
    assert stack.interface_roughness_sigmas_bottom_up(0.3) == [0.8, 1.2]


def test_substrate_roughness_alone_marks_per_layer_roughness() -> None:
    stack = build_single_layer_stack(
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=10.0,
        substrate_roughness_sigma_nm=0.5,
    )

    assert stack.has_per_layer_roughness() is True
    # Only the substrate boundary is overridden; the layer falls back to default.
    assert stack.interface_roughness_sigmas_bottom_up(0.0) == [0.5, 0.0]


def test_single_layer_stack_per_layer_and_top_cap_sigmas() -> None:
    stack = build_single_layer_stack(
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=10.0,
        top_cap_material="C",
        top_cap_thickness_nm=2.0,
        layer_roughness_sigma_nm=1.0,
        top_cap_roughness_sigma_nm=0.0,
    )

    assert stack.has_per_layer_roughness() is True
    # interfaces: [substrate boundary, top of layer, top of cap]
    assert stack.interface_roughness_sigmas_bottom_up(0.5) == [0.5, 1.0, 0.0]


def test_custom_stack_maps_each_layer_top_interface() -> None:
    stack = assemble_custom_stack(
        substrate_material="Si",
        layers_bottom_up=[
            LayerSpec(material="Cr", thickness_nm=2.0, roughness_sigma_nm=1.0),
            LayerSpec(material="C", thickness_nm=3.0),  # defers to default
        ],
        top_cap_material="Au",
        top_cap_thickness_nm=1.0,
        top_cap_roughness_sigma_nm=2.0,
    )

    # interfaces: [substrate, top of Cr, top of C (default), top of cap]
    assert stack.interface_roughness_sigmas_bottom_up(0.3) == [0.3, 1.0, 0.3, 2.0]
    assert stack.has_per_layer_roughness() is True


def test_custom_stack_without_overrides_is_uniform() -> None:
    stack = assemble_custom_stack(
        substrate_material="Si",
        layers_bottom_up=[LayerSpec(material="Cr", thickness_nm=2.0)],
    )

    assert stack.has_per_layer_roughness() is False
    assert stack.interface_roughness_sigmas_bottom_up(0.4) == [0.4, 0.4]


def test_multilayer_stack_per_material_sigmas() -> None:
    stack = build_multilayer_stack(
        substrate_material="Si",
        material_a="Cr",
        material_b="C",
        d_period_nm=6.0,
        gamma=0.5,
        n_bilayers=2,
        top_material="C",
        material_a_roughness_sigma_nm=1.0,
        material_b_roughness_sigma_nm=2.0,
    )

    sigmas = stack.interface_roughness_sigmas_bottom_up(0.5)
    # Length matches n_layers (2 * n_bilayers) + 1 interface.
    assert len(sigmas) == len(stack.layer_sequence_bottom_up()) + 1
    # top_material is C (material_b). Bottom-up bilayer order is [Cr, C], so each
    # layer's top-interface sigma follows the layer's material.
    assert sigmas == [0.5, 1.0, 2.0, 1.0, 2.0]


def test_interface_sigma_count_matches_layers_for_all_stacks() -> None:
    stacks = [
        SingleLayerStack(substrate_material="Si", layer_material="Pt", layer_thickness_nm=10.0),
        MultilayerStack(substrate_material="Si", n_bilayers=3),
        CustomStack(
            substrate_material="Si",
            layers_bottom_up=[LayerSpec(material="Cr", thickness_nm=2.0)],
        ),
    ]
    for stack in stacks:
        sigmas = stack.interface_roughness_sigmas_bottom_up(0.0)
        assert len(sigmas) == len(stack.layer_sequence_bottom_up()) + 1
        assert np.all(np.asarray(sigmas) == 0.0)
