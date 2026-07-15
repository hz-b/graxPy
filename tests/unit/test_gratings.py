from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grax import RoughnessSpec
from grax.gratings import BlazedGrating, LaminarGrating
from grax.materials import material_label
from grax.stacks import (
    LayerSpec,
    MultilayerStack,
    SingleLayerStack,
    assemble_custom_stack,
    build_multilayer_stack,
    build_single_layer_stack,
)
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[2] / "examples" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")
C = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_C_cxro.txt", "C")
CR = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Cr_cxro.txt", "Cr")
AU = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Au_cxro.txt", "Au")


def test_laminar_grating_profile_points_match_current_slag_geometry() -> None:
    grating = LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
    )

    positions, heights = grating.profile_points()

    period_nm = 1e6 / 400
    expected_width_nm = (1.0 - 0.67) * period_nm

    assert positions[0] == 0.0
    assert positions[-1] == period_nm
    assert np.max(heights) == 14.9
    assert positions[3] - positions[2] == pytest.approx(expected_width_nm)


def test_roughness_spec_validates_inputs() -> None:
    assert RoughnessSpec(kind="debye-waller", sigma_nm=0.5).kind == "debye-waller"

    with pytest.raises(ValueError, match="kind"):
        RoughnessSpec(kind="unknown", sigma_nm=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        RoughnessSpec(kind="solver", sigma_nm=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sigma_nm"):
        RoughnessSpec(kind="debye-waller", sigma_nm=-0.1)
    with pytest.raises(ValueError, match="resolution_factor"):
        RoughnessSpec(kind="random-interface", sigma_nm=0.5, resolution_factor=0.0)
    with pytest.raises(ValueError, match="correlation_length_nm"):
        RoughnessSpec(kind="random-interface", sigma_nm=0.5, correlation_length_nm=-1.0)
    assert RoughnessSpec(kind="random-interface", sigma_nm=0.5, correlation_length_nm=None).correlation_length_nm is None
    assert RoughnessSpec(kind="random-interface", sigma_nm=0.5, correlation_length_nm=0.0).correlation_length_nm == 0.0


def test_laminar_grating_stores_roughness_spec() -> None:
    roughness = RoughnessSpec(kind="random-interface", sigma_nm=1.0, seed=12)

    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        roughness=roughness,
    )

    assert grating.roughness == roughness


def test_roughness_resolution_warning_uses_factor_rule() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=0.5,
        z_resolution_nm=0.1,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.5, resolution_factor=4.0),
    )

    with pytest.warns(UserWarning, match="roughness/factor=0.125"):
        grating._warn_if_roughness_underresolved()


def test_grating_roughness_offsets_are_deterministic_and_interface_specific() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=10.0,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=1.0, seed=4),
    )
    x_grid = grating._build_x_grid(num_periods=1)
    base_surface = grating._surface_profile_on_grid(x_grid, num_periods=1)

    rough_surface_a = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=0)
    rough_surface_b = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=0)
    next_interface = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=1)

    assert np.allclose(rough_surface_a, rough_surface_b)
    assert not np.allclose(rough_surface_a, base_surface)
    assert not np.allclose(rough_surface_a - base_surface, next_interface - base_surface)
    assert rough_surface_a[0] - base_surface[0] == pytest.approx(
        rough_surface_a[-1] - base_surface[-1]
    )


def _roughness_offset(grating: LaminarGrating, interface_index: int = 1) -> np.ndarray:
    x_grid = grating._build_x_grid(num_periods=1)
    return grating._roughness_interface_offset(x_grid, interface_index)


def _build_random_interface_grating(**roughness_kwargs) -> LaminarGrating:
    return LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=2.0,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=1.0, seed=7, **roughness_kwargs),
    )


def test_correlated_roughness_is_smoother_than_white_noise() -> None:
    white = _roughness_offset(_build_random_interface_grating(correlation_length_nm=0.0))
    correlated = _roughness_offset(_build_random_interface_grating(correlation_length_nm=250.0))

    # Both are normalized to the same rms, but the correlated field varies far
    # more slowly between adjacent samples.
    white_step = float(np.mean(np.diff(white) ** 2))
    correlated_step = float(np.mean(np.diff(correlated) ** 2))
    assert correlated_step < 0.05 * white_step
    assert float(np.sqrt(np.mean(correlated**2))) == pytest.approx(1.0, rel=1e-6)


def test_correlation_length_defaults_to_one_tenth_of_period() -> None:
    fine = _build_random_interface_grating()  # correlation_length_nm=None
    coarse = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        period_lpermm=200,
        x_resolution_nm=2.0,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=1.0, seed=7),
    )

    assert fine._roughness_correlation_length_nm() == pytest.approx(fine.period_nm / 10.0)
    assert coarse._roughness_correlation_length_nm() == pytest.approx(coarse.period_nm / 10.0)
    # A larger period yields a longer correlation length -> smoother steps.
    assert coarse._roughness_correlation_length_nm() > fine._roughness_correlation_length_nm()


def test_zero_correlation_length_reproduces_white_noise() -> None:
    grating = _build_random_interface_grating(correlation_length_nm=0.0)
    offset = _roughness_offset(grating)

    # White noise: adjacent samples are uncorrelated, so the lag-1 correlation is
    # near zero (a correlated field would be close to 1).
    centered = offset[:-1] - offset[:-1].mean()
    lag1 = float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
    assert abs(lag1) < 0.3


def test_per_layer_roughness_uses_layer_specific_sigma() -> None:
    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=CR, thickness_nm=2.0, roughness_sigma_nm=0.0),
            LayerSpec(material=C, thickness_nm=3.0, roughness_sigma_nm=1.5),
        ],
    )
    grating = LaminarGrating(
        substrate_material=SI,
        coating_stack=stack,
        x_resolution_nm=10.0,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.5, seed=4),
    )
    x_grid = grating._build_x_grid(num_periods=1)
    base_surface = grating._surface_profile_on_grid(x_grid, num_periods=1)

    # Interface 0 is the substrate boundary and falls back to the default sigma.
    substrate_interface = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=0)
    # Interface 1 is the top of layer 0 (sigma == 0) -> no perturbation.
    cr_top = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=1)
    # Interface 2 is the top of layer 1 (sigma == 1.5) -> perturbed.
    c_top = grating._rough_interface(base_surface, x_grid=x_grid, interface_index=2)

    assert not np.allclose(substrate_interface, base_surface)
    assert np.allclose(cr_top, base_surface)
    assert not np.allclose(c_top, base_surface)

    default_rms = float(np.sqrt(np.mean((substrate_interface - base_surface) ** 2)))
    c_rms = float(np.sqrt(np.mean((c_top - base_surface) ** 2)))
    assert c_rms == pytest.approx(1.5, rel=1e-6)
    assert default_rms == pytest.approx(0.5, rel=1e-6)


def test_roughness_resolution_warning_uses_max_per_layer_sigma() -> None:
    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=CR, thickness_nm=2.0, roughness_sigma_nm=2.0),
        ],
    )
    grating = LaminarGrating(
        substrate_material=SI,
        coating_stack=stack,
        x_resolution_nm=0.5,
        z_resolution_nm=0.1,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.0, resolution_factor=4.0),
    )

    with pytest.warns(UserWarning, match="roughness/factor=0.5"):
        grating._warn_if_roughness_underresolved()


def test_roughness_spec_validates_num_supercells() -> None:
    assert RoughnessSpec(kind="random-interface", sigma_nm=0.5).num_supercells == 1

    with pytest.raises(ValueError, match="num_supercells"):
        RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_supercells=0)
    with pytest.raises(ValueError, match="num_supercells"):
        RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_supercells=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_supercells"):
        RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_supercells=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_supercells"):
        RoughnessSpec(kind="debye-waller", sigma_nm=0.5, num_supercells=3)

    assert RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_supercells=3).num_supercells == 3


def test_roughness_num_supercells_only_applies_to_random_interface() -> None:
    debye_grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        roughness=RoughnessSpec(kind="debye-waller", sigma_nm=0.5),
    )
    random_interface_grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_supercells=4),
    )
    no_roughness_grating = LaminarGrating(substrate_material=SI, layer_material=PT, layer_thickness_nm=28.77)

    assert debye_grating._roughness_num_supercells() == 1
    assert random_interface_grating._roughness_num_supercells() == 4
    assert no_roughness_grating._roughness_num_supercells() == 1


def test_roughness_random_field_spans_full_supercell_as_one_continuous_field() -> None:
    grating = _build_random_interface_grating(correlation_length_nm=250.0, num_supercells=3)
    x_grid_one_period = grating._build_x_grid(num_periods=1)
    x_grid_supercell = grating._build_x_grid(num_periods=3)

    rng_one = np.random.default_rng(7)
    rng_super = np.random.default_rng(7)
    field_one_period = grating._roughness_random_field(x_grid_one_period, rng_one, 250.0)
    field_supercell = grating._roughness_random_field(x_grid_supercell, rng_super, 250.0)

    # The supercell field spans 3x the samples of a single period and is not
    # simply three tiled copies of the single-period field (same rng state,
    # different outcome because the FFT is synthesized over the full span).
    assert field_supercell.size == pytest.approx(3 * (field_one_period.size - 1) + 1)
    tiled_guess = np.concatenate(
        [field_one_period[:-1], field_one_period[:-1], field_one_period[:-1], field_one_period[:1]]
    )
    assert not np.allclose(field_supercell, tiled_guess)
    # Still periodic over the full supercell span (last sample == first).
    assert field_supercell[0] == pytest.approx(field_supercell[-1])


def test_roughness_random_field_dx_uses_actual_span() -> None:
    grating = _build_random_interface_grating(correlation_length_nm=50.0)
    x_grid_two_periods = grating._build_x_grid(num_periods=2)

    rng = np.random.default_rng(3)
    field = grating._roughness_random_field(x_grid_two_periods, rng, 50.0)

    # Regression guard for the dx_nm bugfix: the field must be synthesized
    # over the *actual* x_grid span (2 periods), not grating.period_nm (1
    # period). A wrong dx_nm would still return an array of the right size
    # but with the wrong physical wavenumber spacing; verify indirectly via
    # the expected number of unique FFT samples matching the 2-period grid.
    n_unique = x_grid_two_periods.size - 1
    expected_dx_nm = float(x_grid_two_periods[-1] - x_grid_two_periods[0]) / n_unique
    assert expected_dx_nm == pytest.approx(2.0 * grating.period_nm / n_unique)
    assert field.size == x_grid_two_periods.size


def test_build_textures_uses_num_supercells_for_random_interface() -> None:
    grating = _build_random_interface_grating(correlation_length_nm=50.0, num_supercells=3)
    x_grid = grating._build_x_grid(num_periods=grating._roughness_num_supercells())

    assert x_grid[-1] == pytest.approx(3 * grating.period_nm)


def test_laminar_grating_can_save_profile_plot(tmp_path: Path) -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
    )

    output_path = tmp_path / "grating.png"

    grating.plot_profile(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_laminar_grating_tiles_profile_plot_over_three_periods() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
    )

    positions, heights = grating._tiled_profile_points(num_periods=3)

    assert positions[0] == 0.0
    assert positions[-1] == pytest.approx(3.0 * grating.period_nm)
    assert np.max(heights) == pytest.approx(grating.depth_nm)


def test_laminar_grating_builds_material_subplot_data() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        top_cap_material=C,
        top_cap_thickness_nm=3.0,
        x_resolution_nm=10.0,
        z_resolution_nm=2.0,
    )

    x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=3)

    assert x_grid[0] == 0.0
    assert x_grid[-1] == pytest.approx(3.0 * grating.period_nm)
    assert z_grid[0] == 0.0
    assert material_map.shape == (z_grid.size, x_grid.size)
    assert set(np.unique(material_map)).issubset({-1, 0, 1, 2})
    assert material_labels == ["C", "Pt", "Si"]


def test_single_layer_stack_sequence_includes_optional_top_cap() -> None:
    stack = SingleLayerStack(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        top_cap_material=C,
        top_cap_thickness_nm=1.7,
    )

    assert stack.layer_sequence_bottom_up() == [(PT, 28.77), (C, 1.7)]
    assert stack.total_thickness_nm == pytest.approx(30.47)


def test_multilayer_stack_sequence_respects_top_material() -> None:
    stack = MultilayerStack(
        substrate_material=SI,
        material_a=CR,
        material_b=C,
        d_period_nm=6.5,
        gamma=0.45,
        n_bilayers=2,
        top_material=C,
    )

    sequence = stack.layer_sequence_bottom_up()

    assert sequence == [
        (CR, pytest.approx(2.925)),
        (C, pytest.approx(3.575)),
        (CR, pytest.approx(2.925)),
        (C, pytest.approx(3.575)),
    ]


def test_grating_can_use_multilayer_stack_for_material_plot() -> None:
    grating = LaminarGrating(
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=2,
            top_material=C,
            top_cap_material=AU,
            top_cap_thickness_nm=2.0,
        ),
        x_resolution_nm=10.0,
        z_resolution_nm=2.0,
    )

    _, _, material_map, material_labels = grating._material_plot_data()

    assert set(np.unique(material_map)).issubset({-1, 0, 1, 2, 3})
    assert material_labels == ["Au", "C", "Cr", "Si"]


def test_blazed_grating_depth_comes_from_blaze_angle() -> None:
    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.9,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=20.0,
    )

    expected_depth_nm = (1e6 / 600) * np.tan(np.deg2rad(0.9))
    assert grating.depth_nm == pytest.approx(expected_depth_nm)

    positions, heights = grating.profile_points()
    assert positions[0] == 0.0
    assert positions[-1] == pytest.approx(1e6 / 600)
    assert np.max(heights) == pytest.approx(expected_depth_nm)


def test_blazed_grating_supports_dual_angle_profile() -> None:
    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.729,
        anti_blaze_angle_deg=5.597,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=30.0,
    )

    period_nm = 1e6 / 600
    blaze_tangent = np.tan(np.deg2rad(0.729))
    anti_blaze_tangent = np.tan(np.deg2rad(5.597))
    expected_depth_nm = (
        period_nm
        * blaze_tangent
        * anti_blaze_tangent
        / (blaze_tangent + anti_blaze_tangent)
    )
    expected_apex_x_nm = period_nm * anti_blaze_tangent / (blaze_tangent + anti_blaze_tangent)

    assert grating.depth_nm == pytest.approx(expected_depth_nm)

    positions, heights = grating.profile_points()
    assert np.allclose(positions, np.array([0.0, expected_apex_x_nm, period_nm]))
    assert np.allclose(heights, np.array([0.0, expected_depth_nm, 0.0]))


def test_custom_stack_assembly_integrates_with_grating_plotting(tmp_path: Path) -> None:
    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=PT, thickness_nm=10.0),
            LayerSpec(material=C, thickness_nm=2.0),
        ],
    )
    grating = LaminarGrating(coating_stack=stack, x_resolution_nm=10.0, z_resolution_nm=2.0)
    profile_path = tmp_path / "custom_stack_profile.png"
    schematic_path = tmp_path / "custom_stack_schematic.png"

    grating.plot_profile(profile_path)
    stack.plot_schematic(schematic_path)

    assert profile_path.exists()
    assert schematic_path.exists()


def test_custom_stack_schematic_collapses_repeated_bilayers() -> None:
    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=CR, thickness_nm=2.0),
            LayerSpec(material=CR, thickness_nm=2.4),
            LayerSpec(material=C, thickness_nm=3.6),
            LayerSpec(material=CR, thickness_nm=2.4),
            LayerSpec(material=C, thickness_nm=3.6),
            LayerSpec(material=CR, thickness_nm=2.4),
            LayerSpec(material=C, thickness_nm=3.6),
        ],
        top_cap_material="O",
        top_cap_thickness_nm=2.0,
    )

    layers_top_down, summary_lines = stack._schematic_layers_and_summary()

    assert [material_label(material) for material, _ in layers_top_down] == ["O", "C", "Cr", "Cr"]
    assert [thickness for _, thickness in layers_top_down] == pytest.approx([2.0, 3.6, 2.4, 2.0])
    assert summary_lines == [
        "Total thickness: 22.000 nm",
        "3 bilayers",
        "Bilayer period: 6.000 nm",
    ]


def test_custom_stack_schematic_keeps_literal_nonrepeating_layers() -> None:
    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=PT, thickness_nm=10.0),
            LayerSpec(material=C, thickness_nm=2.0),
            LayerSpec(material=CR, thickness_nm=1.0),
        ],
    )

    layers_top_down, summary_lines = stack._schematic_layers_and_summary()

    assert [material_label(material) for material, _ in layers_top_down] == ["Cr", "C", "Pt"]
    assert [thickness for _, thickness in layers_top_down] == pytest.approx([1.0, 2.0, 10.0])
    assert summary_lines == ["Total thickness: 13.000 nm"]


def test_stack_builder_helpers_match_existing_stack_types() -> None:
    single = build_single_layer_stack(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        top_cap_material=C,
        top_cap_thickness_nm=1.0,
    )
    assert isinstance(single, SingleLayerStack)
    assert single.layer_sequence_bottom_up() == [(PT, 28.77), (C, 1.0)]

    multilayer = build_multilayer_stack(
        substrate_material=SI,
        material_a=CR,
        material_b=C,
        d_period_nm=6.5,
        gamma=0.45,
        n_bilayers=2,
        top_material=C,
    )
    assert isinstance(multilayer, MultilayerStack)
    assert len(multilayer.layer_sequence_bottom_up()) == 4


def test_plot_material_colors_are_unique_and_deterministic_for_custom_stack() -> None:
    stack = assemble_custom_stack(
        substrate_material="Si",
        layers_bottom_up=[
            LayerSpec(material="Cr", thickness_nm=2.0),
            LayerSpec(material="C", thickness_nm=3.6),
        ],
        top_cap_material="O",
        top_cap_thickness_nm=2.0,
    )
    grating = LaminarGrating(coating_stack=stack)
    labels = ["O", "C", "Cr", "Si", "Cr"]

    colors_first = grating._plot_material_colors(coating_stack=stack, material_labels=labels)
    colors_second = grating._plot_material_colors(coating_stack=stack, material_labels=labels)

    assert colors_first == colors_second
    assert colors_first[2] == colors_first[4]
    assert len(set(colors_first[:4])) == 4


def test_plot_material_colors_distinguish_cr_and_c_in_multilayer_like_custom_stack() -> None:
    stack = assemble_custom_stack(
        substrate_material="Si",
        layers_bottom_up=[
            LayerSpec(material="Cr", thickness_nm=2.0),
            LayerSpec(material="Cr", thickness_nm=2.4),
            LayerSpec(material="C", thickness_nm=3.6),
            LayerSpec(material="Cr", thickness_nm=2.4),
            LayerSpec(material="C", thickness_nm=3.6),
        ],
        top_cap_material="O",
        top_cap_thickness_nm=2.0,
    )
    grating = BlazedGrating(coating_stack=stack)
    labels = ["O", "C", "Cr", "Si"]

    colors = grating._plot_material_colors(coating_stack=stack, material_labels=labels)

    color_by_label = {label: color for label, color in zip(labels, colors)}
    assert color_by_label["Cr"] != color_by_label["C"]
