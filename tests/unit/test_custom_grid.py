"""Focused tests for custom parameter-grid validation and expansion."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grax.web import app as web_app_module
from grax.web.custom_grid import (
    LARGE_GRID_WARNING,
    common_parameter_catalog,
    expand_custom_grid,
    parameter_catalog,
    write_custom_grid_all_orders,
    write_custom_grid_summary,
)
from grax.web.persistence import build_grating_from_spec


def _laminar_spec(**updates):
    spec = {
        "name": "Test laminar",
        "grating_type": "laminar",
        "period_lpermm": 400,
        "x_resolution_nm": 1.0,
        "z_resolution_nm": 0.5,
        "width_to_period_ratio": 0.67,
        "depth_nm": 14.9,
        "left_wall_angle_deg": 15.0,
        "right_wall_angle_deg": 15.0,
        "roughness": None,
        "stack": {"type": "bare", "substrate_material": "Si"},
    }
    spec.update(updates)
    return spec


def _config(spec, *, independent=None, coupled=None, variants=None):
    return {
        "schema_version": 1,
        "variants": variants or [{"id": "base", "name": "Base", "spec": spec}],
        "independent": independent or [],
        "coupled_groups": coupled or [],
    }


def test_catalog_contains_conditions_numerics_geometry_and_coating() -> None:
    spec = _laminar_spec(
        stack={
            "type": "bare",
            "substrate_material": {"name": "Si", "density_g_cm3": 2.329},
        }
    )
    ids = {item.id for item in parameter_catalog(spec)}

    assert {"energy_ev", "grazing_angle_deg", "fourier_orders", "solver"} <= ids
    assert {"period_lpermm", "x_resolution_nm", "z_resolution_nm"} <= ids
    assert {"width_to_period_ratio", "depth_nm", "left_wall_angle_deg"} <= ids
    assert "stack.substrate_material" in ids
    assert "stack.substrate_material.density_g_cm3" in ids


def test_integer_range_is_rounded_deduplicated_and_ordered() -> None:
    spec = _laminar_spec()
    normalized, cases = expand_custom_grid(
        _config(
            spec,
            independent=[
                {
                    "parameter": "fourier_orders",
                    "mode": "range",
                    "minimum": 5,
                    "maximum": 10,
                    "count": 31,
                }
            ],
        ),
        spec,
    )

    assert normalized["independent"][0]["values"] == [5, 6, 7, 8, 9, 10]
    assert normalized["independent"][0] == {
        "parameter": "fourier_orders",
        "mode": "range",
        "values": [5, 6, 7, 8, 9, 10],
        "minimum": 5.0,
        "maximum": 10.0,
        "count": 31,
    }
    assert [case["fourier_orders"] for case in cases] == [5, 6, 7, 8, 9, 10]
    assert cases[0]["custom_resolved_parameters"]["fourier_orders"] == 5
    assert cases[0]["custom_resolved_parameters"]["depth_nm"] == pytest.approx(14.9)


def test_explicit_values_override_inactive_ranges_when_building_cases() -> None:
    """Use explicit energies despite stale axis bounds and legacy sweep inputs."""
    spec = _laminar_spec()
    config = _config(
        spec,
        independent=[
            {
                "parameter": "energy_ev",
                "mode": "list",
                "values": [137, 251, 409],
                "minimum": 100,
                "maximum": 120,
                "count": 2,
            }
        ],
    )
    # The builder uses the saved grating and no longer submits structural variants.
    config.pop("variants")
    config, _ = web_app_module._custom_grid_from_form(
        {"custom_grid_config": json.dumps(config)}, spec
    )
    cases = web_app_module._cases_for_workflow(
        simulation=SimpleNamespace(),
        workflow="custom_grid",
        grating=build_grating_from_spec(spec),
        energies=np.asarray([100.0, 110.0, 120.0]),
        form={"custom_grid_config": json.dumps(config)},
        diffraction_order=-1,
        fourier_orders=5,
    )

    assert [case["energy_ev"] for case in cases] == [137, 251, 409]
    assert all(case["grating"].depth_nm == spec["depth_nm"] for case in cases)


def test_coupled_rows_stay_zipped_and_axes_form_cartesian_product() -> None:
    spec = _laminar_spec()
    _, cases = expand_custom_grid(
        _config(
            spec,
            independent=[{"parameter": "depth_nm", "mode": "list", "values": [10, 20]}],
            coupled=[
                {
                    "name": "conditions",
                    "parameters": ["energy_ev", "grazing_angle_deg"],
                    "rows": [
                        {"energy_ev": 100, "grazing_angle_deg": 1},
                        {"energy_ev": 200, "grazing_angle_deg": 2},
                    ],
                }
            ],
        ),
        spec,
    )

    assert len(cases) == 4
    assert [(case["energy_ev"], case["grazing_angle_deg"]) for case in cases] == [
        (100, 1),
        (200, 2),
        (100, 1),
        (200, 2),
    ]
    assert [case["grating"].depth_nm for case in cases] == [10, 10, 20, 20]


def test_multiple_dependents_share_source_rows_without_deduplicating_values() -> None:
    """Keep generated integer repeats aligned with the other coupled values."""
    spec = _laminar_spec()
    config = {
        "schema_version": 1,
        "independent": [],
        "coupled_groups": [
            {
                "parameters": ["energy_ev", "grazing_angle_deg", "fourier_orders"],
                "rows": [
                    {"energy_ev": 100, "grazing_angle_deg": 10, "fourier_orders": 5},
                    {"energy_ev": 650, "grazing_angle_deg": 40, "fourier_orders": 6},
                    {"energy_ev": 1200, "grazing_angle_deg": 70, "fourier_orders": 6},
                ],
            }
        ],
    }
    _, cases = web_app_module._custom_grid_from_form(
        {"custom_grid_config": json.dumps(config)}, spec
    )

    assert [
        (case["energy_ev"], case["grazing_angle_deg"], case["fourier_orders"])
        for case in cases
    ] == [(100, 10, 5), (650, 40, 6), (1200, 70, 6)]



def test_duplicate_parameter_and_malformed_coupled_row_are_rejected() -> None:
    spec = _laminar_spec()
    with pytest.raises(ValueError, match="only appear once"):
        expand_custom_grid(
            _config(
                spec,
                independent=[{"parameter": "energy_ev", "mode": "list", "values": [100]}],
                coupled=[
                    {
                        "parameters": ["energy_ev", "grazing_angle_deg"],
                        "rows": [{"energy_ev": 100, "grazing_angle_deg": 1}],
                    }
                ],
            ),
            spec,
        )

    with pytest.raises(ValueError, match="one value for every parameter"):
        expand_custom_grid(
            _config(
                spec,
                coupled=[
                    {
                        "parameters": ["energy_ev", "grazing_angle_deg"],
                        "rows": [{"energy_ev": 100}],
                    }
                ],
            ),
            spec,
        )


def test_variant_axis_rebuilds_each_complete_structure() -> None:
    laminar = _laminar_spec()
    sinusoidal = {
        key: value
        for key, value in laminar.items()
        if key not in {"width_to_period_ratio", "left_wall_angle_deg", "right_wall_angle_deg"}
    }
    sinusoidal.update({"name": "Sine", "grating_type": "sinusoidal", "depth_nm": 8})
    variants = [
        {"id": "laminar", "name": "Laminar", "spec": laminar},
        {"id": "sine", "name": "Sine", "spec": sinusoidal},
    ]
    common_ids = {item.id for item in common_parameter_catalog([laminar, sinusoidal])}
    assert "depth_nm" in common_ids
    assert "width_to_period_ratio" not in common_ids

    _, cases = expand_custom_grid(
        _config(
            laminar,
            variants=variants,
            independent=[{"parameter": "variant", "mode": "list", "values": ["laminar", "sine"]}],
        ),
        laminar,
    )
    assert [type(case["grating"]).__name__ for case in cases] == [
        "LaminarGrating",
        "SinusoidalGrating",
    ]
    assert [case["custom_variant_name"] for case in cases] == ["Laminar", "Sine"]


def test_large_grid_has_no_hard_limit() -> None:
    spec = _laminar_spec()
    values = list(range(LARGE_GRID_WARNING + 1))
    _, cases = expand_custom_grid(
        _config(
            spec, independent=[{"parameter": "diffraction_order", "mode": "list", "values": values}]
        ),
        spec,
    )
    assert len(cases) == LARGE_GRID_WARNING + 1


def test_roughness_parameters_can_be_added_to_a_smooth_base_variant() -> None:
    spec = _laminar_spec(roughness=None)
    _, cases = expand_custom_grid(
        _config(
            spec,
            coupled=[
                {
                    "name": "roughness",
                    "parameters": ["roughness.kind", "roughness.sigma_nm"],
                    "rows": [
                        {"roughness.kind": "debye-waller", "roughness.sigma_nm": 0.2},
                        {"roughness.kind": "random-interface", "roughness.sigma_nm": 0.4},
                    ],
                }
            ],
        ),
        spec,
    )

    assert [case["grating"].roughness.kind for case in cases] == [
        "debye-waller",
        "random-interface",
    ]
    assert [case["grating"].roughness.sigma_nm for case in cases] == [0.2, 0.4]


def test_dynamic_result_columns_are_written_to_both_outputs(tmp_path) -> None:
    result = SimpleNamespace(
        case_id="custom-grid-00000000",
        energy_ev=100.0,
        grazing_angle_deg=1.0,
        selected_efficiency=0.4,
        selected_diffraction_angle_deg=2.0,
        status="ok",
        error_message=None,
        orders=np.asarray([-1, 0, 1]),
        efficiency_all=np.asarray([0.1, 0.2, 0.4]),
        diffraction_angle_all=np.asarray([-2.0, 0.0, 2.0]),
        case_data={
            "custom_parameter_order": ["energy_ev", "grazing_angle_deg"],
            "custom_parameters": {"energy_ev": 100.0, "grazing_angle_deg": 1.0},
        },
    )
    summary = tmp_path / "summary.csv"
    all_orders = tmp_path / "all_orders.csv"
    write_custom_grid_summary([result], summary)
    write_custom_grid_all_orders([result], all_orders)

    with summary.open(newline="") as handle:
        assert next(csv.reader(handle))[:3] == ["case_id", "energy_ev", "grazing_angle_deg"]
    with all_orders.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:3] == ["case_id", "energy_ev", "grazing_angle_deg"]
    assert len(rows) == 4


def test_afm_variant_profile_is_snapshotted_into_run_directory(tmp_path) -> None:
    source = tmp_path / "source-profile.csv"
    source.write_text("x_nm,z_nm\n0,0\n1,1\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    config = {
        "variants": [
            {
                "id": "afm one",
                "name": "AFM one",
                "spec": {
                    "grating_type": "afm",
                    "profile": {"profile_path": str(source)},
                },
            }
        ]
    }

    web_app_module._snapshot_custom_grid_afm_profiles(config, run_dir)

    snapshot = run_dir / "custom_grid_profiles" / "afm_one.csv"
    assert snapshot.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert config["variants"][0]["spec"]["profile"]["profile_path"] == str(snapshot.resolve())


def test_web_workflow_reconstructs_custom_cases_with_per_case_settings() -> None:
    spec = _laminar_spec()
    config = _config(
        spec,
        independent=[{"parameter": "depth_nm", "mode": "list", "values": [10, 20]}],
        coupled=[
            {
                "name": "conditions",
                "parameters": ["energy_ev", "solver"],
                "rows": [
                    {"energy_ev": 100, "solver": "rcwa"},
                    {"energy_ev": 200, "solver": "neviere"},
                ],
            }
        ],
    )
    form = {
        "custom_grid_config": json.dumps(config),
        "custom_default_grazing_angle_deg": "1.25",
        "custom_default_diffraction_order": "-1",
        "custom_default_fourier_orders": "6",
        "custom_default_polarization": "p",
        "custom_default_solver": "rcwa",
    }

    cases = web_app_module._cases_for_workflow(
        simulation=SimpleNamespace(),
        workflow="custom_grid",
        grating=build_grating_from_spec(spec),
        energies=np.asarray([100.0, 200.0]),
        form=form,
        diffraction_order=-1,
        fourier_orders=6,
    )

    assert len(cases) == 4
    assert [case["solver"] for case in cases] == ["rcwa", "neviere", "rcwa", "neviere"]
    assert [case["grating"].depth_nm for case in cases] == [10, 10, 20, 20]
    assert all(case["grazing_angle_deg"] == pytest.approx(1.25) for case in cases)
    assert all(case["diffraction_order"] == -1 for case in cases)
    assert all(case["fourier_orders"] == 6 for case in cases)
    assert all(case["polarization"] == "p" for case in cases)


def test_yaml_metadata_payload_contains_complete_custom_builder_configuration(tmp_path) -> None:
    spec = _laminar_spec()
    config = _config(
        spec,
        independent=[{"parameter": "energy_ev", "mode": "list", "values": [100, 200]}],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "all_orders.csv").write_text(
        "case_id,energy_ev,order,efficiency\ncase-1,100,1,0.2\n",
        encoding="utf-8",
    )
    manifest = {
        "id": "custom-run",
        "workflow": "custom_grid",
        "grating_id": "grating-1",
        "grating_name": "Test laminar",
        "grating_spec": spec,
        "run_input": {"custom_grid_config": config},
        "custom_grid": config,
        "cases": [],
        "total_points": 2,
    }

    payload = web_app_module._run_metadata_payload(manifest, run_dir)

    assert payload["custom_grid"] == config
    assert payload["custom_grid"]["variants"][0]["spec"]["stack"]["type"] == "bare"


@pytest.mark.parametrize("material", ["Si", {"name": "Si", "density_g_cm3": 2.329}])
def test_density_scan_supports_saved_material_strings_and_objects(material) -> None:
    """Density scans work with both supported saved material formats."""
    spec = _laminar_spec(stack={"type": "bare", "substrate_material": material})
    density_id = "stack.substrate_material.density_g_cm3"
    assert density_id in {item.id for item in parameter_catalog(spec)}
    _, cases = expand_custom_grid(_config(spec, independent=[
        {"parameter": density_id, "mode": "range", "minimum": 2, "maximum": 3, "count": 3},
    ]), spec)
    assert [case["grating"].resolved_stack().substrate_material.density_g_cm3
            for case in cases] == [2, 2.5, 3]


def test_material_rows_apply_density_and_interface_roughness() -> None:
    """Material-specific properties reach the constructed stack."""
    spec = _laminar_spec(stack={
        "type": "bare", "substrate_material": "Si", "substrate_roughness_sigma_nm": 0.7,
    })
    _, cases = expand_custom_grid(_config(spec, independent=[{
        "parameter": "stack.substrate_material", "mode": "list", "values": [
            {"name": "Au", "density_g_cm3": 19, "roughness_sigma_nm": 0.2},
            {"name": "Pt", "density_g_cm3": 21, "roughness_sigma_nm": 0.4},
            {"name": "Si", "density_g_cm3": None, "roughness_sigma_nm": None},
        ],
    }]), spec)
    stacks = [case["grating"].resolved_stack() for case in cases]
    assert [stack.substrate_material.density_g_cm3 for stack in stacks] == [19, 21, None]
    assert [stack.substrate_roughness_sigma_nm for stack in stacks] == [0.2, 0.4, 0.7]
    assert spec["stack"]["substrate_material"] == "Si"


def test_separate_property_scans_override_embedded_material_properties() -> None:
    """Explicit property axes override material defaults regardless of axis order."""
    spec = _laminar_spec()
    _, cases = expand_custom_grid(_config(spec, independent=[
        {"parameter": "stack.substrate_roughness_sigma_nm", "mode": "list", "values": [0.8]},
        {"parameter": "stack.substrate_material.density_g_cm3", "mode": "list", "values": [18]},
        {"parameter": "stack.substrate_material", "mode": "list", "values": [
            {"name": "Au", "density_g_cm3": 19, "roughness_sigma_nm": 0.2},
        ]},
    ]), spec)
    stack = cases[0]["grating"].resolved_stack()
    assert stack.substrate_material.density_g_cm3 == 18
    assert stack.substrate_roughness_sigma_nm == 0.8


def test_custom_layer_material_properties_target_the_selected_interface() -> None:
    """A custom layer uses its own roughness field and density."""
    spec = _laminar_spec(stack={"type": "custom", "substrate_material": "Si",
                               "layers_bottom_up": [{"material": "Au", "thickness_nm": 10}]})
    _, cases = expand_custom_grid(_config(spec, independent=[{
        "parameter": "stack.layers_bottom_up.0.material", "mode": "list",
        "values": [{"name": "Pt", "density_g_cm3": 20, "roughness_sigma_nm": 0.3}],
    }]), spec)
    layer = cases[0]["grating"].resolved_stack().layers_bottom_up[0]
    assert layer.material.density_g_cm3 == 20
    assert layer.roughness_sigma_nm == 0.3


@pytest.mark.parametrize("parameter", [
    "stack.material_b.density_g_cm3", "stack.top_material.density_g_cm3",
])
def test_multilayer_density_scans_preserve_top_layer_selection(parameter) -> None:
    """The selected top material follows the scanned bilayer material density."""
    spec = _laminar_spec(stack={"type": "multilayer", "substrate_material": "Si",
                               "material_a": "Si", "material_b": "Mo", "top_material": "Mo",
                               "d_period_nm": 6, "gamma": 0.5, "n_bilayers": 5})
    _, cases = expand_custom_grid(_config(spec, independent=[
        {"parameter": parameter, "mode": "list", "values": [9, 10]},
    ]), spec)
    for case, density in zip(cases, [9, 10], strict=True):
        stack = case["grating"].resolved_stack()
        assert stack.top_material == stack.material_b
        assert stack.material_b.density_g_cm3 == density


@pytest.mark.parametrize("properties", [
    {"density_g_cm3": 0}, {"density_g_cm3": -1}, {"density_g_cm3": float("inf")},
    {"roughness_sigma_nm": -1}, {"roughness_sigma_nm": float("nan")},
])
def test_invalid_material_properties_are_rejected(properties) -> None:
    """Reject invalid physical values before constructing simulation cases."""
    spec = _laminar_spec()
    with pytest.raises(ValueError, match="finite"):
        expand_custom_grid(_config(spec, independent=[{
            "parameter": "stack.substrate_material", "mode": "list",
            "values": [{"name": "Au", **properties}],
        }]), spec)
