"""Tests for the web app's plane-mirror tab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grax import build_multilayer_stack, parratt_reflectivity
from grax.materials import MaterialSpec
from grax.web.app import create_app

pytest.importorskip("flask")

FORM = {
    "stack_type": "multilayer",
    "substrate_material": "Si",
    "material_a": "Ru",
    "material_b": "C",
    "d_period_nm": "7.0",
    "gamma": "0.4",
    "n_bilayers": "10",
    "top_material": "C",
    "scan_mode": "energy",
    "energy_min": "200",
    "energy_max": "300",
    "energy_points": "11",
    "angle_min": "5",
    "angle_max": "30",
    "angle_points": "4",
    "map_energy_points": "11",
    "map_angle_points": "4",
    "fixed_angle_deg": "20",
    "fixed_energy_ev": "259",
    "polarization": "s",
    "roughness_sigma_nm": "0.2",
    "grading_mode": "none",
}


@pytest.fixture
def client(tmp_path: Path):
    return create_app(data_dir=tmp_path).test_client()


def test_page_renders_and_is_linked_from_navigation(client) -> None:
    page = client.get("/plane-mirror")
    assert page.status_code == 200
    assert b"data-plane-mirror-form" in page.data
    assert b"/plane-mirror" in client.get("/").data


def test_compute_matches_direct_parratt(client) -> None:
    response = client.post("/_compute/plane-mirror", data=FORM)
    payload = response.get_json()

    stack = build_multilayer_stack(
        substrate_material=MaterialSpec("Si"),
        material_a=MaterialSpec("Ru"),
        material_b=MaterialSpec("C"),
        d_period_nm=7.0,
        gamma=0.4,
        n_bilayers=10,
        top_material=MaterialSpec("C"),
    )
    expected = parratt_reflectivity(stack, np.linspace(200, 300, 11), 20.0, roughness_sigma_nm=0.2)[:, 0]
    assert response.status_code == 200 and payload["ok"] is True
    assert payload["graded"] is None
    np.testing.assert_allclose(payload["reflectivity"], expected, rtol=1e-9)
    assert all(0.0 <= value <= 1.0 for value in payload["reflectivity"])


def test_graded_curve_and_angle_scan(client) -> None:
    graded = client.post(
        "/_compute/plane-mirror",
        data={**FORM, "grading_mode": "linear", "grading_percent_per_mm": "0.5", "footprint_points": "9"},
    ).get_json()
    assert len(graded["graded"]) == 11
    assert max(graded["graded"]) <= max(graded["reflectivity"])

    angle = client.post(
        "/_compute/plane-mirror",
        data={**FORM, "scan_mode": "angle"},
    ).get_json()
    assert angle["x_label"].startswith("Grazing angle") and len(angle["x"]) == 4


def test_map_scan_matches_direct_parratt_and_csv(client) -> None:
    payload = client.post("/_compute/plane-mirror", data={**FORM, "scan_mode": "map"}).get_json()

    stack = build_multilayer_stack(
        substrate_material=MaterialSpec("Si"),
        material_a=MaterialSpec("Ru"),
        material_b=MaterialSpec("C"),
        d_period_nm=7.0,
        gamma=0.4,
        n_bilayers=10,
        top_material=MaterialSpec("C"),
    )
    expected = parratt_reflectivity(
        stack, np.linspace(200, 300, 11), np.linspace(5, 30, 4), roughness_sigma_nm=0.2
    )
    assert payload["mode"] == "map" and payload["graded"] is False
    np.testing.assert_allclose(np.array(payload["z"]), expected.T, rtol=1e-9)

    lines = client.post("/plane-mirror/csv", data={**FORM, "scan_mode": "map"}).get_data(as_text=True).splitlines()
    assert lines[0] == "energy_ev,grazing_angle_deg,reflectivity" and len(lines) == 1 + 11 * 4


def test_unused_scan_fields_are_not_required_or_validated(client) -> None:
    form = {key: value for key, value in FORM.items() if not key.startswith("angle_")}
    assert client.post("/_compute/plane-mirror", data={**form, "angle_min": "-5"}).status_code == 200


def test_default_roughness_is_zero(client) -> None:
    assert b'name="roughness_sigma_nm" type="number" step="0.01" min="0" value="0"' in client.get("/plane-mirror").data


@pytest.mark.parametrize(
    "override",
    [{"energy_points": "0"}, {"scan_mode": "bogus"}, {"energy_min": "abc"}, {"polarization": "x"}, {"substrate_material": "Zz"}],
)
def test_bad_input_returns_400(client, override) -> None:
    response = client.post("/_compute/plane-mirror", data={**FORM, **override})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_csv_download(client) -> None:
    response = client.post("/plane-mirror/csv", data={**FORM, "grading_mode": "linear"})
    lines = response.get_data(as_text=True).strip().splitlines()
    assert response.mimetype == "text/csv"
    assert lines[0].endswith("reflectivity_graded") and len(lines) == 12
    assert client.post("/plane-mirror/csv", data={**FORM, "energy_points": "0"}).status_code == 400


CUSTOM_FORM = {
    **FORM,
    "stack_type": "custom",
    "cl_material": ["C", "Pt", "Cr"],
    "cl_density_g_cm3": ["", "", ""],
    "cl_thickness_nm": ["2", "10", "3"],
    "cl_roughness_sigma_nm": ["", "0.5", ""],
}


def test_custom_layers_match_direct_parratt_and_order_is_top_down(client) -> None:
    from grax import LayerSpec, assemble_custom_stack

    payload = client.post("/_compute/plane-mirror", data=CUSTOM_FORM).get_json()
    stack = assemble_custom_stack(
        substrate_material=MaterialSpec("Si"),
        layers_bottom_up=[
            LayerSpec(MaterialSpec("Cr"), 3.0),
            LayerSpec(MaterialSpec("Pt"), 10.0, roughness_sigma_nm=0.5),
            LayerSpec(MaterialSpec("C"), 2.0),
        ],
    )
    expected = parratt_reflectivity(stack, np.linspace(200, 300, 11), 20.0, roughness_sigma_nm=0.2)[:, 0]
    assert payload["ok"] is True
    np.testing.assert_allclose(payload["reflectivity"], expected, rtol=1e-9)


def test_single_custom_layer_is_allowed(client) -> None:
    form = {**CUSTOM_FORM, "cl_material": ["Pt"], "cl_density_g_cm3": [""], "cl_thickness_nm": ["10"], "cl_roughness_sigma_nm": [""]}
    assert client.post("/_compute/plane-mirror", data=form).status_code == 200


@pytest.mark.parametrize(
    "override",
    [
        {"cl_material": [], "cl_density_g_cm3": [], "cl_thickness_nm": [], "cl_roughness_sigma_nm": []},
        {"cl_thickness_nm": ["2", "x", "3"]},
        {"cl_thickness_nm": ["2", "-1", "3"]},
        {"cl_material": ["C", "", "Cr"]},
        {"cl_roughness_sigma_nm": ["", "-1", ""]},
    ],
)
def test_bad_custom_layers_return_400(client, override) -> None:
    assert client.post("/_compute/plane-mirror", data={**CUSTOM_FORM, **override}).status_code == 400


def test_stack_schematic_endpoint_and_custom_option_scope(client) -> None:
    for form in (CUSTOM_FORM, FORM):
        payload = client.post("/_preview/plane-mirror-stack", data=form).get_json()
        assert payload["ok"] is True and payload["image"].startswith("data:image/png;base64,")
    assert client.post("/_preview/plane-mirror-stack", data={**FORM, "d_period_nm": "x"}).status_code == 400

    assert b'value="custom"' in client.get("/plane-mirror").data
    assert b'value="custom"' not in client.get("/gratings/new").data


def test_map_defaults_stay_under_the_sample_cap(client) -> None:
    from grax.web.plane_mirror import MAX_TOTAL_SAMPLES, PLANE_MIRROR_DEFAULTS, parse_plane_mirror_options

    options = parse_plane_mirror_options({**PLANE_MIRROR_DEFAULTS, "scan_mode": "map"})
    assert options.energy_points * options.angle_points <= MAX_TOTAL_SAMPLES
    angle = parse_plane_mirror_options({**PLANE_MIRROR_DEFAULTS, "scan_mode": "angle"})
    assert (angle.angle_min, angle.angle_max, angle.angle_points, angle.fixed_energy_ev) == (0.1, 8.0, 5000, 1000.0)


def test_energy_scan_defaults(client) -> None:
    page = client.get("/plane-mirror").get_data(as_text=True)
    for name, value in (("energy_min", "100"), ("energy_max", "6000"), ("energy_points", "1000"), ("fixed_angle_deg", "0.4")):
        assert f'name="{name}" type="number"' in page and f'value="{value}"' in page.split(f'name="{name}"')[1].split(">")[0]


def test_too_many_points_messages_state_limits_and_suggest_counts() -> None:
    from grax.web.plane_mirror import MAX_TOTAL_SAMPLES, PLANE_MIRROR_DEFAULTS, parse_plane_mirror_options

    def error(**override: str) -> str:
        with pytest.raises(ValueError) as caught:
            parse_plane_mirror_options({**PLANE_MIRROR_DEFAULTS, **override})
        return str(caught.value)

    single = error(energy_points="6000")
    assert "5,000" in single and "per scan" in single

    axis = error(scan_mode="map", map_angle_points="9000")
    assert "5,000" in axis and "per axis" in axis

    graded_map = error(scan_mode="map", grading_mode="linear", footprint_points="41")
    assert f"{MAX_TOTAL_SAMPLES:,}" in graded_map and "1,000 x 1,000 = 1,000,000" in graded_map
    suggestion = graded_map.split("For example use ")[1].split(" energy x ")
    energy, angle = int(suggestion[0].replace(",", "")), int(suggestion[1].split(" ")[0].replace(",", ""))
    assert energy * angle * 41 <= MAX_TOTAL_SAMPLES and energy >= 100

    graded_single = error(grading_mode="linear", energy_points="5000", footprint_points="500")
    assert "at most 4,000 points" in graded_single
