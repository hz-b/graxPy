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
    "scan_min": "200",
    "scan_max": "300",
    "scan_points": "11",
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
        data={**FORM, "scan_mode": "angle", "scan_min": "5", "scan_max": "30"},
    ).get_json()
    assert angle["x_label"].startswith("Grazing angle")


@pytest.mark.parametrize(
    "override",
    [{"scan_points": "0"}, {"scan_mode": "bogus"}, {"scan_min": "abc"}, {"polarization": "x"}, {"substrate_material": "Zz"}],
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
    assert client.post("/plane-mirror/csv", data={**FORM, "scan_points": "0"}).status_code == 400
