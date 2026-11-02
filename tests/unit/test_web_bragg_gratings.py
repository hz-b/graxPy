# ruff: noqa: D100,D103

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("flask")

from grax.web import bragg_simulations  # noqa: E402
from grax.web.app import create_app  # noqa: E402
from grax.web.bragg_gratings import DEFAULTS as GRATING_DEFAULTS  # noqa: E402

pytestmark = pytest.mark.unit

GRATING_FORM = {**{key: str(value) for key, value in GRATING_DEFAULTS.items()}, "name": "LNB comb"}


@pytest.fixture
def client(tmp_path):
    return create_app(data_dir=tmp_path).test_client()


def _create(client, **changes):
    return client.post("/bragg-gratings", data={**GRATING_FORM, **changes})


def _scan_form():
    return {key: str(value) for key, value in bragg_simulations.DEFAULTS.items()}


def test_pages_render(client):
    for path in ("/bragg-gratings", "/bragg-gratings/new", "/bragg-runs"):
        assert client.get(path).status_code == 200
    _create(client)
    assert client.get("/bragg-gratings/lnb-comb").status_code == 200


def test_grating_is_created_edited_and_deleted(client):
    response = _create(client)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/bragg-gratings/lnb-comb")
    assert b"LNB comb" in client.get("/bragg-gratings").data

    edit = client.get("/bragg-gratings/lnb-comb/edit")
    assert edit.status_code == 200
    assert b"LNB comb" in edit.data

    changed = client.post(
        "/bragg-gratings/lnb-comb", data={**GRATING_FORM, "name": "LNB comb", "finger_um": "0.7"}
    )
    assert changed.status_code == 302
    assert b'value="0.7"' in client.get("/bragg-gratings/lnb-comb/edit").data

    assert client.post("/bragg-gratings/lnb-comb/delete").status_code == 302
    assert client.get("/bragg-gratings/lnb-comb/edit").status_code == 404


def test_same_name_gets_a_new_id(client):
    _create(client)
    _create(client)

    assert client.get("/bragg-gratings/lnb-comb-2/edit").status_code == 200


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": " "}, "name"),
        ({"crystal": "quartz"}, "crystal"),
        ({"h": "x"}, "whole number"),
        ({"h": "2", "k": "0", "l": "0", "crystal": "Si"}, "forbidden"),
        ({"finger_um": "-1"}, "Finger width"),
    ],
)
def test_bad_grating_input_is_explained(client, changes, message):
    response = _create(client, **changes)

    assert response.status_code == 400
    assert message in response.get_data(as_text=True)


def test_scan_needs_a_saved_grating(client):
    response = client.post("/bragg-gratings/missing/run", data=_scan_form())

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("energy_ev", "5", "Photon energy"),
        ("amplitudes_nm", "0, x", "Amplitudes"),
        ("amplitudes_nm", "0, 101", "between 0 and 100"),
        ("n_orders", "8", "odd"),
        ("beam_x_um", "-1", "Beam size"),
        ("step_deg", "0.0000001", "more than"),
        ("scan_type", "magic", "scan type"),
        ("amplitude_mode", "magic", "Unknown way"),
    ],
)
def test_bad_scan_input_is_explained(client, field, value, message):
    _create(client)

    form = {**_scan_form(), "amplitude_mode": "list", field: value}
    response = client.post("/bragg-gratings/lnb-comb/run", data=form)

    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_amplitudes_default_to_a_linear_range():
    assert bragg_simulations.DEFAULTS["amplitude_mode"] == "range"


def test_amplitudes_can_be_a_linear_range():
    from grax.web.bragg_simulations import parse_params

    grating = {"id": "g", "name": "g"}
    form = {
        **_scan_form(),
        "amplitude_mode": "range",
        "amplitude_start_nm": "0",
        "amplitude_stop_nm": "0.1",
        "amplitude_points": "5",
        "amplitudes_nm": "ignored",
    }

    assert parse_params(form, grating)["amplitudes_nm"] == [0.0, 0.025, 0.05, 0.075, 0.1]
    assert parse_params({**form, "amplitude_points": "1"}, grating)["amplitudes_nm"] == [0.0]


def _wait_for(client, run_id):
    deadline = time.monotonic() + 10.0
    while True:
        payload = client.get(f"/bragg-scans/{run_id}/status").get_json()
        if payload["state"] not in {"queued", "running", "aborting"}:
            return payload
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_scan_shows_curves_while_running_and_opens_its_own_page(client, monkeypatch):
    release = threading.Event()

    def fake(params, progress=None, stop=None):
        amplitudes = params["amplitudes_nm"]
        progress(amplitudes[0], [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        release.wait(5.0)
        for amplitude in amplitudes[1:]:
            progress(amplitude, [-1.0, 0.0, 1.0], [0.0, 0.5, 0.0])
        return {
            "angles_deg": [-1.0, 0.0, 1.0],
            "curves": {f"{a:g}": [0.0, 1.0, 0.0] for a in amplitudes},
            "summary": {},
        }

    monkeypatch.setattr(bragg_simulations, "compute_scan", fake)
    _create(client)

    form = {**_scan_form(), "amplitude_mode": "list", "amplitudes_nm": "0, 0.05"}
    started = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()
    run_id = started["run_id"]
    assert started["url"].endswith(f"/bragg-runs/{run_id}")
    page = client.get(started["url"])
    assert page.status_code == 200
    assert b"data-live-run-monitor" in page.data
    assert b"data-bragg-run-page" in page.data

    deadline = time.monotonic() + 10.0
    while not (live := client.get(f"/bragg-scans/{run_id}/status").get_json())["data"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert live["state"] == "running"
    assert live["total_points"] == 2
    assert live["can_abort"] is True
    assert list(live["data"]["curves"]) == ["0"]

    release.set()
    payload = _wait_for(client, run_id)

    assert payload["state"] == "completed"
    assert payload["completed_points"] == 2
    assert sorted(payload["data"]["curves"]) == ["0", "0.05"]
    assert run_id.encode() in client.get("/bragg-gratings/lnb-comb").data


def test_saved_scans_are_listed_and_deleted(client, monkeypatch):
    def fake(params, progress=None, stop=None):
        progress(0.0, [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        return {"angles_deg": [-1.0, 0.0, 1.0], "curves": {"0": [0.0, 1.0, 0.0]}, "summary": {}}

    monkeypatch.setattr(bragg_simulations, "compute_scan", fake)
    _create(client)
    assert b"No Bragg scans" in client.get("/bragg-runs").data

    form = {**_scan_form(), "amplitude_mode": "list", "amplitudes_nm": "0"}
    run_id = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()["run_id"]
    _wait_for(client, run_id)

    page = client.get("/bragg-runs").data
    assert b"LNB comb" in page
    assert f"/bragg-runs/{run_id}".encode() in page

    assert client.post(f"/bragg-runs/{run_id}/delete").status_code == 302
    assert b"No Bragg scans" in client.get("/bragg-runs").data
    assert client.get(f"/bragg-scans/{run_id}/status").status_code == 404
    assert client.get(f"/bragg-runs/{run_id}").status_code == 404


def test_a_running_scan_can_be_aborted_and_keeps_its_curves(client, monkeypatch):
    def slow(params, progress=None, stop=None):
        amplitudes = params["amplitudes_nm"]
        progress(amplitudes[0], [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        stop.wait(5.0)
        return {"angles_deg": [-1.0, 0.0, 1.0], "curves": {f"{amplitudes[0]:g}": [0.0, 1.0, 0.0]}}

    monkeypatch.setattr(bragg_simulations, "compute_scan", slow)
    _create(client)
    form = {**_scan_form(), "amplitude_mode": "list", "amplitudes_nm": "0, 0.05"}
    run_id = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()["run_id"]
    deadline = time.monotonic() + 10.0
    while not client.get(f"/bragg-scans/{run_id}/status").get_json()["data"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    assert client.post(f"/bragg-runs/{run_id}/abort").status_code == 302
    payload = _wait_for(client, run_id)

    assert payload["state"] == "aborted"
    assert list(payload["data"]["curves"]) == ["0"]
    assert client.get(f"/bragg-runs/{run_id}").status_code == 200


def test_failed_scan_reports_the_error(client, monkeypatch):
    def broken(params, progress=None, stop=None):
        raise RuntimeError("cannot be reached")

    monkeypatch.setattr(bragg_simulations, "compute_scan", broken)
    _create(client)

    run_id = client.post("/bragg-gratings/lnb-comb/run", data=_scan_form()).get_json()["run_id"]
    payload = _wait_for(client, run_id)

    assert payload["state"] == "failed"
    assert "cannot be reached" in payload["error_text"]
    assert client.get("/bragg-runs").data.count(b"failed") >= 1


def test_unknown_run_is_not_found(client):
    assert client.get("/bragg-scans/doesnotexist/status").status_code == 404


def test_preview_draws_the_electrodes(client):
    response = client.post(
        "/_preview/bragg-grating", data={**GRATING_FORM, "preview_amplitude_nm": "0.1"}
    )

    assert response.status_code == 200
    assert response.get_json()["image"].startswith("data:image/png;base64,")


def test_preview_rejects_bad_geometry(client):
    response = client.post("/_preview/bragg-grating", data={**GRATING_FORM, "finger_um": "x"})

    assert response.status_code == 400
    assert "Finger width" in response.get_json()["error"]


def test_default_amplitudes_reproduce_the_paper_series():
    amplitudes = [float(item) for item in bragg_simulations.PAPER_AMPLITUDES_NM.split(",")]

    assert amplitudes[0] == 0.0
    assert amplitudes[-1] == pytest.approx(0.12)
    assert len(amplitudes) == 11


COATING_FORM = {
    "coated": "on",
    "cl_kind": ["single", "multilayer"],
    "cl_material": ["Pt", "W"],
    "cl_density_g_cm3": ["", ""],
    "cl_thickness_nm": ["10", "1.2"],
    "cl_planarization_nm": ["0", "0"],
    "cl_material_b": ["", "B4C"],
    "cl_density_b_g_cm3": ["", "2.52"],
    "cl_thickness_b_nm": ["", "1.8"],
    "cl_repeats": ["1", "5"],
}


def test_coated_grating_is_saved_and_preselects_rcwa(client):
    assert _create(client, **COATING_FORM).status_code == 302
    edit = client.get("/bragg-gratings/lnb-comb/edit").data
    assert b'name="coated" value="on" data-bragg-coated checked' in edit
    assert b'value="B4C"' in edit and b'name="cl_planarization_nm"' in edit
    assert b'name="cl_roughness_sigma_nm"' not in edit
    detail = client.get("/bragg-gratings/lnb-comb").data
    assert b"data-bragg-cross-section" not in detail
    assert b'alt="Cross-section of one period' in detail
    assert b'<option value="rcwa" selected>' in detail
    assert b"coating Pt 10 nm on [W 1.2 / B4C 1.8 nm] \xc3\x97 5" in detail

    from grax.web.bragg_gratings import BraggGratingStore, coating_layers

    spec = BraggGratingStore(client.application.config["GRAx_DATA_DIR"] / "bragg_gratings").load("lnb-comb")
    layers = coating_layers(spec)
    assert len(layers) == 11 and layers[-1].material.name == "Pt"


def test_switching_the_coating_off_keeps_the_layers_but_uses_takagi(client):
    _create(client, **{**COATING_FORM, "coated": ""})
    detail = client.get("/bragg-gratings/lnb-comb").data
    assert b'<option value="takagi" selected>' in detail
    assert b'value="B4C"' in client.get("/bragg-gratings/lnb-comb/edit").data


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cl_material": ["Pt", "Unobtainium"]}, b"Unobtainium"),
        ({"cl_thickness_nm": ["0", "1.2"]}, b"thickness must be above 0"),
        ({"cl_planarization_nm": ["-1", "0"]}, b"planarization must be at least 0"),
        ({"cl_repeats": ["1", "0"]}, b"repeats must be between 1 and 250"),
    ],
)
def test_bad_coating_is_explained(client, changes, message):
    response = _create(client, **{**COATING_FORM, **changes})
    assert response.status_code == 400
    assert message in response.data
    assert b'value="W"' in response.data


def test_coating_layers_expand_multilayers_bottom_up():
    from grax.web.bragg_gratings import coating_layers

    spec = {
        "coated": True,
        "coating_blocks": [
            {"kind": "single", "material": "Pt", "thickness_nm": "10", "planarization_nm": "2"},
            {"kind": "multilayer", "material": "W", "thickness_nm": "1", "material_b": "C",
             "thickness_b_nm": "2", "repeats": "2", "planarization_nm": ""},
        ],
    }
    layers = coating_layers(spec)
    assert [(layer.material.name, layer.thickness_nm) for layer in layers] == [
        ("C", 2.0), ("W", 1.0), ("C", 2.0), ("W", 1.0), ("Pt", 10.0)
    ]
    assert layers[-1].planarization_nm == 2.0
    assert coating_layers({**spec, "coated": False}) == ()


def test_old_gratings_without_a_coating_still_work(client):
    _create(client)
    from grax.web.bragg_gratings import default_solver

    assert default_solver({"crystal": "LiNbO3"}) == "takagi"
    assert b'<option value="takagi" selected>' in client.get("/bragg-gratings/lnb-comb").data


def test_preview_draws_the_coating_cross_section(client):
    payload = client.post("/_preview/bragg-grating", data={**GRATING_FORM, **COATING_FORM}).get_json()
    assert payload["ok"] and payload["cross_section"].startswith("data:image/png;base64,")
    bare = client.post("/_preview/bragg-grating", data=GRATING_FORM).get_json()
    assert bare["ok"] and bare["cross_section"] == ""
    bad = client.post("/_preview/bragg-grating", data={**GRATING_FORM, **COATING_FORM, "cl_thickness_nm": ["x", "1"]})
    assert bad.status_code == 400


def test_grazing_scan_fields_are_validated(client):
    _create(client, **COATING_FORM)
    form = {**_scan_form(), "solver": "rcwa", "theta_center_deg": "0.2", "theta_half_range_deg": "0.5"}
    response = client.post("/bragg-gratings/lnb-comb/run", data=form)
    assert response.status_code == 400
    assert "between 0° and 90°" in response.get_json()["error"]
    form = {**_scan_form(), "solver": "magic"}
    assert client.post("/bragg-gratings/lnb-comb/run", data=form).status_code == 400


def test_grazing_scan_runs_every_order(client):
    _create(client, **{**COATING_FORM, "cl_repeats": ["1", "1"]})
    form = {
        **_scan_form(),
        "solver": "rcwa",
        "amplitude_mode": "list",
        "amplitudes_nm": "0, 0.1",
        "theta_center_deg": "0.35",
        "theta_half_range_deg": "0.05",
        "theta_step_deg": "0.05",
        "fourier_orders": "6",
        "workers": "1",
        "x_resolution_nm": "4",
        "z_resolution_nm": "2",
    }
    started = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()
    assert started["ok"], started
    payload = _wait_for(client, started["run_id"])
    assert payload["state"] == "completed", payload["error_text"]
    data = payload["data"]
    assert data["geometry"] == "grazing"
    assert data["angles_deg"] == pytest.approx([0.3, 0.35, 0.4])
    assert sorted(data["curves"]) == ["0", "0.1"]
    assert "1" in data["curves"]["0.1"] and "2" in data["curves"]["0"]
    assert max(data["curves"]["0"].get("1", [0.0])) < 1e-12 < max(data["curves"]["0.1"]["1"])
    assert len(data["exit_angles_deg"]["0"]) == 3
    page = client.get(started["url"]).data
    assert b"data-bragg-order" in page and b"data-bragg-detector-amplitude" not in page
    assert (payload["completed_points"], payload["total_points"]) == (6, 6)
    assert b"RCWA (grazing incidence)" in page


def test_grazing_resolutions_are_validated_and_default_to_the_grating_pages(client):
    from grax.web.bragg_simulations import parse_params

    grating = {"id": "g", "coated": True, "coating_blocks": [{"kind": "layer"}]}
    params = parse_params({"solver": "rcwa"}, grating)
    assert (params["x_resolution_nm"], params["z_resolution_nm"]) == (1.0, 1.0)
    with pytest.raises(ValueError, match="Z resolution"):
        parse_params({"solver": "rcwa", "z_resolution_nm": "0"}, grating)


def test_theta_window_follows_the_energy_and_finds_the_multilayer_peak(client):
    _create(client, **COATING_FORM)
    low = client.get("/bragg-gratings/lnb-comb/theta-window?energy_ev=8000").get_json()
    high = client.get("/bragg-gratings/lnb-comb/theta-window?energy_ev=16000").get_json()
    assert low["ok"] and high["ok"]
    assert low["multilayer_peak_deg"] is not None
    assert high["multilayer_peak_deg"] == pytest.approx(low["multilayer_peak_deg"] / 2.0, rel=0.1)
    assert low["theta_center_deg"] - low["theta_half_range_deg"] > 0.0
    assert client.get("/bragg-gratings/lnb-comb/theta-window?energy_ev=x").status_code == 400
    page = client.get("/bragg-gratings/lnb-comb").get_data(as_text=True)
    assert "Multilayer Bragg peak at" in page and 'name="theta_center_deg"' in page


def test_energy_scan_runs_takagi_rocking_curves_for_each_energy(client):
    _create(client)
    form = {
        **_scan_form(),
        "scan_type": "energy_rocking_curve",
        "solver": "takagi",
        "energy_mode": "list",
        "energies_ev": "8000, 9000",
        "amplitude_nm": "0.05",
        "half_range_deg": "0.01",
        "step_deg": "0.002",
        "n_orders": "3",
    }
    started = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()
    assert started["ok"], started
    payload = _wait_for(client, started["run_id"])
    assert payload["state"] == "completed", payload["error_text"]
    assert sorted(payload["data"]["curves"]) == ["8000", "9000"]
    assert (payload["completed_points"], payload["total_points"]) == (2, 2)
    page = client.get(started["url"]).get_data(as_text=True)
    assert 'data-series-label="Photon energy (eV)"' in page and "2 energies from 8000.0 to 9000.0 eV" in page
    assert "amplitude 0.05 nm" in page
    listing = client.get("/bragg-runs").get_data(as_text=True)
    assert "2 energies, 8000 to 9000 eV at 0.05 nm" in listing


def test_energy_scan_can_follow_the_multilayer_peak(client, tmp_path):
    _create(client, **COATING_FORM)
    form = {
        **_scan_form(),
        "scan_type": "energy_rocking_curve",
        "solver": "rcwa",
        "energy_mode": "range",
        "energy_start_ev": "8000",
        "energy_stop_ev": "10000",
        "energy_points": "2",
        "amplitude_nm": "0.1",
        "theta_follow_peak": "on",
        "theta_center_deg": "5",
        "theta_half_range_deg": "0.02",
        "theta_step_deg": "0.02",
        "fourier_orders": "6",
        "workers": "1",
        "x_resolution_nm": "4",
        "z_resolution_nm": "2",
    }
    started = client.post("/bragg-gratings/lnb-comb/run", data=form).get_json()
    assert started["ok"], started
    payload = _wait_for(client, started["run_id"])
    assert payload["state"] == "completed", payload["error_text"]
    data = payload["data"]
    assert data["angles_deg"] == pytest.approx([-0.02, 0.0, 0.02])
    assert sorted(data["curves"]) == ["10000", "8000"]
    assert (payload["completed_points"], payload["total_points"]) == (6, 6)
    saved = bragg_simulations.load_scan(tmp_path, started["run_id"])
    centres = saved["summary"]["centre_deg"]
    assert centres["10000"] == pytest.approx(centres["8000"] * 0.8, rel=0.05)
    page = client.get(started["url"]).get_data(as_text=True)
    assert "around the multilayer peak of each energy" in page
    assert "Grazing angle from the multilayer peak" in page


def test_energy_scan_validation(client):
    _create(client)
    base = {**_scan_form(), "scan_type": "energy_rocking_curve", "solver": "takagi"}
    for changes, message in (
        ({"energy_mode": "list", "energies_ev": "8000, 40000"}, "between 10 and 30000 eV"),
        ({"energy_mode": "list", "energies_ev": "8000, x"}, "Energies must be numbers"),
        ({"amplitude_nm": "-1"}, "Deformation amplitude"),
        ({"solver": "rcwa", "theta_follow_peak": "on"}, "no multilayer"),
    ):
        response = client.post("/bragg-gratings/lnb-comb/run", data={**base, **changes})
        assert response.status_code == 400, changes
        assert message in response.get_json()["error"]
