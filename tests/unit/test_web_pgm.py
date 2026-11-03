# ruff: noqa: D103
"""PGM website input, job and result behavior."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

import grax
from grax.pgm import PGMCancelled, WorkingRangeResult, calculate_working_range
from grax.web.app import create_app
from grax.web.grating_forms import _spec_from_form
from grax.web.persistence import (
    GratingStore,
    PlaneMirrorStore,
    build_grating_from_spec,
    grating_to_spec,
)
from grax.web.pgm_jobs import PGM_DEFAULTS, parse_pgm_form, pgm_setup_store, pgm_store
from grax.web.pgm_presets import PGM_PRESETS
from grax.web.plane_mirror_views import _plane_mirror_stack
from grax.web.resource_manager import resource_status
from tests.unit.test_web_plane_mirror import FORM as MIRROR_FORM


@pytest.fixture
def pgm_client(tmp_path: Path):
    app = create_app(data_dir=tmp_path)
    grating = grax.BlazedGrating(
        period_lpermm=400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=None,
        substrate_material="Pt",
        layer_material="Pt",
        layer_thickness_nm=1,
        x_resolution_nm=4,
        z_resolution_nm=1,
    )
    saved_grating = GratingStore(tmp_path / "saved_gratings").save(
        grating_to_spec(grating, name="PGM grating")
    )
    saved_mirror = PlaneMirrorStore(tmp_path / "saved_plane_mirrors").save(
        {
            "name": "PGM mirror",
            "stack_form": dict(MIRROR_FORM),
            "scans": [],
        }
    )
    return app, app.test_client(), saved_grating, saved_mirror


GRATING_FIELDS = {
    "name": "Edited grating",
    "grating_type": "blazed",
    "period_lpermm": "400",
    "x_resolution_nm": "4",
    "z_resolution_nm": "1",
    "blaze_angle_deg": "0.9",
    "anti_blaze_angle_deg": "",
    "substrate_material": "Pt",
    "stack_type": "single_layer",
    "layer_material": "Pt",
    "layer_thickness_nm": "1",
    "roughness_kind": "none",
}


def _merged(grating_fields=None, mirror_fields=None, **overrides):
    data = {
        **{f"grating.{key}": value for key, value in (grating_fields or GRATING_FIELDS).items()},
        **{f"mirror.{key}": value for key, value in (mirror_fields or MIRROR_FORM).items()},
        "display_name": "Test PGM",
    }
    data.update({key.replace("__", "."): value for key, value in overrides.items()})
    return data


def _submit(client, **overrides):
    return client.post("/pgm/runs", data=_merged(**overrides))


def _run_id(response) -> str:
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["redirect"].rsplit("/", 1)[-1]


def _wait_for(client, run_id: str, expected: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/pgm/runs/{run_id}/status")
        if response.get_json()["state"] == expected:
            return
        time.sleep(0.02)
    pytest.fail(f"PGM run did not reach {expected}")


def test_pgm_form_validation_and_security(pgm_client):
    _, client, grating, mirror = pgm_client
    assert client.get("/pgm").status_code == 200
    assert _submit(client, c_min="4", c_max="2").status_code == 422
    assert _submit(client, n_energy="1000", n_c="1000").status_code == 422
    assert _submit(client, rcwa_energy_nodes="100", rcwa_cff_nodes="100").status_code == 422
    bad_grating = _submit(client, grating__period_lpermm="-1")
    assert bad_grating.status_code == 422
    assert bad_grating.get_json()["error"].startswith("Grating:")
    bad_mirror = _submit(client, mirror__n_bilayers="x")
    assert bad_mirror.status_code == 422
    assert bad_mirror.get_json()["error"].startswith("Mirror:")
    for url in ("/pgm/runs", "/pgm/saved", "/pgm/saved/x/rename", "/pgm/saved/x/delete"):
        assert (
            client.post(url, data=_merged(), headers={"Origin": "https://example.org"}).status_code
            == 403
        )
    assert client.get("/pgm/runs/../point").status_code == 404
    assert parse_pgm_form({"n_energy": "101", "n_c": "81"})["n_energy"] == 101
    assert parse_pgm_form({"mirror_surface_axis_offset_mm": "0"})[
        "mirror_surface_axis_offset_mm"
    ] == 0


def test_pgm_run_snapshots_maps_point_exports_and_delete(pgm_client, monkeypatch, tmp_path):
    app, client, grating, mirror = pgm_client

    def fake_run(parameters):
        energies, cff, geometry = calculate_working_range(parameters)
        shape = geometry["throughput_percent"].shape
        parameters.progress_callback(120, 120)
        optics = {
            "mirror_reflectivity": np.full(shape, 0.7),
            "grating_efficiency": np.full(shape, 0.3),
            "optical_percent": np.full(shape, 21.0),
            "total_percent": geometry["throughput_percent"] * 0.21,
        }
        return WorkingRangeResult(energies, cff, geometry, optics)

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", fake_run)
    run_id = _run_id(
        _submit(
            client,
            n_energy="6",
            n_c="4",
            grating_source_id=grating["id"],
            mirror_source_id=mirror["id"],
        )
    )
    _wait_for(client, run_id, "completed")
    manifest = pgm_store(tmp_path).load(run_id)
    assert manifest["grating_snapshot"]["id"] == grating["id"]
    assert manifest["mirror_snapshot"]["id"] == mirror["id"]
    GratingStore(tmp_path / "saved_gratings").delete(grating["id"])
    PlaneMirrorStore(tmp_path / "saved_plane_mirrors").delete(mirror["id"])
    assert client.get(f"/pgm/runs/{run_id}").status_code == 200
    maps = client.get(f"/pgm/runs/{run_id}/maps").get_json()
    assert len(maps["maps"]["total"]) == 4
    point = client.get(f"/pgm/runs/{run_id}/point?energy=0&cff=0").get_json()
    assert point["efficiencies"]["mirror"] == pytest.approx(70)
    assert point["geometry"]["alpha_deg"] > 0
    assert client.get(f"/pgm/runs/{run_id}/export.csv").status_code == 200
    assert client.get(f"/pgm/runs/{run_id}/export.png").status_code == 200
    assert client.get(f"/pgm/runs/{run_id}/point?energy=999&cff=0").status_code == 404
    assert client.post(f"/pgm/runs/{run_id}/delete").status_code == 302
    assert not (tmp_path / "pgm_runs" / run_id).exists()


def test_pgm_stop_marks_run_aborted(pgm_client, monkeypatch):
    _, client, grating, mirror = pgm_client
    started = threading.Event()

    def wait_until_stopped(parameters):
        started.set()
        while not parameters.stop_event.wait(0.01):
            pass
        raise PGMCancelled("stopped")

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", wait_until_stopped)
    run_id = _run_id(_submit(client))
    assert started.wait(5)
    page = client.get(f"/pgm/runs/{run_id}").get_data(as_text=True)
    assert "data-live-run-monitor" in page
    assert "data-run-eta" in page
    assert "data-run-abort-action" in page
    status = client.get(f"/pgm/runs/{run_id}/status").get_json()
    assert status["state"] == "running"
    assert status["remaining_points"] == status["total_points"] > 0
    assert status["worker_mode"] == "auto"
    assert status["requested_workers"] is None
    assert status["resolved_workers"] >= 1
    assert status["can_abort"] is True
    assert client.post(f"/pgm/runs/{run_id}/stop").status_code == 302
    _wait_for(client, run_id, "aborted")
    status = client.get(f"/pgm/runs/{run_id}/status").get_json()
    assert status["can_abort"] is False
    assert status["elapsed_seconds"] >= 0
    assert "data-live-run-monitor" not in client.get(f"/pgm/runs/{run_id}").get_data(as_text=True)


def test_pgm_dashboard_fills_in_during_a_run_and_keeps_it_when_stopped(pgm_client, monkeypatch):
    _, client, _, _ = pgm_client
    first_nodes, more_nodes = threading.Event(), threading.Event()

    def partial_run(parameters):
        energies, cff, geometry = calculate_working_range(parameters)
        mirror = np.full(geometry["throughput_percent"].shape, 0.5)
        parameters.maps_ready_callback(energies, cff, geometry, mirror)
        e_nodes = np.linspace(energies[0], energies[-1], 10)
        c_nodes = np.linspace(cff[0], cff[-1], 10)
        eta = np.full((10, 10), np.nan)
        eta[:, :5] = 0.4
        parameters.nodes_callback(e_nodes, c_nodes, eta)
        first_nodes.set()
        more_nodes.wait(5)
        while not parameters.stop_event.wait(0.01):
            pass
        raise PGMCancelled("stopped")

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", partial_run)
    run_id = _run_id(_submit(client, n_energy="21", n_c="11"))
    assert first_nodes.wait(5)

    page = client.get(f"/pgm/runs/{run_id}").get_data(as_text=True)
    assert "data-pgm-dashboard" in page
    assert "SX700 PARAMETERS" in page
    assert "data-pgm-map" not in page

    data = client.get(f"/pgm/runs/{run_id}/maps").get_json()
    assert data["maps"]["mirror"][0][0] == pytest.approx(50)
    assert data["colorscale"][0] == [0.0, "rgb(48,18,59)"]
    assert len(data["colorscale"]) == 64
    rows = data["maps"]["grating"]
    solved = [row[0] for row in rows if row[0] not in (None, 0)]
    assert solved and solved[0] == pytest.approx(40)
    assert any(row[-1] is None for row in rows)
    product = data["maps"]["optical"]
    assert any(row[-1] is None for row in product)

    assert client.get(
        f"/pgm/runs/{run_id}/live?v={data['token']}"
    ).get_json() == {"unchanged": True}
    live = client.get(f"/pgm/runs/{run_id}/live").get_json()
    assert set(live["maps"]) == {"grating", "optical", "total"}
    assert live["token"] == data["token"]

    last_energy = len(data["energies"]) - 1
    start = data["start"]
    point = client.get(
        f"/pgm/runs/{run_id}/point?energy={start['energy_index']}&cff={start['cff_index']}"
    ).get_json()
    assert point["side_view"]["mirror"]["x"]
    far = client.get(f"/pgm/runs/{run_id}/point?energy={last_energy}&cff=5").get_json()
    if far["geometry"] is not None:
        assert far["efficiencies"]["grating"] is None

    assert client.get(f"/pgm/runs/{run_id}/export.csv").status_code == 409
    more_nodes.set()
    assert client.post(f"/pgm/runs/{run_id}/stop").status_code == 302
    _wait_for(client, run_id, "aborted")
    stopped = client.get(f"/pgm/runs/{run_id}/maps").get_json()
    assert any(row[0] not in (None, 0) for row in stopped["maps"]["grating"])


def test_pgm_finished_run_has_full_maps_and_overlays(pgm_client, monkeypatch, tmp_path):
    _, client, _, _ = pgm_client
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    run_id = _run_id(_submit(client, n_energy="6", n_c="4"))
    _wait_for(client, run_id, "completed")
    run_dir = tmp_path / "pgm_runs" / run_id
    assert not (run_dir / "live_base.npz").exists()
    data = client.get(f"/pgm/runs/{run_id}/maps").get_json()
    assert len(data["maps"]["total"]) == 4
    assert len(data["maps"]["total"][0]) == 6
    assert data["overlays"]["distance_limit"]["level"] == pytest.approx(6.35)
    assert data["maps"]["optical"][0][0] == pytest.approx(21)


def test_pgm_run_without_maps_reports_conflict(pgm_client, monkeypatch):
    _, client, _, _ = pgm_client

    def fails(parameters):
        raise RuntimeError("no maps")

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", fails)
    run_id = _run_id(_submit(client))
    _wait_for(client, run_id, "failed")
    assert client.get(f"/pgm/runs/{run_id}/maps").status_code == 409
    assert client.get(f"/pgm/runs/{run_id}/point?energy=0&cff=0").status_code == 409
    assert client.get(f"/pgm/runs/{run_id}/live").get_json() == {"unchanged": True}


def _fake_run(parameters):
    energies, cff, geometry = calculate_working_range(parameters)
    shape = geometry["throughput_percent"].shape
    optics = {
        "mirror_reflectivity": np.full(shape, 0.7),
        "grating_efficiency": np.full(shape, 0.3),
        "optical_percent": np.full(shape, 21.0),
        "total_percent": geometry["throughput_percent"] * 0.21,
    }
    return WorkingRangeResult(energies, cff, geometry, optics)


def test_pgm_run_from_edited_prefill_leaves_library_unchanged(pgm_client, monkeypatch, tmp_path):
    _, client, grating, mirror = pgm_client
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    before = (
        GratingStore(tmp_path / "saved_gratings").load(grating["id"]),
        PlaneMirrorStore(tmp_path / "saved_plane_mirrors").load(mirror["id"]),
    )

    filled = client.get(f"/pgm/grating-fields?id={grating['id']}").get_data(as_text=True)
    assert 'value="PGM grating"' in filled
    assert f'value="{mirror["name"]}"' in client.get(
        f"/pgm/mirror-fields?id={mirror['id']}"
    ).get_data(as_text=True)
    assert client.get("/pgm/grating-fields?id=missing").status_code == 404

    run_id = _run_id(
        _submit(
            client,
            n_energy="6",
            n_c="4",
            grating__period_lpermm="1200",
            grating_source_id=grating["id"],
        )
    )
    _wait_for(client, run_id, "completed")
    manifest = pgm_store(tmp_path).load(run_id)
    assert manifest["grating_snapshot"]["period_lpermm"] == 1200
    assert manifest["grating_id"] == grating["id"]
    assert manifest["mirror_id"] is None
    assert before == (
        GratingStore(tmp_path / "saved_gratings").load(grating["id"]),
        PlaneMirrorStore(tmp_path / "saved_plane_mirrors").load(mirror["id"]),
    )


def test_pgm_run_without_any_saved_optic(tmp_path, monkeypatch):
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    client = create_app(data_dir=tmp_path).test_client()
    assert client.get("/pgm").status_code == 200
    run_id = _run_id(_submit(client, n_energy="6", n_c="4"))
    _wait_for(client, run_id, "completed")
    assert pgm_store(tmp_path).load(run_id)["grating_name"] == "Edited grating"
    assert not (tmp_path / "saved_gratings").exists()


def test_saved_pgm_save_reload_rename_delete(pgm_client, tmp_path):
    _, client, grating, mirror = pgm_client
    gratings = len(GratingStore(tmp_path / "saved_gratings").list())
    mirrors = len(PlaneMirrorStore(tmp_path / "saved_plane_mirrors").list())

    assert client.post("/pgm/saved", data=_merged()).status_code == 422
    response = client.post(
        "/pgm/saved", data=_merged(pgm_name="My PGM", grating__period_lpermm="600", n_c="33")
    )
    assert response.status_code == 200
    saved = response.get_json()
    assert saved["name"] == "My PGM"
    assert len(GratingStore(tmp_path / "saved_gratings").list()) == gratings
    assert len(PlaneMirrorStore(tmp_path / "saved_plane_mirrors").list()) == mirrors

    page = client.get(saved["url"]).get_data(as_text=True)
    assert 'name="period_lpermm" type="number" step="1" value="600"' in page
    assert 'name="n_c" step="1" required value="33"' in page
    assert client.get("/pgm?saved=missing").status_code == 404
    assert client.get("/pgm?preset=missing").status_code == 404

    assert "My PGM" in client.get("/pgm/saved").get_data(as_text=True)
    assert "My PGM" not in client.get("/pgm/runs").get_data(as_text=True)
    url = f"/pgm/saved/{saved['id']}"
    assert client.post(f"{url}/rename", data={"name": "Renamed"}).status_code == 302
    assert pgm_setup_store(tmp_path).load(saved["id"])["name"] == "Renamed"
    assert client.post(f"{url}/rename", data={"name": " "}).status_code == 422
    assert client.post(f"{url}/delete").status_code == 302
    assert pgm_setup_store(tmp_path).list() == []
    assert client.post(f"{url}/delete").status_code == 404


def test_runs_can_be_filtered_by_the_pgm_they_started_from(pgm_client, monkeypatch, tmp_path):
    _, client, _, _ = pgm_client
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    saved = client.post("/pgm/saved", data=_merged(pgm_name="Beamline A")).get_json()

    def run(**fields):
        run_id = _run_id(_submit(client, n_energy="6", n_c="4", **fields))
        _wait_for(client, run_id, "completed")
        return run_id

    from_saved = run(pgm_id=saved["id"], display_name="From saved")
    from_preset = run(pgm_id="sx700-example", display_name="From example")
    plain = run(display_name="Plain")
    unknown = run(pgm_id="missing", display_name="Unknown")

    store = pgm_store(tmp_path)
    assert store.load(from_saved)["pgm_name"] == "Beamline A"
    assert store.load(from_preset)["pgm_id"] == "sx700-example"
    assert store.load(plain)["pgm_id"] is None
    assert store.load(unknown)["pgm_id"] is None

    page = client.get("/pgm/runs").get_data(as_text=True)
    assert f'<option value="{saved["id"]}">Beamline A</option>' in page
    assert '<option value="sx700-example">' in page
    assert '<option value="none">No saved PGM</option>' in page
    assert f'data-run-filter-key="{saved["id"]}"' in page

    # A rename shows up in the filter, and a deleted PGM keeps the name the run recorded.
    url = f"/pgm/saved/{saved['id']}"
    client.post(f"{url}/rename", data={"name": "Beamline B"})
    assert f'<option value="{saved["id"]}">Beamline B</option>' in client.get(
        "/pgm/runs"
    ).get_data(as_text=True)
    client.post(f"{url}/delete")
    assert f'<option value="{saved["id"]}">Beamline A</option>' in client.get(
        "/pgm/runs"
    ).get_data(as_text=True)


def test_pgm_workers_follow_the_form_and_return_to_the_pool(pgm_client, monkeypatch, tmp_path):
    _, client, _, _ = pgm_client
    seen = []

    def recording_run(parameters):
        seen.append((parameters.rcwa_workers, parameters.worker_pids_callback is not None))
        return _fake_run(parameters)

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", recording_run)
    monkeypatch.setattr("grax.web.resource_manager._pool_total", lambda: 6)

    manual = _run_id(
        _submit(client, n_energy="6", n_c="4", max_workers_mode="manual", max_workers="2")
    )
    _wait_for(client, manual, "completed")
    status = client.get(f"/pgm/runs/{manual}/status").get_json()
    assert (status["worker_mode"], status["requested_workers"]) == ("manual", 2)
    assert status["resolved_workers"] == 2

    automatic = _run_id(_submit(client, n_energy="6", n_c="4", max_workers_mode="auto"))
    _wait_for(client, automatic, "completed")
    status = client.get(f"/pgm/runs/{automatic}/status").get_json()
    assert (status["worker_mode"], status["requested_workers"]) == ("auto", None)
    assert status["resolved_workers"] == 6

    assert seen == [(2, True), (6, True)]
    assert resource_status()["pool_used"] == 0
    assert _submit(client, max_workers_mode="manual", max_workers="0").status_code == 422


def test_failed_pgm_run_returns_its_workers(pgm_client, monkeypatch):
    _, client, _, _ = pgm_client

    def fails(parameters):
        raise RuntimeError("solver broke")

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", fails)
    run_id = _run_id(_submit(client))
    _wait_for(client, run_id, "failed")
    assert client.get(f"/pgm/runs/{run_id}/status").get_json()["error_text"] == "solver broke"
    assert resource_status()["pool_used"] == 0


def test_manage_pgm_runs_saves_comments_renames_and_bulk_deletes(pgm_client, monkeypatch, tmp_path):
    _, client, _, _ = pgm_client
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    store = pgm_store(tmp_path)
    first = _run_id(_submit(client, n_energy="6", n_c="4", comment="  First note "))
    second = _run_id(_submit(client, n_energy="6", n_c="4", display_name="Second"))
    for run_id in (first, second):
        _wait_for(client, run_id, "completed")
    assert store.load(first)["comment"] == "First note"

    page = client.get("/pgm/runs").get_data(as_text=True)
    assert "data-run-filter-key=" in page
    assert "Delete selected" in page
    assert f'/pgm/runs/{first}/export.csv' in page

    # Saving one card leaves the other card's typed fields alone.
    assert client.post(
        "/pgm/runs/manage",
        data={
            "save_run_id": first,
            f"display_name_{first}": "Renamed",
            f"comment_{first}": "New note",
            f"display_name_{second}": "Ignored",
        },
    ).status_code == 302
    assert store.load(first)["display_name"] == "Renamed"
    assert store.load(first)["comment"] == "New note"
    assert store.load(second)["display_name"] == "Second"

    assert client.post(
        "/pgm/runs/manage",
        data={"action": "delete", "selected_run_id": [first, "not-a-run"]},
    ).status_code == 302
    assert [run["id"] for run in store.list()] == [second]
    assert client.post(
        "/pgm/runs/manage",
        data={"action": "delete"},
        headers={"Origin": "http://evil.example"},
    ).status_code == 403


def test_bulk_delete_stops_a_running_pgm_job_first(pgm_client, monkeypatch, tmp_path):
    _, client, _, _ = pgm_client
    started = threading.Event()

    def wait_until_stopped(parameters):
        started.set()
        while not parameters.stop_event.wait(0.01):
            pass
        raise PGMCancelled("stopped")

    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", wait_until_stopped)
    run_id = _run_id(_submit(client))
    assert started.wait(5)
    assert client.post(
        "/pgm/runs/manage", data={"action": "delete", "selected_run_id": run_id}
    ).status_code == 302
    assert not (tmp_path / "pgm_runs" / run_id).exists()
    assert resource_status()["pool_used"] == 0


def test_custom_layers_round_trip_through_a_saved_pgm(pgm_client):
    _, client, _, _ = pgm_client
    fields = {
        **MIRROR_FORM,
        "stack_type": "custom",
        "cl_kind": ["single", "single"],
        "cl_material": ["Pt", "Cr"],
        "cl_density_g_cm3": ["21.45", "7.19"],
        "cl_thickness_nm": ["11", "3"],
        "cl_roughness_sigma_nm": ["", ""],
    }
    data = _merged(mirror_fields=fields, pgm_name="Layers")
    response = client.post("/pgm/saved", data=data)
    assert response.status_code == 200, response.get_data(as_text=True)
    page = client.get(response.get_json()["url"]).get_data(as_text=True)
    assert 'name="cl_thickness_nm" type="number" step="any" min="0" value="11"' in page


def test_afm_grating_keeps_its_own_profile(pgm_client, tmp_path, monkeypatch):
    _, client, _, _ = pgm_client
    monkeypatch.setattr("grax.web.pgm_jobs.run_working_range", _fake_run)
    source = tmp_path / "saved_gratings" / "afm" / "profile.csv"
    source.parent.mkdir(parents=True)
    x = np.linspace(0, 2500, 101)
    source.write_text(
        "\n".join(f"{a},{b}" for a, b in zip(x, 5 * np.sin(2 * np.pi * x / 2500), strict=True))
    )
    fields = {
        **GRATING_FIELDS,
        "grating_type": "afm",
        "afm_profile_path": str(source),
        "afm_period_nm": "2500",
    }
    response = client.post("/pgm/saved", data=_merged(grating_fields=fields, pgm_name="AFM"))
    assert response.status_code == 200, response.get_data(as_text=True)
    setup = pgm_setup_store(tmp_path).load(response.get_json()["id"])
    kept = Path(setup["grating_form"]["afm_profile_path"])
    assert kept == tmp_path / "saved_pgms" / setup["id"] / "profile.csv" and kept.is_file()

    run_id = _run_id(_submit(client, grating_fields=fields, n_energy="6", n_c="4"))
    _wait_for(client, run_id, "completed")
    snapshot = pgm_store(tmp_path).load(run_id)["grating_snapshot"]
    assert Path(snapshot["profile"]["profile_path"]).parent == tmp_path / "pgm_runs" / run_id

    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    outside.write_text(source.read_text())
    try:
        refused = _submit(client, grating_fields={**fields, "afm_profile_path": str(outside)})
    finally:
        outside.unlink()
    assert refused.status_code == 422

    client.post(f"/pgm/saved/{setup['id']}/delete")
    assert not kept.parent.exists()


def _layers(stack):
    return [
        (getattr(material, "name", material), pytest.approx(thickness))
        for material, thickness in stack.layer_sequence_bottom_up()
    ]


def test_sx700_preset_matches_the_colleagues_example(pgm_client):
    _, client, _, _ = pgm_client
    preset = PGM_PRESETS["sx700-example"]
    example_stack = grax.MultilayerStack(
        substrate_material="Si",
        material_a="Cr",
        material_b="C",
        d_period_nm=4.8,
        gamma=2.0 / 4.8,
        n_bilayers=40,
        top_material="C",
    )
    grating_form = {"roughness_kind": "none", **preset["grating_form"]}
    grating = build_grating_from_spec(_spec_from_form(grating_form))
    assert grating.period_lpermm == 2400
    assert (grating.blaze_angle_deg, grating.anti_blaze_angle_deg) == (0.9, 3.25)
    assert (grating.x_resolution_nm, grating.z_resolution_nm) == (1.0, 0.1)
    assert _layers(grating.resolved_stack()) == _layers(example_stack)

    mirror = _plane_mirror_stack(preset["mirror_form"])
    layers = _layers(mirror)
    assert len(layers) == 120
    assert layers[-1] == ("Cr", pytest.approx(2.0))
    assert layers[-2] == ("C", pytest.approx(2.8))
    assert layers[0] == ("C", pytest.approx(2.8))

    values = parse_pgm_form(preset["parameters"])
    assert values["diffraction_order"] == 2
    assert (values["n_energy"], values["n_c"]) == (501, 450)
    assert (values["rcwa_energy_nodes"], values["rcwa_cff_nodes"]) == (20, 10)
    assert (values["energy_min_eV"], values["energy_max_eV"]) == (1000, 6000)
    assert (values["c_min"], values["c_max"]) == (1.5, 10)
    assert client.get("/pgm?preset=sx700-example").status_code == 200
    assert mirror.material_a.density_g_cm3 == 7.19
    assert grating.resolved_stack().material_a.density_g_cm3 == 7.19


def test_new_pgm_page_opens_on_the_sx700_example(pgm_client):
    _, client, _, _ = pgm_client
    page = client.get("/pgm").get_data(as_text=True)
    assert 'value="SX700 example grating"' in page
    assert 'value="SX700 example mirror"' in page
    assert 'name="period_lpermm" type="number" step="1" value="2400"' in page
    assert 'name="n_energy" step="1" required value="501"' in page
    assert 'name="rcwa_energy_nodes" step="1" required value="20"' in page
    # Going back to the first picker entry restores the example optics.
    grating = client.get("/pgm/grating-fields").get_data(as_text=True)
    assert 'value="SX700 example grating"' in grating
    assert 'value="SX700 example mirror"' in client.get("/pgm/mirror-fields").get_data(as_text=True)


def test_pgm_geometry_section_explains_every_term(pgm_client):
    """Every geometry field keeps its name and has hover help, next to the drawing."""
    _, client, _, _ = pgm_client
    html = client.get("/pgm").get_data(as_text=True)
    assert 'class="pgm-geometry-drawing"' in html
    for name in (
        "mirror_length_mm", "grating_length_mm", "edge_margin_mm", "beam_offset_mm",
        "beam_height_mm", "mirror_grating_axis_dy_mm", "mirror_axis_dx_mm", "optic_dx_mm",
        "mirror_surface_axis_offset_mm", "mirror_downstream_edge_offset_mm",
        "theta_min_deg", "theta_max_deg", "beta_min_deg", "beta_max_deg", "crit_distance",
    ):
        assert html.count(f'name="{name}"') == 1
        tip = html.split(f'id="pgm-help-{name}">', 1)[1].split("</span>", 1)[0]
        assert tip.strip()
    optional = html.split('name="mirror_surface_axis_offset_mm"', 1)[1].split(">", 1)[0]
    assert "required" not in optional


def test_pgm_form_defaults_follow_the_library_except_the_quick_scan():
    assert PGM_DEFAULTS["mirror_length_mm"] == "600"
    assert PGM_DEFAULTS["crit_distance"] == "6.35"
    assert PGM_DEFAULTS["mirror_surface_axis_offset_mm"] == ""
    assert PGM_DEFAULTS["rcwa_solver"] == "rcwa"
    assert (PGM_DEFAULTS["n_energy"], PGM_DEFAULTS["n_c"]) == ("101", "81")
    assert PGM_DEFAULTS["rcwa_energy_nodes"] == "12"


def test_run_dir_rejects_unsafe_run_ids(tmp_path):
    store = pgm_store(tmp_path)
    assert store.run_dir("pgm-abc") == store.directory / "pgm-abc"
    with pytest.raises(ValueError):
        store.run_dir("../outside")
