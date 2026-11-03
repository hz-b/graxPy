# ruff: noqa: D100,D103

from __future__ import annotations

import io
import json
import shutil
import stat
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

import grax  # noqa: E402
from grax.web import transfer  # noqa: E402
from grax.web.app import create_app  # noqa: E402
from grax.web.multilayer_design_studies import (  # noqa: E402
    MultilayerDesignStudyStore,
    MultilayerGratingStore,
)
from grax.web.persistence import (  # noqa: E402
    GratingStore,
    PGMSetupStore,
    PlaneMirrorStore,
    grating_to_spec,
)
from grax.web.runs import RunStore  # noqa: E402
from grax.web.transfer_kinds import KINDS  # noqa: E402
from tests.unit.test_web_multilayer_design import _study_form  # noqa: E402
from tests.unit.test_web_pgm import GRATING_FIELDS, _merged  # noqa: E402
from tests.unit.test_web_plane_mirror import FORM as MIRROR_FORM  # noqa: E402

PLOT_BYTES = b"\x89PNG\r\n\x1a\nresult"


def _workspace(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return create_app(data_dir=path)


def _save_grating(data_dir: Path, name: str = "Shared grating") -> dict:
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
    return GratingStore(data_dir / "saved_gratings").save(grating_to_spec(grating, name=name))


def _write_profile(path: Path, amplitude: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 2500, 101)
    rows = (f"{a},{b}" for a, b in zip(x, amplitude * np.sin(2 * np.pi * x / 2500), strict=True))
    path.write_text("x_nm,z_nm\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _save_afm_grating(data_dir: Path, name: str = "Scan", amplitude: float = 5.0) -> dict:
    store = GratingStore(data_dir / "saved_gratings")
    spec = {**_save_grating(data_dir, name), "grating_type": "afm"}
    profile = store.directory / spec["id"] / "profile.csv"
    _write_profile(profile, amplitude)
    (profile.parent / "source_upload").write_bytes(b"raw scan")
    spec["profile"] = {
        "profile_path": str(profile),
        "source_filename": "scan.csv",
        "units": "nm",
        "profile_type": "topography",
        "period_nm": 2500,
        "import_settings": {},
    }
    return store.save(spec)


def _write_run(
    root: Path,
    run_id: str,
    *,
    status: str = "completed",
    extra: dict | None = None,
    files: dict[str, bytes] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "id": run_id,
        "created_at": "2026-01-02T03:04:05",
        "updated_at": "2026-01-02T03:04:05",
        "display_name": f"Run {run_id}",
        "workflow": "energy_sweep",
        "status": status,
        "artifacts": ["summary.csv", "selected_efficiency.png"],
        **(extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.csv").write_text("energy_ev,efficiency\n100,0.5\n", encoding="utf-8")
    (run_dir / "selected_efficiency.png").write_bytes(PLOT_BYTES)
    for name, content in (files or {}).items():
        target = run_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return run_dir


def _export(client, kind: str, object_id: str, run_ids: list[str]) -> bytes:
    response = client.post(f"/transfer/{kind}/{object_id}/export", data={"run_id": run_ids})
    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.mimetype == "application/zip"
    return response.get_data()


def _upload(client, package: bytes, name: str = "package.grax.zip"):
    return client.post(
        "/transfer/import",
        data={"package": (io.BytesIO(package), name)},
        content_type="multipart/form-data",
    )


def _stage(client, package: bytes) -> str:
    response = _upload(client, package)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["url"]


def _commit(client, url: str, run_ids: list[str], **form):
    return client.post(url, data={"run_id": run_ids, "mode": "copy", **form})


def _snapshot(data_dir: Path) -> dict[str, bytes]:
    """Return every saved file, leaving out the staging folder of uploads."""
    return {
        path.relative_to(data_dir).as_posix(): path.read_bytes()
        for path in data_dir.rglob("*")
        if path.is_file() and ".imports" not in path.parts
    }


def _build_package(
    *,
    kind: str = "grating",
    runs: list[dict] | None = None,
    spec: dict | None = None,
    extra_members: dict[str, bytes] | None = None,
    header: dict | None = None,
) -> bytes:
    """Build a package by hand, as someone other than this program might."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(transfer.PACKAGE_FILE, json.dumps({
            "format": "grax-package",
            "format_version": 1,
            "kind": kind,
            "grax_version": "0",
            "exported_at": "2026-01-01T00:00:00",
            "object": {"id": "imported", "name": "Imported"},
            "runs": [{"id": run["id"], "name": run["id"]} for run in runs or []],
            **(header or {}),
        }))
        archive.writestr("object/object.json", json.dumps(spec if spec is not None else _valid_spec()))
        for run in runs or []:
            archive.writestr(f"runs/{run['id']}/manifest.json", json.dumps(run))
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _valid_spec() -> dict:
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
    return {"id": "imported", **grating_to_spec(grating, name="Imported")}


def _valid_run(run_id: str = "run-a", **overrides) -> dict:
    return {
        "id": run_id,
        "status": "completed",
        "created_at": "2026-01-01T00:00:00",
        "workflow": "energy_sweep",
        "grating_id": "imported",
        **overrides,
    }


@pytest.fixture
def source(tmp_path: Path):
    app = _workspace(tmp_path / "source")
    return app, app.test_client(), tmp_path / "source"


@pytest.fixture
def target(tmp_path: Path):
    app = _workspace(tmp_path / "target")
    return app, app.test_client(), tmp_path / "target"


def test_grating_round_trip_keeps_results_and_shows_the_run(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-a", extra={"grating_id": grating["id"], "grating_name": grating["name"]})

    url = _stage(receiver, _export(sender, "grating", grating["id"], ["run-a"]))
    preview = receiver.get(url)
    assert preview.status_code == 200
    page = preview.get_data(as_text=True)
    assert "Import grating: Shared grating" in page
    assert 'value="run-a" checked' in page
    assert "may already be here" not in page

    response = _commit(receiver, url, ["run-a"])
    assert response.status_code == 200, response.get_data(as_text=True)
    assert "/gratings/manage" in response.get_json()["url"]
    assert GratingStore(target_dir / "saved_gratings").load(grating["id"])["name"] == "Shared grating"
    manifest = RunStore(target_dir / "runs").load("run-a")
    assert manifest["grating_id"] == grating["id"]
    run_dir = target_dir / "runs" / "run-a"
    assert (run_dir / "selected_efficiency.png").read_bytes() == PLOT_BYTES
    assert (run_dir / "summary.csv").read_bytes() == (source_dir / "runs" / "run-a" / "summary.csv").read_bytes()
    assert receiver.get("/runs/run-a").status_code == 200
    assert receiver.get(f"/gratings/{grating['id']}").status_code == 200
    assert receiver.get("/").status_code == 200
    assert not list((target_dir / ".imports").iterdir())


def test_plane_mirror_round_trip_with_saved_scans(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    mirror = PlaneMirrorStore(source_dir / "saved_plane_mirrors").save({
        "name": "Mirror",
        "stack_form": dict(MIRROR_FORM),
        "scans": [{"id": "scan-1", "name": "Scan", "label": "", "form": {"scan_mode": "energy"}}],
    })
    _write_run(source_dir / "runs", "plane-mirror-a", extra={
        "workflow": "plane_mirror", "mirror_id": mirror["id"], "mirror_name": "Mirror",
    })
    url = _stage(receiver, _export(sender, "plane_mirror", mirror["id"], ["plane-mirror-a"]))
    response = _commit(receiver, url, ["plane-mirror-a"])
    assert response.status_code == 200, response.get_data(as_text=True)
    saved = PlaneMirrorStore(target_dir / "saved_plane_mirrors").load(mirror["id"])
    assert saved["scans"][0]["id"] == "scan-1"
    assert RunStore(target_dir / "runs").load("plane-mirror-a")["mirror_id"] == mirror["id"]
    assert receiver.get(f"/plane-mirrors/{mirror['id']}").status_code == 200
    assert receiver.get("/plane-mirror-runs").status_code == 200


def test_pgm_round_trip_clears_the_senders_optics_and_drops_the_cache(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    response = sender.post("/pgm/saved", data=_merged(pgm_name="Beamline PGM"))
    assert response.status_code == 200, response.get_data(as_text=True)
    pgm_id = response.get_json()["id"]
    _write_run(source_dir / "pgm_runs", "pgm-a", extra={
        "workflow": "pgm", "pgm_id": pgm_id, "pgm_name": "Beamline PGM",
        "grating_id": "senders-grating", "mirror_id": "senders-mirror",
    }, files={"cache/rcwa_x.npz": b"cache", "maps.npz": b"maps"})

    package = _export(sender, "pgm", pgm_id, ["pgm-a"])
    assert not [name for name in zipfile.ZipFile(io.BytesIO(package)).namelist() if "/cache/" in name]
    response = _commit(receiver, _stage(receiver, package), ["pgm-a"])
    assert response.status_code == 200, response.get_data(as_text=True)
    assert PGMSetupStore(target_dir / "saved_pgms").load(pgm_id)["name"] == "Beamline PGM"
    manifest = RunStore(target_dir / "pgm_runs").load("pgm-a")
    assert (manifest["pgm_id"], manifest["grating_id"], manifest["mirror_id"]) == (pgm_id, None, None)
    assert (target_dir / "pgm_runs" / "pgm-a" / "maps.npz").read_bytes() == b"maps"


def test_multilayer_round_trip_brings_its_studies(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    assert sender.post("/multilayer-design", data=_study_form()).status_code == 302
    study = MultilayerDesignStudyStore(source_dir / "multilayer_designs").list()[0]
    grating = MultilayerGratingStore(source_dir / "multilayer_gratings").list()[0]
    runs = sender.get(f"/transfer/multilayer_grating/{grating['id']}/runs").get_json()["runs"]
    assert [run["id"] for run in runs] == [study["id"]]
    survey = source_dir / "multilayer_designs" / study["id"] / "survey" / "survey.csv"
    survey.parent.mkdir(parents=True)
    survey.write_text("d,blaze\n1,2\n", encoding="utf-8")

    url = _stage(receiver, _export(sender, "multilayer_grating", grating["id"], [study["id"]]))
    response = _commit(receiver, url, [study["id"]])
    assert response.status_code == 200, response.get_data(as_text=True)
    imported = MultilayerDesignStudyStore(target_dir / "multilayer_designs").load(study["id"])
    assert imported["grating_id"] == grating["id"]
    assert (target_dir / "multilayer_designs" / study["id"] / "survey" / "survey.csv").is_file()
    assert receiver.get("/multilayer-gratings").status_code == 200
    assert receiver.get(f"/multilayer-design/{study['id']}").status_code == 200


def test_afm_grating_and_run_survive_without_the_sender_and_hide_local_paths(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_afm_grating(source_dir)
    profile_path = grating["profile"]["profile_path"]
    _write_run(source_dir / "runs", "run-afm", extra={
        "grating_id": grating["id"],
        "grating_spec": grating,
        "error_text": f"Failed reading {source_dir / 'somewhere'}",
    }, files={
        "checkpoints/metadata.json": json.dumps({"summary": {"profile_path": profile_path}}).encode(),
        "checkpoints/results.jsonl": (json.dumps({"profile_path": profile_path}) + "\n").encode(),
        "error.txt": f"Traceback in {source_dir}".encode(),
    })

    package = _export(sender, "grating", grating["id"], ["run-afm"])
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        for name in archive.namelist():
            content = archive.read(name)
            assert str(source_dir).encode() not in content, name
            assert json.dumps(str(source_dir))[1:-1].encode() not in content, name

    response = _commit(receiver, _stage(receiver, package), ["run-afm"])
    assert response.status_code == 200, response.get_data(as_text=True)
    shutil.rmtree(source_dir)

    saved = GratingStore(target_dir / "saved_gratings").load(grating["id"])
    kept = Path(saved["profile"]["profile_path"])
    assert kept == target_dir / "saved_gratings" / grating["id"] / "profile.csv"
    assert kept.is_file() and (kept.parent / "source_upload").read_bytes() == b"raw scan"
    assert receiver.get(f"/gratings/{grating['id']}").status_code == 200

    manifest = RunStore(target_dir / "runs").load("run-afm")
    run_profile = Path(manifest["grating_spec"]["profile"]["profile_path"])
    assert run_profile.is_relative_to(target_dir / "runs" / "run-afm") and run_profile.is_file()
    checkpoint = json.loads((target_dir / "runs" / "run-afm" / "checkpoints" / "metadata.json").read_text())
    assert Path(checkpoint["summary"]["profile_path"]) == run_profile
    assert receiver.get("/runs/run-afm").status_code == 200


def test_afm_pgm_round_trip_keeps_its_profile(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    profile = source_dir / "saved_gratings" / "afm" / "profile.csv"
    _write_profile(profile)
    fields = {**GRATING_FIELDS, "grating_type": "afm", "afm_profile_path": str(profile), "afm_period_nm": "2500"}
    pgm_id = sender.post("/pgm/saved", data=_merged(grating_fields=fields, pgm_name="AFM")).get_json()["id"]

    response = _commit(receiver, _stage(receiver, _export(sender, "pgm", pgm_id, [])), [])
    assert response.status_code == 200, response.get_data(as_text=True)
    saved = PGMSetupStore(target_dir / "saved_pgms").load(pgm_id)
    assert Path(saved["grating_form"]["afm_profile_path"]) == target_dir / "saved_pgms" / pgm_id / "profile.csv"
    assert Path(saved["grating_form"]["afm_profile_path"]).is_file()


def test_only_the_selected_runs_travel(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    for run_id in ("run-a", "run-b", "run-c"):
        _write_run(source_dir / "runs", run_id, extra={"grating_id": grating["id"]})

    listed = sender.get(f"/transfer/grating/{grating['id']}/runs").get_json()["runs"]
    assert {run["id"] for run in listed} == {"run-a", "run-b", "run-c"}

    url = _stage(receiver, _export(sender, "grating", grating["id"], ["run-a", "run-b"]))
    page = receiver.get(url).get_data(as_text=True)
    assert "run-a" in page and "run-b" in page and "run-c" not in page
    _commit(receiver, url, ["run-b"])
    assert [run["id"] for run in RunStore(target_dir / "runs").list()] == ["run-b"]


def test_export_refuses_a_running_run_and_another_objects_run(source) -> None:
    _, sender, source_dir = source
    grating = _save_grating(source_dir)
    other = _save_grating(source_dir, "Other")
    _write_run(source_dir / "runs", "run-live", status="running", extra={"grating_id": grating["id"]})
    _write_run(source_dir / "runs", "run-other", extra={"grating_id": other["id"]})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("grax.web.transfer_kinds._is_run_active", lambda _app, run_id: run_id == "run-live")
        rows = sender.get(f"/transfer/grating/{grating['id']}/runs").get_json()["runs"]
        assert rows[0]["problem"] == "It is still running."
        refused = sender.post(f"/transfer/grating/{grating['id']}/export", data={"run_id": ["run-live"]})
        assert refused.status_code == 422
        assert "still running" in refused.get_json()["error"]

    refused = sender.post(f"/transfer/grating/{grating['id']}/export", data={"run_id": ["run-other"]})
    assert refused.status_code == 422


def test_a_run_left_running_by_a_stopped_app_arrives_aborted(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-stuck", status="running", extra={"grating_id": grating["id"]})
    url = _stage(receiver, _export(sender, "grating", grating["id"], ["run-stuck"]))
    assert _commit(receiver, url, ["run-stuck"]).status_code == 200
    manifest = RunStore(target_dir / "runs").load("run-stuck")
    assert manifest["status"] == "aborted"
    assert manifest["error_text"]


def test_afm_run_with_a_missing_profile_cannot_be_exported(source) -> None:
    _, sender, source_dir = source
    grating = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-gone", extra={
        "grating_id": grating["id"],
        "grating_spec": {"profile": {"profile_path": str(source_dir / "nowhere.csv")}},
    })
    rows = sender.get(f"/transfer/grating/{grating['id']}/runs").get_json()["runs"]
    assert "missing" in rows[0]["problem"]


def test_adding_a_copy_gets_new_ids_and_a_suffix(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-a", extra={"grating_id": grating["id"]})
    package = _export(sender, "grating", grating["id"], ["run-a"])

    assert _commit(receiver, _stage(receiver, package), ["run-a"]).status_code == 200
    url = _stage(receiver, package)
    page = receiver.get(url).get_data(as_text=True)
    assert "may already be here" in page
    assert "Already here" in page
    assert 'value="run-a" checked' not in page

    assert _commit(receiver, url, ["run-a"]).status_code == 200
    names = sorted(spec["name"] for spec in GratingStore(target_dir / "saved_gratings").list())
    assert names == ["Shared grating", "Shared grating (imported)"]
    copy = next(s for s in GratingStore(target_dir / "saved_gratings").list() if "(imported)" in s["name"])
    assert copy["id"] != grating["id"]
    assert RunStore(target_dir / "runs").load("run-a-2")["grating_id"] == copy["id"]
    assert RunStore(target_dir / "runs").load("run-a")["grating_id"] == grating["id"]


def test_replacing_keeps_local_runs_and_adds_imported_ones(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    sent = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-new", extra={"grating_id": sent["id"]})
    package = _export(sender, "grating", sent["id"], ["run-new"])

    local = _save_grating(target_dir)
    local_store = GratingStore(target_dir / "saved_gratings")
    local_store.save({**local, "period_lpermm": 999})
    _write_run(target_dir / "runs", "run-old", extra={"grating_id": local["id"]})

    response = _commit(receiver, _stage(receiver, package), ["run-new"], mode="replace", target_id=local["id"])
    assert response.status_code == 200, response.get_data(as_text=True)
    assert "Replaced" in response.get_json()["url"]
    assert local_store.load(local["id"])["period_lpermm"] == sent["period_lpermm"]
    assert len(local_store.list()) == 1
    assert {run["id"] for run in RunStore(target_dir / "runs").list()} == {"run-old", "run-new"}
    assert RunStore(target_dir / "runs").load("run-old")["grating_id"] == local["id"]


def test_replacing_an_afm_grating_keeps_the_scan_of_local_runs(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    sent = _save_afm_grating(source_dir, amplitude=9.0)
    package = _export(sender, "grating", sent["id"], [])

    local = _save_afm_grating(target_dir, amplitude=2.0)
    original_profile = Path(local["profile"]["profile_path"]).read_bytes()
    _write_run(target_dir / "runs", "run-old", extra={"grating_id": local["id"], "grating_spec": local})

    response = _commit(receiver, _stage(receiver, package), [], mode="replace", target_id=local["id"])
    assert response.status_code == 200, response.get_data(as_text=True)
    assert Path(local["profile"]["profile_path"]).read_bytes() != original_profile
    manifest = RunStore(target_dir / "runs").load("run-old")
    kept = Path(manifest["grating_spec"]["profile"]["profile_path"])
    assert kept.is_relative_to(target_dir / "runs" / "run-old")
    assert kept.read_bytes() == original_profile


def test_replacing_a_multilayer_grating_marks_local_results_stale(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    assert sender.post("/multilayer-design", data=_study_form()).status_code == 302
    sent = MultilayerGratingStore(source_dir / "multilayer_gratings").list()[0]
    package = _export(sender, "multilayer_grating", sent["id"], [])

    assert receiver.post("/multilayer-design", data=_study_form(gamma="0.6")).status_code == 302
    studies = MultilayerDesignStudyStore(target_dir / "multilayer_designs")
    local_study = studies.list()[0]
    local_study["stages"]["survey"]["status"] = "completed"
    studies.save(local_study)
    local = MultilayerGratingStore(target_dir / "multilayer_gratings").list()[0]

    response = _commit(receiver, _stage(receiver, package), [], mode="replace", target_id=local["id"])
    assert response.status_code == 200, response.get_data(as_text=True)
    updated = studies.load(local_study["id"])
    assert updated["stages"]["survey"]["status"] == "stale"
    assert updated["config"]["gamma"] == sent["config"]["gamma"]
    assert len(MultilayerGratingStore(target_dir / "multilayer_gratings").list()) == 1


def test_replacing_is_refused_while_a_run_is_active(source, target, monkeypatch) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    package = _export(sender, "grating", _save_grating(source_dir)["id"], [])
    local = _save_grating(target_dir)
    _write_run(target_dir / "runs", "run-live", status="running", extra={"grating_id": local["id"]})
    monkeypatch.setattr("grax.web.transfer_kinds._is_run_active", lambda _app, run_id: run_id == "run-live")
    before = _snapshot(target_dir)

    url = _stage(receiver, package)
    response = _commit(receiver, url, [], mode="replace", target_id=local["id"])
    assert response.status_code == 422
    assert "Stop" in response.get_json()["error"]
    assert response.get_json()["can_retry"] is True
    assert _snapshot(target_dir) == before
    assert receiver.get(url).status_code == 200


def _zip_with(member: str, content: bytes = b"x", *, mode: int | None = None) -> bytes:
    buffer = io.BytesIO(_build_package())
    with zipfile.ZipFile(buffer, "a") as archive:
        info = zipfile.ZipInfo(member)
        if mode is not None:
            info.external_attr = mode << 16
        archive.writestr(info, content)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        (b"this is not a zip file", "not a GRAx package"),
        (_zip_with("../escape.txt"), "leaves its folder"),
        (_zip_with("runs/a/../../../escape.csv"), "leaves its folder"),
        (_zip_with("/etc/escape.csv"), "absolute path"),
        (_zip_with("C:/escape.csv"), "reserved characters"),
        (_zip_with("runs\\a\\escape.csv"), "not a GRAx package"),
        (_zip_with("runs/run-a/NUL.csv"), "Windows reserves"),
        (_zip_with("object/link", mode=stat.S_IFLNK | 0o777), "link or special file"),
        (_zip_with("runs/run-a/page.html"), "unexpected entry"),
        (_zip_with("runs/run-a/drawing.svg"), "unexpected entry"),
        (_zip_with("object/extra.csv"), "unexpected entry"),
        (_zip_with("other/file.csv"), "unexpected entry"),
    ],
)
def test_unsafe_archives_are_refused_and_change_nothing(target, package: bytes, expected: str) -> None:
    _, receiver, target_dir = target
    before = _snapshot(target_dir)
    response = _upload(receiver, package)
    assert response.status_code == 422, response.get_data(as_text=True)
    assert expected in response.get_json()["error"]
    assert _snapshot(target_dir) == before
    assert not (target_dir / "escape.txt").exists()
    assert not (target_dir.parent / "escape.txt").exists()
    assert not list((target_dir / ".imports").iterdir())


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        (_build_package(header={"format": "other"}), "not a GRAx package"),
        (_build_package(header={"format_version": 99}), "newer version"),
        (_build_package(header={"kind": "toaster"}), "does not know"),
        (_build_package(header={"object": {"id": "../evil", "name": "x"}}), "object is invalid"),
        (_build_package(header={"runs": [{"id": "../evil"}]}), "run list is invalid"),
        (_build_package(spec={**_valid_spec(), "schema_version": 99}), "newer version"),
        (_build_package(spec={"name": "Broken"}), "is invalid"),
        (_build_package(spec=["not", "an", "object"]), "is invalid"),
        (_build_package(spec={**_valid_spec(), "profile": {"profile_path": "/etc/passwd"}}), "outside the package"),
        (_build_package(extra_members={"runs/unlisted/summary.csv": b"x"}), "unlisted runs"),
    ],
)
def test_invalid_packages_are_refused_with_a_message(target, package: bytes, expected: str) -> None:
    _, receiver, target_dir = target
    before = _snapshot(target_dir)
    response = _upload(receiver, package)
    assert response.status_code == 422, response.get_data(as_text=True)
    assert expected in response.get_json()["error"]
    assert _snapshot(target_dir) == before


def test_a_package_without_a_description_is_refused(target) -> None:
    _, receiver, _ = target
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("object/object.json", "{}")
    response = _upload(receiver, buffer.getvalue())
    assert response.status_code == 422
    assert "description is missing" in response.get_json()["error"]


def test_oversized_and_implausible_files_are_refused(target, monkeypatch) -> None:
    _, receiver, _ = target
    monkeypatch.setattr(transfer, "MAX_UPLOAD_BYTES", 10)
    response = _upload(receiver, _build_package())
    assert response.status_code == 422
    assert "too large" in response.get_json()["error"]
    monkeypatch.undo()

    monkeypatch.setattr(transfer, "MAX_MEMBERS", 1)
    response = _upload(receiver, _build_package())
    assert "too many files" in response.get_json()["error"]
    monkeypatch.undo()

    monkeypatch.setattr(transfer, "MAX_COMPRESSION_RATIO", 0)
    monkeypatch.setattr(transfer, "_RATIO_ALLOWANCE_BYTES", 0)
    response = _upload(receiver, _build_package())
    assert "implausible" in response.get_json()["error"]


def test_a_damaged_archive_is_refused(target) -> None:
    _, receiver, _ = target
    package = bytearray(_build_package())
    # Damage the first entry's data, keeping the directory at the end readable.
    package[40:60] = b"\xff" * 20
    response = _upload(receiver, bytes(package))
    assert response.status_code == 422
    assert response.get_json()["error"]


def test_upload_without_a_file_explains_itself(target) -> None:
    _, receiver, _ = target
    response = receiver.post("/transfer/import", data={}, content_type="multipart/form-data")
    assert response.status_code == 422
    assert "Choose a file" in response.get_json()["error"]


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (_valid_run(grating_spec={"profile": {"profile_path": "/etc/passwd"}}), "outside the package"),
        (_valid_run(grating_spec={"profile": {"profile_path": "grax-package://../../secret.csv"}}), "does not contain"),
        (_valid_run(artifacts=["../../secret.png"]), "outside its own folder"),
        (_valid_run(artifacts=["/etc/secret.png"]), "outside its own folder"),
        (_valid_run(artifacts="summary.csv"), "lists its result files"),
        (_valid_run(status=None), "no valid status"),
        (_valid_run(id="something-else"), "invalid"),
    ],
)
def test_a_bad_run_is_disabled_and_the_rest_stay_importable(target, run: dict, expected: str) -> None:
    _, receiver, target_dir = target
    good = _valid_run("run-good")
    bad = {**run, "id": run["id"]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(transfer.PACKAGE_FILE, json.dumps({
            "format": "grax-package", "format_version": 1, "kind": "grating",
            "object": {"id": "imported", "name": "Imported"},
            "runs": [{"id": "run-good"}, {"id": "run-bad"}],
        }))
        archive.writestr("object/object.json", json.dumps(_valid_spec()))
        archive.writestr("runs/run-good/manifest.json", json.dumps(good))
        archive.writestr("runs/run-bad/manifest.json", json.dumps(
            bad if bad["id"] == "something-else" else {**bad, "id": "run-bad"}
        ))
    url = _stage(receiver, buffer.getvalue())
    page = receiver.get(url).get_data(as_text=True)
    assert "Cannot be imported" in page

    assert expected in page or expected in page.replace("&#39;", "'")

    refused = _commit(receiver, url, ["run-good", "run-bad"])
    assert refused.status_code == 422
    assert "cannot be imported" in refused.get_json()["error"]
    assert not (target_dir / "runs" / "run-good").exists()

    assert _commit(receiver, url, ["run-good"]).status_code == 200
    assert (target_dir / "runs" / "run-good").is_dir()
    assert not (target_dir / "runs" / "run-bad").exists()
    assert receiver.get("/").status_code == 200


@pytest.mark.parametrize("failure", [OSError("disk full"), RuntimeError("something unexpected")])
def test_a_failure_midway_is_undone_reported_and_can_be_retried(source, target, monkeypatch, failure) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    _write_run(source_dir / "runs", "run-a", extra={"grating_id": grating["id"]})
    url = _stage(receiver, _export(sender, "grating", grating["id"], ["run-a"]))
    before = _snapshot(target_dir)

    def fail(self, data_dir, spec):
        raise failure

    monkeypatch.setattr(type(KINDS["grating"]), "write_object", fail)
    response = _commit(receiver, url, ["run-a"])
    assert response.status_code == 422, response.get_data(as_text=True)
    body = response.get_json()
    assert "nothing was changed" in body["error"] and str(failure) in body["error"]
    assert body["can_retry"] is True
    assert _snapshot(target_dir) == before
    assert not (target_dir / "runs" / "run-a").exists()
    assert receiver.get("/").status_code == 200
    assert receiver.get("/gratings/manage").status_code == 200

    monkeypatch.undo()
    assert _commit(receiver, url, ["run-a"]).status_code == 200
    assert (target_dir / "runs" / "run-a").is_dir()


def test_a_failed_replace_restores_the_local_object_and_files(source, target, monkeypatch) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    package = _export(sender, "grating", _save_afm_grating(source_dir, amplitude=9.0)["id"], [])
    local = _save_afm_grating(target_dir, amplitude=2.0)
    url = _stage(receiver, package)
    before = _snapshot(target_dir)

    calls = []
    real_write = type(KINDS["grating"]).write_object

    def fail_once(self, data_dir, spec):
        calls.append(spec)
        if len(calls) == 1:
            raise OSError("disk full")
        real_write(self, data_dir, spec)

    monkeypatch.setattr(type(KINDS["grating"]), "write_object", fail_once)
    assert _commit(receiver, url, [], mode="replace", target_id=local["id"]).status_code == 422
    monkeypatch.undo()
    after = _snapshot(target_dir)
    assert after[f"saved_gratings/{local['id']}/profile.csv"] == before[f"saved_gratings/{local['id']}/profile.csv"]
    assert after[f"saved_gratings/{local['id']}/source_upload"] == before[f"saved_gratings/{local['id']}/source_upload"]
    assert json.loads(after[f"saved_gratings/{local['id']}.json"]) == json.loads(
        before[f"saved_gratings/{local['id']}.json"]
    )


def test_a_failure_that_breaks_the_upload_says_to_choose_the_file_again(source, target, monkeypatch) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    url = _stage(receiver, _export(sender, "grating", _save_grating(source_dir)["id"], []))
    monkeypatch.setattr(type(KINDS["grating"]), "write_object", lambda *_: (_ for _ in ()).throw(OSError("full")))
    monkeypatch.setattr(transfer, "_unpack_package", lambda *_: (_ for _ in ()).throw(OSError("gone")))
    body = _commit(receiver, url, []).get_json()
    assert body["can_retry"] is False
    assert "Choose the file again" in body["error"]


def test_unexpected_errors_on_every_route_become_messages(source, monkeypatch) -> None:
    _, client, source_dir = source
    grating = _save_grating(source_dir)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    url = _stage(client, _export(client, "grating", grating["id"], []))
    monkeypatch.setattr("grax.web.routes.transfer.stage_upload", boom)
    monkeypatch.setattr("grax.web.routes.transfer.install_package", boom)
    monkeypatch.setattr("grax.web.routes.transfer.cancel_import", boom)
    monkeypatch.setattr("grax.web.routes.transfer.import_preview", boom)
    monkeypatch.setattr("grax.web.routes.transfer.export_runs", boom)
    monkeypatch.setattr("grax.web.routes.transfer.write_package", boom)

    for response in (
        _upload(client, b"x"),
        client.post(url, data={"mode": "copy"}),
        client.post(f"{url}/cancel"),
        client.get(f"/transfer/grating/{grating['id']}/runs"),
        client.post(f"/transfer/grating/{grating['id']}/export"),
    ):
        assert response.is_json
        assert "boom" in response.get_json()["error"]
    page = client.get(url)
    assert page.status_code == 410 and "boom" in page.get_data(as_text=True)


def test_broken_upload_stream_is_reported(target, monkeypatch) -> None:
    _, receiver, _ = target

    def boom(*_args, **_kwargs):
        raise OSError("connection lost")

    monkeypatch.setattr(transfer, "_save_upload", boom)
    response = _upload(receiver, b"anything")
    assert response.status_code == 422
    assert "connection lost" in response.get_json()["error"]


def test_expired_import_explains_itself(target) -> None:
    _, receiver, _ = target
    page = receiver.get("/transfer/import/0123456789abcdef")
    assert page.status_code == 410
    assert "no longer available" in page.get_data(as_text=True)
    response = _commit(receiver, "/transfer/import/0123456789abcdef", [])
    assert response.status_code == 422
    assert response.get_json()["can_retry"] is False


def _wait_for_run(client, run_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 90
    status: dict = {}
    while time.monotonic() < deadline:
        status = client.get(f"/runs/{run_id}/status").get_json()
        if status["state"] in {"completed", "failed", "aborted"}:
            break
        time.sleep(0.1)
    assert status["state"] == expected, status
    return status


def test_real_aborted_run_resumes_and_restarts_after_import(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, target_dir = target
    grating = _save_grating(source_dir)
    response = sender.post(f"/gratings/{grating['id']}/runs", data={
        "workflow": "energy_sweep",
        "energy_start_ev": "800",
        "energy_stop_ev": "810",
        "energy_points": "2",
        "grazing_angle_deg": "1.5",
        "diffraction_order": "1",
        "fourier_orders": "3",
        "polarization": "p",
        "run_x_resolution_nm": "4",
        "run_z_resolution_nm": "1",
        "max_workers_mode": "manual",
        "max_workers": "1",
    })
    assert response.status_code == 302, response.get_data(as_text=True)
    run_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    _wait_for_run(sender, run_id, "completed")

    # Leave the run as an abort would: one point done and no final results.
    run_dir = source_dir / "runs" / run_id
    results = run_dir / "checkpoints" / "results.jsonl"
    results.write_text(results.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    for name in ("summary.csv", "all_orders.csv"):
        (run_dir / name).unlink(missing_ok=True)
    RunStore(source_dir / "runs").update(run_id, lambda saved: saved.update(status="aborted"))

    url = _stage(receiver, _export(sender, "grating", grating["id"], [run_id]))
    assert "cannot be resumed" not in receiver.get(url).get_data(as_text=True)
    assert _commit(receiver, url, [run_id]).status_code == 200

    assert receiver.post(f"/runs/{run_id}/resume").status_code == 302
    status = _wait_for_run(receiver, run_id, "completed")
    assert status["completed_points"] == 2
    assert (target_dir / "runs" / run_id / "summary.csv").is_file()

    assert receiver.post(f"/runs/{run_id}/restart").status_code == 302
    _wait_for_run(receiver, run_id, "completed")


def test_preview_warns_when_afm_checkpoints_block_resuming(source, target) -> None:
    _, sender, source_dir = source
    _, receiver, _ = target
    grating = _save_afm_grating(source_dir)
    _write_run(source_dir / "runs", "run-afm", extra={"grating_id": grating["id"]}, files={
        "checkpoints/metadata.json": json.dumps(
            {"profile_path": grating["profile"]["profile_path"]}
        ).encode(),
    })
    url = _stage(receiver, _export(sender, "grating", grating["id"], ["run-afm"]))
    assert "cannot be resumed" in receiver.get(url).get_data(as_text=True)


def test_cancel_discards_the_upload(source) -> None:
    _, client, source_dir = source
    url = _stage(client, _export(client, "grating", _save_grating(source_dir)["id"], []))
    assert client.post(f"{url}/cancel").get_json()["url"] == "/"
    assert not list((source_dir / ".imports").iterdir())


def test_cross_site_posts_are_refused_on_every_route(source) -> None:
    _, client, source_dir = source
    grating = _save_grating(source_dir)
    headers = {"Origin": "https://evil.example"}
    for url in (
        f"/transfer/grating/{grating['id']}/export",
        "/transfer/import",
        "/transfer/import/0123456789abcdef",
        "/transfer/import/0123456789abcdef/cancel",
    ):
        assert client.post(url, headers=headers).status_code == 403


def test_every_manage_page_and_home_offer_export(source) -> None:
    _, client, source_dir = source
    grating = _save_grating(source_dir)
    mirror = PlaneMirrorStore(source_dir / "saved_plane_mirrors").save(
        {"name": "Mirror", "stack_form": dict(MIRROR_FORM), "scans": []}
    )
    pgm_id = client.post("/pgm/saved", data=_merged(pgm_name="PGM")).get_json()["id"]
    assert client.post("/multilayer-design", data=_study_form()).status_code == 302
    multilayer = MultilayerGratingStore(source_dir / "multilayer_gratings").list()[0]
    for page, kind, object_id in (
        ("/gratings/manage", "grating", grating["id"]),
        ("/plane-mirrors", "plane_mirror", mirror["id"]),
        ("/pgm/saved", "pgm", pgm_id),
        ("/multilayer-gratings", "multilayer_grating", multilayer["id"]),
    ):
        text = client.get(page).get_data(as_text=True)
        assert f"/transfer/{kind}/{object_id}/export" in text, page
        assert "data-object-export-dialog" in text
    home = client.get("/").get_data(as_text=True)
    assert "data-object-export-choose" in home and "data-object-import-form" in home
    catalog = client.get("/transfer/objects").get_json()
    assert {group["kind"]: len(group["objects"]) for group in catalog} == {
        "grating": 1, "plane_mirror": 1, "pgm": 1, "multilayer_grating": 1,
    }


def test_success_messages_appear_on_every_manage_page(source) -> None:
    _, client, _ = source
    for page in ("/gratings/manage", "/plane-mirrors", "/pgm/saved", "/multilayer-gratings"):
        text = client.get(page, query_string={"message": "Imported it", "kind": "success"}).get_data(as_text=True)
        assert "Imported it" in text, page


def test_unknown_kind_and_object_are_answered_in_json(source) -> None:
    _, client, _ = source
    assert client.get("/transfer/toaster/some-id/runs").status_code == 404
    response = client.get("/transfer/grating/missing/runs")
    assert response.status_code == 404
    assert response.get_json()["error"]
