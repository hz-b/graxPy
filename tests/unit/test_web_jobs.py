# ruff: noqa: D100,D103

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("flask")

from grax.web import files, jobs, plane_mirror_runs  # noqa: E402
from grax.web.app import create_app  # noqa: E402
from grax.web.multilayer_design_studies import MultilayerDesignStudyStore  # noqa: E402
from grax.web.multilayer_jobs import _multilayer_design_job_key  # noqa: E402
from grax.web.run_state import _is_run_active  # noqa: E402
from grax.web.runs import RunStore  # noqa: E402
from tests.unit.test_web_multilayer_design import _create_study  # noqa: E402
from tests.unit.test_web_plane_mirror import CUSTOM_FORM  # noqa: E402

MIRROR_SCAN = {
    "scan_mode": "energy_sweep",
    "energy_min": "200",
    "energy_max": "300",
    "energy_points": "3",
    "fixed_angle_deg": "0.4",
    "grading_mode": "none",
}


def _wait_until(condition, timeout: float = 10.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _start_waiting_job(app, job_id: str, *, started: threading.Event | None = None):  # noqa: ANN001, ANN202
    """Start a job that runs until it is asked to stop."""

    def work() -> None:
        if started is not None:
            started.set()
        jobs.stop_event(app, job_id).wait(10)
        jobs._finish_active_run(app, job_id, state="aborted")

    return jobs.start_job(
        app, job_id, target=work, kwargs={}, thread_name=f"test-{job_id}", workflow="test", total_points=1
    )


def test_a_job_cannot_be_started_twice_at_once(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    barrier = threading.Barrier(8)
    started: list[object] = []

    def start() -> None:
        barrier.wait()
        started.append(_start_waiting_job(app, "same-job"))

    threads = [threading.Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len([state for state in started if state is not None]) == 1
    jobs.stop_job(app, "same-job")
    assert not _is_run_active(app, "same-job")


def test_stop_job_waits_for_the_worker_to_exit(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    started = threading.Event()
    state = _start_waiting_job(app, "job", started=started)
    started.wait(5)

    jobs.stop_job(app, "job")

    assert not state.worker_thread.is_alive()
    assert state.state == "aborted"
    assert state.abort_requested


@pytest.mark.parametrize(
    ("saved_status", "running", "expected"),
    [
        ("running", False, True),
        ("queued", False, True),
        ("aborting", False, True),
        ("running", True, False),
        ("completed", False, False),
        ("failed", False, False),
    ],
)
def test_work_left_running_by_a_stopped_app_is_orphaned(tmp_path: Path, saved_status, running, expected) -> None:  # noqa: ANN001
    app = create_app(data_dir=tmp_path)
    if running:
        started = threading.Event()
        _start_waiting_job(app, "job", started=started)
        started.wait(5)

    assert jobs.is_orphaned(app, "job", saved_status) is expected
    jobs.stop_job(app, "job")


def test_concurrent_manifest_updates_are_not_lost(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    store.save({"id": "run", "count": 0})

    def bump() -> None:
        for _ in range(20):
            store.update("run", lambda manifest: manifest.update(count=manifest["count"] + 1))

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.load("run")["count"] == 160
    assert sorted(path.name for path in (tmp_path / "runs" / "run").iterdir()) == ["manifest.json"]


def test_replacing_a_file_retries_while_windows_holds_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, destination = tmp_path / "new.txt", tmp_path / "old.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    real_replace = files.os.replace
    attempts: list[int] = []

    def flaky_replace(src, dst):  # noqa: ANN001, ANN202
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(13, "The process cannot access the file")
        real_replace(src, dst)

    monkeypatch.setattr(files.os, "replace", flaky_replace)
    monkeypatch.setattr(files.os, "name", "nt")

    files.replace_file(source, destination)

    assert len(attempts) == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_replacing_a_file_that_another_program_keeps_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An image viewer or editor holding the plot or results must not make the save fail."""

    source, destination = tmp_path / "new.txt", tmp_path / "old.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old content", encoding="utf-8")
    monkeypatch.setattr(files, "_RETRY_SECONDS", 0.05)

    with destination.open("rb") as reader:
        files.replace_file(source, destination)
        # Windows cannot swap an open file, so the new content is written into it;
        # elsewhere the swap succeeds and the open handle keeps the old file.
        assert reader.read() == (b"new" if os.name == "nt" else b"old content")

    assert destination.read_text(encoding="utf-8") == "new"
    assert not source.exists()


@pytest.mark.skipif(not hasattr(__import__("ctypes"), "windll"), reason="Windows file locks")
def test_loading_a_manifest_waits_while_windows_holds_it(tmp_path: Path) -> None:
    """A page that loads a run while its job replaces the manifest must not fail."""

    import ctypes
    from ctypes import wintypes

    store = RunStore(tmp_path / "runs")
    store.save({"id": "run", "status": "running"})
    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.restype = wintypes.HANDLE
    generic_read, no_sharing, open_existing = 0x80000000, 0, 3
    handle = create_file(
        str(tmp_path / "runs" / "run" / "manifest.json"), generic_read, no_sharing, None, open_existing, 0, None
    )
    assert handle != wintypes.HANDLE(-1).value
    releaser = threading.Timer(0.3, ctypes.windll.kernel32.CloseHandle, args=(wintypes.HANDLE(handle),))
    releaser.start()
    try:
        assert store.load("run")["status"] == "running"
    finally:
        releaser.join()


def test_replacing_a_file_does_not_retry_permission_errors_off_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def denied(src, dst):  # noqa: ANN001, ANN202
        raise PermissionError(13, "denied")

    monkeypatch.setattr(files.os, "replace", denied)
    monkeypatch.setattr(files.os, "name", "posix")

    with pytest.raises(PermissionError):
        files.replace_file(tmp_path / "a", tmp_path / "b")


def test_remove_tree_clears_read_only_files(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "locked.txt").write_text("x", encoding="utf-8")
    (folder / "locked.txt").chmod(0o444)

    files.remove_tree(folder)
    files.remove_tree(folder)

    assert not folder.exists()
    assert list(tmp_path.iterdir()) == []


def _save_finished_grating_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "done-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"id": "done-run", "grating_name": "G", "status": "completed", "workflow": "energy_sweep"}),
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "summary.csv").write_text("energy_ev\n100\n", encoding="utf-8")
    return run_dir


@pytest.mark.skipif(not hasattr(__import__("ctypes"), "windll"), reason="only Windows refuses to delete open files")
def test_deleting_a_run_whose_csv_is_open_elsewhere_keeps_it_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(files, "_RETRY_SECONDS", 0.05)
    run_dir = _save_finished_grating_run(tmp_path)
    client = create_app(data_dir=tmp_path).test_client()
    delete = {"action": "delete", "selected_run_id": "done-run", "confirm_delete": "1"}

    with (run_dir / "summary.csv").open("rb"):
        response = client.post("/runs/gratings", data=delete)

    assert response.status_code == 409
    assert "Close it and try again" in response.get_data(as_text=True)
    assert sorted(path.name for path in run_dir.iterdir()) == ["checkpoints", "manifest.json", "summary.csv"]
    assert [path.name for path in (tmp_path / "runs").iterdir()] == ["done-run"]

    assert client.post("/runs/gratings", data=delete).status_code == 302
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.skipif(not hasattr(__import__("ctypes"), "windll"), reason="only Windows refuses to delete open files")
def test_restarting_a_run_whose_csv_is_open_elsewhere_asks_to_close_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(files, "_RETRY_SECONDS", 0.05)
    run_dir = _save_finished_grating_run(tmp_path)
    client = create_app(data_dir=tmp_path).test_client()

    with (run_dir / "summary.csv").open("rb"):
        response = client.post("/runs/done-run/restart")

    assert response.status_code == 409
    assert "Close it and try again" in response.get_data(as_text=True)
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def _start_mirror_run(client, monkeypatch: pytest.MonkeyPatch, release: threading.Event) -> str:  # noqa: ANN001
    real_compute = plane_mirror_runs.compute_plane_mirror

    def slow_compute(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        release.wait(10)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(plane_mirror_runs, "compute_plane_mirror", slow_compute)
    response = client.post("/plane-mirrors", data={**CUSTOM_FORM, "mirror_name": "Job mirror", "scan_name": "E"})
    mirror_id = response.headers["Location"].split("/plane-mirrors/")[1].split("?")[0]
    response = client.post(f"/plane-mirrors/{mirror_id}/runs", data={**MIRROR_SCAN, "run_name": "Job run"})
    assert response.status_code == 302
    return response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]


def test_plane_mirror_runs_are_tracked_like_other_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    release = threading.Event()
    run_id = _start_mirror_run(client, monkeypatch, release)

    try:
        assert _is_run_active(app, run_id)
        # Two resumes at once must not start a second worker.
        assert client.post(f"/plane-mirror-runs/{run_id}/resume").status_code == 409
        # The database folder cannot be switched while it runs.
        switched = client.post("/workspace/pick", environ_base={"REMOTE_ADDR": "127.0.0.1"})
        assert "Cannot change the database folder" in switched.headers["Location"].replace("+", " ")
    finally:
        release.set()
    _wait_until(lambda: client.get(f"/plane-mirror-runs/{run_id}/status").get_json()["status"] == "completed")
    assert not _is_run_active(app, run_id)


def test_renaming_a_running_plane_mirror_run_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    release = threading.Event()
    run_id = _start_mirror_run(client, monkeypatch, release)
    _wait_until(lambda: RunStore(tmp_path / "runs").load(run_id)["status"] == "running")

    RunStore(tmp_path / "runs").rename(run_id, "Renamed while running")
    release.set()
    _wait_until(lambda: client.get(f"/plane-mirror-runs/{run_id}/status").get_json()["status"] == "completed")

    assert RunStore(tmp_path / "runs").load(run_id)["display_name"] == "Renamed while running"


def test_aborting_a_plane_mirror_run_stops_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    release = threading.Event()
    run_id = _start_mirror_run(client, monkeypatch, release)

    client.post(f"/plane-mirror-runs/{run_id}/abort")
    assert RunStore(tmp_path / "runs").load(run_id)["status"] == "aborting"
    release.set()

    _wait_until(lambda: client.get(f"/plane-mirror-runs/{run_id}/status").get_json()["status"] == "aborted")
    assert not _is_run_active(app, run_id)


def test_a_failed_plane_mirror_run_reads_as_failed_also_when_saved_as_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = create_app(data_dir=tmp_path)
    client = app.test_client()

    def broken(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ValueError("solver exploded")

    release = threading.Event()
    release.set()
    run_id = _start_mirror_run(client, monkeypatch, release)
    monkeypatch.setattr(plane_mirror_runs, "compute_plane_mirror", broken)
    client.post(f"/plane-mirror-runs/{run_id}/restart")
    _wait_until(lambda: client.get(f"/plane-mirror-runs/{run_id}/status").get_json()["status"] == "failed")
    assert RunStore(tmp_path / "runs").load(run_id)["status"] == "failed"

    RunStore(tmp_path / "runs").update(run_id, lambda manifest: manifest.update(status="error"))
    assert client.get(f"/plane-mirror-runs/{run_id}/status").get_json()["status"] == "failed"
    assert client.post(f"/plane-mirror-runs/{run_id}/resume").status_code == 302


def test_deleting_a_running_grating_run_stops_it_first(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    run_dir = tmp_path / "runs" / "busy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"id": "busy-run", "grating_name": "G", "status": "running", "workflow": "energy_sweep"}),
        encoding="utf-8",
    )
    started = threading.Event()
    state = _start_waiting_job(app, "busy-run", started=started)
    started.wait(5)

    response = app.test_client().post(
        "/runs/gratings", data={"action": "delete", "selected_run_id": "busy-run", "confirm_delete": "1"}
    )

    assert response.status_code == 302
    assert not state.worker_thread.is_alive()
    assert not run_dir.exists()


def test_deleting_a_study_stops_its_running_stage_first(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    study_id = _create_study(client)
    started = threading.Event()
    state = _start_waiting_job(app, _multilayer_design_job_key(study_id, "survey"), started=started)
    started.wait(5)

    client.post(f"/multilayer-design/{study_id}/delete")

    assert not state.worker_thread.is_alive()
    assert not MultilayerDesignStudyStore(tmp_path / "multilayer_designs").study_dir(study_id).exists()
