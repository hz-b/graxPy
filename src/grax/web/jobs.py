"""Start, stop, finish and inspect background jobs.

Grating runs, plane-mirror runs and multilayer design stages all run as one
thread each, registered in the in-memory registry of :mod:`.run_state`. Only
this module creates, stops and finishes those entries, so every kind of job is
tracked, stopped and reported the same way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .run_files import _load_run_manifest, _update_manifest_fields
from .run_state import (
    ActiveRunState,
    _active_runs,
    _active_runs_lock,
    _elapsed_seconds,
    _is_active_run_entry_live,
    _terminate_run_processes,
    _wait_for_run_shutdown,
)

LIVE_STATES = frozenset({"queued", "running", "aborting"})


def start_job(
    app: Any,
    job_id: str,
    *,
    target: Callable[..., None],
    kwargs: dict[str, Any],
    thread_name: str,
    workflow: str,
    total_points: int,
    worker_mode: str = "auto",
    requested_workers: int | None = None,
    **state_fields: Any,
) -> ActiveRunState | None:
    """Register a job and start its worker thread, unless that job is already running.

    Checking, registering and starting happen under the registry lock, so two
    requests can never start the same job twice. Returns the new entry, or
    ``None`` when the job was already running.
    """

    with _active_runs_lock(app):
        existing = _active_runs(app).get(job_id)
        if existing is not None and _is_active_run_entry_live(existing):
            return None
        state = ActiveRunState(
            run_id=job_id,
            workflow=workflow,
            total_points=total_points,
            worker_mode=worker_mode,
            requested_workers=requested_workers,
            resolved_workers=None,
            **state_fields,
        )
        worker = threading.Thread(target=target, kwargs=kwargs, daemon=True, name=thread_name)
        state.worker_thread = worker
        _active_runs(app)[job_id] = state
        worker.start()
    return state


def stop_event(app: Any, job_id: str) -> threading.Event:
    """Return the event a job's worker watches to know it should stop."""

    with _active_runs_lock(app):
        entry = _active_runs(app).get(job_id)
    return entry.stop_event if entry is not None else threading.Event()


def request_stop(app: Any, job_id: str, *, delete: bool = False) -> bool:
    """Ask a queued or running job to stop; return whether a request was made."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(job_id)
        if active is None or active.state not in {"queued", "running"}:
            return False
        active.abort_requested = True
        active.delete_requested = delete
        active.stop_event.set()
        active.state = "aborting"
    return True


def wait_for_stop(app: Any, job_id: str) -> bool:
    """Stop a job's worker processes and wait for its thread; return whether it stopped."""

    _terminate_run_processes(app, job_id)
    return _wait_for_run_shutdown(app, job_id)


def stop_job(app: Any, job_id: str, *, delete: bool = False) -> None:
    """Stop a job if it is running and wait until its worker has exited."""

    request_stop(app, job_id, delete=delete)
    wait_for_stop(app, job_id)


def is_orphaned(app: Any, job_id: str, saved_status: Any) -> bool:
    """Return whether saved work says it is running although no worker in this app runs it.

    That happens when the app stopped while the job was running.
    """

    if saved_status not in LIVE_STATES:
        return False
    with _active_runs_lock(app):
        active = _active_runs(app).get(job_id)
        return active is None or active.state not in LIVE_STATES


def _finish_active_run(
    app: Any,
    run_id: str,
    *,
    state: str,
    plot_relative_path: str | None = None,
    error_text: str | None = None,
    completed_points: int | None = None,
) -> None:
    """Mark one job as finished."""

    elapsed_seconds: float | None = None
    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        if active is None:
            return
        active.state = state
        active.finished_at_monotonic = time.monotonic()
        active.completed_points = active.total_points if completed_points is None else completed_points
        if plot_relative_path is not None:
            active.plot_relative_path = plot_relative_path
            active.plot_token = uuid4().hex
        if error_text is not None:
            active.error_text = error_text
        active.simulation_pids = set()
        elapsed_seconds = _elapsed_seconds(active)

    # Ordinary runs have a manifest under ``runs/<run_id>``. Persist timing so
    # it survives registry cleanup and process restarts; design-stage registry
    # keys have no run manifest and are intentionally ignored here.
    data_dir = Path(app.config["GRAx_DATA_DIR"])
    if _load_run_manifest(data_dir, run_id) is not None:
        _update_manifest_fields(data_dir, run_id, elapsed_seconds=elapsed_seconds)
