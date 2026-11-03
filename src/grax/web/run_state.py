"""In-memory registry of runs and design stages executing in this server process."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import Any
from uuid import uuid4

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime.
    class _MissingPsutil:
        """Fallback object that keeps the psutil interface patchable in tests."""

        def virtual_memory(self) -> Any:
            """Raise a stable error when psutil is unavailable."""

            raise RuntimeError("psutil is not installed.")

    psutil = _MissingPsutil()


@dataclass
class ActiveRunState:
    """Track one live run while it executes in the background."""

    run_id: str
    workflow: str
    total_points: int
    worker_mode: str
    requested_workers: int | None
    resolved_workers: int | None
    state: str = "queued"
    created_at_monotonic: float = field(default_factory=time.monotonic)
    started_at_monotonic: float | None = None
    finished_at_monotonic: float | None = None
    completed_points: int = 0
    plot_relative_path: str | None = None
    plot_token: str = ""
    last_plot_publish_monotonic: float | None = None
    error_text: str = ""
    completion_timestamps: list[float] = field(default_factory=list)
    abort_requested: bool = False
    delete_requested: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: threading.Thread | None = None
    simulation_pids: set[int] = field(default_factory=set)
    # Points already on disk when the run started (a resumed checkpoint). They
    # count towards progress but not towards this run's rate, so the ETA has to
    # subtract them.
    resumed_points: int = 0


def _active_runs(app: Any) -> dict[str, ActiveRunState]:
    """Return the active in-memory run registry for one Flask app."""

    return app.extensions.setdefault("grax_active_runs", {})


def _active_runs_lock(app: Any) -> threading.Lock:
    """Return the lock guarding active-run state for one Flask app."""

    return app.extensions.setdefault("grax_active_runs_lock", threading.Lock())


def _terminate_run_processes(app: Any, run_id: str, *, grace_seconds: float = 3.0) -> None:
    """Stop one active run's simulation subprocesses, children first."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        pids = set() if active is None else set(active.simulation_pids)
    pids.discard(os.getpid())
    if not pids or not hasattr(psutil, "Process"):
        return
    targets: list[Any] = []
    for pid in pids:
        try:
            process = psutil.Process(pid)
        except Exception:
            continue
        try:
            targets.extend(process.children(recursive=True))
        except Exception:
            pass
        targets.append(process)
    for process in targets:
        try:
            process.terminate()
        except Exception:
            continue
    try:
        _, alive = psutil.wait_procs(targets, timeout=grace_seconds)
    except Exception:
        return
    for process in alive:
        try:
            process.kill()
        except Exception:
            continue


def _is_abort_requested(app: Any, run_id: str) -> bool:
    """Return whether one active run has a pending abort request."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        return active is not None and active.abort_requested


def _is_delete_requested(app: Any, run_id: str) -> bool:
    """Return whether one active run should be deleted after abort."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        return active is not None and active.delete_requested


def _wait_for_run_shutdown(app: Any, run_id: str, *, timeout_seconds: float = 10.0) -> bool:
    """Wait briefly for one active run worker to stop."""

    deadline = time.monotonic() + timeout_seconds
    worker: threading.Thread | None = None
    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        if active is not None:
            worker = active.worker_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=max(deadline - time.monotonic(), 0.0))
    return not _is_run_active(app, run_id)


def _update_active_run(
    app: Any,
    run_id: str,
    *,
    state: str | None = None,
    started: bool = False,
    completed_increment: int = 0,
    completed_points: int | None = None,
    resolved_workers: int | None = None,
    plot_relative_path: str | None = None,
    bump_plot_token: bool = False,
    error_text: str | None = None,
    simulation_pids: set[int] | None = None,
) -> None:
    """Apply a thread-safe update to one active run state."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        if active is None:
            return
        if state is not None:
            active.state = state
        if started and active.started_at_monotonic is None:
            active.started_at_monotonic = time.monotonic()
        if completed_points is not None:
            active.completed_points = completed_points
        if completed_increment:
            active.completed_points += completed_increment
            active.completion_timestamps.append(time.monotonic())
        if resolved_workers is not None:
            active.resolved_workers = resolved_workers
        if plot_relative_path is not None:
            active.plot_relative_path = plot_relative_path
        if bump_plot_token:
            active.plot_token = uuid4().hex
        if error_text is not None:
            active.error_text = error_text
        if simulation_pids is not None:
            active.simulation_pids = set(simulation_pids)


def _is_active_run_entry_live(active: ActiveRunState) -> bool:
    """Return whether one in-memory active-run entry still has a live worker."""

    if active.state not in {"queued", "running", "aborting"}:
        return False
    worker = active.worker_thread
    return worker is not None and worker.is_alive()


def _elapsed_seconds(active: ActiveRunState) -> float | None:
    """Return elapsed time in seconds for one active run."""

    anchor = active.started_at_monotonic or active.created_at_monotonic
    finished = active.finished_at_monotonic or time.monotonic()
    return max(finished - anchor, 0.0)


def _eta_seconds(active: ActiveRunState) -> float | None:
    """Return a simple completion ETA from observed completion rate."""

    if active.completed_points < 1 or active.started_at_monotonic is None:
        return None
    elapsed = max(time.monotonic() - active.started_at_monotonic, 1e-6)
    rate = active.completed_points / elapsed
    if rate <= 0.0:
        return None
    remaining = max(active.total_points - active.completed_points, 0)
    return remaining / rate


def _cleanup_finished_runs(app: Any, *, retention_seconds: float = 300.0) -> None:
    """Drop finished active runs from memory after a short retention window."""

    cutoff = time.monotonic() - retention_seconds
    with _active_runs_lock(app):
        stale_ids = [
            run_id
            for run_id, active in _active_runs(app).items()
            if active.finished_at_monotonic is not None and active.finished_at_monotonic < cutoff
        ]
        for run_id in stale_ids:
            _active_runs(app).pop(run_id, None)


def _has_active_runs(app: Any) -> bool:
    """Return whether any run is still queued or running."""

    with _active_runs_lock(app):
        return any(_is_active_run_entry_live(active) for active in _active_runs(app).values())


def _is_run_active(app: Any, run_id: str) -> bool:
    """Return whether one specific run is still active."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        return active is not None and _is_active_run_entry_live(active)
