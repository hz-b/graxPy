"""Progress and memory readouts for the run status endpoints."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import _file_token
from .run_files import (
    _checkpoint_counts,
    _load_run_manifest,
    _preferred_run_plot_path,
    _run_completed_points,
    _run_plot_url,
)
from .run_state import (
    _active_runs,
    _active_runs_lock,
    _cleanup_finished_runs,
    _elapsed_seconds,
    _eta_seconds,
    _is_active_run_entry_live,
    psutil,
)


def _run_status_payload(*, app: Any, data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Return the current live-status payload for one run."""

    _cleanup_finished_runs(app)
    manifest = _load_run_manifest(data_dir, run_id)
    checkpoint_completed_points = 0
    checkpoint_total_points = 0
    if manifest is not None:
        checkpoint_completed_points, checkpoint_total_points = _checkpoint_counts(data_dir / "runs" / run_id)
    active_payload: dict[str, Any] | None = None
    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        if active is not None and _is_active_run_entry_live(active):
            elapsed = _elapsed_seconds(active)
            eta = _eta_seconds(active)
            active_payload = {
                "run_id": run_id,
                "state": active.state,
                "completed_points": active.completed_points,
                "total_points": active.total_points,
                "remaining_points": max(active.total_points - active.completed_points, 0),
                "checkpoint_completed_points": checkpoint_completed_points,
                "checkpoint_total_points": checkpoint_total_points or active.total_points,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "worker_mode": active.worker_mode,
                "requested_workers": active.requested_workers,
                "resolved_workers": active.resolved_workers,
                "plot_url": _run_plot_url(
                    data_dir=data_dir,
                    run_id=run_id,
                    relative_path=active.plot_relative_path,
                    token=active.plot_token,
                ),
                "plot_token": active.plot_token,
                "error_text": active.error_text,
                "can_abort": active.state in {"queued", "running"} and not active.abort_requested,
                "active_job_missing": False,
            }

    if active_payload is not None:
        return active_payload

    if manifest is None:
        return None
    manifest_state = str(manifest.get("status", "completed"))
    completed_points = _run_completed_points(manifest, checkpoint_completed_points)
    total_points = int(manifest.get("total_points") or checkpoint_total_points or completed_points)
    normalized_state = "completed" if manifest_state == "ok" else manifest_state
    can_abort = manifest_state not in {"ok", "completed", "failed", "aborted"}
    active_job_missing = normalized_state in {"queued", "running", "aborting"}
    if active_job_missing:
        normalized_state = "aborted"
    elif normalized_state in {"paused", "interrupted"}:
        normalized_state = "aborted"
    return {
        "run_id": run_id,
        "state": normalized_state,
        "completed_points": completed_points,
        "total_points": total_points,
        "remaining_points": max(total_points - completed_points, 0),
        "checkpoint_completed_points": checkpoint_completed_points,
        "checkpoint_total_points": checkpoint_total_points or total_points,
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "eta_seconds": 0.0 if normalized_state in {"completed", "failed", "aborted"} else None,
        "worker_mode": manifest.get("worker_mode", "auto"),
        "requested_workers": manifest.get("requested_workers"),
        "resolved_workers": manifest.get("resolved_workers"),
        "plot_url": _run_plot_url(
            data_dir=data_dir,
            run_id=run_id,
            relative_path=_preferred_run_plot_path(data_dir, run_id),
            token=_file_token(data_dir / "runs" / run_id / _preferred_run_plot_path(data_dir, run_id))
            if _preferred_run_plot_path(data_dir, run_id)
            else "",
        ),
        "plot_token": "",
        "error_text": manifest.get("error_text", ""),
        "can_abort": can_abort,
        "active_job_missing": active_job_missing,
    }


def _memory_status_payload() -> dict[str, Any]:
    """Return live memory metrics for the current machine."""

    try:
        memory = psutil.virtual_memory()
        return {
            "ok": True,
            "total_bytes": int(memory.total),
            "used_bytes": int(memory.used),
            "available_bytes": int(memory.available),
            "percent_used": float(memory.percent),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception:
        fallback_payload = _memory_status_from_proc_meminfo()
        if fallback_payload is not None:
            return fallback_payload
        return {"ok": False, "error": "Memory metrics are unavailable."}


def _simulation_process_memory_payload(app: Any, run_id: str) -> dict[str, Any]:
    """Return current RSS for the web process and one run's worker pool."""

    if not hasattr(psutil, "Process"):
        return {
            "ok": False,
            "web_process_rss_bytes": None,
            "simulation_process_rss_bytes": None,
        }
    try:
        current_process = psutil.Process(os.getpid())
        web_process_rss_bytes = int(current_process.memory_info().rss)
        with _active_runs_lock(app):
            active = _active_runs(app).get(run_id)
            worker_pids = set() if active is None else set(active.simulation_pids)
        simulation_process_rss_bytes = 0
        for pid in worker_pids:
            try:
                simulation_process_rss_bytes += int(psutil.Process(pid).memory_info().rss)
            except Exception:
                continue
        return {
            "ok": True,
            "web_process_rss_bytes": web_process_rss_bytes,
            "simulation_process_rss_bytes": simulation_process_rss_bytes,
        }
    except Exception:
        return {
            "ok": False,
            "web_process_rss_bytes": None,
            "simulation_process_rss_bytes": None,
        }


def _memory_status_from_proc_meminfo() -> dict[str, Any] | None:
    """Return memory metrics parsed from Linux /proc/meminfo when available."""

    meminfo_path = Path("/proc/meminfo")
    try:
        payload = meminfo_path.read_text(encoding="utf-8")
    except OSError:
        return None

    values_kb: dict[str, int] = {}
    for line in payload.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        number = raw_value.strip().split(" ", 1)[0]
        try:
            values_kb[key] = int(number)
        except ValueError:
            continue

    total_kb = values_kb.get("MemTotal")
    available_kb = values_kb.get("MemAvailable", values_kb.get("MemFree"))
    if total_kb is None or available_kb is None or total_kb <= 0:
        return None
    used_kb = max(total_kb - available_kb, 0)
    return {
        "ok": True,
        "total_bytes": int(total_kb * 1024),
        "used_bytes": int(used_kb * 1024),
        "available_bytes": int(available_kb * 1024),
        "percent_used": float((used_kb / total_kb) * 100.0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
