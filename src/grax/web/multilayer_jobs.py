"""Background execution and status of multilayer design-study stages."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    _file_token,
    _relative_to,
    _unlink_relative,
)
from .multilayer_design_studies import (
    MultilayerDesignStudyStore,
    build_design_config,
    downstream_stages,
    resolve_designs,
    survey_cell_count,
    survey_design_options,
)
from .jobs import (
    _finish_active_run,
    is_orphaned,
    start_job,
    stop_job,
)
from .jobs import stop_event as job_stop_event
from .run_state import (
    _active_runs,
    _active_runs_lock,
    _cleanup_finished_runs,
    _elapsed_seconds,
    _eta_seconds,
    _is_active_run_entry_live,
    _update_active_run,
)


# --------------------------------------------------------------------------- #
# Multilayer-design studies                                                    #
# --------------------------------------------------------------------------- #
def _multilayer_design_job_key(study_id: str, stage: str) -> str:
    """Registry key for one running design-study stage."""

    return f"mldesign:{study_id}:{stage}"


ORPHANED_STAGE_ERROR = "The app stopped before this stage finished."


def _mark_orphaned_stages(app: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Show stages left "running" by an app that has since stopped as aborted.

    Only the in-memory manifest changes; it is saved when the study is next edited or run.
    """

    study_id = str(manifest.get("id", ""))
    for stage, stage_state in (manifest.get("stages") or {}).items():
        if is_orphaned(app, _multilayer_design_job_key(study_id, stage), stage_state.get("status")):
            stage_state.update(status="aborted", error_text=ORPHANED_STAGE_ERROR)
    return manifest


def _multilayer_design_stage_total(manifest: dict[str, Any], stage: str) -> int:
    """Best-effort count of scan items for one stage, for the progress bar."""

    if stage == "survey":
        return survey_cell_count(manifest["config"])
    return len(manifest["stages"][stage].get("designs") or [])


def _start_multilayer_design_worker(
    *, app: Any, data_dir: Path, study_id: str, stage: str
) -> None:
    """Register an active-run entry for a design stage and start its worker thread."""

    store = MultilayerDesignStudyStore(data_dir / "multilayer_designs")
    manifest = store.load(study_id)

    key = _multilayer_design_job_key(study_id, stage)
    resumed_points = 0
    if stage == "energy_scan":
        resumed_points = _energy_scan_checkpoint_progress(
            study_dir=store.study_dir(study_id),
            designs=manifest["stages"][stage].get("designs") or [],
            energy_points=int(manifest["config"].get("energy_scan_points") or 0),
        )[0]
    start_job(
        app,
        key,
        target=_execute_multilayer_design_job,
        kwargs={"app": app, "data_dir": data_dir, "study_id": study_id, "stage": stage},
        thread_name=f"grax-mldesign-{study_id}-{stage}",
        workflow=f"multilayer_design_{stage}",
        total_points=_multilayer_design_stage_total(manifest, stage),
        resumed_points=resumed_points,
    )


def _execute_multilayer_design_job(
    *, app: Any, data_dir: Path, study_id: str, stage: str
) -> None:
    """Run one design-study stage in a background thread and record the outcome."""

    from grax.multilayer_design import MultilayerGratingDesigner

    from .resource_manager import allocate_workers, release_workers

    key = _multilayer_design_job_key(study_id, stage)
    store = MultilayerDesignStudyStore(data_dir / "multilayer_designs")

    def _set_stage(**updates: Any) -> None:
        store.update(study_id, lambda manifest: manifest["stages"][stage].update(updates))

    allocate_workers(key)
    _update_active_run(app, key, state="running", started=True)
    _set_stage(status="running", error_text="", aborted=False)
    auto_chain = False
    stop_event = job_stop_event(app, key)
    try:
        manifest = store.load(study_id)
        config = build_design_config(store.study_dir(study_id), manifest["config"])
        designer = MultilayerGratingDesigner(config)

        def _progress(report: Any) -> None:
            _update_active_run(app, key, completed_points=report.completed)
            with _active_runs_lock(app):
                entry = _active_runs(app).get(key)
                if entry is not None and report.total:
                    entry.total_points = report.total

        def _keep_going() -> bool:
            with _active_runs_lock(app):
                entry = _active_runs(app).get(key)
                return entry is None or not entry.stop_event.is_set()

        def _worker_pids_changed(worker_pids: set[int]) -> None:
            # The live pool size is the only honest worker count -- the config
            # only ever says "auto".
            _update_active_run(
                app,
                key,
                simulation_pids=worker_pids,
                resolved_workers=len(worker_pids) or None,
            )

        if stage == "survey":
            result = designer.run_survey(
                progress_callback=_progress,
                should_continue=_keep_going,
                stop_event=stop_event,
                on_worker_pids_changed=_worker_pids_changed,
            )
            aborted = bool(result.aborted)
            artifacts = {
                "optimal_blaze_plot": _relative_to(result.plot_path, store.study_dir(study_id)),
                "max_efficiency_plot": _relative_to(
                    result.efficiency_plot_path, store.study_dir(study_id)
                ),
                "heatmap_plot": _relative_to(
                    result.heatmap_plot_path, store.study_dir(study_id)
                ),
                "survey_csv": _relative_to(
                    result.combined_csv_path, store.study_dir(study_id)
                ),
            }
            auto_chain = (
                not aborted
                and manifest.get("auto_energy_scan") == "best"
                and manifest["stages"]["energy_scan"]["status"] != "running"
            )
        else:
            designs = [tuple(pair) for pair in manifest["stages"][stage].get("designs") or []]
            scans = designer.run_energy_scan(
                designs,
                progress_callback=_progress,
                should_continue=_keep_going,
                stop_event=stop_event,
                on_worker_pids_changed=_worker_pids_changed,
            )
            aborted = len(scans) < len(designs)
            artifacts = {
                "designs": [
                    {
                        "d_spacing_nm": scan.d_spacing_nm,
                        "blaze_angle_deg": scan.blaze_angle_deg,
                        "plot": _relative_to(scan.titled_plot_path, store.study_dir(study_id)),
                        "summary_csv": _relative_to(
                            scan.summary_csv_path, store.study_dir(study_id)
                        ),
                    }
                    for scan in scans
                ],
            }
            if len(scans) >= 2:
                artifacts["overlay_plot"] = _relative_to(
                    designer.plot_energy_scan_overlay(scans), store.study_dir(study_id)
                )

        _set_stage(
            status="aborted" if aborted else "completed",
            ran_at=datetime.now().isoformat(timespec="seconds"),
            aborted=aborted,
            artifacts=artifacts,
            error_text="",
        )
        _finish_active_run(app, key, state="aborted" if aborted else "completed")
    except Exception as error:  # noqa: BLE001 - surfaced to the study page
        if stop_event.is_set():
            # Killing the workers raises out of the stage; that is an abort, not
            # a failure, and the user does not need the traceback for it.
            _set_stage(
                status="aborted",
                ran_at=datetime.now().isoformat(timespec="seconds"),
                aborted=True,
                error_text="",
            )
            _finish_active_run(app, key, state="aborted")
        else:
            _set_stage(status="failed", error_text=str(error))
            _finish_active_run(app, key, state="failed", error_text=str(error))
    finally:
        release_workers(key)

    if auto_chain:
        _start_auto_energy_scan(app=app, data_dir=data_dir, study_id=study_id)


def _start_auto_energy_scan(*, app: Any, data_dir: Path, study_id: str) -> None:
    """Queue the energy scan on the survey's best design, for ``auto_energy_scan``."""

    store = MultilayerDesignStudyStore(data_dir / "multilayer_designs")
    try:
        options = survey_design_options(store.study_dir(study_id))
        designs = resolve_designs("best", options, [])
    except (ValueError, OSError):
        return
    store.update(
        study_id,
        lambda manifest: manifest["stages"]["energy_scan"].update(
            status="queued", error_text="", designs=designs
        ),
    )
    _start_multilayer_design_worker(
        app=app, data_dir=data_dir, study_id=study_id, stage="energy_scan"
    )


def _energy_scan_checkpoint_path(study_dir: Path, pair: Any) -> Path | None:
    """Return one design's checkpoint file, or ``None`` if the pair is malformed."""

    try:
        d_spacing, blaze = float(pair[0]), float(pair[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (
        study_dir
        / "energy_scan"
        / f"d{d_spacing:.3f}nm_blaze{blaze:.3f}deg"
        / "checkpoints"
        / "results.jsonl"
    )


def _energy_scan_checkpoint_series(*, study_dir: Path, designs: list[Any]) -> list[dict[str, Any]]:
    """Return each design's solved ``(energy, efficiency)`` points, energy-sorted.

    The summary CSV only lands when a design finishes, so the live plot reads the
    checkpoint instead -- one JSON record per solved energy, written as the sweep
    goes. Energies come back in completion order, hence the sort.
    """

    series: list[dict[str, Any]] = []
    for pair in designs:
        results = _energy_scan_checkpoint_path(study_dir, pair)
        if results is None:
            continue
        points: list[tuple[float, float]] = []
        if results.is_file():
            try:
                with results.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue  # a line still being written
                        if record.get("status") != "ok":
                            continue
                        energy = record.get("energy_ev")
                        efficiency = record.get("selected_efficiency")
                        if energy is None or efficiency is None:
                            continue
                        points.append((float(energy), float(efficiency)))
            except OSError:
                points = []
        points.sort()
        series.append(
            {
                "d_spacing_nm": float(pair[0]),
                "blaze_angle_deg": float(pair[1]),
                "energies_ev": [energy for energy, _ in points],
                "efficiencies": [efficiency for _, efficiency in points],
            }
        )
    return series


def _energy_scan_checkpoint_progress(
    *, study_dir: Path, designs: list[Any], energy_points: int
) -> tuple[int, int]:
    """Return ``(completed, total)`` energy points across an energy scan's designs.

    ``run_energy_scan`` reports progress once per *design*, so a single-design
    scan would otherwise sit at ``0 / 1`` for its whole run -- indistinguishable
    from a job that never started. Each design's sweep checkpoints one line per
    solved energy, so counting those gives a bar that actually moves.
    """

    total = max(0, int(energy_points)) * len(designs)
    completed = 0
    for pair in designs:
        results = _energy_scan_checkpoint_path(study_dir, pair)
        if results is None or not results.is_file():
            continue
        try:
            with results.open("r", encoding="utf-8") as handle:
                completed += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return min(completed, total) if total else completed, total


def _multilayer_design_stage_status_payload(
    *, app: Any, data_dir: Path, study_id: str, stage: str
) -> dict[str, Any]:
    """Return a status payload for one design stage in the shape ``initRunMonitor`` reads."""

    _cleanup_finished_runs(app)
    key = _multilayer_design_job_key(study_id, stage)
    manifest = _mark_orphaned_stages(
        app, MultilayerDesignStudyStore(data_dir / "multilayer_designs").load(study_id)
    )
    stage_state = manifest["stages"][stage]

    with _active_runs_lock(app):
        active = _active_runs(app).get(key)
        if active is not None and _is_active_run_entry_live(active):
            completed = active.completed_points
            total = active.total_points
            elapsed = _elapsed_seconds(active)
            eta = _eta_seconds(active)
            label = ""
            if stage == "energy_scan":
                # Count solved energies, not finished designs, so the bar moves.
                designs = stage_state.get("designs") or []
                checkpoint_completed, checkpoint_total = _energy_scan_checkpoint_progress(
                    study_dir=data_dir / "multilayer_designs" / study_id,
                    designs=designs,
                    energy_points=int(manifest["config"].get("energy_scan_points") or 0),
                )
                if checkpoint_total:
                    completed, total = checkpoint_completed, checkpoint_total
                    # Only energies solved by *this* run measure its rate;
                    # resumed checkpoints cost it no time.
                    solved_here = completed - active.resumed_points
                    eta = (
                        elapsed * (total - completed) / solved_here
                        if solved_here > 0 and elapsed and total > completed
                        else None
                    )
                if designs:
                    index = min(active.completed_points, len(designs) - 1)
                    d_spacing, blaze = designs[index]
                    label = (
                        f"design {index + 1} of {len(designs)}: "
                        f"d = {float(d_spacing):.3f} nm, blaze = {float(blaze):.3f} deg"
                    )
            return {
                "state": active.state,
                "completed_points": completed,
                "total_points": total,
                "remaining_points": max(total - completed, 0),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "current_label": (
                    "stopping workers…" if active.state == "aborting" else label
                ),
                "worker_mode": "auto",
                "requested_workers": None,
                "resolved_workers": active.resolved_workers,
                "plot_url": None,
                "plot_token": "",
                "error_text": active.error_text,
                "can_abort": active.state in {"queued", "running"}
                and not active.abort_requested,
            }

    status = stage_state.get("status", "not_run")
    normalized = {"queued": "running", "aborting": "running"}.get(status, status)
    if normalized not in {"completed", "failed", "aborted", "stale", "not_run", "running"}:
        normalized = "not_run"
    artifacts = stage_state.get("artifacts") or {}
    plot_rel = artifacts.get("heatmap_plot") if stage == "survey" else artifacts.get("overlay_plot")
    plot_url = None
    plot_token = ""
    if plot_rel:
        plot_path = data_dir / "multilayer_designs" / study_id / plot_rel
        if plot_path.exists():
            plot_token = _file_token(plot_path)
            plot_url = f"/_data/multilayer_designs/{study_id}/{plot_rel}?v={plot_token}"
    return {
        "state": normalized if normalized != "stale" else "completed",
        "completed_points": 0,
        "total_points": 0,
        "remaining_points": 0,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "current_label": "",
        "worker_mode": "auto",
        "requested_workers": None,
        "resolved_workers": None,
        "plot_url": plot_url,
        "plot_token": plot_token,
        "error_text": stage_state.get("error_text", ""),
        "can_abort": False,
    }


def _abort_multilayer_design_stage(
    *, app: Any, store: Any, study_id: str, stage: str, discard: bool
) -> None:
    """Request a cooperative stop for a running stage and wait for it to finish."""

    # The sweep kills its own pool from the inside; stopping the job's worker
    # processes is the backstop for a worker stuck somewhere the stop check does
    # not reach. It also matters for `discard`, which deletes the stage
    # directory the worker is writing into.
    stop_job(app, _multilayer_design_job_key(study_id, stage))
    if discard:
        _reset_multilayer_design_stage(store=store, study_id=study_id, stage=stage)


def _reset_multilayer_design_stage(*, store: Any, study_id: str, stage: str) -> None:
    """Delete one stage's outputs and blank its manifest entry."""

    study_dir = store.study_dir(study_id)
    artifacts = store.load(study_id)["stages"][stage].get("artifacts") or {}

    if stage == "survey":
        shutil.rmtree(study_dir / "survey", ignore_errors=True)
        for key in ("optimal_blaze_plot", "max_efficiency_plot", "heatmap_plot"):
            _unlink_relative(study_dir, artifacts.get(key))
    else:
        shutil.rmtree(study_dir / "energy_scan", ignore_errors=True)
        _unlink_relative(study_dir, artifacts.get("overlay_plot"))
        for design in artifacts.get("designs") or []:
            _unlink_relative(study_dir, design.get("plot"))

    def blank_stage(manifest: dict[str, Any]) -> None:
        manifest["stages"][stage] = {
            "status": "not_run",
            "ran_at": None,
            "error_text": "",
            "aborted": False,
            "designs": [],
            "artifacts": {},
        }
        for later in downstream_stages(stage):
            if manifest["stages"][later]["status"] in {"completed", "aborted"}:
                manifest["stages"][later]["status"] = "stale"

    store.update(study_id, blank_stage)
