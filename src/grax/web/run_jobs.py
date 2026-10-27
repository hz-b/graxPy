"""Queueing, executing, aborting and reporting grating simulation runs."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import parameter_scans
from .common import (
    _file_token,
    _normalized_polarization,
    _normalized_solver,
    _run_id,
    _write_text_atomically,
)
from .custom_grid import (
    expand_custom_grid,
    plot_custom_grid,
)
from .grating_forms import (
    _attach_roughness,
)
from .run_files import (
    _checkpoint_counts,
    _load_checkpoint_results,
    _load_run_manifest,
    _manifest_cases,
    _parameter_study_artifacts,
    _persist_run_outputs,
    _preferred_run_plot_path,
    _replace_plot_with_retry,
    _run_artifacts_for_results,
    _run_completed_points,
    _run_error_summary,
    _run_plot_url,
    _update_manifest_fields,
)
from .run_inputs import (
    _cases_for_workflow,
    _custom_grid_case_count_from_form,
    _custom_grid_defaults_from_form,
    _custom_grid_from_form,
    _parameter_scan_point_count,
    _parameter_scans_from_form,
    _parameter_study_configuration,
    _parameter_study_fixed_configuration,
    _parameter_study_total_points,
    _prepare_monochromator_cff_config,
    _run_input_from_form,
    _snapshot_custom_grid_afm_profiles,
    _worker_settings_from_form,
)
from .run_state import (
    ActiveRunState,
    _active_runs,
    _active_runs_lock,
    _cleanup_finished_runs,
    _elapsed_seconds,
    _eta_seconds,
    _is_abort_requested,
    _is_active_run_entry_live,
    _is_delete_requested,
    _is_run_active,
    _terminate_run_processes,
    _update_active_run,
    _wait_for_run_shutdown,
)
from .runs import RunStore

matplotlib.use("Agg")

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime.
    class _MissingPsutil:
        """Fallback object that keeps the psutil interface patchable in tests."""

        def virtual_memory(self) -> Any:
            """Raise a stable error when psutil is unavailable."""

            raise RuntimeError("psutil is not installed.")

    psutil = _MissingPsutil()


def _queue_run(
    *,
    app: Any,
    data_dir: Path,
    grating_id: str,
    grating_name: str,
    grating_spec: dict[str, Any],
    grating: Any,
    form: Any,
) -> dict[str, Any]:
    """Create a queued run manifest and start background execution."""

    workflow = str(form["workflow"])
    if workflow == "monochromator":
        prepared_config = _prepare_monochromator_cff_config(
            form, parameter_scans.parse_config(form.get("parameter_sweep_config"))
        )
        # Keep Werkzeug's MultiDict interface: run-input persistence uses
        # getlist() to retain repeated form fields such as selected orders.
        form = form.copy() if hasattr(form, "copy") else dict(form)
        form["parameter_sweep_config"] = json.dumps(prepared_config)
    run_id = _run_id(workflow, grating_id)
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    custom_config: dict[str, Any] | None = None
    custom_cases: list[dict[str, Any]] = []
    sweep = None
    if workflow in {"angle_sweep", "fixed_angle"}:
        energy = float(form.get("energy_ev", 100))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Energy must be finite and greater than 0 eV.")
        energies = np.asarray([energy], dtype=float)
    elif workflow == "custom_grid":
        custom_config, custom_cases = _custom_grid_from_form(form, grating_spec)
        _snapshot_custom_grid_afm_profiles(custom_config, run_dir)
        custom_config, custom_cases = expand_custom_grid(
            custom_config,
            grating_spec,
            defaults=_custom_grid_defaults_from_form(form),
        )
        energies = np.asarray([case["energy_ev"] for case in custom_cases], dtype=float)
    elif workflow == "parameter_study" and str(form.get("study_energy_ev", "")).strip():
        energies = np.asarray([float(form["study_energy_ev"])], dtype=float)
        if energies[0] <= 0:
            raise ValueError("Study energy must be greater than 0 eV.")
    else:
        energies = np.linspace(
            float(form["energy_start_ev"]),
            float(form["energy_stop_ev"]),
            int(float(form["energy_points"])),
        )
    if workflow == "angle_sweep" or (
        workflow == "fixed_angle" and parameter_scans.has_variations(parameter_scans.parse_config(form.get("parameter_sweep_config")))
    ) or (
        workflow in {"energy_sweep", "monochromator", "constant_deviation"}
        and parameter_scans.has_variations(parameter_scans.parse_config(form.get("parameter_sweep_config")))
    ):
        sweep, sweep_cases = _parameter_scans_from_form(form, grating_spec, energies)
        _snapshot_custom_grid_afm_profiles(sweep["grid"], run_dir)
        scan_form = {**dict(form), "parameter_sweep_config": sweep["grid"]}
        sweep, sweep_cases = _parameter_scans_from_form(scan_form, grating_spec, energies)
    diffraction_order = (
        int(custom_cases[0]["diffraction_order"])
        if custom_cases else int(float(form.get("diffraction_order", 1)))
    )
    if workflow == "parameter_study":
        study_config = _parameter_study_configuration(form, grating)
        _parameter_study_fixed_configuration(form, study_config[2], study_config[3], study_config[4])
        fourier_orders = int(np.max(study_config[2]))
    elif custom_cases:
        fourier_orders = int(custom_cases[0]["fourier_orders"])
    else:
        fourier_orders = int(float(form.get("fourier_orders", 15)))
    polarization = (
        str(custom_cases[0]["polarization"])
        if custom_cases else _normalized_polarization(form.get("polarization", "p"))
    )
    solver = (
        str(custom_cases[0]["solver"])
        if custom_cases else _normalized_solver(form.get("solver", "rcwa"))
    )
    worker_mode, max_workers_setting, requested_workers = _worker_settings_from_form(form)
    manifest: dict[str, Any] = {
        "id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": workflow,
        "grating_id": grating_id,
        "grating_name": grating_name,
        "grating_spec": dict(grating_spec),
        "display_name": f"{grating_name} · {workflow}",
        "comment": str(form.get("comment", "")).strip(),
        "polarization": polarization,
        "solver": solver,
        "status": "queued",
        "artifacts": [],
        "worker_mode": worker_mode,
        "requested_workers": requested_workers,
        "resolved_workers": None,
        "total_points": (
            _parameter_study_total_points(form, grating, energies.size)
            if workflow == "parameter_study"
            else len(custom_cases) if workflow == "custom_grid"
            else len(sweep_cases) if sweep is not None
            else int(energies.size)
        ),
        "run_input": _run_input_from_form(form),
        "resume_count": 0,
    }
    manifest["run_input"]["polarization"] = polarization
    manifest["run_input"]["solver"] = solver
    manifest["run_input"]["diffraction_order"] = diffraction_order
    manifest["run_input"]["fourier_orders"] = fourier_orders
    if custom_config is not None:
        manifest["custom_grid"] = custom_config
        manifest["run_input"]["custom_grid_config"] = custom_config
    if sweep is not None:
        manifest["parameter_sweep"] = sweep
        manifest["run_input"]["parameter_sweep_config"] = sweep["grid"]
    RunStore(data_dir / "runs").save(manifest)

    worker_form_data = dict(form)
    if sweep is not None:
        worker_form_data["parameter_sweep_config"] = sweep["grid"]
    if custom_config is not None:
        worker_form_data["custom_grid_config"] = custom_config
    _start_run_worker(
        app=app,
        data_dir=data_dir,
        run_id=run_id,
        grating_id=grating_id,
        grating_name=grating_name,
        grating=grating,
        workflow=workflow,
        energies=energies,
        form_data=worker_form_data,
        diffraction_order=diffraction_order,
        fourier_orders=fourier_orders,
        max_workers_setting=max_workers_setting,
        worker_mode=worker_mode,
        requested_workers=requested_workers,
        completed_points=0,
        resume=False,
    )
    return manifest


def _abort_run(
    *,
    app: Any,
    data_dir: Path,
    run_id: str,
    delete_after_abort: bool,
) -> dict[str, Any] | None:
    """Abort one run, optionally deleting it after the worker stops."""

    manifest = _load_run_manifest(data_dir, run_id)
    if manifest is None:
        return None
    if _is_run_active(app, run_id):
        _request_run_abort(app, data_dir, run_id, delete_after_abort=delete_after_abort)
        _terminate_run_processes(app, run_id)
        _wait_for_run_shutdown(app, run_id)
        if delete_after_abort and _load_run_manifest(data_dir, run_id) is not None:
            RunStore(data_dir / "runs").delete_many([run_id])
            return None
        return _load_run_manifest(data_dir, run_id)
    if delete_after_abort:
        RunStore(data_dir / "runs").delete_many([run_id])
        return None
    _persist_aborted_run_state(data_dir, run_id)
    return _load_run_manifest(data_dir, run_id)


def _start_run_worker(
    *,
    app: Any,
    data_dir: Path,
    run_id: str,
    grating_id: str,
    grating_name: str,
    grating: Any,
    workflow: str,
    energies: np.ndarray,
    form_data: dict[str, Any],
    diffraction_order: int,
    fourier_orders: int,
    max_workers_setting: str | int,
    worker_mode: str,
    requested_workers: int | None,
    completed_points: int,
    resume: bool,
) -> None:
    """Register one active run state and start its worker thread."""

    active_state = ActiveRunState(
        run_id=run_id,
        workflow=workflow,
        total_points=(
            _parameter_study_total_points(form_data, grating, energies.size)
            if workflow == "parameter_study"
            else _custom_grid_case_count_from_form(form_data) if workflow == "custom_grid"
            else _parameter_scan_point_count(form_data, int(energies.size))
        ),
        worker_mode=worker_mode,
        requested_workers=requested_workers,
        resolved_workers=None,
        completed_points=completed_points,
    )
    with _active_runs_lock(app):
        _active_runs(app)[run_id] = active_state

    worker = threading.Thread(
        target=_execute_run_job,
        kwargs={
            "app": app,
            "data_dir": data_dir,
            "run_id": run_id,
            "grating_id": grating_id,
            "grating_name": grating_name,
            "grating": grating,
            "workflow": workflow,
            "energies": energies,
            "form_data": form_data,
            "diffraction_order": diffraction_order,
            "fourier_orders": fourier_orders,
            "max_workers_setting": max_workers_setting,
            "resume": resume,
        },
        daemon=True,
        name=f"grax-run-{run_id}",
    )
    active_state.worker_thread = worker
    worker.start()


def _execute_run_job(
    *,
    app: Any,
    data_dir: Path,
    run_id: str,
    grating_id: str,
    grating_name: str,
    grating: Any,
    workflow: str,
    energies: np.ndarray,
    form_data: dict[str, Any],
    diffraction_order: int,
    fourier_orders: int,
    max_workers_setting: str | int,
    resume: bool = False,
) -> None:
    """Execute one queued run in a background thread and persist its outputs."""

    from grax import parameter_sweep, simulation

    from .resource_manager import allocate_workers, release_workers

    _attach_roughness(grating, form_data)

    run_dir = data_dir / "runs" / run_id
    run_manifest = _load_run_manifest(data_dir, run_id) or {}

    # Block here until the global pool has room — run stays "queued" while waiting.
    granted_workers = allocate_workers(run_id)

    # Honour manual cap if set; otherwise use all granted workers.
    if isinstance(max_workers_setting, int):
        effective_max_workers: int = min(granted_workers, max_workers_setting)
    else:
        effective_max_workers = granted_workers

    _update_active_run(app, run_id, state="running", started=True)
    _update_manifest_fields(data_dir, run_id, status="running")
    try:
        if workflow == "parameter_study":
            _study_energy, study_angle, fourier_values, x_values, z_values = _parameter_study_configuration(form_data, grating)
            (
                fourier_fixed_x,
                fourier_fixed_z,
                x_fixed_fourier,
                x_fixed_z,
                z_fixed_fourier,
                z_fixed_x,
            ) = _parameter_study_fixed_configuration(form_data, fourier_values, x_values, z_values)

            def _parameter_study_progress(completed: int, total: int, partial_result: Any) -> None:
                """Persist visible progress and a partial plot after each sweep."""

                _update_active_run(app, run_id, completed_points=completed)
                _update_manifest_fields(
                    data_dir,
                    run_id,
                    status="running",
                    completed_points=completed,
                    total_points=total,
                    artifacts=_parameter_study_artifacts(run_dir),
                )
                temporary_plot_path = run_dir / "parameter_study.tmp.png"
                partial_plot_path = run_dir / "parameter_study.png"
                parameter_sweep.plot_parameter_study(
                    partial_result,
                    output_filename=temporary_plot_path,
                )
                if _replace_plot_with_retry(temporary_plot_path, partial_plot_path):
                    _update_active_run(
                        app,
                        run_id,
                        plot_relative_path=partial_plot_path.name,
                        bump_plot_token=True,
                    )
                else:
                    # Windows can deny replacement while Flask is serving the
                    # previous preview. Keep the sweep running and retain the
                    # last plot until a later update can be published.
                    app.logger.debug(
                        "Could not refresh the live plot for run %s; keeping the previous preview",
                        run_id,
                    )
                    temporary_plot_path.unlink(missing_ok=True)

            result = parameter_sweep.run_parameter_study(
                grating=grating,
                energies_ev=energies,
                grazing_angle_deg=study_angle,
                diffraction_order=diffraction_order,
                polarization=_normalized_polarization(form_data.get("polarization", "p")),
                solver=_normalized_solver(form_data.get("solver", "rcwa")),
                fourier_orders_values=fourier_values,
                x_resolution_values=x_values,
                z_resolution_values=z_values,
                fourier_fixed_x_resolution_nm=fourier_fixed_x,
                fourier_fixed_z_resolution_nm=fourier_fixed_z,
                x_fixed_fourier_orders=x_fixed_fourier,
                x_fixed_z_resolution_nm=x_fixed_z,
                z_fixed_fourier_orders=z_fixed_fourier,
                z_fixed_x_resolution_nm=z_fixed_x,
                output_dir=run_dir,
                save_csv=True,
                show_progress=False,
                sweep_progress_callback=_parameter_study_progress,
                resume=resume,
            )
            plot_path = run_dir / "parameter_study.png"
            temporary_plot_path = run_dir / "parameter_study.tmp.png"
            parameter_sweep.plot_parameter_study(result, output_filename=temporary_plot_path)
            if not _replace_plot_with_retry(temporary_plot_path, plot_path):
                temporary_plot_path.unlink(missing_ok=True)
                app.logger.warning(
                    "Could not publish the final parameter-study plot for run %s; keeping the last available preview",
                    run_id,
                )
            _update_active_run(
                app,
                run_id,
                completed_points=_parameter_study_total_points(form_data, grating, energies.size),
                plot_relative_path="parameter_study.png",
                bump_plot_token=True,
            )
            _update_manifest_fields(
                data_dir,
                run_id,
                status="completed",
                artifacts=_parameter_study_artifacts(run_dir),
                completed_points=_parameter_study_total_points(form_data, grating, energies.size),
                total_points=_parameter_study_total_points(form_data, grating, energies.size),
            )
            _finish_active_run(app, run_id, state="completed")
            return

        cases = _cases_for_workflow(
            simulation=simulation,
            workflow=workflow,
            grating=grating,
            energies=energies,
            form=form_data,
            diffraction_order=diffraction_order,
            fourier_orders=fourier_orders,
        )
        runner = simulation.BatchSimulationRunner(
            diffraction_order=diffraction_order,
            fourier_orders=fourier_orders,
            solver=_normalized_solver(form_data.get("solver", "rcwa")),
            max_workers=effective_max_workers,
            show_progress=False,
            on_error="continue",
            checkpoint_dir=run_dir / "checkpoints",
            resume=resume,
            stop_event=_active_runs(app)[run_id].stop_event if run_id in _active_runs(app) else None,
            on_worker_pids_changed=lambda worker_pids: _update_active_run(
                app,
                run_id,
                simulation_pids=worker_pids,
            ),
        )
        _update_active_run(
            app,
            run_id,
            resolved_workers=int(runner.resolved_max_workers),
        )
        _update_manifest_fields(data_dir, run_id, resolved_workers=int(runner.resolved_max_workers))

        results: list[Any] = _load_checkpoint_results(run_dir) if resume else []
        live_plot_path = run_dir / "live_progress.png"
        for result in runner.run_cases(
            cases,
            metadata={
                "workflow": workflow,
                "grating_id": grating_id,
                "grating_name": grating_name,
            },
        ):
            results.append(result)
            _update_active_run(
                app,
                run_id,
                completed_increment=1,
                resolved_workers=int(runner.resolved_max_workers),
            )
            _publish_live_progress_if_due(
                app=app,
                run_id=run_id,
                results=results,
                output_path=live_plot_path,
                diffraction_order=diffraction_order,
                title=f"{grating_name} {workflow}",
            )

        if results:
            _publish_live_progress_if_due(
                app=app,
                run_id=run_id,
                results=results,
                output_path=live_plot_path,
                diffraction_order=diffraction_order,
                title=f"{grating_name} {workflow}",
                min_interval_seconds=0.0,
            )

        is_aborted = _is_abort_requested(app, run_id) or runner.stopped_early
        _persist_run_outputs(
            data_dir=data_dir,
            run_id=run_id,
            results=results,
            diffraction_order=diffraction_order,
            title=f"{grating_name} {workflow}",
            include_plot=bool(results),
        )

        if is_aborted:
            _update_manifest_fields(
                data_dir,
                run_id,
                status="aborted",
                cases=_manifest_cases(results),
                total_points=len(cases),
                last_checkpoint_at=datetime.now().isoformat(timespec="seconds"),
                aborted_at=datetime.now().isoformat(timespec="seconds"),
                artifacts=_run_artifacts_for_results(run_dir, include_plot=bool(results)),
            )
            _finish_active_run(
                app,
                run_id,
                state="aborted",
                plot_relative_path=_preferred_run_plot_path(data_dir, run_id),
                completed_points=len(results),
            )
            if _is_delete_requested(app, run_id):
                RunStore(data_dir / "runs").delete_many([run_id])
            return

        _update_manifest_fields(
            data_dir,
            run_id,
            status="completed",
            artifacts=_run_artifacts_for_results(run_dir, include_plot=bool(results)),
            cases=_manifest_cases(results),
            resolved_workers=int(runner.resolved_max_workers),
            total_points=len(cases),
        )
        _finish_active_run(
            app,
            run_id,
            state="completed",
            plot_relative_path=_preferred_run_plot_path(data_dir, run_id),
        )
    except Exception as error:  # pragma: no cover - exercised by integration behavior.
        results = _load_checkpoint_results(run_dir)
        if workflow != "parameter_study":
            _persist_run_outputs(
                data_dir=data_dir,
                run_id=run_id,
                results=results,
                diffraction_order=diffraction_order,
                title=f"{grating_name} {workflow}",
                include_plot=bool(results),
            )
        _write_text_atomically(run_dir / "error.txt", traceback.format_exc())
        _update_manifest_fields(
            data_dir,
            run_id,
            status="failed",
            error_text=_run_error_summary(workflow, error),
            cases=_manifest_cases(results),
            artifacts=(
                _parameter_study_artifacts(run_dir)
                if workflow == "parameter_study"
                else _run_artifacts_for_results(run_dir, include_plot=bool(results))
            ),
            completed_points=(
                _run_completed_points(_load_run_manifest(data_dir, run_id) or {}, 0)
                if workflow == "parameter_study"
                else None
            ),
            total_points=(
                _parameter_study_total_points(form_data, grating, energies.size)
                if workflow == "parameter_study" else None
            ),
        )
        _finish_active_run(
            app,
            run_id,
            state="failed",
            error_text=_run_error_summary(workflow, error),
        )
    finally:
        release_workers(run_id)


def _request_run_abort(app: Any, data_dir: Path, run_id: str, *, delete_after_abort: bool) -> bool:
    """Request cooperative abort for one active run."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        if active is None or active.state not in {"queued", "running"}:
            return False
        active.abort_requested = True
        active.delete_requested = delete_after_abort
        active.stop_event.set()
        active.state = "aborting"
    _update_manifest_fields(data_dir, run_id, status="aborting")
    return True


def _persist_aborted_run_state(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Persist the current checkpoint-backed state for one aborted run."""

    manifest = _load_run_manifest(data_dir, run_id)
    if manifest is None:
        return None
    run_dir = data_dir / "runs" / run_id
    results = _load_checkpoint_results(run_dir)
    run_input = dict(manifest.get("run_input", {}))
    diffraction_order = int(float(run_input.get("diffraction_order", 1)))
    title = (
        f"{manifest.get('grating_name', manifest.get('display_name', run_id))} "
        f"{manifest.get('workflow', 'run')}"
    )
    _persist_run_outputs(
        data_dir=data_dir,
        run_id=run_id,
        results=results,
        diffraction_order=diffraction_order,
        title=title,
        include_plot=bool(results),
    )
    return _update_manifest_fields(
        data_dir,
        run_id,
        status="aborted",
        cases=_manifest_cases(results),
        total_points=int(manifest.get("total_points") or len(results)),
        aborted_at=datetime.now().isoformat(timespec="seconds"),
        artifacts=_run_artifacts_for_results(run_dir, include_plot=bool(results)),
        error_text="",
    )


def _results_sorted_for_live_plot(results: list[Any], *, x_key: str = "energy_ev") -> list[Any]:
    """Return successful case results sorted by the chosen live-plot x axis."""

    def x_value(result: Any) -> float:
        if x_key == "index":
            return float(result.index + 1)
        if x_key in getattr(result, "case_data", {}):
            return float(result.case_data[x_key])
        if hasattr(result, x_key):
            return float(getattr(result, x_key))
        raise KeyError(f"Unable to sort live plot using x key '{x_key}'.")

    return sorted(
        [result for result in results if result.status == "ok"],
        key=x_value,
    )


def _publish_live_progress_snapshot(
    *,
    state: ActiveRunState,
    results: list[Any],
    output_path: Path,
    diffraction_order: int,
    title: str,
    now: float | None = None,
    min_interval_seconds: float = 1.5,
) -> bool:
    """Render and atomically publish one live progress image when due."""

    from grax import simulation

    publish_time = time.monotonic() if now is None else float(now)
    if not results:
        return False
    if (
        state.last_plot_publish_monotonic is not None
        and publish_time - state.last_plot_publish_monotonic < min_interval_seconds
    ):
        return False

    is_custom_grid = any("custom_parameters" in result.case_data for result in results)
    ordered_results = _results_sorted_for_live_plot(
        results,
        x_key="index" if is_custom_grid else "energy_ev",
    )
    grouped_results = any("scan_id" in result.case_data for result in results)
    if not ordered_results and not grouped_results:
        return False

    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if grouped_results:
        manifest = _load_run_manifest(output_path.parent.parent.parent, output_path.parent.name) or {}
        parameter_scans.plot_scans(results, manifest["parameter_sweep"], temp_path, title=title)
    elif is_custom_grid:
        plot_custom_grid(ordered_results, temp_path, title=title)
    else:
        simulation.plot_order_subset(
            ordered_results,
            temp_path,
            diffraction_orders=[diffraction_order],
            title=title,
        )
    temp_path.replace(output_path)
    state.plot_relative_path = output_path.name
    state.plot_token = uuid4().hex
    state.last_plot_publish_monotonic = publish_time
    return True


def _publish_live_progress_if_due(
    *,
    app: Any,
    run_id: str,
    results: list[Any],
    output_path: Path,
    diffraction_order: int,
    title: str,
    min_interval_seconds: float = 1.5,
) -> None:
    """Publish the live progress image on a throttled interval."""

    with _active_runs_lock(app):
        state = _active_runs(app).get(run_id)
    if state is None:
        return
    published = _publish_live_progress_snapshot(
        state=state,
        results=results,
        output_path=output_path,
        diffraction_order=diffraction_order,
        title=title,
        min_interval_seconds=min_interval_seconds,
    )
    if not published:
        return
    _update_active_run(
        app,
        run_id,
        plot_relative_path=output_path.name,
    )


def _finish_active_run(
    app: Any,
    run_id: str,
    *,
    state: str,
    plot_relative_path: str | None = None,
    error_text: str | None = None,
    completed_points: int | None = None,
) -> None:
    """Mark one active run as finished."""

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


def _repair_saved_run_plot(app: Any, data_dir: Path, run_id: str, manifest: dict[str, Any]) -> None:
    """Recreate a missing completed plot from checkpoints without rerunning cases."""

    if manifest.get("status") not in {"completed", "aborted", "failed", "ok"}:
        return
    run_dir = data_dir / "runs" / run_id
    study = manifest.get("workflow") == "parameter_study"
    destination = run_dir / ("parameter_study.png" if study else "selected_efficiency.png")
    if destination.exists():
        return
    results = [] if study else _load_checkpoint_results(run_dir)
    if not study and not results and not (run_dir / "summary.csv").exists():
        return
    temporary = run_dir / f"{destination.stem}.{uuid4().hex}.tmp.png"
    try:
        sweep = manifest.get("parameter_sweep")
        title = f"{manifest.get('grating_name', run_id)} {manifest.get('workflow', 'run')}"
        if study:
            if not _plot_saved_parameter_study(run_dir, manifest, temporary):
                return
        elif not results:
            _plot_saved_summary(run_dir, manifest, temporary, title=title)
        elif sweep:
            parameter_scans.plot_scans(results, sweep, temporary, title=title)
        elif any("custom_parameters" in result.case_data for result in results):
            plot_custom_grid(results, temporary, title=title)
        else:
            from grax import simulation

            simulation.plot_order_subset(
                results, temporary,
                diffraction_orders=[int(manifest.get("run_input", {}).get("diffraction_order", 1))],
                title=title,
            )
        temporary.replace(destination)
        _update_manifest_fields(
            data_dir, run_id,
            artifacts=_parameter_study_artifacts(run_dir) if study else _run_artifacts_for_results(run_dir, include_plot=True),
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        app.logger.exception("Could not rebuild plot for run %s", run_id)


def _plot_saved_summary(run_dir: Path, manifest: dict[str, Any], output_path: Path, *, title: str) -> None:
    """Plot saved CSV rows when a completed run has no checkpoint file."""

    from types import SimpleNamespace

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Saved summary has no result rows.")
    sweep = manifest.get("parameter_sweep")
    if sweep:
        scan_orders = {scan["id"]: int(scan["settings"]["diffraction_order"]) for scan in sweep["scans"]}
        results = []
        for row in rows:
            scan_id = row["scan_id"]
            selected = float(row["selected_efficiency"] or "nan")
            results.append(SimpleNamespace(
                case_data={"scan_id": scan_id},
                energy_ev=float(row["energy_ev"]),
                grazing_angle_deg=float(row["grazing_angle_deg"]),
                status=row["status"],
                selected_efficiency=selected,
                orders=np.asarray([scan_orders[scan_id]]),
                efficiency_all=np.asarray([selected]),
            ))
        parameter_scans.plot_scans(results, sweep, output_path, title=title)
        return
    workflow = manifest.get("workflow")
    x_key = "grazing_angle_deg" if workflow == "angle_sweep" else "energy_ev"
    x_label = "Grazing angle (deg)" if workflow == "angle_sweep" else "Photon Energy (eV)"
    if workflow == "custom_grid":
        x_key, x_label = None, "Case"
    points = [
        (float(row[x_key]) if x_key else float(index + 1),
         float(row["selected_efficiency"] or "nan") if row["status"] == "ok" else np.nan)
        for index, row in enumerate(rows)
    ]
    points.sort(key=lambda item: item[0])
    figure, axis = plt.subplots(figsize=(10, 6))
    try:
        axis.plot([x for x, _ in points], [y for _, y in points], marker=".", linewidth=1.2)
        axis.set(xlabel=x_label, ylabel="Selected-order efficiency", title=title)
        axis.grid(True, alpha=0.3)
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_saved_parameter_study(run_dir: Path, manifest: dict[str, Any], output_path: Path) -> bool:
    """Rebuild the numerical-stability figure from its saved sweep CSV files."""

    from grax import parameter_sweep

    by_energy: dict[float, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("parameter_study_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        parameter = rows[0].get("parameter")
        if parameter not in {"fourier_orders", "x_resolution_nm", "z_resolution_nm"}:
            continue
        energy = float(rows[0]["energy_ev"])
        by_energy.setdefault(energy, {})[parameter] = parameter_sweep.ParameterSweepSeries(
            parameter=parameter,
            values=np.asarray([float(row["value"]) for row in rows]),
            efficiencies=np.asarray([float(row["efficiency"]) for row in rows]),
            errors=np.asarray([row["error"].strip().lower() == "true" for row in rows]),
            error_messages=np.asarray([row.get("error_message", "") for row in rows]),
        )
    required = {"fourier_orders", "x_resolution_nm", "z_resolution_nm"}
    complete = [(energy, sweeps) for energy, sweeps in sorted(by_energy.items()) if required <= sweeps.keys()]
    if not complete:
        return False
    form = manifest.get("run_input", {})
    result = parameter_sweep.ParameterStudyResult(
        energies_ev=np.asarray([energy for energy, _ in complete]),
        grazing_angle_deg=float(form.get("study_grazing_angle_deg", form.get("grazing_angle_deg", 1.5))),
        diffraction_order=int(form.get("diffraction_order", 1)),
        fourier_orders_values=complete[0][1]["fourier_orders"].values,
        x_resolution_values=complete[0][1]["x_resolution_nm"].values,
        z_resolution_values=complete[0][1]["z_resolution_nm"].values,
        results=[
            parameter_sweep.ParameterStudyEnergyResult(
                energy_ev=energy,
                grazing_angle_deg=float(form.get("study_grazing_angle_deg", form.get("grazing_angle_deg", 1.5))),
                sweeps=sweeps,
            )
            for energy, sweeps in complete
        ],
    )
    parameter_sweep.plot_parameter_study(result, output_filename=output_path)
    return True
