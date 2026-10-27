"""Queueing, executing, aborting and reporting grating simulation runs."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from . import parameter_scans
from .common import (
    _normalized_polarization,
    _normalized_solver,
    _run_id,
    _write_text_atomically,
)
from .custom_grid import (
    expand_custom_grid,
)
from .grating_forms import (
    _attach_roughness,
)
from .jobs import (
    _finish_active_run,
    request_stop,
    start_job,
    wait_for_stop,
)
from .jobs import stop_event as job_stop_event
from .run_files import (
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
from .run_plots import _publish_live_progress_if_due
from .run_state import (
    _is_abort_requested,
    _is_delete_requested,
    _is_run_active,
    _update_active_run,
)
from .runs import RunStore

matplotlib.use("Agg")


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
        wait_for_stop(app, run_id)
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

    start_job(
        app,
        run_id,
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
        thread_name=f"grax-run-{run_id}",
        workflow=workflow,
        total_points=(
            _parameter_study_total_points(form_data, grating, energies.size)
            if workflow == "parameter_study"
            else _custom_grid_case_count_from_form(form_data) if workflow == "custom_grid"
            else _parameter_scan_point_count(form_data, int(energies.size))
        ),
        worker_mode=worker_mode,
        requested_workers=requested_workers,
        completed_points=completed_points,
    )


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
            stop_event=job_stop_event(app, run_id),
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

        # Parallel workers finish in any order; save results in case order like a resumed run.
        results.sort(key=parameter_scans.result_order)
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

    if not request_stop(app, run_id, delete=delete_after_abort):
        return False
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


