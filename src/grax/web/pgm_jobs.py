"""Parse, run and persist SX700 PGM working-range jobs."""

from __future__ import annotations

import dataclasses
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from werkzeug.datastructures import MultiDict

from grax.pgm import PGMCancelled, SX700Parameters, run_working_range

from .files import remove_tree
from .grating_forms import (
    _custom_layer_rows,
    _default_form_values,
    _form_values_from_spec,
    _grating_custom_layer_rows,
    _material_density_placeholders,
    _spec_from_form,
)
from .grating_roughness import validate_roughness_geometry
from .jobs import _finish_active_run, is_orphaned, start_job, stop_event
from .persistence import PGMSetupStore, build_grating_from_spec
from .pgm_dashboard import FINAL_FILE, MAP_KEYS, save_base_maps, save_final_maps, save_nodes
from .pgm_presets import DEFAULT_PGM_PRESET, PGM_PRESETS
from .plane_mirror import clean_name, split_plane_mirror_form
from .plane_mirror_views import _plane_mirror_page_defaults, _plane_mirror_stack
from .resource_manager import allocate_workers, release_workers
from .run_state import (
    _active_runs,
    _active_runs_lock,
    _elapsed_seconds,
    _eta_seconds,
    _is_active_run_entry_live,
    _update_active_run,
)
from .runs import RunStore

PGM_FORM_FIELDS = (
    "mirror_length_mm",
    "grating_length_mm",
    "optic_dx_mm",
    "beam_offset_mm",
    "mirror_grating_axis_dy_mm",
    "mirror_axis_dx_mm",
    "mirror_surface_axis_offset_mm",
    "mirror_downstream_edge_offset_mm",
    "theta_min_deg",
    "theta_max_deg",
    "beta_min_deg",
    "beta_max_deg",
    "beam_height_mm",
    "edge_margin_mm",
    "crit_distance",
    "mirror_roughness_nm",
    "energy_min_eV",
    "energy_max_eV",
    "c_min",
    "c_max",
    "n_energy",
    "n_c",
    "diffraction_order",
    "rcwa_harmonics",
    "rcwa_energy_nodes",
    "rcwa_cff_nodes",
    "rcwa_solver",
)
# A request that leaves these out gets a quicker scan than the library default.
# The page always sends them: presets and saved PGMs set their own values.
PGM_QUICK_SCAN_DEFAULTS = {"n_energy": "101", "n_c": "81", "rcwa_energy_nodes": "12"}


def _form_default(value: Any) -> str:
    """Return a parameter default as the text a form field shows."""
    if value is None:
        return ""
    return value if isinstance(value, str) else f"{value:g}"


_PARAMETER_DEFAULTS = {field.name: field.default for field in dataclasses.fields(SX700Parameters)}
PGM_DEFAULTS: dict[str, str] = {
    name: PGM_QUICK_SCAN_DEFAULTS.get(name, _form_default(_PARAMETER_DEFAULTS[name]))
    for name in PGM_FORM_FIELDS
}
LIVE_SAVE_SECONDS = 1.0
NO_PGM_KEY = "none"
NO_PGM_LABEL = "No saved PGM"
GRATING_PREFIX = "grating."
MIRROR_PREFIX = "mirror."
OPTIONAL_FIELDS = {"mirror_surface_axis_offset_mm", "mirror_downstream_edge_offset_mm"}
INTEGER_FIELDS = {
    "n_energy",
    "n_c",
    "diffraction_order",
    "rcwa_harmonics",
    "rcwa_energy_nodes",
    "rcwa_cff_nodes",
}


def pgm_store(data_dir: Path) -> RunStore:
    """Return the dedicated PGM run store in the selected workspace."""
    return RunStore(Path(data_dir) / "pgm_runs")


def pgm_setup_store(data_dir: Path) -> PGMSetupStore:
    """Return the store of named PGM setups in the selected workspace."""
    return PGMSetupStore(Path(data_dir) / "saved_pgms")


def split_optics_form(form: Any) -> tuple[MultiDict, MultiDict]:
    """Return the grating and mirror editor fields of a merged PGM request."""
    grating, mirror = MultiDict(), MultiDict()
    for key in form.keys():
        for prefix, target in ((GRATING_PREFIX, grating), (MIRROR_PREFIX, mirror)):
            if key.startswith(prefix):
                target.setlist(key.removeprefix(prefix), form.getlist(key))
    return grating, mirror


def grating_from_form(form: Any, data_dir: Path, *, source_id: str = "") -> dict[str, Any]:
    """Validate the grating editor and return a grating spec without saving it."""
    try:
        spec = _spec_from_form(form, require_name=False)
        if spec["grating_type"] == "afm":
            profile = Path(spec["profile"]["profile_path"]).resolve()
            if not profile.is_relative_to(Path(data_dir).resolve()):
                raise ValueError("The processed AFM profile is unavailable in this workspace.")
        validate_roughness_geometry(build_grating_from_spec(spec))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Grating: {error}") from error
    spec["name"] = str(form.get("name") or "").strip() or "Custom grating"
    spec["id"] = source_id or None
    return spec


def mirror_from_form(form: Any, *, source_id: str = "") -> dict[str, Any]:
    """Validate the mirror editor and return a mirror record without saving it."""
    try:
        stack_form, _ = split_plane_mirror_form(form)
        _plane_mirror_stack(form)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Mirror: {error}") from error
    return {
        "id": source_id or None,
        "name": str(form.get("mirror_name") or "").strip() or "Custom mirror",
        "stack_form": stack_form,
    }


def _form_fields(form: MultiDict) -> dict[str, Any]:
    """Return form fields for saving; repeated custom-layer columns stay lists."""
    return {
        key: list(form.getlist(key)) if key.startswith("cl_") else str(form.get(key))
        for key in form.keys()
    }


def _keep_afm_profile(path: Any, directory: Path) -> str:
    """Copy a processed AFM profile so the setup or run outlives the source grating."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "profile.csv"
    shutil.copy2(Path(str(path)), target)
    return str(target)


def save_pgm_setup(data_dir: Path, form: Any) -> dict[str, Any]:
    """Save the page as a new named PGM with its own copies of both optics."""
    name = clean_name(form.get("pgm_name"), "PGM name")
    grating_form, mirror_form = split_optics_form(form)
    spec = grating_from_form(grating_form, data_dir)
    mirror_from_form(mirror_form)
    store = pgm_setup_store(data_dir)
    saved = store.save(
        {
            "name": name,
            "grating_form": _form_fields(grating_form),
            "mirror_form": _form_fields(mirror_form),
            "parameters": parse_pgm_form(form),
        }
    )
    if spec["grating_type"] == "afm":
        saved["grating_form"]["afm_profile_path"] = _keep_afm_profile(
            spec["profile"]["profile_path"], store.directory / saved["id"]
        )
        saved = store.save(saved)
    return saved


def rename_pgm_setup(data_dir: Path, setup_id: str, name: Any) -> None:
    """Rename one saved PGM."""
    store = pgm_setup_store(data_dir)
    store.save({**store.load(setup_id), "name": clean_name(name, "PGM name")})


def delete_pgm_setup(data_dir: Path, setup_id: str) -> None:
    """Delete one saved PGM and its kept AFM profile."""
    store = pgm_setup_store(data_dir)
    store.load(setup_id)
    store.delete(setup_id)
    remove_tree(store.directory / setup_id)


def _editor_context(
    defaults: dict[str, Any], custom_layers: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "defaults": defaults,
        "custom_layers": custom_layers,
        "density_placeholders": _material_density_placeholders(defaults),
    }


def grating_editor(
    *, spec: dict[str, Any] | None = None, form: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the grating fields from a library grating, saved fields or the default example."""
    if spec is not None:
        return _editor_context(_form_values_from_spec(spec), _grating_custom_layer_rows(spec))
    if form is None:
        form = PGM_PRESETS[DEFAULT_PGM_PRESET]["grating_form"]
    defaults = _default_form_values()
    defaults.update({key: value for key, value in form.items() if isinstance(value, str)})
    rows = _custom_layer_rows(form) if "cl_material" in form else _grating_custom_layer_rows({})
    return _editor_context(defaults, rows)


def mirror_editor(stack_form: dict[str, Any] | None = None, *, name: str = "") -> dict[str, Any]:
    """Return the mirror fields from a library mirror, saved fields or the default example."""
    if stack_form is None:
        stack_form = PGM_PRESETS[DEFAULT_PGM_PRESET]["mirror_form"]
    fields = dict(stack_form)
    defaults = _plane_mirror_page_defaults(fields, {})
    defaults["mirror_name"] = name or str(fields.get("mirror_name", ""))
    return _editor_context(defaults, _custom_layer_rows(fields))


def pgm_setup_page_values(setup: dict[str, Any]) -> dict[str, Any]:
    """Return the template values for a preset or a saved PGM."""
    parameters = {
        key: "" if value is None else str(value) for key, value in setup["parameters"].items()
    }
    return {
        "values": {**PGM_DEFAULTS, **parameters},
        "grating_editor": grating_editor(form=setup["grating_form"]),
        "mirror_editor": mirror_editor(setup["mirror_form"]),
    }


def pgm_preset(preset_id: str) -> dict[str, Any] | None:
    """Return one ready-made setup, or None when the name is unknown."""
    return PGM_PRESETS.get(preset_id)


def parse_pgm_form(form: Any) -> dict[str, float | int | str | None]:
    """Validate the website's bounded geometry, scan and solver inputs."""
    values: dict[str, float | int | str | None] = {}
    for name, default in PGM_DEFAULTS.items():
        raw = str(form.get(name, default)).strip().replace(",", ".")
        if name == "rcwa_solver":
            values[name] = raw.lower()
            continue
        if name in OPTIONAL_FIELDS and raw == "":
            values[name] = None
            continue
        try:
            number = float(raw)
        except ValueError as error:
            raise ValueError(f"{name.replace('_', ' ')} must be a number") from error
        if not np.isfinite(number):
            raise ValueError(f"{name.replace('_', ' ')} must be finite")
        if name in INTEGER_FIELDS:
            if not number.is_integer():
                raise ValueError(f"{name.replace('_', ' ')} must be a whole number")
            values[name] = int(number)
        else:
            values[name] = number
    if int(values["n_energy"]) * int(values["n_c"]) > 250_000:
        raise ValueError("The final map cannot exceed 250,000 points")
    if int(values["rcwa_energy_nodes"]) * int(values["rcwa_cff_nodes"]) > 2500:
        raise ValueError("The grating solve cannot exceed 2,500 sampling nodes")
    if int(values["rcwa_energy_nodes"]) < 10 or int(values["rcwa_cff_nodes"]) < 10:
        raise ValueError("Grating sampling needs at least 10 nodes on each axis")
    # This validates the remaining physical and range constraints without
    # loading the selected optics in the request thread.
    SX700Parameters(**values, rcwa_workers=1, rcwa_convergence_check=False)
    return values


def _mirror_form(saved: dict[str, Any]) -> MultiDict:
    pairs = []
    for name, value in saved.get("stack_form", {}).items():
        for item in value if isinstance(value, list) else [value]:
            pairs.append((name, item))
    return MultiDict(pairs)


def _parameters(
    manifest: dict[str, Any],
    data_dir: Path,
    *,
    stop=None,
    progress=None,
    maps_ready=None,
    nodes_ready=None,
    workers: int = 1,
    worker_pids=None,
) -> SX700Parameters:
    grating = build_grating_from_spec(manifest["grating_snapshot"])
    mirror_stack = _plane_mirror_stack(_mirror_form(manifest["mirror_snapshot"]))
    return SX700Parameters(
        **manifest["parameters"],
        grating=grating,
        mirror_stack=mirror_stack,
        rcwa_workers=workers,
        rcwa_convergence_check=False,
        cache_dir=pgm_store(data_dir).run_dir(str(manifest["id"])) / "cache",
        stop_event=stop,
        progress_callback=progress,
        maps_ready_callback=maps_ready,
        nodes_callback=nodes_ready,
        worker_pids_callback=worker_pids,
    )


def pgm_source(data_dir: Path, pgm_id: Any) -> dict[str, str] | None:
    """Return the id and name of the saved PGM or example a run was started from."""
    pgm_id = str(pgm_id or "")
    preset = pgm_preset(pgm_id)
    if preset is not None:
        return {"id": pgm_id, "name": preset["name"]}
    try:
        return {"id": pgm_id, "name": pgm_setup_store(data_dir).load(pgm_id)["name"]}
    except (FileNotFoundError, ValueError):
        return None


def label_runs_by_pgm(data_dir: Path, runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Add each run's PGM key and current name; return the PGMs for the filter menu."""
    names: dict[str, str] = {}
    for run in runs:
        pgm_id = str(run.get("pgm_id") or "")
        source = pgm_source(data_dir, pgm_id)
        name = source["name"] if source else str(run.get("pgm_name") or "")
        run["pgm_key"] = pgm_id or NO_PGM_KEY
        run["pgm_label"] = name or NO_PGM_LABEL
        names[run["pgm_key"]] = run["pgm_label"]
    return [
        {"key": key, "name": name}
        for key, name in sorted(names.items(), key=lambda item: item[1].casefold())
    ]


def create_pgm_run(
    data_dir: Path,
    *,
    grating: dict[str, Any],
    mirror: dict[str, Any],
    values: dict[str, float | int | str | None],
    display_name: str,
    pgm: dict[str, str] | None = None,
    comment: str = "",
    workers: tuple[str, str | int, int | None] = ("manual", 1, 1),
) -> dict[str, Any]:
    """Save the inputs as immutable snapshots before starting a background job."""
    run_id = f"pgm-{uuid4().hex[:16]}"
    now = datetime.now().isoformat(timespec="seconds")
    if grating.get("grating_type") == "afm":
        kept = _keep_afm_profile(
            grating["profile"]["profile_path"], pgm_store(data_dir).run_dir(run_id)
        )
        grating = {**grating, "profile": {**grating["profile"], "profile_path": kept}}
    manifest = {
        "id": run_id,
        "workflow": "pgm",
        "display_name": display_name or "PGM working range",
        "comment": comment,
        "worker_mode": workers[0],
        "max_workers_setting": workers[1],
        "requested_workers": workers[2],
        "pgm_id": pgm["id"] if pgm else None,
        "pgm_name": pgm["name"] if pgm else None,
        "grating_id": grating.get("id"),
        "grating_name": grating["name"],
        "grating_snapshot": grating,
        "mirror_id": mirror.get("id"),
        "mirror_name": mirror["name"],
        "mirror_snapshot": mirror,
        "parameters": values,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "completed_points": 0,
        "total_points": int(values["rcwa_energy_nodes"]) * int(values["rcwa_cff_nodes"]),
    }
    # Construct both optics now so malformed saved specs fail before creating a run.
    _parameters(manifest, data_dir)
    return pgm_store(data_dir).save(manifest)


def start_pgm_run(app: Any, data_dir: Path, run_id: str) -> None:
    """Register and start one PGM worker through the shared job manager."""
    manifest = pgm_store(data_dir).load(run_id)
    start_job(
        app,
        run_id,
        target=_worker,
        kwargs={"app": app, "data_dir": data_dir, "run_id": run_id},
        thread_name=f"grax-pgm-{run_id}",
        workflow="pgm",
        total_points=int(manifest["total_points"]),
        worker_mode=manifest.get("worker_mode", "manual"),
        requested_workers=manifest.get("requested_workers", 1),
    )


def pgm_status_payload(app: Any, run: dict[str, Any]) -> dict[str, Any]:
    """Return one run's progress in the shape the shared run monitor reads."""
    with _active_runs_lock(app):
        active = _active_runs(app).get(run["id"])
        live = active is not None and _is_active_run_entry_live(active)
        state = active.state if live else str(run["status"])
        completed = active.completed_points if live else int(run.get("completed_points", 0))
        total = active.total_points if live else int(run.get("total_points", 0))
        elapsed = _elapsed_seconds(active) if live else run.get("elapsed_seconds")
        eta = _eta_seconds(active) if live else None
        can_abort = live and state in {"queued", "running"} and not active.abort_requested
        # Runs saved before the worker choice ran on one worker.
        resolved = (
            active.resolved_workers
            if live
            else run.get("resolved_workers", run.get("requested_workers", 1))
        )
    return {
        "state": state,
        "completed_points": completed,
        "total_points": total,
        "remaining_points": max(total - completed, 0),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "worker_mode": run.get("worker_mode", "manual"),
        "requested_workers": run.get("requested_workers", 1),
        "resolved_workers": resolved,
        "error_text": run.get("error_text", ""),
        "can_abort": can_abort,
    }


def _worker(app: Any, data_dir: Path, run_id: str) -> None:
    store = pgm_store(data_dir)
    state = "failed"
    error_text = ""
    completed = 0
    last_saved = -10
    started = time.monotonic()
    try:
        manifest = store.load(run_id)
        # Wait here for room in the global pool; the run stays queued meanwhile.
        granted = allocate_workers(run_id)
        setting = manifest.get("max_workers_setting", 1)
        workers = min(granted, setting) if isinstance(setting, int) else granted
        _update_active_run(app, run_id, state="running", started=True, resolved_workers=workers)
        store.update(
            run_id, lambda saved: saved.update(status="running", resolved_workers=workers)
        )

        def progress(done: int, total: int) -> None:
            nonlocal completed, last_saved
            completed = done
            _update_active_run(app, run_id, completed_points=done)
            percent = int(100 * done / max(total, 1))
            if percent >= last_saved + 5 or done >= total:
                store.update(
                    run_id,
                    lambda saved: saved.update(
                        completed_points=done,
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                last_saved = percent

        run_dir = store.run_dir(run_id)
        last_nodes_saved = 0.0

        def maps_ready(energies, cff, geometry, mirror_reflectivity) -> None:
            try:
                save_base_maps(run_dir, energies, cff, geometry, mirror_reflectivity)
            except OSError:
                app.logger.warning("Could not save live PGM maps for %s", run_id, exc_info=True)

        def nodes_ready(e_nodes, c_nodes, eta_nodes) -> None:
            nonlocal last_nodes_saved
            now = time.monotonic()
            if now - last_nodes_saved < LIVE_SAVE_SECONDS:
                return
            last_nodes_saved = now
            try:
                save_nodes(run_dir, e_nodes, c_nodes, eta_nodes)
            except OSError:
                app.logger.warning("Could not save live PGM nodes for %s", run_id, exc_info=True)

        parameters = _parameters(
            manifest,
            data_dir,
            stop=stop_event(app, run_id),
            progress=progress,
            maps_ready=maps_ready,
            nodes_ready=nodes_ready,
            workers=workers,
            worker_pids=lambda pids: _update_active_run(app, run_id, simulation_pids=pids),
        )
        result = run_working_range(parameters)
        parameters.check_stopped()
        save_final_maps(
            run_dir,
            result.energies_eV,
            result.cff_values,
            result.geometry,
            result.optics,
        )
        parameters.check_stopped()
        state = "completed"
    except PGMCancelled:
        state = "aborted"
    except Exception as error:
        # Stopping ends the worker processes, which can surface as a solver error.
        if stop_event(app, run_id).is_set():
            state = "aborted"
        else:
            app.logger.exception("PGM run %s failed", run_id)
            error_text = str(error)
    finally:
        release_workers(run_id)
        store.update(
            run_id,
            lambda saved: saved.update(
                status=state,
                error_text=error_text,
                completed_points=completed,
                elapsed_seconds=time.monotonic() - started,
                updated_at=datetime.now().isoformat(timespec="seconds"),
            ),
        )
        _finish_active_run(
            app, run_id, state=state, error_text=error_text, completed_points=completed
        )


def abort_if_orphaned(app: Any, data_dir: Path, run: dict[str, Any]) -> None:
    """Record a run the stopped app left as running as aborted, also in the given manifest."""
    if is_orphaned(app, run["id"], run.get("status")):
        pgm_store(data_dir).update(
            run["id"],
            lambda saved: saved.update(
                status="aborted",
                error_text="The app stopped before this run finished.",
            ),
        )
        run["status"] = "aborted"


def load_pgm_maps(data_dir: Path, run_id: str) -> dict[str, np.ndarray]:
    """Load completed map arrays without allowing a path outside this run."""
    path = pgm_store(data_dir).run_dir(run_id) / FINAL_FILE
    with np.load(path, allow_pickle=False) as saved:
        maps = {key: saved[key] for key in saved.files}
    if "optical_map" in maps:
        maps["optical"] = maps.pop("optical_map")
    return {key: maps[key] for key in ("energies", "cff", *MAP_KEYS)}
