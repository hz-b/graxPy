"""Flask application for the local grax web MVP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

try:
    import plotly.graph_objects as go
    import plotly.io as plotly_io
    from plotly.offline import get_plotlyjs
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime.
    go = None
    plotly_io = None
    get_plotlyjs = None
    make_subplots = None

from grax.materials import available_material_symbols, material_density_catalog, material_density_g_cm3
from grax.simulation.core import normalize_polarization

from .persistence import GratingStore, build_grating_from_spec
from .runs import RunStore

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime.
    class _MissingPsutil:
        """Fallback object that keeps the psutil interface patchable in tests."""

        def virtual_memory(self) -> Any:
            """Raise a stable error when psutil is unavailable."""

            raise RuntimeError("psutil is not installed.")

    psutil = _MissingPsutil()


def _require_plotly() -> None:
    """Require Plotly runtime dependencies for the Web UI."""

    if go is None or plotly_io is None or get_plotlyjs is None or make_subplots is None:
        raise RuntimeError(
            "Interactive Web UI plots require Plotly. Install them with either "
            "python -m pip install \"graxpy[web]\" or, from a local checkout, "
            "python -m pip install -e \".[web]\""
        )


def _plotly_bundle_text() -> str:
    """Return the bundled Plotly JavaScript source."""

    _require_plotly()
    assert get_plotlyjs is not None
    return str(get_plotlyjs())


PLOTLY_MARKER_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("circle", "Circle"),
    ("square", "Square"),
    ("diamond", "Diamond"),
    ("triangle-up", "Triangle up"),
    ("triangle-down", "Triangle down"),
    ("cross", "Cross"),
    ("x", "X"),
    ("star", "Star"),
)
PLOTLY_DEFAULT_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


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


def _active_runs(app: Any) -> dict[str, ActiveRunState]:
    """Return the active in-memory run registry for one Flask app."""

    return app.extensions.setdefault("grax_active_runs", {})


def _active_runs_lock(app: Any) -> threading.Lock:
    """Return the lock guarding active-run state for one Flask app."""

    return app.extensions.setdefault("grax_active_runs_lock", threading.Lock())


def create_app(*, data_dir: str | Path | None = None):
    """Create the local Flask app.

    Args:
        data_dir: Directory used for saved gratings, run artifacts, and previews.

    Returns:
        Configured Flask application.
    """
    try:
        from flask import (
            Flask,
            abort,
            jsonify,
            redirect,
            render_template,
            request,
            send_from_directory,
            url_for,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by user startup.
        raise RuntimeError(
            "The web app requires Flask. Install it with either "
            "python -m pip install \"graxpy[web]\" or, from a local checkout, "
            "python -m pip install -e \".[web]\""
        ) from error

    app = Flask(__name__)
    app.config["GRAx_DATA_DIR"] = Path(data_dir or Path.cwd() / ".grax-web").resolve()
    def store() -> GratingStore:
        return GratingStore(app.config["GRAx_DATA_DIR"] / "saved_gratings")

    def run_store() -> RunStore:
        return RunStore(app.config["GRAx_DATA_DIR"] / "runs")

    def preview_root() -> Path:
        return app.config["GRAx_DATA_DIR"] / "previews" / "live"

    def active_data_dir() -> Path:
        return Path(app.config["GRAx_DATA_DIR"]).resolve()

    @app.context_processor
    def inject_site_metadata() -> dict[str, Any]:
        """Inject shared site attribution metadata into templates."""

        return {"site_metadata": _site_metadata()}

    @app.get("/")
    def index():
        data_dir = active_data_dir()
        return render_template(
            "index.html",
            gratings=store().list(),
            runs=_list_plot_runs(data_dir / "runs"),
            results_dir=data_dir / "runs",
            plots_dir=data_dir / "plots",
            active_workspace=data_dir,
            message=str(request.args.get("message", "")).strip(),
            message_kind=str(request.args.get("kind", "info")).strip() or "info",
        )

    @app.get("/docs")
    def web_docs():
        return render_template(
            "web_docs.html",
            workspace_root=active_data_dir(),
        )

    @app.post("/workspace")
    def switch_workspace():
        if _has_active_runs(app):
            return redirect(url_for("index", message="Cannot switch workspace while active runs exist.", kind="error"))
        try:
            target_dir = _validate_workspace_root(request.form.get("workspace_root"))
        except (OSError, ValueError) as error:
            return redirect(url_for("index", message=str(error), kind="error"))
        app.config["GRAx_DATA_DIR"] = target_dir
        return redirect(url_for("index", message=f"Workspace switched to {target_dir}", kind="success"))

    @app.get("/gratings/new")
    def new_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=_default_form_values(),
        )
        defaults = _default_form_values()
        return render_template(
            "grating_form.html",
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            defaults=defaults,
            density_placeholders=_material_density_placeholders(defaults),
            action_url=url_for("create_grating"),
            submit_label="Save grating",
            preview=preview,
        )

    @app.post("/gratings")
    def create_grating():
        spec = _spec_from_form(request.form)
        try:
            build_grating_from_spec(spec)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        saved = store().save(spec)
        return redirect(url_for("grating_detail", grating_id=saved["id"]))

    @app.get("/gratings/manage")
    def manage_gratings():
        return render_template(
            "grating_manage.html",
            gratings=store().list(),
        )

    @app.post("/gratings/manage")
    def update_gratings():
        action = str(request.form.get("action", "delete"))
        if action != "delete":
            return redirect(url_for("manage_gratings"))
        delete_mode = str(request.form.get("delete_mode", "grating_only")).strip()
        selected_ids = [str(grating_id) for grating_id in request.form.getlist("delete_grating_id")]
        if not selected_ids:
            return redirect(url_for("manage_gratings"))
        if delete_mode == "grating_and_runs":
            active_linked = [
                run_id
                for grating_id in selected_ids
                for run_id in [str(run["id"]) for run in _linked_runs(active_data_dir(), grating_id)]
                if _is_run_active(app, run_id)
            ]
            if active_linked:
                return redirect(url_for("manage_gratings", message="Cannot delete linked runs while active runs exist."))
        for grating_id in selected_ids:
            if delete_mode == "grating_and_runs":
                run_store().delete_many([str(run["id"]) for run in _linked_runs(active_data_dir(), grating_id)])
            store().delete(grating_id)
            preview_path = active_data_dir() / "previews" / f"{grating_id}.png"
            if preview_path.exists():
                preview_path.unlink()
        return redirect(url_for("manage_gratings"))

    @app.get("/gratings/<grating_id>")
    def grating_detail(grating_id: str):
        grating_store = store()
        spec = grating_store.load(grating_id)
        try:
            grating = build_grating_from_spec(spec)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        preview_path = active_data_dir() / "previews" / f"{grating_id}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        grating.plot_profile(preview_path)
        return render_template(
            "grating_detail.html",
            grating=spec,
            preview_url=url_for("data_file", filename=f"previews/{grating_id}.png"),
        )

    @app.get("/gratings/<grating_id>/delete")
    def delete_grating_prompt(grating_id: str):
        spec = store().load(grating_id)
        linked_runs = _linked_runs(active_data_dir(), grating_id)
        return render_template(
            "grating_delete.html",
            grating=spec,
            linked_runs=linked_runs,
        )

    @app.post("/gratings/<grating_id>/delete")
    def delete_grating(grating_id: str):
        spec = store().load(grating_id)
        delete_mode = str(request.form.get("delete_mode", "grating_only")).strip()
        linked_runs = _linked_runs(active_data_dir(), grating_id)
        if delete_mode == "grating_and_runs":
            active_linked = [run_id for run_id in [str(run["id"]) for run in linked_runs] if _is_run_active(app, run_id)]
            if active_linked:
                return redirect(
                    url_for(
                        "delete_grating_prompt",
                        grating_id=grating_id,
                        message="Cannot delete linked runs while active runs exist.",
                    )
                )
            run_store().delete_many([str(run["id"]) for run in linked_runs])
        store().delete(grating_id)
        preview_path = active_data_dir() / "previews" / f"{grating_id}.png"
        if preview_path.exists():
            preview_path.unlink()
        message = (
            f"Deleted grating '{spec['name']}' and {len(linked_runs)} linked runs."
            if delete_mode == "grating_and_runs"
            else f"Deleted grating '{spec['name']}'."
        )
        return redirect(url_for("index", message=message, kind="success"))

    @app.get("/gratings/<grating_id>/edit")
    def edit_grating(grating_id: str):
        spec = store().load(grating_id)
        defaults = _form_values_from_spec(spec)
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=defaults,
        )
        return render_template(
            "grating_form.html",
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            defaults=defaults,
            density_placeholders=_material_density_placeholders(defaults),
            action_url=url_for("update_grating", grating_id=grating_id),
            submit_label="Update grating",
            preview=preview,
        )

    @app.post("/gratings/<grating_id>")
    def update_grating(grating_id: str):
        grating_store = store()
        previous = grating_store.load(grating_id)
        spec = _spec_from_form(request.form)
        spec["id"] = grating_id
        spec["created_at"] = previous.get("created_at")
        try:
            build_grating_from_spec(spec)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        grating_store.save(spec)
        return redirect(url_for("grating_detail", grating_id=grating_id))

    @app.post("/gratings/<grating_id>/runs")
    def create_run(grating_id: str):
        spec = store().load(grating_id)
        try:
            grating = build_grating_from_spec(spec)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        run = _queue_run(
            app=app,
            data_dir=app.config["GRAx_DATA_DIR"],
            grating_id=grating_id,
            grating_name=str(spec["name"]),
            grating_spec=spec,
            grating=grating,
            form=request.form,
        )
        return redirect(url_for("run_detail", run_id=run["id"]))

    @app.post("/_preview/grating")
    def preview_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=request.form,
        )
        return jsonify(preview)

    @app.get("/runs/manage")
    def manage_runs():
        return render_template(
            "run_manage.html",
            runs=_list_plot_runs(active_data_dir() / "runs"),
        )

    @app.post("/runs/manage")
    def update_runs():
        action = str(request.form.get("action", "save"))
        store = run_store()
        if action == "delete":
            store.delete_many(request.form.getlist("delete_run_id"))
            return redirect(url_for("manage_runs"))

        for run in store.list():
            run_id = str(run["id"])
            field_name = f"display_name_{run_id}"
            comment_field_name = f"comment_{run_id}"
            if field_name in request.form:
                store.rename(run_id, str(request.form.get(field_name, "")))
            if comment_field_name in request.form:
                store.update_comment(run_id, str(request.form.get(comment_field_name, "")))
        return redirect(url_for("manage_runs"))

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        manifest["results_dir"] = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        return render_template(
            "run_detail.html",
            run=manifest,
            initial_status=_run_status_payload(app=app, data_dir=app.config["GRAx_DATA_DIR"], run_id=run_id),
            status_url=url_for("run_status", run_id=run_id),
            memory_url=url_for("system_memory"),
            abort_url=url_for("abort_run_dialog", run_id=run_id),
        )

    @app.get("/runs/<run_id>/abort")
    def abort_run_dialog(run_id: str):
        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        status = _run_status_payload(app=app, data_dir=app.config["GRAx_DATA_DIR"], run_id=run_id)
        if status is None:
            abort(404)
        return render_template("run_abort.html", run=manifest, status=status)

    @app.post("/runs/<run_id>/abort")
    def abort_run(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None:
            abort(404)
        disposition = str(request.form.get("disposition", "save")).strip().lower()
        if disposition not in {"save", "delete"}:
            abort(400)
        _abort_run(
            app=app,
            data_dir=active_data_dir(),
            run_id=run_id,
            delete_after_abort=disposition == "delete",
        )
        if disposition == "delete" and _load_run_manifest(active_data_dir(), run_id) is None:
            return redirect(url_for("manage_runs"))
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/status")
    def run_status(run_id: str):
        payload = _run_status_payload(app=app, data_dir=app.config["GRAx_DATA_DIR"], run_id=run_id)
        if payload is None:
            abort(404)
        return jsonify(payload)

    @app.get("/system/memory")
    def system_memory():
        return jsonify(_memory_status_payload())

    @app.get("/system/resource-status")
    def system_resource_status():
        from .resource_manager import resource_status
        return jsonify(resource_status())

    @app.get("/plots")
    def plot_index():
        plots = _list_plots(active_data_dir() / "plots")
        return render_template("plot_list.html", plots=plots)

    @app.get("/plots/new")
    @app.get("/runs/compare")
    def plot_new():
        runs = _list_plot_runs(active_data_dir() / "runs")
        return render_template(
            "plot_form.html",
            runs=runs,
            run_options=runs,
            results_dir=active_data_dir() / "runs",
            preview=None,
            plotly_bundle=_plotly_bundle_text(),
            marker_symbols=PLOTLY_MARKER_SYMBOLS,
        )

    @app.post("/_preview/plot")
    def preview_plot():
        preview = _build_plot_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=request.form,
        )
        return jsonify(preview)

    @app.get("/plots/export")
    def plot_export_dialog():
        preview_id = str(request.args.get("preview_id", "")).strip()
        browse_path = _safe_browse_path(request.args.get("path"))
        return render_template(
            "plot_export_dialog.html",
            preview_id=preview_id,
            current_path=str(browse_path),
            parent_path=str(browse_path.parent) if browse_path.parent != browse_path else None,
            breadcrumbs=_path_breadcrumbs(browse_path),
            entries=_list_directory_entries(browse_path),
        )

    @app.post("/plots/export")
    def export_plot():
        preview_id = str(request.form.get("preview_id", "")).strip()
        directory = _safe_browse_path(request.form.get("directory"))
        filename = str(request.form.get("filename", "")).strip()
        overwrite = str(request.form.get("overwrite", "")).strip() == "1"
        if not filename:
            return jsonify({"ok": False, "error": "Choose a filename."})
        if not filename.lower().endswith(".png"):
            filename = f"{filename}.png"
        source_path = preview_root() / f"{preview_id}.png"
        if not source_path.exists():
            return jsonify({"ok": False, "error": "Preview image is no longer available."})
        output_path = directory / Path(filename).name
        if output_path.exists() and not overwrite:
            return jsonify({"ok": False, "error": "Target file already exists.", "exists": True})
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output_path)
        return jsonify({"ok": True, "output_path": str(output_path)})

    @app.post("/plots")
    @app.post("/runs/compare")
    def create_plot():
        run_ids = request.form.getlist("run_ids")
        if not run_ids:
            abort(400, "Select at least one saved run.")
        order_selection = {
            run_id: [int(value) for value in request.form.getlist(f"orders_{run_id}")]
            for run_id in run_ids
        }
        for run_id, orders in order_selection.items():
            if not orders:
                abort(400, f"Select at least one diffraction order for run {run_id}.")
        plot_options = _plot_options_from_form(
            form_data=request.form,
            run_ids=run_ids,
            order_selection=order_selection,
        )
        plot = _build_combined_plot(
            data_dir=app.config["GRAx_DATA_DIR"],
            run_ids=run_ids,
            order_selection=order_selection,
            title=plot_options["title"],
            plot_options=plot_options,
        )
        return redirect(url_for("plot_detail", plot_id=plot["id"]))

    @app.get("/plots/<plot_id>")
    def plot_detail(plot_id: str):
        plot_dir = app.config["GRAx_DATA_DIR"] / "plots" / plot_id
        manifest_path = plot_dir / "manifest.json"
        if not manifest_path.exists():
            abort(404)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["plots_dir"] = app.config["GRAx_DATA_DIR"] / "plots"
        if "figure_json" not in manifest and manifest.get("plot_config") is not None:
            run_ids = [str(run["id"]) for run in manifest.get("selected_runs", [])]
            order_selection = {
                str(run["id"]): [int(value) for value in run.get("orders", [])]
                for run in manifest.get("selected_runs", [])
            }
            try:
                _, figure_json, _ = _build_plotly_compare_payload(
                    data_dir=app.config["GRAx_DATA_DIR"],
                    run_ids=run_ids,
                    order_selection=order_selection,
                    plot_options=dict(manifest["plot_config"]),
                )
                manifest["figure_json"] = figure_json
            except ValueError:
                manifest["figure_json"] = None
        return render_template(
            "plot_detail.html",
            plot=manifest,
            plotly_bundle=_plotly_bundle_text() if manifest.get("figure_json") else None,
        )

    @app.get("/plots/<plot_id>/delete")
    @app.post("/plots/<plot_id>/delete")
    def plot_delete(plot_id: str):
        plot_dir = app.config["GRAx_DATA_DIR"] / "plots" / plot_id
        manifest_path = plot_dir / "manifest.json"
        if not manifest_path.exists():
            abort(404)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if request.method == "POST":
            shutil.rmtree(plot_dir, ignore_errors=True)
            return redirect(url_for("plot_index"))
        return render_template("plot_delete.html", plot=manifest)

    @app.get("/_data/<path:filename>")
    def data_file(filename: str):
        return send_from_directory(app.config["GRAx_DATA_DIR"], filename)

    return app


def main() -> None:
    """Run the local development server."""

    parser = argparse.ArgumentParser(prog="grax-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    app = create_app()
    _maybe_open_browser(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=True)


def _maybe_open_browser(host: str, port: int) -> None:
    """Open the local web UI in the default browser when appropriate.

    Args:
        host: Flask bind host.
        port: Flask bind port.
    """

    reloader_state = os.environ.get("WERKZEUG_RUN_MAIN")
    if reloader_state not in (None, "true"):
        return
    if reloader_state is None and os.environ.get("FLASK_RUN_FROM_CLI") is None:
        return

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{int(port)}"
    timer = threading.Timer(1.0, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def _default_form_values() -> dict[str, str]:
    """Return display defaults for the create-grating form."""
    return {
        "period_lpermm": "400",
        "grating_type": "laminar",
        "x_resolution_nm": "1.0",
        "z_resolution_nm": "1.0",
        "width_to_period_ratio": "0.67",
        "depth_nm": "14.9",
        "left_wall_angle_deg": "15.0",
        "right_wall_angle_deg": "15.0",
        "blaze_angle_deg": "0.75",
        "anti_blaze_angle_deg": "",
        "substrate_material": "Si",
        "substrate_material_density_g_cm3": _default_density_text("Si"),
        "stack_type": "single_layer",
        "layer_material": "Pt",
        "layer_material_density_g_cm3": _default_density_text("Pt"),
        "layer_thickness_nm": "28.77",
        "material_a": "Cr",
        "material_a_density_g_cm3": _default_density_text("Cr"),
        "material_b": "C",
        "material_b_density_g_cm3": _default_density_text("C"),
        "d_period_nm": "6.5",
        "gamma": "0.45",
        "n_bilayers": "40",
        "top_material": "C",
        "top_material_density_g_cm3": _default_density_text("C"),
        "top_cap_material": "",
        "top_cap_material_density_g_cm3": "",
        "top_cap_thickness_nm": "0.0",
        "substrate_roughness_sigma_nm": "",
        "layer_roughness_sigma_nm": "",
        "material_a_roughness_sigma_nm": "",
        "material_b_roughness_sigma_nm": "",
        "top_cap_roughness_sigma_nm": "",
    }


def _default_density_text(material_name: str) -> str:
    """Return the default density string for one built-in material."""

    density = material_density_g_cm3(material_name)
    return "" if density is None else str(density)


def _material_density_placeholders(defaults: dict[str, str]) -> dict[str, str]:
    """Return density placeholders for the currently selected materials."""
    placeholders = {}
    for field_name in (
        "substrate_material",
        "layer_material",
        "material_a",
        "material_b",
        "top_material",
        "top_cap_material",
    ):
        material_name = defaults.get(field_name, "")
        density = material_density_g_cm3(material_name)
        placeholders[field_name] = "" if density is None else str(density)
    return placeholders


def _run_result_location(data_dir: Path, run_id: str) -> Path:
    """Return the directory holding one run's saved artifacts."""
    return data_dir / "runs" / run_id


def _form_values_from_spec(spec: dict[str, Any]) -> dict[str, str]:
    """Return form defaults populated from a saved grating spec."""
    values = _default_form_values()
    values.update({key: "" if value is None else str(value) for key, value in spec.items()})
    stack = dict(spec.get("stack", {}))
    values["stack_type"] = str(stack.get("type", "single_layer"))
    for key, value in stack.items():
        if isinstance(value, dict):
            values[key] = "" if value.get("name") is None else str(value.get("name"))
            density_key = f"{key}_density_g_cm3"
            density_value = value.get("density_g_cm3")
            if density_value in (None, ""):
                values[density_key] = _default_density_text(str(value.get("name", "")))
            else:
                values[density_key] = str(density_value)
            continue
        values[key] = "" if value is None else str(value)
        if key in {
            "substrate_material",
            "layer_material",
            "material_a",
            "material_b",
            "top_material",
            "top_cap_material",
        }:
            density_key = f"{key}_density_g_cm3"
            values[density_key] = _default_density_text(values[key])
    return values


def _spec_from_form(form: Any) -> dict[str, Any]:
    """Build a saved grating spec from submitted form data."""
    grating_type = str(form["grating_type"])
    spec: dict[str, Any] = {
        "name": str(form["name"]).strip() or "Untitled grating",
        "grating_type": grating_type,
        "period_lpermm": int(float(form["period_lpermm"])),
        "x_resolution_nm": float(form["x_resolution_nm"]),
        "z_resolution_nm": float(form["z_resolution_nm"]),
        "stack": _stack_spec_from_form(form),
    }
    if grating_type == "laminar":
        spec.update(
            {
                "width_to_period_ratio": float(form["width_to_period_ratio"]),
                "depth_nm": float(form["depth_nm"]),
                "left_wall_angle_deg": float(form["left_wall_angle_deg"]),
                "right_wall_angle_deg": float(form["right_wall_angle_deg"]),
            }
        )
        return spec
    if grating_type == "blazed":
        anti_blaze_text = str(form.get("anti_blaze_angle_deg", "")).strip()
        spec.update(
            {
                "blaze_angle_deg": float(form["blaze_angle_deg"]),
                "anti_blaze_angle_deg": None if anti_blaze_text == "" else float(anti_blaze_text),
            }
        )
        return spec
    raise ValueError("Unsupported grating_type.")


def _build_grating_preview(
    *,
    data_dir: Path,
    form_data: Any,
) -> dict[str, Any]:
    """Build a live grating preview payload from form data."""
    try:
        spec = _spec_from_form(form_data)
        grating = build_grating_from_spec(spec)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "ok": False,
            "error": str(error),
            "preview_id": None,
            "preview_url": None,
        }

    preview_id = uuid4().hex
    preview_path = data_dir / "previews" / "live" / f"{preview_id}.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    grating.plot_profile(preview_path)
    return {
        "ok": True,
        "error": "",
        "preview_id": preview_id,
        "preview_url": f"/_data/previews/live/{preview_id}.png",
    }


def _stack_spec_from_form(form: Any) -> dict[str, Any]:
    """Build a stack spec from submitted form data."""
    stack_type = str(form["stack_type"])
    top_cap_thickness = float(form.get("top_cap_thickness_nm", 0.0) or 0.0)
    if stack_type == "multilayer":
        return {
            "type": "multilayer",
            "substrate_material": _material_spec_from_form(form, "substrate_material"),
            "material_a": _material_spec_from_form(form, "material_a"),
            "material_b": _material_spec_from_form(form, "material_b"),
            "d_period_nm": float(form["d_period_nm"]),
            "gamma": float(form["gamma"]),
            "n_bilayers": int(float(form["n_bilayers"])),
            "top_material": _material_spec_from_form(form, "top_material"),
            "top_cap_material": _optional_material_spec_from_form(form, "top_cap_material"),
            "top_cap_thickness_nm": top_cap_thickness,
            "substrate_roughness_sigma_nm": _optional_float(form, "substrate_roughness_sigma_nm"),
            "material_a_roughness_sigma_nm": _optional_float(form, "material_a_roughness_sigma_nm"),
            "material_b_roughness_sigma_nm": _optional_float(form, "material_b_roughness_sigma_nm"),
            "top_cap_roughness_sigma_nm": _optional_float(form, "top_cap_roughness_sigma_nm"),
        }
    return {
        "type": "single_layer",
        "substrate_material": _material_spec_from_form(form, "substrate_material"),
        "layer_material": _material_spec_from_form(form, "layer_material"),
        "layer_thickness_nm": float(form["layer_thickness_nm"]),
        "top_cap_material": _optional_material_spec_from_form(form, "top_cap_material"),
        "top_cap_thickness_nm": top_cap_thickness,
        "substrate_roughness_sigma_nm": _optional_float(form, "substrate_roughness_sigma_nm"),
        "layer_roughness_sigma_nm": _optional_float(form, "layer_roughness_sigma_nm"),
        "top_cap_roughness_sigma_nm": _optional_float(form, "top_cap_roughness_sigma_nm"),
    }


def _optional_float(form: Any, field_name: str) -> float | None:
    """Return an optional float from a form field, treating blanks as unset."""
    text = str(form.get(field_name, "")).strip()
    if text == "":
        return None
    return float(text)


def _material_spec_from_form(form: Any, field_name: str) -> dict[str, Any]:
    """Return one serialized material spec from form fields."""
    name = str(form[field_name]).strip()
    density_text = str(form.get(f"{field_name}_density_g_cm3", "")).strip()
    density = None if density_text == "" else float(density_text)
    return {"name": name, "density_g_cm3": density}


def _optional_material_spec_from_form(form: Any, field_name: str) -> dict[str, Any] | None:
    """Return one optional serialized material spec from form fields."""
    name = str(form.get(field_name, "")).strip()
    if name == "":
        return None
    return _material_spec_from_form(form, field_name)


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
    run_id = _run_id(workflow, grating_id)
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    energies = np.linspace(
        float(form["energy_start_ev"]),
        float(form["energy_stop_ev"]),
        int(float(form["energy_points"])),
    )
    diffraction_order = int(float(form.get("diffraction_order", 1)))
    fourier_orders = int(float(form.get("fourier_orders", 5)))
    polarization = _normalized_polarization(form.get("polarization", "s"))
    solver = _normalized_solver(form.get("solver", "rcwa"))
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
        "total_points": int(energies.size),
        "run_input": _run_input_from_form(form),
        "resume_count": 0,
    }
    RunStore(data_dir / "runs").save(manifest)

    _start_run_worker(
        app=app,
        data_dir=data_dir,
        run_id=run_id,
        grating_id=grating_id,
        grating_name=grating_name,
        grating=grating,
        workflow=workflow,
        energies=energies,
        form_data=dict(form),
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
        total_points=int(energies.size),
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
            run_x_resolution_nm = float(form_data.get("run_x_resolution_nm") or grating.x_resolution_nm)
            run_z_resolution_nm = float(form_data.get("run_z_resolution_nm") or grating.z_resolution_nm)
            result = parameter_sweep.run_parameter_study(
                grating=grating,
                energies_ev=energies,
                grazing_angle_deg=float(form_data["grazing_angle_deg"]),
                diffraction_order=diffraction_order,
                polarization=_normalized_polarization(form_data.get("polarization", "s")),
                solver=_normalized_solver(form_data.get("solver", "rcwa")),
                fourier_orders_values=[fourier_orders],
                x_resolution_values=[run_x_resolution_nm],
                z_resolution_values=[run_z_resolution_nm],
                output_dir=run_dir,
                save_csv=True,
                show_progress=False,
            )
            plot_path = run_dir / "parameter_study.png"
            parameter_sweep.plot_parameter_study(result, output_filename=plot_path)
            _update_active_run(
                app,
                run_id,
                completed_points=int(energies.size),
                plot_relative_path="parameter_study.png",
                bump_plot_token=True,
            )
            _update_manifest_fields(
                data_dir,
                run_id,
                status="completed",
                artifacts=["parameter_study.png"],
                total_points=int(energies.size),
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
            _publish_live_progress_if_due(
                app=app,
                run_id=run_id,
                results=results,
                output_path=live_plot_path,
                diffraction_order=diffraction_order,
                title=f"{grating_name} {workflow}",
                min_interval_seconds=0.0,
            )
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
        _persist_run_outputs(
            data_dir=data_dir,
            run_id=run_id,
            results=results,
            diffraction_order=diffraction_order,
            title=f"{grating_name} {workflow}",
            include_plot=bool(results),
        )
        _update_manifest_fields(
            data_dir,
            run_id,
            status="failed",
            error_text=str(error),
            cases=_manifest_cases(results),
            artifacts=_run_artifacts_for_results(run_dir, include_plot=bool(results)),
        )
        _finish_active_run(app, run_id, state="failed", error_text=str(error))
    finally:
        release_workers(run_id)


def _worker_settings_from_form(form: Any) -> tuple[str, str | int, int | None]:
    """Return worker-mode metadata and the runner max_workers setting."""

    worker_mode = str(form.get("max_workers_mode", "auto") or "auto").strip().lower()
    if worker_mode == "manual":
        requested_workers = int(float(form.get("max_workers", 1)))
        if requested_workers < 1:
            raise ValueError("Manual worker count must be at least 1.")
        return "manual", requested_workers, requested_workers
    return "auto", "auto", None


def _run_input_from_form(form: Any) -> dict[str, Any]:
    """Return persisted run-input metadata needed for checkpoint resume."""

    input_data: dict[str, Any] = {}
    for key in form.keys():
        values = form.getlist(key)
        input_data[key] = values if len(values) > 1 else form.get(key)
    input_data["polarization"] = _normalized_polarization(input_data.get("polarization", "s"))
    input_data["solver"] = _normalized_solver(input_data.get("solver", "rcwa"))
    return input_data


def _normalized_polarization(value: Any) -> str:
    """Return a canonical polarization code for Web UI run inputs.

    Delegates to :func:`grax.normalize_polarization`, so the form accepts the
    same ``TE``/``TM`` aliases the Python API does and stores the canonical
    ``"s"``/``"p"``.
    """

    return normalize_polarization(value or "s")


def _normalized_solver(value: Any) -> str:
    """Return a validated electromagnetic solver name for Web UI run inputs."""

    solver = str(value or "rcwa").strip().lower()
    if solver not in {"rcwa", "neviere"}:
        raise ValueError("solver must be 'rcwa' or 'neviere'.")
    return solver


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


def _load_checkpoint_results(run_dir: Path) -> list[Any]:
    """Load deduplicated checkpointed case results for one run directory."""

    from grax.simulation.serialization import _load_checkpoint_case_results

    checkpoint_path = run_dir / "checkpoints" / "results.jsonl"
    if not checkpoint_path.exists():
        return []
    results = list(_load_checkpoint_case_results(checkpoint_path).values())
    return sorted(results, key=lambda result: (float(result.energy_ev), int(result.index)))


def _checkpoint_counts(run_dir: Path) -> tuple[int, int]:
    """Return completed and total point counts for one run directory."""

    manifest = _load_run_manifest(run_dir.parent.parent, run_dir.name)
    total_points = int((manifest or {}).get("total_points") or 0)
    completed_points = len(_load_checkpoint_results(run_dir))
    return completed_points, total_points


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


def _manifest_cases(results: list[Any]) -> list[dict[str, Any]]:
    """Return manifest-ready case summaries for persisted run metadata."""

    return [
        {
            "case_id": result.case_id,
            "energy_ev": result.energy_ev,
            "grazing_angle_deg": result.grazing_angle_deg,
            "selected_efficiency": result.selected_efficiency,
            "status": result.status,
            "error_message": result.error_message,
        }
        for result in results
    ]


def _run_artifacts_for_results(run_dir: Path, *, include_plot: bool) -> list[str]:
    """Return the artifact filenames currently available for one run."""

    artifact_names = ["summary.csv", "all_orders.csv"]
    if include_plot and (run_dir / "selected_efficiency.png").exists():
        artifact_names.append("selected_efficiency.png")
    if (run_dir / "parameter_study.png").exists():
        artifact_names.append("parameter_study.png")
    return artifact_names


def _persist_run_outputs(
    *,
    data_dir: Path,
    run_id: str,
    results: list[Any],
    diffraction_order: int,
    title: str,
    include_plot: bool,
) -> None:
    """Persist current partial or final run artifacts from collected results."""

    if not results:
        return
    from grax import simulation

    run_dir = data_dir / "runs" / run_id
    _write_summary_csv(results, run_dir / "summary.csv")
    simulation.write_all_orders_csv(results, run_dir / "all_orders.csv")
    if include_plot:
        simulation.plot_order_subset(
            results,
            run_dir / "selected_efficiency.png",
            diffraction_orders=[diffraction_order],
            title=title,
        )


def _site_metadata() -> dict[str, str]:
    """Return shared attribution metadata for the web UI."""

    return {
        "organization": "Helmholtz-Zentrum Berlin",
        "contact_name": "Simone Vadilonga",
        "contact_email": "simone.vadilonga@helmholtz-berlin.de",
        "github_url": "https://github.com/hz-b/graxPy",
        "docs_url": "https://graxpy.readthedocs.io/en/latest/",
    }


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

    ordered_results = _results_sorted_for_live_plot(results, x_key="energy_ev")
    if not ordered_results:
        return False

    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
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


def _load_run_manifest(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Load one run manifest if it exists."""

    manifest_path = data_dir / "runs" / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("id", run_id)
    payload.setdefault("display_name", f"{payload.get('grating_name', 'Run')} · {payload.get('workflow', 'run')}")
    payload.setdefault("polarization", _normalized_polarization(payload.get("run_input", {}).get("polarization", "s")))
    payload.setdefault("solver", _normalized_solver(payload.get("run_input", {}).get("solver", "rcwa")))
    if isinstance(payload.get("run_input"), dict):
        payload["run_input"].setdefault("polarization", payload["polarization"])
    return payload


def _update_manifest_fields(data_dir: Path, run_id: str, **updates: Any) -> dict[str, Any]:
    """Update one run manifest in place and return the saved payload."""

    store = RunStore(data_dir / "runs")
    payload = store.load(run_id)
    payload.update({key: value for key, value in updates.items() if value is not None})
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return store.save(payload)


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


def _is_active_run_entry_live(active: ActiveRunState) -> bool:
    """Return whether one in-memory active-run entry still has a live worker."""

    if active.state not in {"queued", "running", "aborting"}:
        return False
    worker = active.worker_thread
    return worker is not None and worker.is_alive()


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
    completed_points = max(
        checkpoint_completed_points,
        int(len(manifest.get("cases", [])) or 0),
    )
    total_points = int(manifest.get("total_points") or checkpoint_total_points or completed_points)
    normalized_state = "completed" if manifest_state == "ok" else manifest_state
    can_abort = manifest_state not in {"ok", "completed", "failed", "aborted"}
    active_job_missing = normalized_state in {"queued", "running", "aborting"}
    if active_job_missing and checkpoint_completed_points > 0:
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
        "elapsed_seconds": None,
        "eta_seconds": None,
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


def _preferred_run_plot_path(data_dir: Path, run_id: str) -> str | None:
    """Return the best available plot file for one run."""

    run_dir = data_dir / "runs" / run_id
    for candidate in ("selected_efficiency.png", "parameter_study.png", "live_progress.png"):
        if (run_dir / candidate).exists():
            return candidate
    return None


def _file_token(path: Path) -> str:
    """Return a cheap cache-busting token for one file path."""

    if not path.exists():
        return ""
    return str(path.stat().st_mtime_ns)


def _run_plot_url(*, data_dir: Path, run_id: str, relative_path: str | None, token: str) -> str | None:
    """Return the cache-busted plot URL for one run image."""

    if not relative_path:
        return None
    plot_path = data_dir / "runs" / run_id / relative_path
    if not plot_path.exists():
        return None
    suffix = f"?v={token or _file_token(plot_path)}"
    return f"/_data/runs/{run_id}/{relative_path}{suffix}"


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


def _attach_roughness(grating: Any, form_data: Any) -> None:
    """Attach a run-time roughness kind to the grating from the run form.

    The per-layer sigma magnitudes live on the grating's coating stack. The kind
    is chosen per run; ``sigma_nm=0.0`` is the fallback for any interface left
    unset. ``"none"`` (or blank) leaves the grating unroughened.
    """

    kind = str(form_data.get("roughness_kind", "none") or "none").strip()
    if kind in {"none", ""}:
        grating.roughness = None
        return
    from grax import RoughnessSpec

    grating.roughness = RoughnessSpec(kind=kind, sigma_nm=0.0, seed=0)


def _cases_for_workflow(
    *,
    simulation: Any,
    workflow: str,
    grating: Any,
    energies: np.ndarray,
    form: Any,
    diffraction_order: int,
    fourier_orders: int,
) -> list[dict[str, Any]]:
    """Return batch cases for one supported workflow."""
    run_x_resolution_nm = float(form.get("run_x_resolution_nm") or grating.x_resolution_nm)
    run_z_resolution_nm = float(form.get("run_z_resolution_nm") or grating.z_resolution_nm)
    polarization = _normalized_polarization(form.get("polarization", "s"))
    solver = _normalized_solver(form.get("solver", "rcwa"))

    if workflow == "fixed_angle":
        cases = list(
            simulation.fixed_angle_cases(
                grating=grating,
                energies_ev=energies,
                grazing_angle_deg=float(form["grazing_angle_deg"]),
                polarization=polarization,
            )
        )
    elif workflow == "monochromator":
        cases = list(
            simulation.monochromator_cases(
                grating=grating,
                energies_ev=energies,
                diffraction_order=diffraction_order,
                cff=float(form.get("cff", 2.25) or 2.25),
                polarization=polarization,
            )
        )
    elif workflow == "multilayer_theta_search":
        cases = list(
            simulation.multilayer_theta_search_cases(
                grating=grating,
                energies_ev=energies,
                diffraction_order=diffraction_order,
                rough_x_resolution_nm=run_x_resolution_nm,
                rough_z_resolution_nm=run_z_resolution_nm,
                fine_x_resolution_nm=run_x_resolution_nm,
                fine_z_resolution_nm=run_z_resolution_nm,
                final_fourier_orders=fourier_orders,
                final_x_resolution_nm=run_x_resolution_nm,
                final_z_resolution_nm=run_z_resolution_nm,
            )
        )
    else:
        raise ValueError("Unsupported workflow.")
    for index, case in enumerate(cases):
        case["case_id"] = f"{workflow}-{index:08d}"
        case["fourier_orders"] = fourier_orders
        case["polarization"] = polarization
        case["solver"] = solver
        if workflow != "multilayer_theta_search":
            case["x_resolution_nm"] = run_x_resolution_nm
            case["z_resolution_nm"] = run_z_resolution_nm
    return cases


def _write_summary_csv(results: list[Any], output_path: Path) -> None:
    """Write selected-order results to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "energy_ev",
                "grazing_angle_deg",
                "selected_efficiency",
                "selected_diffraction_angle_deg",
                "status",
                "error_message",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.energy_ev,
                    result.grazing_angle_deg,
                    result.selected_efficiency,
                    result.selected_diffraction_angle_deg,
                    result.status,
                    result.error_message or "",
                ]
            )


def _list_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Return persisted run manifests newest first."""
    return RunStore(run_dir).list()


def _run_summary_label(run: dict[str, Any]) -> str:
    """Return the primary saved-run descriptor used in plot selection."""

    return (
        f"name: {run.get('grating_name', 'Run')} · "
        f"date/time: {run.get('created_at', '')} · "
        f"grating type: {_run_grating_type(run)}"
    )


def _run_summary_meta(run: dict[str, Any]) -> str:
    """Return a secondary metadata line for one saved run."""

    comment = str(run.get("comment", "")).strip() or "—"
    return (
        f"sweep type: {run.get('workflow', 'run')} · "
        f"comment: {comment}"
    )


def _run_grating_type(run: dict[str, Any]) -> str:
    """Return the grating type recorded for one run."""

    grating_spec = run.get("grating_spec")
    if isinstance(grating_spec, dict):
        value = str(grating_spec.get("grating_type", "")).strip()
        if value:
            return value
    value = str(run.get("grating_type", "")).strip()
    return value or "unknown"


def _run_selector_text(run: dict[str, Any]) -> str:
    """Return the full run text shown in plot-selection UIs."""

    return f"{_run_summary_label(run)} · {_run_summary_meta(run)}"


def _list_plot_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Return run manifests augmented with available plot orders."""
    runs = []
    for run in _list_runs(run_dir):
        run_id = str(run["id"])
        run_dir_for_id = _run_result_location(run_dir.parent, run_id)
        checkpoint_completed_points, checkpoint_total_points = _checkpoint_counts(run_dir_for_id)
        state = str(run.get("status", "completed"))
        if state in {"queued", "running", "aborting"} and checkpoint_completed_points > 0:
            state = "aborted"
        elif state in {"paused", "interrupted"}:
            state = "aborted"
        enriched_run = {
            **run,
            "status": state,
        }
        runs.append(
            {
                **enriched_run,
                "results_dir": run_dir_for_id,
                "available_orders": _available_orders(run_dir_for_id),
                "summary_label": _run_summary_label(enriched_run),
                "summary_meta": _run_summary_meta(enriched_run),
                "selector_text": _run_selector_text(enriched_run),
                "checkpoint_completed_points": checkpoint_completed_points,
                "checkpoint_total_points": checkpoint_total_points or int(run.get("total_points") or 0),
            }
        )
    return runs


def _list_plots(plot_dir: Path) -> list[dict[str, Any]]:
    """Return saved combined plot manifests newest first."""
    if not plot_dir.exists():
        return []
    plots = []
    for path in plot_dir.glob("*/manifest.json"):
        with path.open("r", encoding="utf-8") as handle:
            plots.append(json.load(handle))
    return sorted(plots, key=lambda plot: str(plot.get("created_at", "")), reverse=True)


def _build_plot_preview(*, data_dir: Path, form_data: Any) -> dict[str, Any]:
    """Build a live combined-plot preview from saved runs and selected orders."""
    run_ids = [str(run_id) for run_id in form_data.getlist("run_ids")]
    if not run_ids:
        return {
            "ok": False,
            "error": "Select at least one saved run.",
            "figure_json": None,
            "selected_runs": [],
            "series_controls": [],
        }

    order_selection: dict[str, list[int]] = {}
    for run_id in run_ids:
        orders = [int(value) for value in form_data.getlist(f"orders_{run_id}") if str(value).strip() != ""]
        if not orders:
            return {
                "ok": False,
                "error": f"Select at least one diffraction order for run {run_id}.",
                "figure_json": None,
                "selected_runs": [],
                "series_controls": [],
            }
        order_selection[run_id] = orders

    plot_options = _plot_options_from_form(
        form_data=form_data,
        run_ids=run_ids,
        order_selection=order_selection,
    )
    try:
        selected_runs, figure_json, series_controls = _build_plotly_compare_payload(
            data_dir=data_dir,
            run_ids=run_ids,
            order_selection=order_selection,
            plot_options=plot_options,
        )
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "figure_json": None,
            "selected_runs": [],
            "series_controls": [],
        }

    return {
        "ok": True,
        "error": "",
        "figure_json": figure_json,
        "selected_runs": selected_runs,
        "series_controls": series_controls,
        "plot_options": plot_options,
    }


def _available_orders(run_dir: Path) -> list[int]:
    """Return positive diffraction orders available for one run."""
    all_orders_path = run_dir / "all_orders.csv"
    orders: set[int] = set()
    if all_orders_path.exists():
        with all_orders_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    orders.add(abs(int(float(row["order"]))))
                except (KeyError, TypeError, ValueError):
                    continue
    if not orders:
        for result in _load_checkpoint_results(run_dir):
            if getattr(result, "status", "") != "ok":
                continue
            for value in np.asarray(getattr(result, "orders", []), dtype=int).tolist():
                orders.add(abs(int(value)))
    return sorted(orders)


def _build_combined_plot(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    title: str,
    plot_options: dict[str, Any],
) -> dict[str, Any]:
    """Build and save a combined plot for several runs and selected orders."""
    plot_root = data_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_id = _plot_id()
    plot_dir = plot_root / plot_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_summaries, figure_json, series_controls = _build_plotly_compare_payload(
        data_dir=data_dir,
        run_ids=run_ids,
        order_selection=order_selection,
        plot_options=plot_options,
    )

    manifest = {
        "id": plot_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "selected_runs": run_summaries,
        "plot_config": {
            "title": title,
            "x_axis_type": plot_options["x_axis_type"],
            "y_axis_type": plot_options["y_axis_type"],
            "series_styles": {
                item["series_key"]: {
                    "color": item["color"],
                    "marker_symbol": item["marker_symbol"],
                    "marker_size": item["marker_size"],
                }
                for item in series_controls
            },
        },
        "figure_json": figure_json,
    }
    with (plot_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def _plot_options_from_form(
    *,
    form_data: Any,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
) -> dict[str, Any]:
    """Return normalized compare-plot configuration from form data."""

    plot_options = {
        "title": str(form_data.get("title", "")).strip() or "Combined plot",
        "x_axis_type": _normalize_axis_type(form_data.get("x_axis_type")),
        "y_axis_type": _normalize_axis_type(form_data.get("y_axis_type")),
        "series_styles": {},
    }
    for run_id in run_ids:
        for order in order_selection.get(run_id, []):
            token = _series_token(run_id, int(order))
            series_key = _series_key(run_id, int(order))
            marker_symbol = str(form_data.get(f"marker_symbol_{token}", "")).strip() or "circle"
            valid_marker_symbols = {value for value, _ in PLOTLY_MARKER_SYMBOLS}
            if marker_symbol not in valid_marker_symbols:
                marker_symbol = "circle"
            try:
                marker_size = int(float(form_data.get(f"marker_size_{token}", 3) or 3))
            except (TypeError, ValueError):
                marker_size = 3
            marker_size = max(1, min(marker_size, 24))
            plot_options["series_styles"][series_key] = {
                "color": _normalize_hex_color(form_data.get(f"color_{token}")),
                "marker_symbol": marker_symbol,
                "marker_size": marker_size,
            }
    return plot_options


def _normalize_axis_type(raw_value: Any) -> str:
    """Return a safe Plotly axis type."""

    value = str(raw_value or "").strip().lower()
    if value == "log":
        return "log"
    return "linear"


def _normalize_hex_color(raw_value: Any) -> str | None:
    """Return a normalized hex color or None when invalid."""

    value = str(raw_value or "").strip()
    if len(value) == 7 and value.startswith("#"):
        hex_digits = value[1:]
        if all(character in "0123456789abcdefABCDEF" for character in hex_digits):
            return value.lower()
    return None


def _series_key(run_id: str, order: int) -> str:
    """Return the stable manifest key for one compare-plot series."""

    return f"{run_id}::order::{int(order)}"


def _series_token(run_id: str, order: int) -> str:
    """Return the form-safe token for one compare-plot series."""

    safe_run_id = "".join(character if character.isalnum() else "_" for character in run_id)
    return f"{safe_run_id}__order_{int(order)}"


def _default_series_style(index: int) -> dict[str, Any]:
    """Return the default style for one compare-plot series."""

    marker_cycle = [value for value, _ in PLOTLY_MARKER_SYMBOLS]
    return {
        "color": PLOTLY_DEFAULT_COLORS[index % len(PLOTLY_DEFAULT_COLORS)],
        "marker_symbol": marker_cycle[index % len(marker_cycle)],
        "marker_size": 3,
    }


def _build_plotly_compare_payload(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    plot_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Return selected-run summaries, serialized Plotly figure, and style controls."""

    _require_plotly()
    run_summaries, selected_series = _load_selected_plot_series(
        data_dir=data_dir,
        run_ids=run_ids,
        order_selection=order_selection,
    )
    if not selected_series:
        raise ValueError("Select at least one run and one diffraction order.")
    series_controls = _series_controls_for_selected_series(
        selected_series=selected_series,
        saved_styles=dict(plot_options.get("series_styles", {})),
    )
    figure = _build_plotly_compare_figure(
        selected_series=selected_series,
        title=str(plot_options["title"]),
        x_axis_type=str(plot_options["x_axis_type"]),
        y_axis_type=str(plot_options["y_axis_type"]),
        series_controls=series_controls,
    )
    assert plotly_io is not None
    return run_summaries, plotly_io.to_json(figure, validate=False), series_controls


def _load_selected_plot_series(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected run summaries plus all order series requested for comparison."""

    run_summaries: list[dict[str, Any]] = []
    selected_series: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = data_dir / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_label = _run_selector_text(manifest)
        orders = order_selection.get(run_id) or _available_orders(run_dir)
        run_summaries.append({"id": run_id, "name": run_label, "orders": orders})
        for order in orders:
            series = _load_order_series(run_dir, order=order, label=run_label)
            if series is not None:
                series["series_key"] = _series_key(run_id, int(order))
                series["series_token"] = _series_token(run_id, int(order))
                series["run_id"] = run_id
                selected_series.append(series)
    return run_summaries, selected_series


def _series_controls_for_selected_series(
    *,
    selected_series: list[dict[str, Any]],
    saved_styles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the normalized per-series style controls for one compare plot."""

    series_controls: list[dict[str, Any]] = []
    for index, series in enumerate(selected_series):
        default_style = _default_series_style(index)
        saved_style = dict(saved_styles.get(str(series["series_key"]), {}))
        marker_symbol = str(saved_style.get("marker_symbol", default_style["marker_symbol"]))
        if marker_symbol not in {value for value, _ in PLOTLY_MARKER_SYMBOLS}:
            marker_symbol = str(default_style["marker_symbol"])
        try:
            marker_size = int(float(saved_style.get("marker_size", default_style["marker_size"])))
        except (TypeError, ValueError):
            marker_size = int(default_style["marker_size"])
        marker_size = max(1, min(marker_size, 24))
        series_controls.append(
            {
                "series_key": str(series["series_key"]),
                "series_token": str(series["series_token"]),
                "run_id": str(series["run_id"]),
                "order": int(series["order"]),
                "label": f"{series['label']} · order: {series['order']}",
                "color": _normalize_hex_color(saved_style.get("color")) or str(default_style["color"]),
                "marker_symbol": marker_symbol,
                "marker_size": marker_size,
            }
        )
    return series_controls


def _build_plotly_compare_figure(
    *,
    selected_series: list[dict[str, Any]],
    title: str,
    x_axis_type: str,
    y_axis_type: str,
    series_controls: list[dict[str, Any]],
) -> Any:
    """Build the interactive Plotly figure for one compare plot."""

    _require_plotly()
    assert go is not None
    control_map = {item["series_key"]: item for item in series_controls}
    figure = go.Figure()
    for series in selected_series:
        style = control_map[str(series["series_key"])]
        figure.add_trace(
            go.Scatter(
                x=np.asarray(series["energies"], dtype=float),
                y=np.asarray(series["efficiencies"], dtype=float),
                mode="lines+markers",
                name=style["label"],
                line={"color": style["color"], "width": 1.6},
                marker={
                    "color": style["color"],
                    "symbol": style["marker_symbol"],
                    "size": style["marker_size"],
                },
                hovertemplate="Energy: %{x:.3f} eV<br>Efficiency: %{y:.6f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 64, "r": 24, "t": 72, "b": 60},
    )
    figure.update_xaxes(title_text="Photon Energy (eV)", type=x_axis_type, showgrid=True, zeroline=False)
    figure.update_yaxes(title_text="Diffraction Efficiency", type=y_axis_type, showgrid=True, zeroline=False)
    return figure


def _render_combined_plot_image(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    title: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Render a combined plot image and return the selected run summary."""
    run_summaries: list[dict[str, Any]] = []
    selected_series: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = data_dir / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_label = str(manifest.get("display_name") or manifest.get("grating_name", run_id))
        orders = order_selection.get(run_id) or _available_orders(run_dir)
        run_summaries.append({"id": run_id, "name": run_label, "orders": orders})
        for order in orders:
            series = _load_order_series(run_dir, order=order, label=run_label)
            if series is not None:
                selected_series.append(series)

    if not selected_series:
        raise ValueError("Select at least one run and one diffraction order.")

    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "d", "v", "x", "*"]
    for index, series in enumerate(selected_series):
        axis.plot(
            np.asarray(series["energies"], dtype=float),
            np.asarray(series["efficiencies"], dtype=float),
            marker=markers[index % len(markers)],
            linewidth=1.2,
            label=f"{series['label']} · Order {series['order']}",
        )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return run_summaries


def _load_order_series(run_dir: Path, *, order: int, label: str) -> dict[str, Any] | None:
    """Load one order-vs-energy series from a saved run."""
    all_orders_path = run_dir / "all_orders.csv"

    requested_order = 0 if int(order) == 0 else -abs(int(order))
    rows: list[dict[str, float]] = []
    if all_orders_path.exists():
        with all_orders_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    row_order = int(float(row["order"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if row_order != requested_order:
                    continue
                try:
                    rows.append(
                        {
                            "energy_ev": float(row["energy_ev"]),
                            "efficiency": float(row["efficiency"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    if not rows:
        for result in _load_checkpoint_results(run_dir):
            if getattr(result, "status", "") != "ok":
                continue
            orders = np.asarray(getattr(result, "orders", []), dtype=int)
            efficiencies = np.asarray(getattr(result, "efficiency_all", []), dtype=float)
            if orders.size == 0 or efficiencies.size != orders.size:
                continue
            for row_order, efficiency in zip(orders.tolist(), efficiencies.tolist()):
                if int(row_order) != requested_order:
                    continue
                rows.append(
                    {
                        "energy_ev": float(result.energy_ev),
                        "efficiency": float(efficiency),
                    }
                )

    if not rows:
        return None

    rows.sort(key=lambda row: row["energy_ev"])
    return {
        "label": label,
        "order": int(order),
        "energies": [row["energy_ev"] for row in rows],
        "efficiencies": [row["efficiency"] for row in rows],
    }


def _plot_id() -> str:
    """Return a filesystem-safe plot identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _run_id(workflow: str, grating_id: str) -> str:
    """Return a filesystem-safe run ID."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{grating_id}-{workflow}".replace("_", "-")


def _safe_browse_path(raw_path: Any) -> Path:
    """Return a normalized filesystem path for the export dialog."""
    if raw_path in (None, ""):
        return Path.home().resolve()
    return Path(str(raw_path)).expanduser().resolve()


def _validate_workspace_root(raw_path: Any) -> Path:
    """Return a writable workspace root path for the current session."""

    if raw_path in (None, ""):
        raise ValueError("Choose a workspace folder.")
    target_dir = Path(str(raw_path)).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    probe_path = target_dir / ".grax-web-write-test"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink()
    return target_dir


def _has_active_runs(app: Any) -> bool:
    """Return whether any run is still queued or running."""

    with _active_runs_lock(app):
        return any(_is_active_run_entry_live(active) for active in _active_runs(app).values())


def _is_run_active(app: Any, run_id: str) -> bool:
    """Return whether one specific run is still active."""

    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        return active is not None and _is_active_run_entry_live(active)


def _linked_runs(data_dir: Path, grating_id: str) -> list[dict[str, Any]]:
    """Return saved runs whose manifests reference one grating ID."""

    return [
        run
        for run in _list_plot_runs(data_dir / "runs")
        if str(run.get("grating_id", "")) == grating_id
    ]


def _list_directory_entries(directory: Path) -> list[dict[str, Any]]:
    """Return browseable directory entries for the export dialog."""
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.iterdir(), key=lambda candidate: (not candidate.is_dir(), candidate.name.lower())):
        if path.name.startswith(".") or not path.is_dir():
            continue
        entries.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
            }
        )
    return entries


def _path_breadcrumbs(path: Path) -> list[dict[str, str]]:
    """Return breadcrumb metadata for one absolute filesystem path."""

    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    breadcrumbs = [{"name": str(anchor) if str(anchor) else resolved.parts[0], "path": str(anchor or resolved)}]
    current = anchor
    for part in resolved.parts[len(anchor.parts) :]:
        current = current / part
        breadcrumbs.append({"name": part, "path": str(current)})
    return breadcrumbs


if __name__ == "__main__":
    main()
