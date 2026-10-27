"""Saved comparison plot routes."""

from __future__ import annotations

import json
import shutil
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from flask import (
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from ..catalog import (
    _is_plane_mirror_run,
    _list_plot_runs,
    _list_plots,
)
from ..common import (
    _is_safe_id,
)
from ..context import WebContext
from ..plot_builder import (
    PLOTLY_MARKER_SYMBOLS,
    _build_combined_plot,
    _build_plot_preview,
    _build_plotly_compare_payload,
    _plot_export_zip,
    _plot_filename_stem,
    _plot_options_from_form,
    _plot_selected_run_entries,
    _plotly_bundle_text,
    _saved_plot_figure_json,
)
from ..run_files import (
    _load_run_manifest,
)
from ..workspace import (
    _list_directory_entries,
    _path_breadcrumbs,
    _safe_browse_path,
)


def register(app: Any, ctx: WebContext) -> None:
    """Register the plots routes on ``app``."""
    active_data_dir = ctx.active_data_dir

    def preview_root() -> Path:
        return app.config["GRAx_DATA_DIR"] / "previews" / "live"

    @app.get("/plots")
    def plot_index():
        data_dir = active_data_dir()
        plots = _list_plots(data_dir / "plots")
        for plot in plots:
            plot_path = plot.get("plot_path")
            if plot_path:
                image_path = data_dir / "plots" / str(plot["id"]) / str(plot_path)
                if image_path.is_file():
                    plot["preview_url"] = url_for(
                        "data_file",
                        filename=f"plots/{plot['id']}/{plot_path}",
                    )
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
            editing_plot=None,
        )

    @app.get("/plots/<plot_id>/edit")
    def plot_edit(plot_id: str):
        manifest_path = active_data_dir() / "plots" / plot_id / "manifest.json"
        if not manifest_path.is_file():
            abort(404)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("plot_config") or any(
            _load_run_manifest(active_data_dir(), str(run.get("id", ""))) is None
            for run in manifest.get("selected_runs", [])
        ):
            abort(409, "Editing requires the original saved runs and plot settings.")
        runs = _list_plot_runs(active_data_dir() / "runs")
        return render_template(
            "plot_form.html", runs=runs, run_options=runs,
            results_dir=active_data_dir() / "runs", preview=None,
            plotly_bundle=_plotly_bundle_text(), marker_symbols=PLOTLY_MARKER_SYMBOLS,
            editing_plot=manifest,
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
        if not all(_is_safe_id(run_id) for run_id in run_ids):
            abort(400, "A selected run is unavailable.")
        order_selection: dict[str, list[int]] = {}
        for run_id in run_ids:
            manifest = _load_run_manifest(active_data_dir(), run_id)
            is_mirror = bool(manifest and _is_plane_mirror_run(manifest))
            orders = [int(value) for value in request.form.getlist(f"orders_{run_id}")]
            if is_mirror:
                order_selection[run_id] = [0]
                continue
            if not orders:
                abort(400, f"Select at least one diffraction order for run {run_id}.")
            order_selection[run_id] = orders
        plot_options = _plot_options_from_form(
            form_data=request.form,
            run_ids=run_ids,
            order_selection=order_selection,
            data_dir=app.config["GRAx_DATA_DIR"],
        )
        try:
            plot = _build_combined_plot(
                data_dir=app.config["GRAx_DATA_DIR"],
                run_ids=run_ids,
                order_selection=order_selection,
                title=plot_options["title"],
                plot_options=plot_options,
            )
        except ValueError as error:
            abort(400, str(error))
        return redirect(url_for("plot_detail", plot_id=plot["id"]))

    @app.post("/plots/<plot_id>/edit")
    def update_plot(plot_id: str):
        manifest_path = active_data_dir() / "plots" / plot_id / "manifest.json"
        if not manifest_path.is_file():
            abort(404)
        old_plot = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not old_plot.get("plot_config"):
            abort(409, "This plot has no editable settings.")
        run_ids = list(dict.fromkeys(request.form.getlist("run_ids")))
        if not run_ids:
            abort(400, "Select at least one saved run.")
        if not all(_is_safe_id(run_id) for run_id in run_ids):
            abort(400, "A selected run is unavailable.")
        order_selection = {}
        for run_id in run_ids:
            run = _load_run_manifest(active_data_dir(), run_id)
            if run is None:
                abort(400, "A selected run is unavailable.")
            orders = [int(value) for value in request.form.getlist(f"orders_{run_id}")]
            if _is_plane_mirror_run(run):
                orders = [0]
            if not orders:
                abort(400, f"Select at least one diffraction order for run {run_id}.")
            order_selection[run_id] = orders
        options = _plot_options_from_form(
            form_data=request.form, run_ids=run_ids, order_selection=order_selection,
            data_dir=active_data_dir(),
        )
        try:
            _build_combined_plot(
                data_dir=active_data_dir(), run_ids=run_ids,
                order_selection=order_selection, title=options["title"],
                plot_options=options, existing_plot=old_plot,
            )
        except ValueError as error:
            abort(400, str(error))
        return redirect(url_for("plot_detail", plot_id=plot_id))

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
            selected_runs=_plot_selected_run_entries(app.config["GRAx_DATA_DIR"], manifest),
        )

    @app.get("/plots/<plot_id>/export/<kind>")
    def plot_export(plot_id: str, kind: str):
        if kind not in {"data", "script"}:
            abort(404)
        manifest_path = active_data_dir() / "plots" / plot_id / "manifest.json"
        if not manifest_path.is_file():
            abort(404)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        figure_json = _saved_plot_figure_json(active_data_dir(), manifest)
        if not figure_json:
            abort(409, "This saved plot has no exportable figure.")
        archive = _plot_export_zip(figure_json, include_script=kind == "script")
        return send_file(
            archive, mimetype="application/zip", as_attachment=True,
            download_name=f"{_plot_filename_stem(str(manifest.get('name') or manifest.get('title') or 'plot'))}-{kind}.zip",
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
