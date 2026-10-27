"""Grating run pages, progress, exports and run lifecycle routes."""

from __future__ import annotations

import csv
import io
import shutil
from typing import (
    Any,
)

import numpy as np
from flask import (
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from .. import (
    parameter_scans,
)
from ..catalog import (
    _is_grating_run,
    _list_plot_runs,
    _manage_run_entries,
    _plot_references_for_runs,
    _run_comparison_parameter_rows,
    _scan_run_rows,
)
from ..context import WebContext
from ..persistence import (
    build_grating_from_spec,
)
from ..plot_builder import (
    _build_plotly_compare_payload,
    _plotly_bundle_text,
)
from ..run_exports import (
    _run_export_archive,
    _run_export_rows,
    _run_metadata_yaml,
)
from ..run_files import (
    _available_orders,
    _load_checkpoint_results,
    _load_run_manifest,
    _parameter_study_checkpoint_count,
    _update_manifest_fields,
)
from ..run_inputs import (
    _custom_grid_configuration_from_value,
    _custom_grid_from_form,
    _worker_settings_from_form,
)
from ..run_jobs import (
    _abort_run,
    _repair_saved_run_plot,
    _run_status_payload,
    _start_run_worker,
)
from ..run_state import (
    _is_run_active,
)


def register(app: Any, ctx: WebContext) -> None:
    """Register the runs routes on ``app``."""
    active_data_dir = ctx.active_data_dir
    run_store = ctx.run_store

    @app.get("/runs/gratings")
    def manage_grating_runs():
        return render_template(
            "run_manage.html",
            runs=[
                run for run in _manage_run_entries(app.config["GRAx_DATA_DIR"], app=app)
                if _is_grating_run(run)
            ],
            message=str(request.args.get("message", "")).strip(),
            message_kind=str(request.args.get("kind", "info")).strip() or "info",
        )

    @app.get("/runs/manage")
    def manage_runs():
        """Keep the former run-management URL working for saved links."""
        return redirect(url_for("manage_grating_runs"))

    @app.post("/runs/gratings")
    def update_runs():
        action = str(request.form.get("action", "save"))
        store = run_store()
        if action == "compare":
            grating_run_ids = {
                str(run.get("id")) for run in store.list() if _is_grating_run(run)
            }
            selected_ids = list(dict.fromkeys(
                run_id for run_id in request.form.getlist("selected_run_id")
                if run_id in grating_run_ids
            ))
            if len(selected_ids) != 2:
                return redirect(url_for(
                    "manage_grating_runs",
                    message="Select exactly two grating runs to compare.",
                    kind="error",
                ))
            return redirect(url_for("compare_grating_runs", run_ids=selected_ids))
        if action == "delete":
            grating_run_ids = {
                str(run.get("id")) for run in store.list() if _is_grating_run(run)
            }
            run_ids = [
                run_id for run_id in (
                    request.form.getlist("selected_run_id")
                    or request.form.getlist("delete_run_id")
                )
                if run_id in grating_run_ids
            ]
            if request.form.get("confirm_delete") != "1":
                references = _plot_references_for_runs(active_data_dir(), run_ids)
                if references:
                    return render_template(
                        "run_delete_confirm.html",
                        run_ids=run_ids,
                        plots=references,
                    )
            store.delete_many(run_ids)
            return redirect(url_for("manage_grating_runs"))

        for run in store.list():
            if not _is_grating_run(run):
                continue
            run_id = str(run["id"])
            field_name = f"display_name_{run_id}"
            comment_field_name = f"comment_{run_id}"
            if field_name in request.form:
                store.rename(run_id, str(request.form.get(field_name, "")))
            if comment_field_name in request.form:
                store.update_comment(run_id, str(request.form.get(comment_field_name, "")))
        return redirect(url_for("manage_grating_runs"))

    # Keep POSTs from older saved forms working while the canonical form URL is
    # now the grating-specific run page.
    app.add_url_rule("/runs/manage", endpoint="legacy_update_runs", view_func=update_runs, methods=["POST"])

    @app.get("/runs/gratings/compare")
    def compare_grating_runs():
        selected_ids = list(dict.fromkeys(str(value) for value in request.args.getlist("run_ids")))
        if len(selected_ids) != 2:
            return redirect(url_for(
                "manage_grating_runs",
                message="Select exactly two grating runs to compare.",
                kind="error",
            ))
        manifests = []
        for run_id in selected_ids:
            manifest = _load_run_manifest(active_data_dir(), run_id)
            if manifest is None or not _is_grating_run(manifest):
                return redirect(url_for(
                    "manage_grating_runs",
                    message="One of the selected grating runs is no longer available.",
                    kind="error",
                ))
            manifest["display_name"] = str(
                manifest.get("display_name") or f"{manifest.get('grating_name', 'Run')} · {manifest.get('workflow', 'run')}"
            )
            manifests.append(manifest)

        run_views = _list_plot_runs(active_data_dir() / "runs")
        view_by_id = {str(run["id"]): run for run in run_views}
        available_by_run = {
            run_id: set(view_by_id.get(run_id, {}).get("available_orders", []))
            for run_id in selected_ids
        }
        common_orders = sorted(set.intersection(*available_by_run.values())) if all(available_by_run.values()) else []
        requested_order = request.args.get("order", "")
        try:
            selected_order = int(requested_order)
        except (TypeError, ValueError):
            selected_order = -1
        if selected_order not in common_orders:
            first_default = (manifests[0].get("run_input") or {}).get("diffraction_order")
            try:
                first_default = int(first_default)
            except (TypeError, ValueError):
                first_default = -1
            selected_order = first_default if first_default in common_orders else (common_orders[0] if common_orders else None)

        parameter_rows = _run_comparison_parameter_rows(manifests)
        figure_json = None
        plot_error = ""
        if selected_order is None:
            plot_error = "These runs have no diffraction order in common, so an efficiency overlay is unavailable."
        else:
            try:
                _, figure_json, _ = _build_plotly_compare_payload(
                    data_dir=active_data_dir(),
                    run_ids=selected_ids,
                    order_selection={run_id: [selected_order] for run_id in selected_ids},
                    plot_options={
                        "title": f"{manifests[0]['display_name']} vs {manifests[1]['display_name']}",
                        "x_axis_value": "energy",
                        "x_axis_type": "linear",
                        "y_axis_type": "linear",
                    },
                )
            except (ValueError, RuntimeError) as error:
                plot_error = str(error)

        return render_template(
            "run_compare.html",
            runs=manifests,
            run_ids=selected_ids,
            common_orders=common_orders,
            selected_order=selected_order,
            parameter_rows=parameter_rows,
            figure_json=figure_json,
            plot_error=plot_error,
            plotly_bundle=_plotly_bundle_text() if figure_json else None,
        )

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        _repair_saved_run_plot(app, app.config["GRAx_DATA_DIR"], run_id, manifest)
        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id) or manifest
        manifest["results_dir"] = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        manifest["scan_rows"] = _scan_run_rows(manifest, manifest["results_dir"])
        return render_template(
            "run_detail.html",
            run=manifest,
            related_plots=_plot_references_for_runs(app.config["GRAx_DATA_DIR"], [run_id]),
            initial_status=_run_status_payload(app=app, data_dir=app.config["GRAx_DATA_DIR"], run_id=run_id),
            status_url=url_for("run_status", run_id=run_id),
            memory_url=url_for("system_memory"),
            abort_url=url_for("abort_run_dialog", run_id=run_id),
            plotly_bundle=_plotly_bundle_text() if manifest.get("parameter_sweep") else None,
            scan_orders=_available_orders(manifest["results_dir"]) or sorted({int(scan["settings"]["diffraction_order"]) for scan in manifest.get("parameter_sweep", {}).get("scans", [])}),
        )

    @app.get("/runs/<run_id>/scan-series")
    def scan_series(run_id: str):
        """Return separate, gap-preserving energy curves for the selected order."""
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if not manifest or not manifest.get("parameter_sweep"):
            abort(404)
        sweep = manifest["parameter_sweep"]
        try:
            order = int(request.args.get("order", 1))
        except ValueError:
            abort(400, "Choose an integer diffraction order.")
        run_dir = active_data_dir() / "runs" / run_id
        series = parameter_scans.load_all_series(run_dir, sweep, order)
        return jsonify(
            {
                "energies": parameter_scans.sample_values(sweep),
                "sample_values": parameter_scans.sample_values(sweep),
                "sample_axis": sweep.get("sample_axis", {"id": "energy_ev", "label": "Photon Energy (eV)"}),
                "scans": [
                    {
                        "id": scan["id"],
                        "label": scan["label"],
                        "axis_indices": scan["axis_indices"],
                        "efficiencies": [
                            value if np.isfinite(value) else None for value in series[scan["id"]]
                        ],
                    }
                    for scan in sweep["scans"]
                ],
            }
        )

    @app.post("/runs/<run_id>/resume")
    def resume_parameter_run(run_id: str):
        """Resume any checkpoint-backed run without changing its saved inputs."""
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if not manifest:
            abort(404)
        if _is_run_active(app, run_id):
            abort(409, "This run is already active.")
        form = dict(manifest["run_input"])
        workflow = str(manifest.get("workflow", ""))
        sweep = manifest.get("parameter_sweep")
        run_dir = active_data_dir() / "runs" / run_id
        completed = _parameter_study_checkpoint_count(run_dir) if workflow == "parameter_study" else len(_load_checkpoint_results(run_dir))
        if sweep:
            energies = np.asarray(sweep["energy_values"], dtype=float)
        elif workflow in {"angle_sweep", "fixed_angle"}:
            energies = np.asarray([float(form.get("energy_ev", 100.0))])
        elif workflow == "custom_grid":
            config = _custom_grid_configuration_from_value(
                manifest.get("custom_grid") or form.get("custom_grid_config")
            )
            variants = config.get("variants") or []
            if not variants or not isinstance(variants[0].get("spec"), dict):
                abort(409, "Saved custom-grid inputs are incomplete and cannot be resumed.")
            _, cases = _custom_grid_from_form(form, dict(variants[0]["spec"]))
            energies = np.asarray([float(case["energy_ev"]) for case in cases], dtype=float)
        elif workflow == "parameter_study":
            energies = np.asarray([float(form.get("study_energy_ev", form.get("energy_ev", 100.0)))])
        else:
            energies = np.linspace(
                float(form.get("energy_start_ev", 100.0)),
                float(form.get("energy_stop_ev", 100.0)),
                int(float(form.get("energy_points", 1))),
            )
        worker_mode, setting, requested = _worker_settings_from_form(form)
        _update_manifest_fields(
            active_data_dir(),
            run_id,
            status="queued",
            error_text="",
            resume_count=int(manifest.get("resume_count", 0)) + 1,
        )
        _start_run_worker(
            app=app,
            data_dir=active_data_dir(),
            run_id=run_id,
            grating_id=manifest["grating_id"],
            grating_name=manifest["grating_name"],
            grating=build_grating_from_spec(manifest["grating_spec"]),
            workflow=workflow,
            energies=energies,
            form_data=form,
            diffraction_order=int(form["diffraction_order"]),
            fourier_orders=int(form["fourier_orders"]),
            max_workers_setting=setting,
            worker_mode=worker_mode,
            requested_workers=requested,
            completed_points=completed,
            resume=True,
        )
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/restart")
    def restart_parameter_run(run_id: str):
        """Restart a terminal run from its saved inputs after clearing old outputs."""
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if not manifest:
            abort(404)
        if _is_run_active(app, run_id):
            abort(409, "This run is already active. Abort it before restarting.")
        run_dir = active_data_dir() / "runs" / run_id
        for child in run_dir.iterdir():
            if child.name == "manifest.json":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        form = dict(manifest["run_input"])
        workflow = str(manifest.get("workflow", ""))
        if workflow in {"angle_sweep", "fixed_angle"}:
            energies = np.asarray([float(form.get("energy_ev", 100.0))])
        elif workflow == "custom_grid":
            config = _custom_grid_configuration_from_value(manifest.get("custom_grid") or form.get("custom_grid_config"))
            variants = config.get("variants") or []
            if not variants or not isinstance(variants[0].get("spec"), dict):
                abort(409, "Saved custom-grid inputs are incomplete and cannot be restarted.")
            _, cases = _custom_grid_from_form(form, dict(variants[0]["spec"]))
            energies = np.asarray([float(case["energy_ev"]) for case in cases], dtype=float)
        elif workflow == "parameter_study":
            energies = np.asarray([float(form.get("study_energy_ev", form.get("energy_ev", 100.0)))])
        else:
            energies = np.linspace(float(form.get("energy_start_ev", 100.0)), float(form.get("energy_stop_ev", 100.0)), int(float(form.get("energy_points", 1))))
        worker_mode, setting, requested = _worker_settings_from_form(form)
        _update_manifest_fields(active_data_dir(), run_id, status="queued", error_text="", completed_points=0, cases=[], restart_count=int(manifest.get("restart_count", 0)) + 1)
        _start_run_worker(
            app=app, data_dir=active_data_dir(), run_id=run_id,
            grating_id=manifest["grating_id"], grating_name=manifest["grating_name"],
            grating=build_grating_from_spec(manifest["grating_spec"]), workflow=workflow,
            energies=energies, form_data=form,
            diffraction_order=int(float(form.get("diffraction_order", 1))),
            fourier_orders=int(float(form.get("fourier_orders", 15))),
            max_workers_setting=setting, worker_mode=worker_mode,
            requested_workers=requested, completed_points=0, resume=False,
        )
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/export/energy-tables")
    def export_energy_tables(run_id: str):
        """Download wide parameter-slice tables with column metadata."""
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if not manifest or not manifest.get("parameter_sweep"):
            abort(404)
        try:
            body = parameter_scans.export_tables(
                active_data_dir() / "runs" / run_id,
                manifest["parameter_sweep"],
                orders=[int(value) for value in request.form.getlist("orders")],
                axis_ids=request.form.getlist("axes"),
                formats=set(request.form.getlist("formats")),
            )
        except (ValueError, TypeError) as error:
            abort(400, str(error))
        return Response(
            body,
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename={run_id}_energy_tables.zip"},
        )

    @app.get("/runs/<run_id>/energy-tables")
    def energy_tables(run_id: str):
        """Offer scan axes, diffraction orders, and wide-table formats."""
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if not manifest or not manifest.get("parameter_sweep"):
            abort(404)
        orders = _available_orders(active_data_dir() / "runs" / run_id)
        if not orders:
            orders = sorted(
                {
                    int(scan["settings"]["diffraction_order"])
                    for scan in manifest["parameter_sweep"]["scans"]
                }
            )
        return render_template("energy_tables.html", run=manifest, orders=orders)

    @app.get("/runs/<run_id>/export/<export_format>")
    def export_run(run_id: str, export_format: str):
        """Download run data as CSV/DAT or its YAML metadata companion."""

        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        export_format = export_format.strip().lower()
        if export_format not in {"csv", "dat", "yaml"}:
            abort(404)

        run_dir = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        if export_format == "yaml":
            try:
                body = _run_metadata_yaml(manifest, run_dir)
            except ModuleNotFoundError:
                abort(500, "YAML export requires PyYAML. Install the web dependencies.")
            return Response(
                body,
                mimetype="application/yaml",
                headers={"Content-Disposition": f"attachment; filename={run_id}_metadata.yaml"},
            )

        header, rows = _run_export_rows(run_dir, workflow=str(manifest.get("workflow", "")))
        if not header:
            abort(404, "No exported data are available for this run yet.")
        buffer = io.StringIO(newline="")
        writer = csv.writer(
            buffer,
            delimiter="\t" if export_format == "dat" else ",",
            lineterminator="\n",
        )
        writer.writerow(header)
        writer.writerows(rows)
        mimetype = "text/tab-separated-values" if export_format == "dat" else "text/csv"
        return Response(
            buffer.getvalue(),
            mimetype=mimetype,
            headers={"Content-Disposition": f"attachment; filename={run_id}.{export_format}"},
        )

    @app.get("/runs/<run_id>/export-all")
    def export_all_run_data(run_id: str):
        """Download all run data formats and metadata in one ZIP archive."""

        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        run_dir = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        try:
            archive = _run_export_archive(manifest, run_dir, {"csv", "dat", "yaml"})
        except ValueError as error:
            abort(404, str(error))
        except ModuleNotFoundError:
            abort(500, "YAML export requires PyYAML. Install the web dependencies.")
        return Response(
            archive,
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename={run_id}_export.zip"},
        )

    @app.post("/runs/<run_id>/export-selected")
    def export_selected_run_data(run_id: str):
        """Download a selected subset of run data and metadata in one ZIP."""

        manifest = _load_run_manifest(app.config["GRAx_DATA_DIR"], run_id)
        if manifest is None:
            abort(404)
        formats = {value.strip().lower() for value in request.form.getlist("formats")}
        if not formats or not formats <= {"csv", "dat", "yaml"}:
            abort(400, "Select at least one valid export format.")
        run_dir = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        try:
            archive = _run_export_archive(manifest, run_dir, formats)
        except ValueError as error:
            abort(404, str(error))
        except ModuleNotFoundError:
            abort(500, "YAML export requires PyYAML. Install the web dependencies.")
        return Response(
            archive,
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename={run_id}_export.zip"},
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
            return redirect(url_for("manage_grating_runs"))
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/status")
    def run_status(run_id: str):
        payload = _run_status_payload(app=app, data_dir=app.config["GRAx_DATA_DIR"], run_id=run_id)
        if payload is None:
            abort(404)
        return jsonify(payload)
