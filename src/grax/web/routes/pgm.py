"""SX700 PGM working-range form, run, results and export routes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from flask import abort, jsonify, redirect, render_template, request, send_file, url_for

from grax.materials import available_material_symbols, material_density_catalog
from grax.pgm import dashboard_title, parameter_blocks

from ..context import WebContext
from ..jobs import stop_job
from ..persistence import PlaneMirrorStore
from ..pgm_dashboard import dashboard_payload, live_payload, load_maps, point_detail
from ..pgm_exports import maps_csv, maps_png
from ..pgm_jobs import (
    _parameters,
    abort_if_orphaned,
    create_pgm_run,
    delete_pgm_setup,
    grating_editor,
    grating_from_form,
    label_runs_by_pgm,
    load_pgm_maps,
    mirror_editor,
    mirror_from_form,
    parse_pgm_form,
    pgm_preset,
    pgm_setup_page_values,
    pgm_setup_store,
    pgm_source,
    pgm_status_payload,
    pgm_store,
    rename_pgm_setup,
    save_pgm_setup,
    split_optics_form,
    start_pgm_run,
)
from ..pgm_presets import DEFAULT_PGM_PRESET, PGM_PRESETS
from ..plot_builder import _plotly_bundle_text
from ..run_inputs import _worker_settings_from_form


def register(app: Any, ctx: WebContext) -> None:
    """Register PGM routes without changing existing endpoint names."""

    def data_dir() -> Path:
        return ctx.active_data_dir()

    def saved_run(run_id: str) -> dict[str, Any]:
        try:
            run = pgm_store(data_dir()).load(run_id)
        except (FileNotFoundError, ValueError):
            abort(404)
        abort_if_orphaned(app, data_dir(), run)
        return run

    def run_directory(run_id: str) -> Path:
        return pgm_store(data_dir()).run_dir(run_id)

    def completed_maps(run_id: str) -> dict[str, np.ndarray]:
        run = saved_run(run_id)
        if run["status"] != "completed":
            abort(409, "This run has no complete results yet.")
        try:
            return load_pgm_maps(data_dir(), run_id)
        except FileNotFoundError:
            abort(404)

    def mirror_store() -> PlaneMirrorStore:
        return PlaneMirrorStore(data_dir() / "saved_plane_mirrors")

    def selected_setup() -> tuple[dict[str, Any], str]:
        """Return the saved PGM or preset named in the page address, or the default example."""
        saved_id = request.args.get("saved")
        if saved_id:
            try:
                return pgm_setup_store(data_dir()).load(saved_id), url_for(
                    "pgm_page", saved=saved_id
                )
            except (FileNotFoundError, ValueError):
                abort(404)
        preset_id = request.args.get("preset", DEFAULT_PGM_PRESET)
        preset = pgm_preset(preset_id)
        if preset is None:
            abort(404)
        return preset, url_for("pgm_page", preset=preset_id)

    @app.get("/pgm")
    def pgm_page():
        setup, start_url = selected_setup()
        return render_template(
            "pgm_form.html",
            **pgm_setup_page_values(setup),
            setup=setup,
            start_url=start_url,
            presets=PGM_PRESETS.values(),
            saved_setups=pgm_setup_store(data_dir()).list(),
            gratings=ctx.store().list(),
            mirrors=mirror_store().list(),
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
        )

    @app.get("/pgm/grating-fields")
    def pgm_grating_fields():
        grating_id = request.args.get("id", "")
        try:
            spec = ctx.store().load(grating_id) if grating_id else None
        except (FileNotFoundError, ValueError):
            abort(404)
        return render_template("_pgm_grating_fields.html", editor=grating_editor(spec=spec))

    @app.get("/pgm/mirror-fields")
    def pgm_mirror_fields():
        mirror_id = request.args.get("id", "")
        try:
            mirror = mirror_store().load(mirror_id) if mirror_id else {}
        except (FileNotFoundError, ValueError):
            abort(404)
        editor = mirror_editor(mirror.get("stack_form"), name=str(mirror.get("name", "")))
        return render_template("_pgm_mirror_fields.html", editor=editor)

    @app.get("/pgm/saved")
    def pgm_setup_index():
        return render_template(
            "pgm_saved.html",
            saved_setups=pgm_setup_store(data_dir()).list(),
            message=str(request.args.get("message", "")).strip(),
            message_kind=str(request.args.get("kind", "info")).strip() or "info",
        )

    @app.post("/pgm/saved")
    def pgm_setup_create():
        try:
            saved = save_pgm_setup(data_dir(), request.form)
        except (KeyError, TypeError, ValueError, OSError) as error:
            return jsonify({"error": str(error)}), 422
        url = url_for("pgm_page", saved=saved["id"])
        return jsonify({"id": saved["id"], "name": saved["name"], "url": url})

    @app.post("/pgm/saved/<setup_id>/rename")
    def pgm_setup_rename(setup_id: str):
        try:
            rename_pgm_setup(data_dir(), setup_id, request.form.get("name"))
        except FileNotFoundError:
            abort(404)
        except ValueError as error:
            abort(422, str(error))
        return redirect(url_for("pgm_setup_index"))

    @app.post("/pgm/saved/<setup_id>/delete")
    def pgm_setup_delete(setup_id: str):
        try:
            delete_pgm_setup(data_dir(), setup_id)
        except (FileNotFoundError, ValueError):
            abort(404)
        return redirect(url_for("pgm_setup_index"))

    @app.post("/pgm/runs")
    def pgm_run_new():
        try:
            grating_form, mirror_form = split_optics_form(request.form)
            run = create_pgm_run(
                data_dir(),
                grating=grating_from_form(
                    grating_form,
                    data_dir(),
                    source_id=str(request.form.get("grating_source_id", "")),
                ),
                mirror=mirror_from_form(
                    mirror_form, source_id=str(request.form.get("mirror_source_id", ""))
                ),
                values=parse_pgm_form(request.form),
                display_name=str(request.form.get("display_name", "")).strip(),
                pgm=pgm_source(data_dir(), request.form.get("pgm_id")),
                comment=str(request.form.get("comment", "")).strip(),
                workers=_worker_settings_from_form(request.form),
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            return jsonify({"error": str(error)}), 422
        start_pgm_run(app, data_dir(), run["id"])
        return jsonify({"redirect": url_for("pgm_run_detail", run_id=run["id"])})

    @app.get("/pgm/runs")
    def pgm_run_index():
        runs = pgm_store(data_dir()).list()
        for run in runs:
            abort_if_orphaned(app, data_dir(), run)
        pgms = label_runs_by_pgm(data_dir(), runs)
        return render_template("pgm_runs.html", runs=runs, pgms=pgms)

    @app.post("/pgm/runs/manage")
    def pgm_runs_update():
        store = pgm_store(data_dir())
        known_ids = [str(run["id"]) for run in store.list()]
        if request.form.get("action") == "delete":
            run_ids = [
                run_id for run_id in request.form.getlist("selected_run_id") if run_id in known_ids
            ]
            # A running worker would keep writing into the folder being removed.
            for run_id in run_ids:
                stop_job(app, run_id)
            store.delete_many(run_ids)
        else:
            store.save_edits(request.form, known_ids, only=str(request.form.get("save_run_id", "")))
        return redirect(url_for("pgm_run_index"))

    @app.get("/pgm/runs/<run_id>")
    def pgm_run_detail(run_id: str):
        run = saved_run(run_id)
        parameters = _parameters(run, data_dir())
        return render_template(
            "pgm_result.html",
            run=run,
            initial_status=pgm_status_payload(app, run),
            status_url=url_for("pgm_run_status", run_id=run_id),
            memory_url=url_for("system_memory"),
            dashboard_title=dashboard_title(parameters),
            parameter_blocks=parameter_blocks(parameters),
            plotly_bundle=_plotly_bundle_text(),
        )

    @app.get("/pgm/runs/<run_id>/status")
    def pgm_run_status(run_id: str):
        return jsonify(pgm_status_payload(app, saved_run(run_id)))

    @app.post("/pgm/runs/<run_id>/stop")
    def pgm_run_stop(run_id: str):
        saved_run(run_id)
        stop_job(app, run_id)
        return redirect(url_for("pgm_run_detail", run_id=run_id))

    @app.post("/pgm/runs/<run_id>/delete")
    def pgm_run_delete(run_id: str):
        saved_run(run_id)
        stop_job(app, run_id)
        pgm_store(data_dir()).delete_many([run_id])
        return redirect(url_for("pgm_run_index"))

    @app.get("/pgm/runs/<run_id>/maps")
    def pgm_run_maps(run_id: str):
        run = saved_run(run_id)
        payload = dashboard_payload(run_directory(run_id), run["parameters"])
        if payload is None:
            abort(409, "This run has no maps yet.")
        return jsonify(payload)

    @app.get("/pgm/runs/<run_id>/live")
    def pgm_run_live(run_id: str):
        saved_run(run_id)
        return jsonify(live_payload(run_directory(run_id), request.args.get("v", "")))

    @app.get("/pgm/runs/<run_id>/point")
    def pgm_run_point(run_id: str):
        run = saved_run(run_id)
        maps = load_maps(run_directory(run_id))
        if maps is None:
            abort(409, "This run has no maps yet.")
        try:
            energy_index = int(request.args["energy"])
            cff_index = int(request.args["cff"])
        except (KeyError, ValueError):
            abort(400, "Choose an energy and CFF grid point.")
        if not (0 <= energy_index < len(maps["energies"]) and 0 <= cff_index < len(maps["cff"])):
            abort(404)
        parameters = _parameters(run, data_dir())
        return jsonify(point_detail(maps, energy_index, cff_index, parameters))

    @app.get("/pgm/runs/<run_id>/export.csv")
    def pgm_run_csv(run_id: str):
        maps = completed_maps(run_id)
        return send_file(
            io.BytesIO(maps_csv(maps)),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{run_id}.csv",
        )

    @app.get("/pgm/runs/<run_id>/export.png")
    def pgm_run_png(run_id: str):
        maps = completed_maps(run_id)
        return send_file(
            maps_png(maps), mimetype="image/png", as_attachment=True, download_name=f"{run_id}.png"
        )
