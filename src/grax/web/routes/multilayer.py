"""Multilayer grating design-study routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
)

from flask import (
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from grax.materials import (
    available_material_symbols,
    material_density_catalog,
)

from ..context import WebContext
from ..jobs import stop_job
from ..multilayer_diagnostics import build_plot as build_diagnostic_plot
from ..multilayer_diagnostics import discover_plots
from ..multilayer_design_studies import (
    STAGE_LABELS,
    STAGES,
    SURVEY_CELL_WARNING_THRESHOLD,
    MultilayerDesignStudyStore,
    build_design_config,
    downstream_stages,
    flatten_config_values,
    offline_script_filename,
    parse_study_config,
    render_offline_script,
    resolve_designs,
    stages_invalidated_by,
    study_config_defaults,
    study_form_sections,
    survey_cell_count,
    survey_design_options,
)
from ..multilayer_jobs import (
    _abort_multilayer_design_stage,
    _energy_scan_checkpoint_series,
    _mark_orphaned_stages,
    _multilayer_design_job_key,
    _multilayer_design_stage_status_payload,
    _reset_multilayer_design_stage,
    _start_multilayer_design_worker,
)
from ..plot_builder import (
    _plotly_bundle_text,
    get_plotlyjs,
)
from ..run_state import (
    _is_run_active,
)


def register(app: Any, ctx: WebContext) -> None:
    """Register the multilayer routes on ``app``."""

    # ------------------------------------------------------------------ #
    # Multilayer-design studies                                          #
    # ------------------------------------------------------------------ #
    def _design_store() -> MultilayerDesignStudyStore:
        return MultilayerDesignStudyStore(app.config["GRAx_DATA_DIR"] / "multilayer_designs")

    def _design_study_or_404(study_id: str) -> dict[str, Any]:
        try:
            return _mark_orphaned_stages(app, _design_store().load(study_id))
        except (ValueError, FileNotFoundError, OSError):
            abort(404)

    def _delete_studies(study_ids: list[str]) -> None:
        # A running stage would keep writing into the folder being removed.
        for study_id in study_ids:
            for stage in STAGES:
                stop_job(app, _multilayer_design_job_key(study_id, stage))
        _design_store().delete_many(study_ids)

    def _design_view_model(manifest: dict[str, Any]) -> dict[str, Any]:
        store = _design_store()
        study_dir = store.study_dir(str(manifest["id"]))
        options = survey_design_options(study_dir)
        stages = []
        for stage in STAGES:
            state = dict(manifest["stages"][stage])
            state["stage"] = stage
            state["label"] = STAGE_LABELS[stage]
            stages.append(state)
        config = manifest["config"]

        def _material_name(key: str) -> str:
            # Materials are stored as [name, density] pairs.
            value = config.get(key)
            return str(value[0]) if isinstance(value, list) and value else ""

        plot_meta = {
            "coating_label": (
                config.get("coating_label")
                or f"{_material_name('material_a')}/{_material_name('material_b')}"
            ),
            "target_energy_ev": config.get("target_energy_ev"),
            "diffraction_order": config.get("diffraction_order"),
        }
        return {
            "study": manifest,
            "stages": stages,
            "design_options": options,
            "design_options_json": json.dumps(options),
            "survey_plot_meta_json": json.dumps(plot_meta),
            # Interactive survey plots need plotly.js inlined; without plotly the
            # template falls back to the PNGs the library already wrote.
            "plotly_bundle": _plotly_bundle_text() if get_plotlyjs is not None else None,
            "survey_cells": survey_cell_count(manifest["config"]),
        }

    @app.get("/multilayer-design")
    def multilayer_design_index():
        return render_template(
            "multilayer_design_index.html",
            studies=[_mark_orphaned_stages(app, study) for study in _design_store().list()],
            stage_labels=STAGE_LABELS,
            stages=STAGES,
        )

    @app.post("/multilayer-design")
    def multilayer_design_create():
        if request.form.get("action") == "delete":
            _delete_studies(request.form.getlist("delete_study_id"))
            return redirect(url_for("multilayer_design_index"))
        display_name = request.form.get("display_name", "").strip() or "Multilayer design"
        try:
            config = parse_study_config(request.form)
            build_design_config(
                app.config["GRAx_DATA_DIR"] / "multilayer_designs" / "_validate", config
            )
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        study = _design_store().create(
            display_name=display_name,
            config=config,
            auto_energy_scan="best" if request.form.get("auto_energy_scan") else "none",
        )
        return redirect(url_for("multilayer_design_detail", study_id=study["id"]))

    @app.get("/multilayer-design/new")
    def multilayer_design_new():
        # Seed from the newest study: the settings someone last tuned -- the
        # advanced scan parameters above all -- are a far better starting point
        # than the dataclass defaults.
        recent = _design_store().list()
        seed = recent[0] if recent else None
        return render_template(
            "multilayer_design_form.html",
            seeded_from=seed,
            defaults=flatten_config_values(
                seed["config"] if seed else study_config_defaults()
            ),
            basic_sections=study_form_sections(advanced=False),
            advanced_sections=study_form_sections(advanced=True),
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            cell_warning_threshold=SURVEY_CELL_WARNING_THRESHOLD,
        )

    @app.get("/multilayer-design/<study_id>/edit")
    def multilayer_design_edit(study_id: str):
        manifest = _design_study_or_404(study_id)
        return render_template(
            "multilayer_design_form.html",
            study=manifest,
            defaults=flatten_config_values(manifest["config"]),
            basic_sections=study_form_sections(advanced=False),
            advanced_sections=study_form_sections(advanced=True),
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            cell_warning_threshold=SURVEY_CELL_WARNING_THRESHOLD,
        )

    @app.post("/multilayer-design/<study_id>/edit")
    def multilayer_design_update(study_id: str):
        store = _design_store()
        manifest = _design_study_or_404(study_id)
        if any(
            _is_run_active(app, _multilayer_design_job_key(study_id, stage)) for stage in STAGES
        ):
            abort(409, "Stop the running stage before changing the parameters.")
        try:
            config = parse_study_config(request.form, base=manifest["config"])
            build_design_config(store.study_dir(study_id), config)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        # Results computed under the old parameters are kept but flagged, so it
        # is obvious which of them the edit left behind.
        for stage in stages_invalidated_by(manifest["config"], config):
            if manifest["stages"][stage]["status"] in {"completed", "aborted"}:
                manifest["stages"][stage]["status"] = "stale"
        manifest["config"] = config
        manifest["display_name"] = (
            request.form.get("display_name", "").strip() or manifest["display_name"]
        )
        manifest["auto_energy_scan"] = (
            "best" if request.form.get("auto_energy_scan") else "none"
        )
        store.save(manifest)
        return redirect(url_for("multilayer_design_detail", study_id=study_id))

    @app.get("/multilayer-design/<study_id>")
    def multilayer_design_detail(study_id: str):
        manifest = _design_study_or_404(study_id)
        return render_template(
            "multilayer_design_detail.html",
            **_design_view_model(manifest),
        )

    @app.post("/multilayer-design/<study_id>/delete")
    def multilayer_design_delete(study_id: str):
        _delete_studies([study_id])
        return redirect(url_for("multilayer_design_index"))

    @app.post("/multilayer-design/<study_id>/stages/<stage>/run")
    def multilayer_design_run_stage(study_id: str, stage: str):
        if stage not in STAGES:
            abort(404)
        store = _design_store()
        manifest = _design_study_or_404(study_id)
        data_dir = app.config["GRAx_DATA_DIR"]
        if any(
            _is_run_active(app, _multilayer_design_job_key(study_id, other)) for other in STAGES
        ):
            abort(409, "Another stage of this study is still running.")

        designs: list[list[float]] = []
        if stage == "energy_scan":
            if manifest["stages"]["survey"]["status"] != "completed":
                abort(409, "Run the survey before scanning designs over energy.")
            try:
                designs = resolve_designs(
                    request.form.get("mode", "best"),
                    survey_design_options(store.study_dir(study_id)),
                    request.form.getlist("design"),
                )
            except ValueError as error:
                abort(400, str(error))

        manifest["stages"][stage]["status"] = "queued"
        manifest["stages"][stage]["error_text"] = ""
        manifest["stages"][stage]["designs"] = designs
        for later in downstream_stages(stage):
            if manifest["stages"][later]["status"] in {"completed", "aborted"}:
                manifest["stages"][later]["status"] = "stale"
        store.save(manifest)
        _start_multilayer_design_worker(
            app=app, data_dir=data_dir, study_id=study_id, stage=stage
        )
        return redirect(url_for("multilayer_design_detail", study_id=study_id))

    @app.get("/multilayer-design/<study_id>/script")
    def multilayer_design_script(study_id: str):
        """Download this study as one standalone script to run outside the web app."""

        manifest = _design_study_or_404(study_id)
        filename = offline_script_filename(manifest)
        return Response(
            render_offline_script(manifest),
            mimetype="text/x-python",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/multilayer-design/<study_id>/diagnostics")
    def multilayer_design_diagnostics(study_id: str):
        """Browse a study's stored result folders and plot its theta scans and other artifacts."""

        manifest = _design_study_or_404(study_id)
        root = _design_store().study_dir(study_id).resolve()
        requested = Path(request.args.get("path", ""))
        current = (root / requested).resolve()
        if requested.is_absolute() or not current.is_relative_to(root) or not current.is_dir():
            current, requested = root, Path("")
        folders = []
        for path in sorted(current.iterdir(), key=lambda item: item.name.lower()):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if path.is_dir() and resolved.is_relative_to(root):
                folders.append({"name": path.name, "path": resolved.relative_to(root).as_posix()})
        breadcrumbs = [{"name": "Study root", "path": ""}]
        for index, part in enumerate(requested.parts):
            breadcrumbs.append(
                {"name": part, "path": Path(*requested.parts[: index + 1]).as_posix()}
            )
        config = manifest["config"]
        plots = discover_plots(root, current, config)
        plot_id = request.args.get("plot_id", "")
        selected_plot = next((item for item in plots if item["id"] == plot_id), None)
        plot_data: dict[str, Any] = {"traces": [], "layout": {}}
        plot_error = ""
        if plot_id and selected_plot is None:
            plot_error = "That plot is not available in this folder."
        elif selected_plot:
            try:
                plot_data = build_diagnostic_plot(selected_plot, root, config)
            except (OSError, ValueError, KeyError, IndexError) as error:
                plot_error = str(error)
        return render_template(
            "multilayer_design_diagnostics.html",
            study=manifest,
            folders=folders,
            plots=[{key: value for key, value in item.items() if key != "source"} for item in plots],
            breadcrumbs=breadcrumbs,
            current_path=requested.as_posix() if requested.parts else "",
            selected_plot=selected_plot,
            plot_data=plot_data,
            plot_error=plot_error,
            show_scan_controls=bool(selected_plot and selected_plot["kind"] == "theta_all"),
            plotly_bundle=_plotly_bundle_text() if get_plotlyjs is not None else None,
        )

    @app.get("/multilayer-design/<study_id>/survey-options")
    def multilayer_design_survey_options(study_id: str):
        """Return the survey grid as it stands, so the plots can follow a live run."""

        _design_study_or_404(study_id)
        return jsonify(survey_design_options(_design_store().study_dir(study_id)))

    @app.get("/multilayer-design/<study_id>/energy-scan-points")
    def multilayer_design_energy_scan_points(study_id: str):
        """Return each design's solved energies, so the scan plot can follow a live run."""

        manifest = _design_study_or_404(study_id)
        return jsonify(
            {
                "designs": _energy_scan_checkpoint_series(
                    study_dir=_design_store().study_dir(study_id),
                    designs=manifest["stages"]["energy_scan"].get("designs") or [],
                )
            }
        )

    @app.get("/multilayer-design/<study_id>/stages/<stage>/status")
    def multilayer_design_stage_status(study_id: str, stage: str):
        if stage not in STAGES:
            abort(404)
        _design_study_or_404(study_id)
        return jsonify(
            _multilayer_design_stage_status_payload(
                app=app,
                data_dir=app.config["GRAx_DATA_DIR"],
                study_id=study_id,
                stage=stage,
            )
        )

    @app.get("/multilayer-design/<study_id>/stages/<stage>/abort")
    def multilayer_design_stage_abort_dialog(study_id: str, stage: str):
        if stage not in STAGES:
            abort(404)
        manifest = _design_study_or_404(study_id)
        return render_template(
            "multilayer_design_stage_abort.html",
            study=manifest,
            stage=stage,
            stage_label=STAGE_LABELS[stage],
        )

    @app.post("/multilayer-design/<study_id>/stages/<stage>/abort")
    def multilayer_design_stage_abort(study_id: str, stage: str):
        if stage not in STAGES:
            abort(404)
        _design_study_or_404(study_id)
        _abort_multilayer_design_stage(
            app=app,
            store=_design_store(),
            study_id=study_id,
            stage=stage,
            discard=request.form.get("disposition") == "discard",
        )
        return redirect(url_for("multilayer_design_detail", study_id=study_id))

    @app.post("/multilayer-design/<study_id>/stages/<stage>/reset")
    def multilayer_design_reset_stage(study_id: str, stage: str):
        if stage not in STAGES:
            abort(404)
        _design_study_or_404(study_id)
        if _is_run_active(app, _multilayer_design_job_key(study_id, stage)):
            abort(409, "This stage is still running.")
        _reset_multilayer_design_stage(
            store=_design_store(), study_id=study_id, stage=stage
        )
        return redirect(url_for("multilayer_design_detail", study_id=study_id))
