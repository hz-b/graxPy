"""Grating library, AFM upload and grating preview routes."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)
from uuid import (
    uuid4,
)

import numpy as np
from flask import (
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from grax.materials import (
    available_material_symbols,
    material_density_catalog,
)

from .. import (
    parameter_scans,
)
from ..afm_workflow import (
    MAX_BYTES,
    preview_afm_upload,
    process_afm_upload,
)
from ..catalog import (
    _linked_runs,
    _list_plot_runs,
)
from ..context import WebContext
from ..custom_grid import (
    default_configuration as default_custom_grid_configuration,
)
from ..custom_grid import (
    parameter_catalog as custom_grid_parameter_catalog,
)
from ..grating_forms import (
    _build_grating_preview,
    _custom_layer_rows,
    _default_form_values,
    _form_values_from_spec,
    _grating_custom_layer_rows,
    _material_density_placeholders,
    _persist_afm_artifacts,
    _render_grating_previews,
    _spec_from_form,
)
from ..grating_roughness import (
    validate_roughness_geometry,
)
from ..persistence import (
    build_grating_from_spec,
)
from ..run_jobs import (
    _queue_run,
)
from ..run_state import (
    _is_run_active,
)


def register(app: Any, ctx: WebContext) -> None:
    """Register the gratings routes on ``app``."""
    active_data_dir = ctx.active_data_dir
    run_store = ctx.run_store
    store = ctx.store

    def afm_request_bytes() -> tuple[bytes, str]:
        upload = request.files.get("afm_file")
        if upload is not None and upload.filename:
            return upload.read(MAX_BYTES + 1), upload.filename
        grating_id = str(request.form.get("afm_existing_grating_id") or "").strip()
        if grating_id:
            spec = store().load(grating_id)
            source = store().directory / grating_id / "source_upload"
            if spec.get("grating_type") == "afm" and source.is_file():
                return source.read_bytes(), str((spec.get("profile") or {}).get("source_filename") or "AFM scan")
        raise ValueError("Choose an AFM text file first.")

    @app.get("/gratings/new")
    def new_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=_default_form_values(),
        )
        defaults = _default_form_values()
        return render_template(
            "grating_form.html", materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()), defaults=defaults,
            density_placeholders=_material_density_placeholders(defaults),
            action_url=url_for("create_grating"), submit_label="Save grating",
            preview=preview, allow_custom=True, allow_bare=True,
            custom_layers=_grating_custom_layer_rows({}),
        )

    @app.post("/gratings/afm/preview")
    def preview_afm():
        """Preview the raw scan and selected preprocessing stages without saving."""

        try:
            raw, _ = afm_request_bytes()
            period_text = str(request.form.get("afm_period_nm", "")).strip()
            preview = preview_afm_upload(
                raw,
                units=str(request.form.get("afm_units", "nm")),
                profile_type=str(request.form.get("afm_profile_type", "blazed")),
                period_nm=float(period_text) if period_text else None,
                min_separation_fraction=float(request.form.get("afm_min_separation") or "0.4"),
                min_prominence_fraction=float(request.form.get("afm_min_prominence") or "0.1"),
                period_index=int(request.form.get("afm_period_index") or "0"),
                average=request.form.get("afm_average") == "1",
                reverse=request.form.get("afm_reverse") == "1",
                zero_baseline=request.form.get("afm_zero_baseline") == "1",
                periodicity_ramp=request.form.get("afm_periodicity_ramp") == "1",
            )
            return jsonify({"ok": True, **preview})
        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
            app.logger.exception("AFM preview failed")
            return jsonify({"ok": False, "error": str(error), "traceback": traceback.format_exc()}), 422
        except Exception as error:
            app.logger.exception("AFM preview crashed")
            return jsonify({
                "ok": False,
                "error": "AFM preview failed unexpectedly. Check the server log or retry.",
                "traceback": traceback.format_exc(),
            }), 500

    @app.post("/gratings/afm/process")
    def process_afm():
        """Process one uploaded AFM scan into a temporary profile workflow."""
        started = time.monotonic()
        workflow_id = uuid4().hex
        workflow_dir = active_data_dir() / "afm_workflows" / workflow_id
        try:
            raw, filename = afm_request_bytes()
            period_text = str(request.form.get("afm_period_nm") or "").strip()
            if period_text:
                period_nm = float(period_text)
            else:
                density_text = str(request.form.get("afm_target_period_lpermm") or "").strip()
                density = float(density_text) if density_text else 0.0
                if not np.isfinite(density) or density <= 0:
                    raise ValueError("Target period, lines/mm must be greater than zero.")
                period_nm = 1e6 / density
            metadata = process_afm_upload(
                raw, workflow_dir, filename=filename,
                units=str(request.form.get("afm_units", "nm")),
                profile_type=str(request.form.get("afm_profile_type", "blazed")),
                period_nm=period_nm,
                min_separation_fraction=float(request.form.get("afm_min_separation") or "0.4"),
                min_prominence_fraction=float(request.form.get("afm_min_prominence") or "0.1"),
                period_index=int(request.form.get("afm_period_index") or "0"),
                average=request.form.get("afm_average") == "1",
                reverse=request.form.get("afm_reverse") == "1",
                zero_baseline=request.form.get("afm_zero_baseline", "1") == "1",
                periodicity_ramp=request.form.get("afm_periodicity_ramp") == "1",
            )
            metadata["workflow_id"] = workflow_id
            metadata["profile_path"] = str((workflow_dir / "profile.csv").resolve())
            metadata["processing_seconds"] = round(time.monotonic() - started, 3)
            (workflow_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            app.logger.info("AFM processing completed in %.3fs (%s)", metadata["processing_seconds"], filename)
            return jsonify({"ok": True, **metadata})
        except (FileNotFoundError, TypeError, ValueError, OSError) as error:
            app.logger.warning("AFM processing failed after %.3fs: %s", time.monotonic() - started, error)
            return jsonify({
                "ok": False, "error": str(error), "traceback": traceback.format_exc(),
                "workflow_id": workflow_id,
            }), 422
        except Exception:
            app.logger.exception("AFM processing crashed after %.3fs", time.monotonic() - started)
            return jsonify({
                "ok": False,
                "error": "AFM processing failed unexpectedly. Check the server log or retry.",
                "traceback": traceback.format_exc(), "workflow_id": workflow_id,
            }), 500

    def invalid_grating_form(error, grating_id=None):
        """Keep submitted values visible when validation fails."""
        defaults = _default_form_values()
        defaults.update(request.form.to_dict())
        return render_template(
            "grating_form.html", defaults=defaults,
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            density_placeholders=_material_density_placeholders(defaults),
            action_url=url_for("update_grating", grating_id=grating_id) if grating_id else url_for("create_grating"),
            submit_label="Update grating" if grating_id else "Save grating",
            preview={"ok": False, "error": str(error)}, form_error=str(error),
            allow_custom=True, allow_bare=True,
            custom_layers=_custom_layer_rows(request.form.to_dict(flat=False)),
            existing_afm_source_url=(url_for("download_afm_source", grating_id=grating_id)
                                     if grating_id and (store().directory / grating_id / "source_upload").is_file()
                                     else None),
            existing_afm_grating_id=grating_id,
            return_to_run=request.form.get("return_to_run") == "1",
        ), 422

    @app.post("/gratings")
    def create_grating():
        try:
            spec = _spec_from_form(request.form)
            validate_roughness_geometry(build_grating_from_spec(spec))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return invalid_grating_form(error)
        saved = store().save(spec)
        if saved.get("grating_type") == "afm":
            try:
                _persist_afm_artifacts(saved, active_data_dir())
                saved = store().save(saved)
            except (OSError, ValueError) as error:
                store().delete(saved["id"])
                shutil.rmtree(store().directory / saved["id"], ignore_errors=True)
                return invalid_grating_form(error)
        return redirect(url_for("grating_detail", grating_id=saved["id"]))

    @app.get("/gratings/manage")
    def manage_gratings():
        data_dir = active_data_dir()
        run_counts: dict[str, int] = {}
        for run in _list_plot_runs(data_dir / "runs"):
            linked_grating_id = str(run.get("grating_id", ""))
            if linked_grating_id:
                run_counts[linked_grating_id] = run_counts.get(linked_grating_id, 0) + 1
        return render_template(
            "grating_manage.html",
            gratings=store().list(),
            grating_run_counts=run_counts,
            message=str(request.args.get("message", "")).strip(),
            message_kind=str(request.args.get("kind", "info")).strip() or "info",
        )

    @app.post("/gratings/manage")
    def update_gratings():
        action = str(request.form.get("action", "delete")).strip()
        selected_ids = [str(grating_id) for grating_id in request.form.getlist("delete_grating_id")]
        if not selected_ids:
            return redirect(url_for("manage_gratings", message="Select at least one grating first.", kind="error"))
        if action == "duplicate":
            duplicated: list[str] = []
            failures: list[str] = []
            for grating_id in selected_ids:
                saved: dict[str, Any] | None = None
                try:
                    source = store().load(grating_id)
                    spec = deepcopy(source)
                    for key in ("id", "created_at", "updated_at"):
                        spec.pop(key, None)
                    spec["name"] = f"{str(source.get('name') or 'Grating').strip()} (copy)"
                    saved = store().save(spec)
                    if saved.get("grating_type") == "afm":
                        _persist_afm_artifacts(saved, active_data_dir())
                        saved = store().save(saved)
                    duplicated.append(str(saved["name"]))
                except (KeyError, OSError, TypeError, ValueError) as error:
                    if saved is not None:
                        store().delete(str(saved["id"]))
                        shutil.rmtree(store().directory / str(saved["id"]), ignore_errors=True)
                    failures.append(str(error))
            if failures:
                message = f"Duplicated {len(duplicated)}; could not duplicate {len(failures)} selected grating{'s' if len(failures) != 1 else ''}."
                if not duplicated:
                    message += f" {failures[0]}"
                return redirect(url_for("manage_gratings", message=message, kind="error"))
            return redirect(url_for(
                "manage_gratings",
                message=f"Created {len(duplicated)} grating {'copy' if len(duplicated) == 1 else 'copies'}.",
                kind="success",
            ))
        if action != "delete":
            return redirect(url_for("manage_gratings"))

        delete_mode = str(request.form.get("delete_mode", "grating_only")).strip()
        if delete_mode not in {"grating_only", "grating_and_runs"}:
            delete_mode = "grating_only"
        if delete_mode == "grating_and_runs":
            active_linked = [
                run_id
                for grating_id in selected_ids
                for run_id in [str(run["id"]) for run in _linked_runs(active_data_dir(), grating_id)]
                if _is_run_active(app, run_id)
            ]
            if active_linked:
                return redirect(url_for("manage_gratings", message="Cannot delete linked runs while active runs exist.", kind="error"))
        for grating_id in selected_ids:
            if delete_mode == "grating_and_runs":
                run_store().delete_many([str(run["id"]) for run in _linked_runs(active_data_dir(), grating_id)])
            store().delete(grating_id)
            preview_path = active_data_dir() / "previews" / f"{grating_id}.png"
            if preview_path.exists():
                preview_path.unlink()
        return redirect(url_for(
            "manage_gratings",
            message=f"Deleted {len(selected_ids)} grating{'s' if len(selected_ids) != 1 else ''}.",
            kind="success",
        ))

    @app.post("/gratings/<grating_id>/rename")
    def rename_grating(grating_id: str):
        grating_store = store()
        spec = grating_store.load(grating_id)
        name = str(request.form.get("name", "")).strip()
        if not name:
            return redirect(url_for("manage_gratings", message="A grating name cannot be empty.", kind="error"))
        if len(name) > 128:
            return redirect(url_for("manage_gratings", message="Grating names must be 128 characters or fewer.", kind="error"))
        spec["name"] = name
        grating_store.save(spec)
        return redirect(url_for("manage_gratings", message=f"Renamed grating to “{name}”.", kind="success"))

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
        rough_path, notices = _render_grating_previews(grating, preview_path)
        version = uuid4().hex
        return render_template(
            "grating_detail.html",
            grating=spec,
            linked_runs=_linked_runs(active_data_dir(), grating_id),
            preview_url=url_for("data_file", filename=f"previews/{grating_id}.png", v=version),
            roughness_preview_url=(url_for("data_file", filename=f"previews/{rough_path.name}", v=version)
                                   if rough_path else None),
            preview_warnings=notices,
            roughness_periods=grating._roughness_num_supercells(),
            custom_grid_catalog=[
                descriptor.to_json() for descriptor in custom_grid_parameter_catalog(spec)
            ],
            custom_grid_default=default_custom_grid_configuration(spec),
            scan_catalogs={mode: [item.to_json() for item in parameter_scans.scan_catalog(spec, mode)] for mode in ("energy_sweep", "angle_sweep", "monochromator", "constant_deviation")},
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
            spec=spec,
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
            allow_custom=True,
            allow_bare=True,
            custom_layers=_grating_custom_layer_rows(spec),
            existing_afm_source_url=(url_for("download_afm_source", grating_id=grating_id)
                                     if spec.get("grating_type") == "afm"
                                     and (store().directory / grating_id / "source_upload").is_file()
                                     else None),
            existing_afm_grating_id=grating_id,
            return_to_run=request.args.get("return_to_run") == "1",
        )

    @app.get("/gratings/<grating_id>/afm/source")
    def download_afm_source(grating_id: str):
        spec = store().load(grating_id)
        source = store().directory / grating_id / "source_upload"
        if spec.get("grating_type") != "afm" or not source.is_file():
            abort(404)
        filename = Path(str((spec.get("profile") or {}).get("source_filename") or "afm_scan.txt")).name
        return send_from_directory(source.parent, source.name, as_attachment=True, download_name=filename)

    @app.post("/gratings/<grating_id>")
    def update_grating(grating_id: str):
        grating_store = store()
        previous = grating_store.load(grating_id)
        try:
            spec = _spec_from_form(request.form)
            spec["id"] = grating_id
            spec["created_at"] = previous.get("created_at")
            validate_roughness_geometry(build_grating_from_spec(spec))
            if spec.get("grating_type") == "afm":
                _persist_afm_artifacts(spec, active_data_dir(), previous=previous)
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as error:
            return invalid_grating_form(error, grating_id)
        grating_store.save(spec)
        if request.form.get("return_to_run") == "1":
            return redirect(url_for("grating_detail", grating_id=grating_id, restore_run="1"))
        return redirect(url_for("grating_detail", grating_id=grating_id))

    @app.post("/gratings/<grating_id>/runs")
    def create_run(grating_id: str):
        spec = store().load(grating_id)
        try:
            grating = build_grating_from_spec(spec)
        except (TypeError, ValueError) as error:
            abort(400, str(error))
        try:
            run = _queue_run(
                app=app,
                data_dir=app.config["GRAx_DATA_DIR"],
                grating_id=grating_id,
                grating_name=str(spec["name"]),
                grating_spec=spec,
                grating=grating,
                form=request.form,
            )
        except (TypeError, ValueError, OverflowError) as error:
            abort(422, str(error))
        return redirect(url_for("run_detail", run_id=run["id"]))

    @app.post("/_preview/grating")
    def preview_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            form_data=request.form,
        )
        return jsonify(preview)
