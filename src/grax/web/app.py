"""Flask application for the local grax web MVP."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from grax.materials import (
    available_material_symbols,
    material_density_catalog,
)
from .grating_roughness import validate_roughness_geometry
from .multilayer_design_studies import (
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
from .persistence import (
    GratingStore,
    PlaneMirrorStore,
    build_grating_from_spec,
)
from .plane_mirror import (
    clean_label,
    clean_name,
    plane_mirror_summary,
    split_plane_mirror_form,
    unique_scan_id,
    stack_schematic_data_uri,
    parse_plane_mirror_options,
    plane_mirror_csv,
)
from .runs import RunStore
from .plane_mirror_runs import (
    MAX_TOTAL_SAMPLES as MAX_PLANE_MIRROR_RUN_SAMPLES,
    expand_variations as expand_plane_mirror_variations,
    start_worker as start_plane_mirror_worker,
    total_sample_count as plane_mirror_sample_count,
)
from . import parameter_scans
from .afm_workflow import (
    MAX_BYTES,
    preview_afm_upload,
    process_afm_upload,
)
from .custom_grid import (
    default_configuration as default_custom_grid_configuration,
    parameter_catalog as custom_grid_parameter_catalog,
)

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
from .catalog import (
    _is_grating_run,
    _is_plane_mirror_run,
    _linked_runs,
    _list_plot_runs,
    _list_plots,
    _manage_run_entries,
    _plot_references_for_runs,
    _run_comparison_parameter_rows,
    _scan_run_rows,
)
from .common import (
    PATH_ID_KEYS,
    _file_token,
    _is_cross_site_request,
    _is_loopback_address,
    _is_safe_id,
    _site_metadata,
    _split_variation_values,
)
from .grating_forms import (
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
from .multilayer_jobs import (
    _abort_multilayer_design_stage,
    _energy_scan_checkpoint_series,
    _multilayer_design_job_key,
    _multilayer_design_stage_status_payload,
    _reset_multilayer_design_stage,
    _start_multilayer_design_worker,
)
from .plane_mirror_views import (
    _load_plane_mirror,
    _plane_mirror_page_defaults,
    _plane_mirror_result,
    _plane_mirror_scan_parameter_details,
    _plane_mirror_stack,
    _plane_mirror_variation_catalog,
)
from .plot_builder import (
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
from .run_exports import (
    _run_export_archive,
    _run_export_rows,
    _run_metadata_yaml,
)
from .run_files import (
    _available_orders,
    _load_checkpoint_results,
    _load_run_manifest,
    _parameter_study_checkpoint_count,
    _update_manifest_fields,
)
from .run_inputs import (
    _custom_grid_configuration_from_value,
    _custom_grid_from_form,
    _worker_settings_from_form,
)
from .run_jobs import (
    _abort_run,
    _memory_status_payload,
    _queue_run,
    _repair_saved_run_plot,
    _run_status_payload,
    _start_run_worker,
)
from .run_state import (
    _has_active_runs,
    _is_run_active,
)
from .workspace import (
    _list_directory_entries,
    _path_breadcrumbs,
    _pick_native_directory,
    _safe_browse_path,
    _validate_workspace_root,
)

# Re-exported so existing imports from grax.web.app keep working.
from .common import (  # noqa: F401
    _grax_version,
    _normalized_polarization,
    _normalized_solver,
    _write_text_atomically,
)
from .grating_forms import (  # noqa: F401
    _attach_roughness,
)
from .plot_builder import (  # noqa: F401
    _compare_x_values,
    _load_selected_plot_series,
    _require_plotly,
    _save_compare_image,
    _series_controls_for_selected_series,
)
from .run_exports import (  # noqa: F401
    _run_metadata_payload,
)
from .multilayer_jobs import (  # noqa: F401
    _energy_scan_checkpoint_progress,
)
from .run_files import (  # noqa: F401
    _load_order_series,
    _parameter_study_artifacts,
    _replace_plot_with_retry,
)
from .run_inputs import (  # noqa: F401
    _cases_for_workflow,
    _custom_grid_defaults_from_form,
    _parameter_scan_point_count,
    _parameter_scans_from_form,
    _run_input_from_form,
    _snapshot_custom_grid_afm_profiles,
)
from .run_jobs import (  # noqa: F401
    _publish_live_progress_snapshot,
    _results_sorted_for_live_plot,
    psutil,
)
from .workspace import platform  # noqa: F401
from .run_state import (  # noqa: F401
    ActiveRunState,
)


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
            Response,
            abort,
            jsonify,
            redirect,
            render_template,
            request,
            send_file,
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

    @app.url_defaults
    def version_frontend_assets(endpoint: str, values: dict[str, Any]) -> None:
        """Load updated frontend assets after a page refresh."""
        if endpoint == "static" and values.get("filename") in {
            "web.js", "web.css", "geometry.js", "afm.js", "graxpy-logo.svg", "hzb-bessy2-logo.png",
            "graxpy-hzb-logo.png",
        }:
            values.setdefault("v", _file_token(Path(app.static_folder) / values["filename"]))

    @app.url_value_preprocessor
    def reject_unsafe_ids(endpoint: str | None, values: dict[str, Any] | None) -> None:
        """Answer 404 for URL IDs that could point outside the workspace folders."""
        for key in PATH_ID_KEYS:
            value = (values or {}).get(key)
            if value is not None and not _is_safe_id(value):
                abort(404)

    @app.before_request
    def reject_cross_site_writes() -> None:
        """Refuse state-changing requests that another website's page sent."""
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return
        if _is_cross_site_request(
            origin=request.headers.get("Origin"),
            fetch_site=request.headers.get("Sec-Fetch-Site"),
            host_url=request.host_url,
        ):
            abort(403, "Requests from other websites are not allowed.")

    def store() -> GratingStore:
        return GratingStore(app.config["GRAx_DATA_DIR"] / "saved_gratings")

    def plane_mirror_store() -> PlaneMirrorStore:
        return PlaneMirrorStore(app.config["GRAx_DATA_DIR"] / "saved_plane_mirrors")

    def run_store() -> RunStore:
        return RunStore(app.config["GRAx_DATA_DIR"] / "runs")

    def preview_root() -> Path:
        return app.config["GRAx_DATA_DIR"] / "previews" / "live"

    def active_data_dir() -> Path:
        return Path(app.config["GRAx_DATA_DIR"]).resolve()

    def load_user_settings() -> dict[str, Any]:
        settings_path = active_data_dir() / "settings.json"
        try:
            with settings_path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        decimal_separator = stored.get("decimal_separator")
        standard_polarization = stored.get("standard_polarization")
        saved = decimal_separator in {"dot", "comma"} and standard_polarization in {"s", "p"}
        return {
            "decimal_separator": decimal_separator if decimal_separator in {"dot", "comma"} else "dot",
            "standard_polarization": standard_polarization if standard_polarization in {"s", "p"} else "p",
            "saved": saved,
        }

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

    @app.context_processor
    def inject_site_metadata() -> dict[str, Any]:
        """Inject shared site attribution and saved user settings into templates."""

        return {"site_metadata": _site_metadata(), "app_settings": load_user_settings()}

    @app.post("/settings")
    def update_user_settings():
        """Persist display preferences beside the active database."""

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "Settings must be sent as a JSON object.")
        decimal_separator = payload.get("decimal_separator")
        standard_polarization = payload.get("standard_polarization")
        if decimal_separator not in {"dot", "comma"}:
            abort(400, "Decimal separator must be dot or comma.")
        if standard_polarization not in {"s", "p"}:
            abort(400, "Standard polarization must be s or p.")
        settings_path = active_data_dir() / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = settings_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps({
                "decimal_separator": decimal_separator,
                "standard_polarization": standard_polarization,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(settings_path)
        return jsonify({"saved": True})

    @app.get("/")
    def index():
        data_dir = active_data_dir()
        gratings = store().list()
        gratings.sort(
            key=lambda grating: (
                str(grating.get("updated_at", "")),
                str(grating.get("created_at", "")),
                str(grating.get("name", "")).lower(),
            ),
            reverse=True,
        )
        return render_template(
            "index.html",
            gratings=gratings,
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

    @app.get("/docs/<topic>")
    def web_docs_topic(topic: str):
        templates = {
            "gratings": "web_docs_gratings.html",
            "plane-mirrors": "web_docs_plane_mirrors.html",
            "multilayer-gratings": "web_docs_multilayer_gratings.html",
        }
        if topic not in templates:
            abort(404)
        return render_template(templates[topic])

    @app.post("/workspace/pick")
    def pick_workspace():
        if _has_active_runs(app):
            return redirect(
                url_for(
                    "index",
                    message="Cannot change the database folder while runs are active.",
                    kind="error",
                )
            )
        if not _is_loopback_address(request.remote_addr):
            return redirect(
                url_for(
                    "index",
                    message=(
                        "System folder selection is available only from the local machine."
                    ),
                    kind="error",
                )
            )
        try:
            selected_dir = _pick_native_directory()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return redirect(
                url_for(
                    "index",
                    message=(
                        f"The system folder picker is unavailable ({error}). "
                        "Use a local desktop session with a supported folder picker."
                    ),
                    kind="error",
                )
            )
        if selected_dir is None:
            return redirect(
                url_for(
                    "index",
                    message="Folder selection canceled. The active database folder was not changed.",
                )
            )
        try:
            target_dir = _validate_workspace_root(selected_dir)
        except (OSError, ValueError) as error:
            return redirect(url_for("index", message=str(error), kind="error"))
        app.config["GRAx_DATA_DIR"] = target_dir
        return redirect(
            url_for("index", message=f"Database folder set to {target_dir}", kind="success")
        )

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

    @app.get("/system/memory")
    def system_memory():
        return jsonify(_memory_status_payload())

    @app.get("/system/resource-status")
    def system_resource_status():
        from .resource_manager import resource_status
        return jsonify(resource_status())

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

    # ------------------------------------------------------------------ #
    # Multilayer-design studies                                          #
    # ------------------------------------------------------------------ #
    def _design_store() -> MultilayerDesignStudyStore:
        return MultilayerDesignStudyStore(app.config["GRAx_DATA_DIR"] / "multilayer_designs")

    def _design_study_or_404(study_id: str) -> dict[str, Any]:
        try:
            return _design_store().load(study_id)
        except (ValueError, FileNotFoundError, OSError):
            abort(404)

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
            studies=_design_store().list(),
            stage_labels=STAGE_LABELS,
            stages=STAGES,
        )

    @app.post("/multilayer-design")
    def multilayer_design_create():
        if request.form.get("action") == "delete":
            _design_store().delete_many(request.form.getlist("delete_study_id"))
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
        _design_store().delete_many([study_id])
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

    def render_plane_mirror_page(
        mirror: dict[str, Any] | None = None,
        scan_id: str | None = None,
        *,
        editing: bool = False,
    ):
        stack_form: dict[str, Any] = {} if mirror is None else dict(mirror.get("stack_form", {}))
        scans = [] if mirror is None else list(mirror.get("scans", []))
        active_scan = next((scan for scan in scans if scan["id"] == scan_id), None)
        if active_scan is None and scans and scan_id is None:
            active_scan = scans[0]
        defaults = _plane_mirror_page_defaults(stack_form, {} if active_scan is None else active_scan["form"])
        if active_scan is None:
            defaults["polarization"] = load_user_settings()["standard_polarization"]
        custom_layers = _custom_layer_rows(stack_form)
        custom_variation_count = len(custom_layers) if stack_form.get("stack_type") == "custom" else 0
        return render_template(
            "plane_mirror.html",
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            defaults=defaults,
            density_placeholders=_material_density_placeholders(defaults),
            allow_custom=True,
            custom_layers=custom_layers,
            mirror=mirror,
            editing=editing,
            scans=scans,
            active_scan_id=None if active_scan is None else active_scan["id"],
            mirror_numeric_parameters=_plane_mirror_variation_catalog(
                numeric_only=True, custom_layer_count=custom_variation_count
            ),
            mirror_variation_parameters=_plane_mirror_variation_catalog(
                custom_layer_count=custom_variation_count
            ),
            message=str(request.args.get("message", "")).strip(),
            plotly_bundle=_plotly_bundle_text() if get_plotlyjs is not None else None,
        )

    @app.get("/plane-mirror")
    def plane_mirror_page():
        return render_plane_mirror_page()

    @app.get("/plane-mirrors")
    def plane_mirror_index():
        mirrors = [{**spec, "summary": plane_mirror_summary(spec)} for spec in plane_mirror_store().list()]
        return render_template(
            "plane_mirror_library.html",
            mirrors=mirrors,
            message=str(request.args.get("message", "")).strip(),
        )

    @app.get("/plane-mirror-runs")
    def plane_mirror_runs():
        manifests = [run for run in run_store().list() if run.get("mirror_id")]
        active = app.extensions.setdefault("grax_mirror_run_events", {})
        for manifest in manifests:
            if manifest.get("status") in {"queued", "running", "aborting"} and manifest["id"] not in active:
                manifest.update(status="aborted", error_text="The app stopped before this run finished.")
                run_store().save(manifest)
        return render_template(
            "plane_mirror_manage.html",
            runs=manifests,
            message=str(request.args.get("message", "")).strip(),
        )

    @app.get("/plane-mirrors/<mirror_id>/stack-preview.png")
    def plane_mirror_saved_stack_preview(mirror_id: str):
        import base64
        from werkzeug.datastructures import MultiDict

        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        form = MultiDict()
        for key, value in mirror.get("stack_form", {}).items():
            if isinstance(value, list):
                for item in value:
                    form.add(key, item)
            else:
                form.add(key, value)
        try:
            data_uri = stack_schematic_data_uri(_plane_mirror_stack(form))
        except (KeyError, TypeError, ValueError):
            abort(404)
        encoded = data_uri.partition(",")[2]
        return Response(
            base64.b64decode(encoded),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.post("/plane-mirrors")
    def plane_mirror_create():
        try:
            name = clean_name(request.form.get("mirror_name"), "Mirror name")
            stack_form, _ = split_plane_mirror_form(request.form)
            _plane_mirror_stack(request.form)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        saved = plane_mirror_store().save(
            {
                "name": name,
                "label": clean_label(request.form.get("mirror_label")),
                "stack_form": stack_form,
                "scans": [],
            }
        )
        return redirect(url_for("plane_mirror_detail", mirror_id=saved["id"]))

    @app.post("/plane-mirrors/<mirror_id>/runs")
    def plane_mirror_run_create(mirror_id: str):
        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        from werkzeug.datastructures import MultiDict
        excluded = {
            "run_name", "variation_axes", "variation_coupled", "scan_name", "scan_label",
            "axis_parameter", "axis_start", "axis_stop", "axis_points",
            "coupled_left_parameter", "coupled_left_values",
            "coupled_right_parameter", "coupled_right_values",
        }
        clean_form = MultiDict()
        for key in request.form.keys():
            if key in excluded:
                continue
            for value in request.form.getlist(key):
                clean_form.add(key, value)
        try:
            _, scan_form = split_plane_mirror_form(clean_form)
            stack_form = dict(mirror.get("stack_form", {}))
            parse_plane_mirror_options(clean_form)
            axis_rows = list(zip(
                request.form.getlist("axis_parameter"),
                request.form.getlist("axis_start"),
                request.form.getlist("axis_stop"),
                request.form.getlist("axis_points"),
            ))
            axes = []
            for parameter, start, stop, points in axis_rows:
                if parameter:
                    axes.append({"parameter": parameter, "start": start, "stop": stop, "points": points})
                elif start.strip() or stop.strip():
                    raise ValueError("Choose a parameter for every independent range.")
            coupled = []
            for left, left_values, right, right_values in zip(
                request.form.getlist("coupled_left_parameter"),
                request.form.getlist("coupled_left_values"),
                request.form.getlist("coupled_right_parameter"),
                request.form.getlist("coupled_right_values"),
            ):
                if left or right or left_values.strip() or right_values.strip():
                    if not left or not right:
                        raise ValueError("Choose both parameters for every coupled value row.")
                    coupled.append({left: _split_variation_values(left_values), right: _split_variation_values(right_values)})
            cases = expand_plane_mirror_variations(scan_form, stack_form, json.dumps(axes), json.dumps(coupled))
            for case in cases:
                case_form = MultiDict()
                for source in (case["stack"], case["scan"]):
                    for key, value in source.items():
                        for item in value if isinstance(value, list) else [value]:
                            case_form.add(key, item)
                _plane_mirror_stack(case_form)
                parse_plane_mirror_options(case_form)
            sample_total = sum(plane_mirror_sample_count(case["scan"]) for case in cases)
            if sample_total > MAX_PLANE_MIRROR_RUN_SAMPLES:
                raise ValueError(
                    f"This run requests {sample_total:,} scan samples across all variations; "
                    f"the limit is {MAX_PLANE_MIRROR_RUN_SAMPLES:,}."
                )
            display_name = clean_name(
                request.form.get("run_name") or f"{mirror['name']} · {scan_form.get('scan_mode', 'energy')} scan",
                "Run name",
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            abort(422, str(error))
        run_id = f"plane-mirror-{mirror_id}-{uuid4().hex[:12]}"
        now = datetime.now().isoformat(timespec="seconds")
        manifest = {
            "id": run_id, "workflow": "plane_mirror", "mirror_id": mirror_id,
            "mirror_name": mirror["name"], "mirror_snapshot": {"name": mirror["name"], "stack_form": stack_form},
            "variation_axes": axes, "variation_coupled": coupled,
            "display_name": display_name, "created_at": now, "updated_at": now,
            "status": "queued", "completed_points": 0, "total_points": len(cases),
            "scan_details": _plane_mirror_scan_parameter_details(cases[0]["scan"]),
            "cases": cases, "artifacts": [],
        }
        run_store().save(manifest)
        start_plane_mirror_worker(app, active_data_dir(), run_id)
        return redirect(url_for("plane_mirror_run_detail", run_id=run_id))

    @app.get("/plane-mirror-runs/<run_id>")
    def plane_mirror_run_detail(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        if manifest.get("status") in {"queued", "running", "aborting"} and run_id not in app.extensions.setdefault("grax_mirror_run_events", {}):
            manifest.update(status="aborted", error_text="The app stopped before this run finished.")
            run_store().save(manifest)
        result_path = active_data_dir() / "runs" / run_id / "mirror_results.json"
        try:
            results = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            results = []
        return render_template("plane_mirror_run_detail.html", run=manifest, results=results)

    @app.get("/plane-mirror-runs/<run_id>/status")
    def plane_mirror_run_status(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        manifest["plot_url"] = url_for("data_file", filename=f"runs/{run_id}/plot.png", v=manifest.get("completed_points", 0)) if manifest.get("completed_points") else None
        return jsonify({key: manifest.get(key) for key in ("status", "completed_points", "total_points", "error_text", "plot_url")})

    @app.get("/plane-mirror-runs/<run_id>/stack-preview.png")
    def plane_mirror_run_stack_preview(run_id: str):
        import base64
        from werkzeug.datastructures import MultiDict
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        form = MultiDict()
        for key, value in manifest.get("mirror_snapshot", {}).get("stack_form", {}).items():
            for item in value if isinstance(value, list) else [value]:
                form.add(key, item)
        try:
            image_uri = stack_schematic_data_uri(_plane_mirror_stack(form))
        except (KeyError, TypeError, ValueError):
            abort(404)
        return Response(base64.b64decode(image_uri.partition(",")[2]), mimetype="image/png", headers={"Cache-Control": "public, max-age=300"})

    @app.post("/plane-mirror-runs/<run_id>/abort")
    def plane_mirror_run_abort(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        event = app.extensions.setdefault("grax_mirror_run_events", {}).get(run_id)
        if event:
            event.set()
            manifest.update(status="aborting", updated_at=datetime.now().isoformat(timespec="seconds"))
            run_store().save(manifest)
        return redirect(url_for("plane_mirror_run_detail", run_id=run_id))

    @app.post("/plane-mirror-runs/<run_id>/resume")
    def plane_mirror_run_resume(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        if run_id in app.extensions.setdefault("grax_mirror_run_events", {}):
            abort(409, "This run is already active.")
        if manifest.get("status") not in {"aborted", "error", "failed"}:
            abort(409, "Only an aborted or failed run can be resumed.")
        start_plane_mirror_worker(app, active_data_dir(), run_id, resume=True)
        return redirect(url_for("plane_mirror_run_detail", run_id=run_id))

    @app.post("/plane-mirror-runs/<run_id>/restart")
    def plane_mirror_run_restart(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        if run_id in app.extensions.setdefault("grax_mirror_run_events", {}):
            abort(409, "Abort this run before restarting it.")
        if manifest.get("status") not in {"completed", "aborted", "error", "failed"}:
            abort(409, "Wait for this run to finish or abort it before restarting.")
        run_dir = active_data_dir() / "runs" / run_id
        (run_dir / "mirror_results.json").unlink(missing_ok=True)
        (run_dir / "plot.png").unlink(missing_ok=True)
        manifest.update(status="queued", completed_points=0, error_text="", updated_at=datetime.now().isoformat(timespec="seconds"))
        run_store().save(manifest)
        start_plane_mirror_worker(app, active_data_dir(), run_id)
        return redirect(url_for("plane_mirror_run_detail", run_id=run_id))

    @app.post("/plane-mirrors/manage")
    def plane_mirror_manage_update():
        for mirror_id in request.form.getlist("delete_mirror_id"):
            plane_mirror_store().delete(str(mirror_id))
        return redirect(url_for("plane_mirror_index"))

    @app.get("/plane-mirrors/<mirror_id>")
    def plane_mirror_detail(mirror_id: str):
        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        scans = list(mirror.get("scans", []))
        scan_id = request.args.get("scan")
        active_scan = next((scan for scan in scans if scan["id"] == scan_id), None)
        if active_scan is None and scans and scan_id is None:
            active_scan = scans[0]
        stack_form = dict(mirror.get("stack_form", {}))
        defaults = _plane_mirror_page_defaults(
            stack_form, {} if active_scan is None else active_scan.get("form", {})
        )
        # Legacy scans remain stored as-is, while their settings load into the
        # equivalent current run workflow when selected as a preset.
        defaults["scan_mode"] = {
            "energy": "energy_sweep",
            "angle": "angle_sweep",
        }.get(defaults.get("scan_mode", "energy"), defaults.get("scan_mode", "energy_sweep"))
        if active_scan is None:
            defaults["polarization"] = load_user_settings()["standard_polarization"]
        custom_layers = _custom_layer_rows(stack_form)
        custom_variation_count = len(custom_layers) if stack_form.get("stack_type") == "custom" else 0
        return render_template(
            "plane_mirror_scan.html",
            mirror=mirror,
            mirror_summary=plane_mirror_summary(mirror),
            scans=scans,
            runs=[run for run in run_store().list() if run.get("mirror_id") == mirror_id][:5],
            active_scan_id=None if active_scan is None else active_scan["id"],
            defaults=defaults,
            custom_layers=custom_layers,
            mirror_numeric_parameters=_plane_mirror_variation_catalog(
                numeric_only=True, custom_layer_count=custom_variation_count
            ),
            mirror_variation_parameters=_plane_mirror_variation_catalog(
                custom_layer_count=custom_variation_count
            ),
            plotly_bundle=_plotly_bundle_text() if get_plotlyjs is not None else None,
        )

    @app.get("/plane-mirrors/<mirror_id>/edit")
    def plane_mirror_edit(mirror_id: str):
        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        return render_plane_mirror_page(mirror, editing=True)

    @app.get("/plane-mirrors/<mirror_id>/scans/<scan_id>/preview.png")
    def plane_mirror_scan_preview(mirror_id: str, scan_id: str):
        from werkzeug.datastructures import MultiDict

        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        scan = next((item for item in mirror.get("scans", []) if item.get("id") == scan_id), None)
        if scan is None:
            abort(404)

        stack_form = mirror.get("stack_form", {})
        form_values = _plane_mirror_page_defaults(stack_form, scan.get("form", {}))
        # Keep thumbnails responsive even when the saved scan has a dense grid.
        for field_name, limit in (
            ("energy_points", 240),
            ("angle_points", 240),
            ("map_energy_points", 100),
            ("map_angle_points", 100),
        ):
            try:
                form_values[field_name] = str(min(int(form_values[field_name]), limit))
            except (KeyError, TypeError, ValueError):
                pass

        form = MultiDict()
        for key, value in form_values.items():
            form.add(key, value)
        for key, values in stack_form.items():
            if key.startswith("cl_") and isinstance(values, list):
                for value in values:
                    form.add(key, value)

        try:
            result = _plane_mirror_result(form)
        except (KeyError, TypeError, ValueError):
            abort(404)

        figure, axis = plt.subplots(figsize=(4.2, 2.2), dpi=110)
        if result["mode"] == "map":
            axis.pcolormesh(
                result["x"], result["y"], result["z"], shading="auto",
                cmap="viridis", vmin=0, vmax=1,
            )
            axis.set_xlabel("Photon energy (eV)")
            axis.set_ylabel("Grazing angle (deg)")
        else:
            axis.plot(result["x"], result["reflectivity"], linewidth=1.4, label="Reflectivity")
            if result["graded"] is not None:
                axis.plot(result["x"], result["graded"], linewidth=1.2, label="Footprint average")
                axis.legend(fontsize=7, frameon=False)
            axis.set_xlabel(result["x_label"])
            axis.set_ylabel("Reflectivity")
            axis.set_ylim(0, 1)
        axis.set_title(scan.get("name", "Saved scan"), fontsize=9)
        axis.grid(True, alpha=0.2, linewidth=0.5)
        figure.tight_layout(pad=1.1)
        image = io.BytesIO()
        figure.savefig(image, format="png", facecolor="white")
        plt.close(figure)
        image.seek(0)
        return send_file(image, mimetype="image/png", max_age=300)

    @app.post("/plane-mirrors/<mirror_id>")
    def plane_mirror_update(mirror_id: str):
        store = plane_mirror_store()
        mirror = _load_plane_mirror(store, mirror_id)
        try:
            mirror["name"] = clean_name(request.form.get("mirror_name"), "Mirror name")
            mirror["label"] = clean_label(request.form.get("mirror_label"))
            mirror["stack_form"], _ = split_plane_mirror_form(request.form)
            _plane_mirror_stack(request.form)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        store.save(mirror)
        return redirect(url_for("plane_mirror_detail", mirror_id=mirror_id, message="Plane mirror updated."))

    @app.post("/plane-mirrors/<mirror_id>/scans")
    def plane_mirror_save_scan(mirror_id: str):
        store = plane_mirror_store()
        mirror = _load_plane_mirror(store, mirror_id)
        try:
            scan_name = clean_name(request.form.get("scan_name"), "Scan name")
            _, scan_form = split_plane_mirror_form(request.form)
            parse_plane_mirror_options(request.form)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        scans = list(mirror.get("scans", []))
        existing = next((scan for scan in scans if scan["name"] == scan_name), None)
        scan_label = clean_label(request.form.get("scan_label"))
        if existing is not None:
            existing["form"] = scan_form
            existing["label"] = scan_label
            scan_id = existing["id"]
        else:
            scan_id = unique_scan_id(scan_name, {scan["id"] for scan in scans})
            scans.append({"id": scan_id, "name": scan_name, "label": scan_label, "form": scan_form})
        mirror["scans"] = scans
        store.save(mirror)
        return redirect(
            url_for("plane_mirror_detail", mirror_id=mirror_id, scan=scan_id, message=f"Saved scan '{scan_name}'.")
        )

    @app.post("/plane-mirrors/<mirror_id>/scans/<scan_id>/rename")
    def plane_mirror_rename_scan(mirror_id: str, scan_id: str):
        store = plane_mirror_store()
        mirror = _load_plane_mirror(store, mirror_id)
        scan = next((scan for scan in mirror.get("scans", []) if scan["id"] == scan_id), None)
        if scan is None:
            abort(404)
        try:
            new_name = clean_name(request.form.get("scan_name"), "Scan name")
        except ValueError as error:
            abort(400, str(error))
        if any(other["name"] == new_name and other["id"] != scan_id for other in mirror["scans"]):
            abort(400, f"A scan named '{new_name}' already exists.")
        scan["name"] = new_name
        scan["label"] = clean_label(request.form.get("scan_label"))
        store.save(mirror)
        return redirect(url_for("plane_mirror_detail", mirror_id=mirror_id, scan=scan_id, message="Scan updated."))

    @app.post("/plane-mirrors/<mirror_id>/scans/<scan_id>/delete")
    def plane_mirror_delete_scan(mirror_id: str, scan_id: str):
        store = plane_mirror_store()
        mirror = _load_plane_mirror(store, mirror_id)
        mirror["scans"] = [scan for scan in mirror.get("scans", []) if scan["id"] != scan_id]
        store.save(mirror)
        return redirect(url_for("plane_mirror_detail", mirror_id=mirror_id, message="Scan deleted."))

    @app.post("/plane-mirrors/<mirror_id>/delete")
    def plane_mirror_delete(mirror_id: str):
        mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
        plane_mirror_store().delete(mirror_id)
        return redirect(url_for("plane_mirror_index", message=f"Deleted plane mirror '{mirror['name']}'."))

    @app.post("/_preview/plane-mirror-stack")
    def plane_mirror_stack_preview():
        try:
            return jsonify({"ok": True, "image": stack_schematic_data_uri(_plane_mirror_stack(request.form))})
        except (KeyError, TypeError, ValueError) as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.post("/_compute/plane-mirror")
    def plane_mirror_compute():
        try:
            mirror_id = str(request.form.get("mirror_id", "")).strip()
            if mirror_id:
                mirror = _load_plane_mirror(plane_mirror_store(), mirror_id)
                from werkzeug.datastructures import MultiDict
                form = MultiDict()
                for key, value in mirror.get("stack_form", {}).items():
                    for item in value if isinstance(value, list) else [value]:
                        form.add(key, item)
                for key in request.form.keys():
                    for value in request.form.getlist(key):
                        form.add(key, value)
                result = _plane_mirror_result(form)
            else:
                result = _plane_mirror_result(request.form)
            return jsonify({"ok": True, **result})
        except (KeyError, TypeError, ValueError) as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.post("/plane-mirror/csv")
    def plane_mirror_download():
        try:
            result = _plane_mirror_result(request.form)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        return Response(
            plane_mirror_csv(result),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=plane_mirror_reflectivity.csv"},
        )

    @app.get("/_data/<path:filename>")
    def data_file(filename: str):
        return send_from_directory(app.config["GRAx_DATA_DIR"], filename)

    return app


def main() -> None:
    """Run the local development server."""

    parser = argparse.ArgumentParser(prog="grax-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable the Flask debugger and auto-reloader (local hosts only; restarts stop running jobs).",
    )
    args = parser.parse_args()
    if args.debug and not (args.host == "localhost" or _is_loopback_address(args.host)):
        parser.error("--debug exposes an interactive debugger and is only allowed with a local --host.")
    app = create_app()
    _maybe_open_browser(args.host, args.port, reloader=args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug)


def _maybe_open_browser(host: str, port: int, *, reloader: bool = False) -> None:
    """Open the local web UI in the default browser when appropriate.

    Args:
        host: Flask bind host.
        port: Flask bind port.
        reloader: Whether the server runs under the Werkzeug reloader, which
            starts the app twice; the browser then opens from the child only.
    """

    if reloader and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{int(port)}"
    timer = threading.Timer(1.0, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()
