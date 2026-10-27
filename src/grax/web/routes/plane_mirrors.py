"""Plane-mirror library, scan and run routes."""

from __future__ import annotations

import io
import json
from datetime import (
    datetime,
)
from typing import (
    Any,
)
from uuid import (
    uuid4,
)

import matplotlib.pyplot as plt
from flask import (
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from grax.materials import (
    available_material_symbols,
    material_density_catalog,
)

from ..common import (
    _split_variation_values,
)
from ..context import WebContext
from ..grating_forms import (
    _custom_layer_rows,
    _material_density_placeholders,
)
from ..jobs import is_orphaned, request_stop
from ..persistence import (
    PlaneMirrorStore,
)
from ..plane_mirror import (
    clean_label,
    clean_name,
    parse_plane_mirror_options,
    plane_mirror_csv,
    plane_mirror_summary,
    split_plane_mirror_form,
    stack_schematic_data_uri,
    unique_scan_id,
)
from ..plane_mirror_runs import (
    MAX_TOTAL_SAMPLES as MAX_PLANE_MIRROR_RUN_SAMPLES,
)
from ..plane_mirror_runs import (
    expand_variations as expand_plane_mirror_variations,
)
from ..plane_mirror_runs import (
    start_worker as start_plane_mirror_worker,
)
from ..plane_mirror_runs import (
    total_sample_count as plane_mirror_sample_count,
)
from ..plane_mirror_views import (
    _load_plane_mirror,
    _plane_mirror_page_defaults,
    _plane_mirror_result,
    _plane_mirror_scan_parameter_details,
    _plane_mirror_stack,
    _plane_mirror_variation_catalog,
)
from ..plot_builder import (
    _plotly_bundle_text,
    get_plotlyjs,
)
from ..run_files import (
    _load_run_manifest,
)
from ..run_state import _is_run_active


def register(app: Any, ctx: WebContext) -> None:
    """Register the plane mirrors routes on ``app``."""
    active_data_dir = ctx.active_data_dir
    load_user_settings = ctx.load_user_settings
    run_store = ctx.run_store

    def plane_mirror_store() -> PlaneMirrorStore:
        return PlaneMirrorStore(app.config["GRAx_DATA_DIR"] / "saved_plane_mirrors")

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

    def _mark_orphaned_run(run_id: str) -> dict[str, Any]:
        """Save a run the app stopped in the middle of as aborted, and return the change."""
        change = {"status": "aborted", "error_text": "The app stopped before this run finished."}
        run_store().update(run_id, lambda saved: saved.update(change))
        return change

    def _normalize_mirror_run_status(manifest: dict[str, Any]) -> None:
        # Runs saved before the job manager recorded failures as "error".
        if manifest.get("status") == "error":
            manifest["status"] = "failed"

    @app.get("/plane-mirror-runs")
    def plane_mirror_runs():
        manifests = [run for run in run_store().list() if run.get("mirror_id")]
        for manifest in manifests:
            if is_orphaned(app, manifest["id"], manifest.get("status")):
                manifest.update(_mark_orphaned_run(manifest["id"]))
            _normalize_mirror_run_status(manifest)
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
        if is_orphaned(app, run_id, manifest.get("status")):
            manifest.update(_mark_orphaned_run(run_id))
        _normalize_mirror_run_status(manifest)
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
        _normalize_mirror_run_status(manifest)
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
        if request_stop(app, run_id):
            run_store().update(
                run_id,
                lambda saved: saved.update(status="aborting", updated_at=datetime.now().isoformat(timespec="seconds")),
            )
        return redirect(url_for("plane_mirror_run_detail", run_id=run_id))

    @app.post("/plane-mirror-runs/<run_id>/resume")
    def plane_mirror_run_resume(run_id: str):
        manifest = _load_run_manifest(active_data_dir(), run_id)
        if manifest is None or not manifest.get("mirror_id"):
            abort(404)
        if _is_run_active(app, run_id):
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
        if _is_run_active(app, run_id):
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
