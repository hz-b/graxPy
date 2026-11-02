"""Bragg-grating library: define a grating, preview its electrodes, run scans."""

from __future__ import annotations

from typing import Any

from flask import abort, jsonify, redirect, render_template, request, url_for

from ..bragg_gratings import (
    DEFAULTS as GRATING_DEFAULTS,
)
from ..bragg_gratings import (
    BraggGratingStore,
    coating_from_form,
    coating_rows,
    coating_summary,
    crystal_choices,
    default_solver,
    geometry_from_form,
    grating_summary,
    has_multilayer,
    idt_grating_from_spec,
    spec_from_form,
    surface_grating_from_spec,
)
from ..bragg_preview import coating_cross_section_data_uri, idt_preview_data_uri
from ..bragg_simulations import (
    DEFAULTS as SCAN_DEFAULTS,
)
from ..bragg_simulations import (
    MAX_ENERGY_EV,
    MIN_ENERGY_EV,
)
from ..bragg_simulations import (
    SCAN_TYPES,
    SOLVERS,
    abort_scan,
    angle_axis_label,
    default_theta_window,
    delete_scan,
    list_scans,
    load_scan,
    parse_params,
    recent_scans,
    run_status,
    series,
    start_scan,
)
from ..context import WebContext

PREVIEW_AMPLITUDE_NM = 0.12


def register(app: Any, ctx: WebContext) -> None:
    """Register the Bragg-grating routes on ``app``."""

    def grating_store() -> BraggGratingStore:
        return BraggGratingStore(ctx.active_data_dir() / "bragg_gratings")

    def load_grating(grating_id: str) -> dict[str, Any]:
        try:
            return grating_store().load(grating_id)
        except (OSError, ValueError):
            abort(404)

    def preview_for(spec: dict[str, Any], amplitude_nm: float = PREVIEW_AMPLITUDE_NM) -> str:
        return idt_preview_data_uri(idt_grating_from_spec(spec), amplitude_nm)

    def cross_section_for(spec: dict[str, Any], amplitude_nm: float = PREVIEW_AMPLITUDE_NM) -> str:
        if not spec.get("coated"):
            return ""
        resolution = {key: SCAN_DEFAULTS[key] for key in ("x_resolution_nm", "z_resolution_nm")}
        return coating_cross_section_data_uri(surface_grating_from_spec(spec, amplitude_nm, **resolution))

    def form_page(*, grating: dict[str, Any] | None, values: Any, error: str = ""):
        from ...materials import available_material_symbols, material_density_catalog

        try:
            preview = preview_for(geometry_from_form(values))
        except ValueError:
            preview = ""
        try:
            spec = values
            if hasattr(values, "getlist"):
                spec = {"crystal": values.get("crystal", "LiNbO3"), **geometry_from_form(values), **coating_from_form(values)}
            cross_section = cross_section_for(spec)
        except (ValueError, KeyError):
            cross_section = ""
        coated = values.get("coated") not in (None, False, "", "0")
        return render_template(
            "bragg_grating_form.html",
            grating=grating,
            values=values,
            coated=coated,
            custom_layers=coating_rows(values),
            hide_layer_roughness=True,
            layer_planarization=True,
            materials=available_material_symbols(),
            material_density_map=dict(material_density_catalog()),
            crystals=crystal_choices(),
            error=error,
            preview=preview,
            cross_section=cross_section,
            preview_amplitude=PREVIEW_AMPLITUDE_NM,
        )

    @app.get("/bragg-gratings")
    def bragg_grating_manage():
        gratings = grating_store().list()
        for grating in gratings:
            grating["summary"] = grating_summary(grating)
        return render_template(
            "bragg_gratings_manage.html",
            gratings=gratings,
            message=str(request.args.get("message", "")).strip(),
        )

    @app.get("/bragg-gratings/new")
    def bragg_grating_new():
        return form_page(grating=None, values=GRATING_DEFAULTS)

    @app.post("/bragg-gratings")
    def bragg_grating_create():
        try:
            spec = spec_from_form(request.form)
        except ValueError as error:
            return form_page(grating=None, values=request.form, error=str(error)), 400
        saved = grating_store().save(spec)
        return redirect(url_for("bragg_grating_detail", grating_id=saved["id"]))

    @app.get("/bragg-gratings/<grating_id>")
    def bragg_grating_detail(grating_id: str):
        grating = load_grating(grating_id)
        return render_template(
            "bragg_grating_detail.html",
            grating=grating,
            summary=grating_summary(grating),
            preview=preview_for(grating),
            cross_section=cross_section_for(grating),
            preview_amplitude=PREVIEW_AMPLITUDE_NM,
            values={**SCAN_DEFAULTS, **default_theta_window(grating, SCAN_DEFAULTS["energy_ev"])},
            solver=default_solver(grating),
            solvers=SOLVERS,
            has_multilayer=has_multilayer(grating),
            scan_types=SCAN_TYPES,
            recent=recent_scans(ctx.active_data_dir(), grating_id),
            message=str(request.args.get("message", "")).strip(),
        )

    @app.get("/bragg-gratings/<grating_id>/theta-window")
    def bragg_grating_theta_window(grating_id: str):
        grating = load_grating(grating_id)
        try:
            energy = float(str(request.args.get("energy_ev", "")).replace(",", "."))
        except ValueError:
            return jsonify(ok=False, error="Photon energy must be a number."), 400
        if not MIN_ENERGY_EV <= energy <= MAX_ENERGY_EV:
            return jsonify(ok=False, error=f"Photon energy must be between {MIN_ENERGY_EV:g} and {MAX_ENERGY_EV:g} eV."), 400
        return jsonify(ok=True, energy_ev=energy, **default_theta_window(grating, energy))

    @app.get("/bragg-gratings/<grating_id>/edit")
    def bragg_grating_edit(grating_id: str):
        grating = load_grating(grating_id)
        return form_page(grating=grating, values=grating)

    @app.post("/bragg-gratings/<grating_id>")
    def bragg_grating_update(grating_id: str):
        grating = load_grating(grating_id)
        try:
            spec = spec_from_form(request.form)
        except ValueError as error:
            return form_page(grating=grating, values=request.form, error=str(error)), 400
        grating_store().save(spec, grating_id)
        return redirect(
            url_for("bragg_grating_detail", grating_id=grating_id, message="Changes saved.")
        )

    @app.post("/bragg-gratings/<grating_id>/delete")
    def bragg_grating_delete(grating_id: str):
        grating = load_grating(grating_id)
        grating_store().delete(grating_id)
        return redirect(
            url_for("bragg_grating_manage", message=f"Deleted Bragg grating '{grating['name']}'.")
        )

    @app.post("/bragg-gratings/<grating_id>/run")
    def bragg_grating_run(grating_id: str):
        grating = load_grating(grating_id)
        try:
            params = parse_params(request.form, grating)
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        run_id = start_scan(app, ctx.active_data_dir(), params)
        return jsonify(
            {"ok": True, "run_id": run_id, "url": url_for("bragg_run_detail", run_id=run_id)}
        )

    @app.get("/bragg-runs")
    def bragg_run_manage():
        data_dir = ctx.active_data_dir()
        existing = {grating["id"] for grating in grating_store().list()}
        runs = list_scans(data_dir)
        for run in runs:
            run["grating_exists"] = run["grating_id"] in existing
        return render_template(
            "bragg_runs_manage.html",
            runs=runs,
            message=str(request.args.get("message", "")).strip(),
        )

    @app.get("/bragg-runs/<run_id>")
    def bragg_run_detail(run_id: str):
        from ..plot_builder import _plotly_bundle_text, get_plotlyjs

        data_dir = ctx.active_data_dir()
        saved = load_scan(data_dir, run_id)
        status = run_status(app, data_dir, run_id)
        if saved is None or status is None:
            abort(404)
        params = saved["params"]
        grating_exists = any(grating["id"] == params["grating_id"] for grating in grating_store().list())
        return render_template(
            "bragg_run_detail.html",
            run_id=run_id,
            params=params,
            scan_label=SCAN_TYPES.get(params["scan_type"], {}).get("label", params["scan_type"]),
            solver_label=SOLVERS.get(params.get("solver", "takagi"), ""),
            series=series(params),
            angle_label=angle_axis_label(params),
            coating=coating_summary(params["grating"]),
            created_at=saved.get("created_at", ""),
            grating_exists=grating_exists,
            initial_status=status,
            status_url=url_for("bragg_scan_status", run_id=run_id),
            memory_url=url_for("system_memory"),
            plotly_bundle=_plotly_bundle_text() if get_plotlyjs is not None else None,
        )

    @app.post("/bragg-runs/<run_id>/abort")
    def bragg_run_abort(run_id: str):
        abort_scan(app, run_id)
        return redirect(url_for("bragg_run_detail", run_id=run_id))

    @app.post("/bragg-runs/<run_id>/delete")
    def bragg_run_delete(run_id: str):
        delete_scan(app, ctx.active_data_dir(), run_id)
        return redirect(url_for("bragg_run_manage", message="Deleted the saved scan."))

    @app.get("/bragg-scans/<run_id>/status")
    def bragg_scan_status(run_id: str):
        payload = run_status(app, ctx.active_data_dir(), run_id)
        if payload is None:
            abort(404)
        return jsonify(payload)

    @app.post("/_preview/bragg-grating")
    def bragg_grating_preview():
        try:
            geometry = geometry_from_form(request.form)
            coating = coating_from_form(request.form)
            crystal = str(request.form.get("crystal", GRATING_DEFAULTS["crystal"]))
            if crystal not in crystal_choices():
                raise ValueError(f"Choose a crystal from {', '.join(crystal_choices())}.")
            amplitude = float(str(request.form.get("preview_amplitude_nm", PREVIEW_AMPLITUDE_NM)))
            if not 0.0 <= amplitude <= 5.0:
                raise ValueError("The preview amplitude must lie between 0 and 5 nm.")
            cross_section = cross_section_for({"crystal": crystal, **geometry, **coating}, amplitude)
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        from ...bragg_grating import IdtGrating

        return jsonify(
            {
                "ok": True,
                "image": idt_preview_data_uri(IdtGrating(**geometry), amplitude),
                "cross_section": cross_section,
            }
        )
