"""Flask application for the local grax web MVP."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from .persistence import GratingStore, build_grating_from_spec, load_material_catalog
from .runs import RunStore


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
            "The web app requires Flask. Install it with: python -m pip install -e '.[web]'"
        ) from error

    app = Flask(__name__)
    app.config["GRAx_DATA_DIR"] = Path(data_dir or Path.cwd() / ".grax-web").resolve()
    catalog = load_material_catalog()

    def store() -> GratingStore:
        return GratingStore(app.config["GRAx_DATA_DIR"] / "saved_gratings")

    def run_store() -> RunStore:
        return RunStore(app.config["GRAx_DATA_DIR"] / "runs")

    def preview_root() -> Path:
        return app.config["GRAx_DATA_DIR"] / "previews" / "live"

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            gratings=store().list(),
            runs=_list_plot_runs(app.config["GRAx_DATA_DIR"] / "runs"),
            results_dir=app.config["GRAx_DATA_DIR"] / "runs",
            plots_dir=app.config["GRAx_DATA_DIR"] / "plots",
        )

    @app.get("/gratings/new")
    def new_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            catalog=catalog,
            form_data=_default_form_values(),
        )
        return render_template(
            "grating_form.html",
            materials=sorted(catalog),
            defaults=_default_form_values(),
            action_url=url_for("create_grating"),
            submit_label="Save grating",
            preview=preview,
        )

    @app.post("/gratings")
    def create_grating():
        spec = _spec_from_form(request.form)
        saved = store().save(spec)
        return redirect(url_for("grating_detail", grating_id=saved["id"]))

    @app.get("/gratings/<grating_id>")
    def grating_detail(grating_id: str):
        grating_store = store()
        spec = grating_store.load(grating_id)
        grating = build_grating_from_spec(spec, catalog)
        preview_path = app.config["GRAx_DATA_DIR"] / "previews" / f"{grating_id}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        grating.plot_profile(preview_path)
        return render_template(
            "grating_detail.html",
            grating=spec,
            preview_url=url_for("data_file", filename=f"previews/{grating_id}.png"),
            materials=sorted(catalog),
        )

    @app.get("/gratings/<grating_id>/edit")
    def edit_grating(grating_id: str):
        spec = store().load(grating_id)
        defaults = _form_values_from_spec(spec)
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            catalog=catalog,
            form_data=defaults,
        )
        return render_template(
            "grating_form.html",
            materials=sorted(catalog),
            defaults=defaults,
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
        grating_store.save(spec)
        return redirect(url_for("grating_detail", grating_id=grating_id))

    @app.post("/gratings/<grating_id>/runs")
    def create_run(grating_id: str):
        spec = store().load(grating_id)
        grating = build_grating_from_spec(spec, catalog)
        run = _run_sweep(
            data_dir=app.config["GRAx_DATA_DIR"],
            grating_id=grating_id,
            grating_name=str(spec["name"]),
            grating=grating,
            form=request.form,
        )
        return redirect(url_for("run_detail", run_id=run["id"]))

    @app.post("/_preview/grating")
    def preview_grating():
        preview = _build_grating_preview(
            data_dir=app.config["GRAx_DATA_DIR"],
            catalog=catalog,
            form_data=request.form,
        )
        return jsonify(preview)

    @app.get("/runs/manage")
    def manage_runs():
        return render_template(
            "run_manage.html",
            runs=_list_plot_runs(app.config["GRAx_DATA_DIR"] / "runs"),
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
            if field_name in request.form:
                store.rename(run_id, str(request.form.get(field_name, "")))
        return redirect(url_for("manage_runs"))

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        run_dir = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            abort(404)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["results_dir"] = run_dir
        return render_template("run_detail.html", run=manifest)

    @app.get("/plots")
    @app.get("/runs/compare")
    def plot_index():
        runs = _list_plot_runs(app.config["GRAx_DATA_DIR"] / "runs")
        return render_template(
            "plot_form.html",
            runs=runs,
            run_options=runs,
            results_dir=app.config["GRAx_DATA_DIR"] / "runs",
            preview=None,
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
        plot = _build_combined_plot(
            data_dir=app.config["GRAx_DATA_DIR"],
            run_ids=run_ids,
            order_selection=order_selection,
            title=str(request.form.get("title", "")).strip() or "Combined plot",
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
        return render_template("plot_detail.html", plot=manifest)

    @app.get("/_data/<path:filename>")
    def data_file(filename: str):
        return send_from_directory(app.config["GRAx_DATA_DIR"], filename)

    return app


def main() -> None:
    """Run the local development server."""
    app = create_app()
    app.run(host="127.0.0.1", port=5050, debug=True)


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
        "stack_type": "single_layer",
        "layer_material": "Pt",
        "layer_thickness_nm": "28.77",
        "material_a": "Cr",
        "material_b": "C",
        "d_period_nm": "6.5",
        "gamma": "0.45",
        "n_bilayers": "40",
        "top_material": "C",
        "top_cap_material": "",
        "top_cap_thickness_nm": "0.0",
    }


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
        values[key] = "" if value is None else str(value)
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
    catalog: dict[str, Any],
    form_data: Any,
) -> dict[str, Any]:
    """Build a live grating preview payload from form data."""
    try:
        spec = _spec_from_form(form_data)
        grating = build_grating_from_spec(spec, catalog)
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
    top_cap = str(form.get("top_cap_material", "")).strip() or None
    top_cap_thickness = float(form.get("top_cap_thickness_nm", 0.0) or 0.0)
    if stack_type == "multilayer":
        return {
            "type": "multilayer",
            "substrate_material": str(form["substrate_material"]),
            "material_a": str(form["material_a"]),
            "material_b": str(form["material_b"]),
            "d_period_nm": float(form["d_period_nm"]),
            "gamma": float(form["gamma"]),
            "n_bilayers": int(float(form["n_bilayers"])),
            "top_material": str(form["top_material"]),
            "top_cap_material": top_cap,
            "top_cap_thickness_nm": top_cap_thickness,
        }
    return {
        "type": "single_layer",
        "substrate_material": str(form["substrate_material"]),
        "layer_material": str(form["layer_material"]),
        "layer_thickness_nm": float(form["layer_thickness_nm"]),
        "top_cap_material": top_cap,
        "top_cap_thickness_nm": top_cap_thickness,
    }


def _run_sweep(
    *,
    data_dir: Path,
    grating_id: str,
    grating_name: str,
    grating: Any,
    form: Any,
) -> dict[str, Any]:
    """Run one supported sweep and persist artifacts."""
    from grax import parameter_sweep, simulation

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
    manifest: dict[str, Any] = {
        "id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": workflow,
        "grating_id": grating_id,
        "grating_name": grating_name,
        "display_name": f"{grating_name} · {workflow}",
        "status": "ok",
        "artifacts": [],
    }

    if workflow == "parameter_study":
        run_x_resolution_nm = float(form.get("run_x_resolution_nm") or grating.x_resolution_nm)
        run_z_resolution_nm = float(form.get("run_z_resolution_nm") or grating.z_resolution_nm)
        result = parameter_sweep.run_parameter_study(
            grating=grating,
            energies_ev=energies,
            grazing_angle_deg=float(form["grazing_angle_deg"]),
            diffraction_order=diffraction_order,
            fourier_orders_values=[fourier_orders],
            x_resolution_values=[run_x_resolution_nm],
            z_resolution_values=[run_z_resolution_nm],
            output_dir=run_dir,
            save_csv=True,
            show_progress=False,
        )
        plot_path = run_dir / "parameter_study.png"
        parameter_sweep.plot_parameter_study(result, output_filename=plot_path)
        manifest["artifacts"].append("parameter_study.png")
    else:
        cases = _cases_for_workflow(
            simulation=simulation,
            workflow=workflow,
            grating=grating,
            energies=energies,
            form=form,
            diffraction_order=diffraction_order,
            fourier_orders=fourier_orders,
        )
        runner = simulation.BatchSimulationRunner(
            default_diffraction_order=diffraction_order,
            default_fourier_orders=fourier_orders,
            max_workers=1,
            show_progress=False,
            on_error="continue",
            checkpoint_dir=run_dir / "checkpoints",
        )
        results = list(
            runner.run_cases(
                cases,
                metadata={
                    "workflow": workflow,
                    "grating_id": grating_id,
                    "grating_name": grating_name,
                },
            )
        )
        summary_path = run_dir / "summary.csv"
        _write_summary_csv(results, summary_path)
        all_orders_path = run_dir / "all_orders.csv"
        simulation.write_all_orders_csv(results, all_orders_path)
        plot_path = run_dir / "selected_efficiency.png"
        simulation.plot_order_subset(
            results,
            plot_path,
            diffraction_orders=[diffraction_order],
            title=f"{grating_name} {workflow}",
        )
        manifest["artifacts"].extend(["summary.csv", "all_orders.csv", "selected_efficiency.png"])
        manifest["cases"] = [
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

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


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

    if workflow == "fixed_angle":
        cases = list(
            simulation.fixed_angle_cases(
                grating=grating,
                energies_ev=energies,
                grazing_angle_deg=float(form["grazing_angle_deg"]),
            )
        )
    elif workflow == "monochromator":
        cases = list(
            simulation.monochromator_cases(
                grating=grating,
                energies_ev=energies,
                diffraction_order=diffraction_order,
                cff=float(form.get("cff", 2.25) or 2.25),
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


def _list_plot_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Return run manifests augmented with available plot orders."""
    runs = []
    for run in _list_runs(run_dir):
        run_id = str(run["id"])
        run_dir_for_id = _run_result_location(run_dir.parent, run_id)
        runs.append(
            {
                **run,
                "results_dir": run_dir_for_id,
                "available_orders": _available_orders(run_dir_for_id),
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
            "preview_id": None,
            "preview_url": None,
            "selected_runs": [],
        }

    order_selection: dict[str, list[int]] = {}
    selected_runs: list[dict[str, Any]] = []
    for run_id in run_ids:
        orders = [int(value) for value in form_data.getlist(f"orders_{run_id}") if str(value).strip() != ""]
        if not orders:
            return {
                "ok": False,
                "error": f"Select at least one diffraction order for run {run_id}.",
                "preview_id": None,
                "preview_url": None,
                "selected_runs": [],
            }
        order_selection[run_id] = orders
        run_manifest = RunStore(data_dir / "runs").load(run_id)
        selected_runs.append(
            {
                "id": run_id,
                "name": str(run_manifest.get("display_name") or run_manifest.get("grating_name", run_id)),
                "orders": orders,
            }
        )

    preview_id = uuid4().hex
    preview_path = data_dir / "previews" / "live" / f"{preview_id}.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_combined_plot_image(
            data_dir=data_dir,
            run_ids=run_ids,
            order_selection=order_selection,
            title=str(form_data.get("title", "")).strip() or "Combined plot",
            output_path=preview_path,
        )
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "preview_id": None,
            "preview_url": None,
            "selected_runs": [],
        }

    return {
        "ok": True,
        "error": "",
        "preview_id": preview_id,
        "preview_url": f"/_data/previews/live/{preview_id}.png",
        "selected_runs": selected_runs,
    }


def _available_orders(run_dir: Path) -> list[int]:
    """Return positive diffraction orders available for one run."""
    all_orders_path = run_dir / "all_orders.csv"
    if not all_orders_path.exists():
        return []
    orders: set[int] = set()
    with all_orders_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                orders.add(abs(int(float(row["order"]))))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(orders)


def _build_combined_plot(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    title: str,
) -> dict[str, Any]:
    """Build and save a combined plot for several runs and selected orders."""
    plot_root = data_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_id = _plot_id()
    plot_dir = plot_root / plot_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / "combined.png"
    run_summaries = _render_combined_plot_image(
        data_dir=data_dir,
        run_ids=run_ids,
        order_selection=order_selection,
        title=title,
        output_path=plot_path,
    )

    manifest = {
        "id": plot_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "selected_runs": run_summaries,
        "plot_path": "combined.png",
    }
    with (plot_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


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
    if not all_orders_path.exists():
        return None

    requested_order = 0 if int(order) == 0 else -abs(int(order))
    rows: list[dict[str, float]] = []
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


def _list_directory_entries(directory: Path) -> list[dict[str, Any]]:
    """Return browseable directory entries for the export dialog."""
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.iterdir(), key=lambda candidate: (not candidate.is_dir(), candidate.name.lower())):
        entries.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "is_dir": path.is_dir(),
            }
        )
    return entries


if __name__ == "__main__":
    main()
