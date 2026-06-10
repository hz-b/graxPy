"""Flask application for the local grax web MVP."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

from .persistence import GratingStore, build_grating_from_spec, load_material_catalog


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

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            gratings=store().list(),
            runs=_list_runs(app.config["GRAx_DATA_DIR"] / "runs"),
        )

    @app.get("/gratings/new")
    def new_grating():
        return render_template(
            "grating_form.html",
            materials=sorted(catalog),
            defaults=_default_form_values(),
            action_url=url_for("create_grating"),
            submit_label="Save grating",
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
        return render_template(
            "grating_form.html",
            materials=sorted(catalog),
            defaults=_form_values_from_spec(spec),
            action_url=url_for("update_grating", grating_id=grating_id),
            submit_label="Update grating",
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

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str):
        run_dir = app.config["GRAx_DATA_DIR"] / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            abort(404)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return render_template("run_detail.html", run=manifest)

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
        "status": "ok",
        "artifacts": [],
    }

    if workflow == "parameter_study":
        result = parameter_sweep.run_parameter_study(
            grating=grating,
            energies_ev=energies,
            grazing_angle_deg=float(form["grazing_angle_deg"]),
            diffraction_order=diffraction_order,
            fourier_orders_values=[fourier_orders],
            x_resolution_values=[float(grating.x_resolution_nm)],
            z_resolution_values=[float(grating.z_resolution_nm)],
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
                final_fourier_orders=fourier_orders,
            )
        )
    else:
        raise ValueError("Unsupported workflow.")
    for index, case in enumerate(cases):
        case["case_id"] = f"{workflow}-{index:08d}"
        case["fourier_orders"] = fourier_orders
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
    if not run_dir.exists():
        return []
    runs = []
    for path in run_dir.glob("*/manifest.json"):
        with path.open("r", encoding="utf-8") as handle:
            runs.append(json.load(handle))
    return sorted(runs, key=lambda run: str(run.get("created_at", "")), reverse=True)


def _run_id(workflow: str, grating_id: str) -> str:
    """Return a filesystem-safe run ID."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{grating_id}-{workflow}".replace("_", "-")


if __name__ == "__main__":
    main()
