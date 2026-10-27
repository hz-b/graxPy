"""Listings of saved runs and plots, and the summaries shown for them on pages."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import parameter_scans
from .common import (
    _polarization_display_label,
)
from .persistence import build_grating_from_spec
from .run_files import (
    _available_orders,
    _checkpoint_counts,
    _list_runs,
    _load_checkpoint_results,
    _preferred_run_plot_path,
    _run_completed_points,
    _run_plot_url,
    _run_result_location,
)
from .run_jobs import (
    _run_status_payload,
)


def _run_summary_label(run: dict[str, Any]) -> str:
    """Return the primary saved-run descriptor used in plot selection."""

    if _is_plane_mirror_run(run):
        return f"mirror: {run.get('mirror_name', 'Plane mirror')}"
    return (
        f"name: {run.get('grating_name', 'Run')} · "
        f"date/time: {run.get('created_at', '')} · "
        f"grating type: {_run_grating_type(run)}"
    )


def _run_summary_meta(run: dict[str, Any]) -> str:
    """Return a secondary metadata line for one saved run."""

    comment = str(run.get("comment", "")).strip() or "—"
    if _is_plane_mirror_run(run):
        return f"scan type: {run.get('scan_details', [['Scan', 'reflectivity']])[0][1]} · comment: {comment}"
    return (
        f"sweep type: {run.get('workflow', 'run')} · "
        f"comment: {comment}"
    )


def _run_grating_type(run: dict[str, Any]) -> str:
    """Return the grating type recorded for one run."""

    grating_spec = run.get("grating_spec")
    if isinstance(grating_spec, dict):
        value = str(grating_spec.get("grating_type", "")).strip()
        if value:
            return value
    value = str(run.get("grating_type", "")).strip()
    return value or "unknown"


def _run_selector_text(run: dict[str, Any]) -> str:
    """Return the full run text shown in plot-selection UIs."""

    return f"{_run_summary_label(run)} · {_run_summary_meta(run)}"


def _is_plane_mirror_run(run: dict[str, Any]) -> bool:
    """Return whether a saved run contains plane-mirror reflectivity results."""

    return bool(run.get("mirror_id") or run.get("workflow") == "plane_mirror")


def _plane_mirror_plot_definition(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    """Adapt saved mirror case results to the scan-and-loop controls used by plot selection."""

    try:
        results = json.loads((run_dir / "mirror_results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cases = manifest.get("cases") or []
    if not isinstance(results, list) or not results or not isinstance(cases, list):
        return None

    axes: list[dict[str, Any]] = []
    covered: set[str] = set()
    for axis in manifest.get("variation_axes") or []:
        parameter = str(axis.get("parameter") or "") if isinstance(axis, dict) else ""
        if not parameter:
            continue
        values = list(dict.fromkeys(
            str((result.get("parameters") or {}).get(parameter))
            for result in results
            if parameter in (result.get("parameters") or {})
        ))
        if values:
            axes.append({"id": parameter, "label": parameter.replace("_", " "),
                         "parameters": [parameter], "values": values})
            covered.add(parameter)

    for index, group in enumerate(manifest.get("variation_coupled") or []):
        if not isinstance(group, dict) or not group:
            continue
        parameters = list(group)
        rows = []
        for result in results:
            values = result.get("parameters") or {}
            if all(parameter in values for parameter in parameters):
                row = {parameter: str(values[parameter]) for parameter in parameters}
                if row not in rows:
                    rows.append(row)
        if rows:
            axis_id = f"coupled-{index}"
            axes.append({"id": axis_id, "label": f"Coupled: {', '.join(parameters)}",
                         "parameters": [axis_id], "values": rows,
                         "row_parameters": parameters})
            covered.update(parameters)

    all_parameters = sorted({
        str(parameter)
        for result in results
        for parameter in (result.get("parameters") or {})
    } - covered)
    for parameter in all_parameters:
        values = list(dict.fromkeys(
            str((result.get("parameters") or {}).get(parameter))
            for result in results
            if parameter in (result.get("parameters") or {})
        ))
        if len(values) > 1:
            axes.append({"id": parameter, "label": parameter.replace("_", " "),
                         "parameters": [parameter], "values": values})

    scans = []
    for index, result in enumerate(results):
        values = {str(key): str(value) for key, value in (result.get("parameters") or {}).items()}
        scan_parameters = dict(values)
        labels = [f"{key}={value}" for key, value in values.items()]
        for axis in axes:
            if axis.get("row_parameters"):
                row = {key: values[key] for key in axis["row_parameters"] if key in values}
                if row:
                    scan_parameters[axis["id"]] = row
        scans.append({"id": f"mirror-case-{index:06d}",
                      "label": "; ".join(labels) or "Base settings",
                      "parameters": scan_parameters})

    mode = str(results[0].get("mode") or "energy_sweep")
    return {
        "axes": axes,
        "scans": scans,
        "energy_values": results[0].get("x", []),
        "sample_axis": {"id": "grazing_angle_deg" if mode == "angle_sweep" else "energy_ev",
                        "label": results[0].get("x_label", "Photon Energy (eV)")},
        "mirror_results": results,
    }


def _list_plot_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Return run manifests augmented with available plot orders."""
    runs = []
    for run in _list_runs(run_dir):
        run_id = str(run["id"])
        run_dir_for_id = _run_result_location(run_dir.parent, run_id)
        checkpoint_completed_points, checkpoint_total_points = _checkpoint_counts(run_dir_for_id)
        state = str(run.get("status", "completed"))
        if state in {"queued", "running", "aborting"} and checkpoint_completed_points > 0:
            state = "aborted"
        elif state in {"paused", "interrupted"}:
            state = "aborted"
        enriched_run = {
            **run,
            "status": state,
        }
        custom_plot = _custom_grid_plot_definition(run) if not run.get("parameter_sweep") else None
        mirror_plot = _plane_mirror_plot_definition(run, run_dir_for_id) if _is_plane_mirror_run(run) else None
        plot_sweep = run.get("parameter_sweep") or custom_plot or mirror_plot
        runs.append(
            {
                **enriched_run,
                "results_dir": run_dir_for_id,
                "available_orders": [0] if _is_plane_mirror_run(run) else _available_orders(run_dir_for_id),
                "is_plane_mirror": _is_plane_mirror_run(run),
                "summary_label": _run_summary_label(enriched_run),
                "summary_meta": _run_summary_meta(enriched_run),
                "selector_text": _run_selector_text(enriched_run),
                "plot_scans": [{"id": scan["id"], "label": scan["label"]} for scan in (plot_sweep or {}).get("scans", [])],
                "plot_loops": _plot_loop_options(plot_sweep),
                "checkpoint_completed_points": checkpoint_completed_points,
                "checkpoint_total_points": checkpoint_total_points or int(run.get("total_points") or 0),
            }
        )
    return runs


def _plot_loop_options(sweep: dict[str, Any] | None) -> list[dict[str, str]]:
    """Offer the parameters recorded as varied scan axes."""
    if not sweep:
        return []
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for axis in sweep.get("axes", []):
        parameters = axis.get("parameters", [])
        for parameter in parameters:
            if parameter in seen:
                continue
            seen.add(parameter)
            label = axis["label"] if len(parameters) == 1 else parameter.replace("_", " ")
            options.append({"id": parameter, "label": label})
    return options


def _custom_grid_plot_definition(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Describe energy curves in a saved Custom grid without changing its manifest."""
    config = manifest.get("custom_grid") or (manifest.get("run_input") or {}).get("custom_grid_config")
    if not isinstance(config, dict):
        return None
    independent = config.get("independent") or []
    energy_axes = [axis for axis in independent if axis.get("parameter") == "energy_ev"]
    if len(energy_axes) != 1 or any(
        "energy_ev" in group.get("parameters", []) for group in config.get("coupled_groups") or []
    ):
        return None
    energy_values = sorted({float(value) for value in energy_axes[0].get("values", [])})
    if not energy_values:
        return None
    axes = []
    for axis in independent:
        if axis.get("parameter") != "energy_ev":
            axes.append({"id": axis["parameter"], "label": axis["parameter"].replace("_", " "),
                         "parameters": [axis["parameter"]], "values": axis["values"]})
    for index, group in enumerate(config.get("coupled_groups") or []):
        axes.append({"id": f"coupled-{index}", "label": group.get("name") or f"Coupled group {index + 1}",
                     "parameters": [f"coupled-{index}"], "values": group["rows"],
                     "row_parameters": group["parameters"]})
    scans = []
    case_map = {}
    scan_keys = {}
    original_axes = [
        ("energy_ev", axis["values"]) if axis.get("parameter") == "energy_ev" else (axis["parameter"], axis["values"])
        for axis in independent
    ] + [(f"coupled-{index}", group["rows"]) for index, group in enumerate(config.get("coupled_groups") or [])]
    for case_index, values in enumerate(itertools.product(*(axis_values for _, axis_values in original_axes))):
        selection = dict(zip((name for name, _ in original_axes), values, strict=True))
        key = tuple(json.dumps(selection[axis["id"]], sort_keys=True) for axis in axes)
        if key not in scan_keys:
            scan_id = f"custom-scan-{len(scans):06d}"
            scan_keys[key] = scan_id
            labels = []
            parameters = {}
            for axis in axes:
                value = selection[axis["id"]]
                parameters[axis["id"]] = value
                if axis.get("row_parameters"):
                    labels.extend(f"{name}={value[name]}" for name in axis["row_parameters"])
                else:
                    labels.append(f"{axis['label']}={value}")
            scans.append({"id": scan_id, "label": "; ".join(labels) or "Default settings",
                          "parameters": parameters})
        case_map[f"custom_grid-{case_index:08d}"] = (scan_keys[key], float(selection["energy_ev"]))
    return {"axes": axes, "scans": scans, "energy_values": energy_values, "case_map": case_map}


def _list_plots(plot_dir: Path) -> list[dict[str, Any]]:
    """Return saved combined plot manifests newest first."""
    if not plot_dir.exists():
        return []
    plots = []
    for path in plot_dir.glob("*/manifest.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                plot = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        plot["unavailable_run_count"] = sum(
            1
            for run in plot.get("selected_runs", [])
            if not (plot_dir.parent / "runs" / str(run.get("id", "")) / "manifest.json").exists()
        )
        plots.append(plot)
    return sorted(plots, key=lambda plot: str(plot.get("created_at", "")), reverse=True)


def _plot_references_for_runs(data_dir: Path, run_ids: list[str]) -> list[dict[str, Any]]:
    """Return saved plots that reference any requested run IDs."""

    wanted = {str(run_id) for run_id in run_ids}
    references: list[dict[str, Any]] = []
    for plot in _list_plots(data_dir / "plots"):
        selected = plot.get("selected_runs", [])
        matches = [run for run in selected if str(run.get("id", "")) in wanted]
        if matches:
            references.append(
                {
                    "id": plot.get("id", ""),
                    "title": plot.get("name") or plot.get("title", "Saved plot"),
                    "url": f"/plots/{plot.get('id', '')}",
                    "runs": matches,
                }
            )
    return references


def _is_grating_run(run: dict[str, Any]) -> bool:
    """Return whether a saved run manifest belongs to the grating workflow."""

    return bool(run.get("grating_id") or run.get("grating_spec") or run.get("grating_name"))


def _manage_run_entries(data_dir: Path, *, app: Any | None = None) -> list[dict[str, Any]]:
    """Build the derived view model used by Manage Runs."""

    entries: list[dict[str, Any]] = []
    for run in _list_plot_runs(data_dir / "runs"):
        run_id = str(run["id"])
        # The status payload knows which runs still have a live worker in this app.
        if app is not None and (run.get("parameter_sweep") or run["status"] in {"queued", "running", "aborting"}):
            run["status"] = _run_status_payload(app=app, data_dir=data_dir, run_id=run_id)["state"]
        input_data = run.get("run_input") if isinstance(run.get("run_input"), dict) else {}
        artifacts = [str(name) for name in run.get("artifacts", [])]
        plot_path = _preferred_run_plot_path(data_dir, run_id)
        material_stack_preview_url = _run_material_stack_preview_url(data_dir, run)
        run.update(
            {
                "detail_url": f"/runs/{run_id}",
                "plot_preview_url": _run_plot_url(
                    data_dir=data_dir,
                    run_id=run_id,
                    relative_path=plot_path,
                    token="",
                ),
                "material_stack_preview_url": material_stack_preview_url,
                "artifacts": artifacts,
                "artifact_links": [
                    {"name": name, "url": f"/_data/runs/{run_id}/{name}"}
                    for name in artifacts
                ],
                "related_plots": _plot_references_for_runs(data_dir, [run_id]),
                "parameter_summary": _run_input_summary(input_data),
                "parameter_details": _run_parameter_details(run),
                "scan_rows": _scan_run_rows(run, data_dir / "runs" / run_id),
                "progress_summary": (
                    f"{_run_completed_points(run, run.get('checkpoint_completed_points', 0))} / "
                    f"{run.get('checkpoint_total_points') or run.get('total_points', 0)} points"
                ),
            }
        )
        entries.append(run)
    return entries


def _run_material_stack_preview_url(data_dir: Path, run: dict[str, Any]) -> str | None:
    """Return a cached material-map image URL for a run's saved grating spec."""
    spec = run.get("grating_spec")
    if not isinstance(spec, dict):
        return None
    run_id = str(run.get("id", ""))
    if not run_id:
        return None
    relative_path = Path("previews") / "run-stacks" / f"{run_id}.png"
    preview_path = data_dir / relative_path
    if not preview_path.is_file():
        try:
            grating = build_grating_from_spec(spec)
            x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=3)
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(figsize=(8, 3.5))
            colors = grating._plot_material_colors(
                coating_stack=grating.resolved_stack(),
                material_labels=material_labels,
            )
            color_map = plt.matplotlib.colors.ListedColormap(colors)
            color_map.set_bad(color="#ffffff", alpha=1.0)
            masked_material_map = np.ma.masked_less(material_map, 0)
            axes.imshow(
                masked_material_map,
                origin="lower",
                aspect="auto",
                extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
                cmap=color_map,
                interpolation="nearest",
                vmin=0,
                vmax=max(len(material_labels) - 1, 1),
            )
            legend_handles = [
                plt.matplotlib.patches.Patch(color=color_map.colors[index], label=label)
                for index, label in enumerate(material_labels)
            ]
            axes.set_xlabel("x (nm)")
            axes.set_ylabel("z (nm)")
            axes.set_title("Material Stack")
            axes.legend(handles=legend_handles, loc="upper right")
            figure.tight_layout()
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(preview_path, dpi=150, bbox_inches="tight")
            plt.close(figure)
        except Exception:
            return None
    return f"/_data/{relative_path.as_posix()}"


def _scan_run_rows(manifest: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Annotate scan definitions with checkpoint-backed progress and images."""
    sweep = manifest.get("parameter_sweep")
    if not sweep:
        return []
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    for result in _load_checkpoint_results(run_dir):
        scan_id = result.case_data.get("scan_id")
        counts[scan_id] = counts.get(scan_id, 0) + 1
        failures[scan_id] = failures.get(scan_id, 0) + (result.status != "ok")
    return [
        {
            **scan,
            "completed": counts.get(scan["id"], 0),
            "failed": failures.get(scan["id"], 0),
            "total": len(parameter_scans.sample_values(sweep)),
            "plot_path": f"scans/{scan['id']}/efficiency.png"
            if (run_dir / "scans" / scan["id"] / "efficiency.png").exists()
            else None,
        }
        for scan in sweep["scans"]
    ]


def _run_input_summary(run_input: dict[str, Any]) -> str:
    """Return compact, stable run parameters for the management list."""

    fields = []
    if run_input.get("workflow") in {"angle_sweep", "fixed_angle"} and run_input.get("energy_ev") is not None:
        fields.append(f"energy {run_input['energy_ev']} eV")
        fields.append(f"grazing {run_input.get('grazing_angle_start_deg')}–{run_input.get('grazing_angle_stop_deg')}°")
    elif run_input.get("energy_start_ev") is not None and run_input.get("energy_stop_ev") is not None:
        fields.append(f"energy {run_input['energy_start_ev']}–{run_input['energy_stop_ev']} eV")
    if run_input.get("grazing_angle_deg") is not None:
        fields.append(f"grazing {run_input['grazing_angle_deg']}°")
    if run_input.get("diffraction_order") is not None:
        fields.append(f"order {run_input['diffraction_order']}")
    return " · ".join(fields) or "Parameters unavailable"


def _run_parameter_details(run: dict[str, Any]) -> list[tuple[str, str]]:
    """Present saved run inputs without requiring a metadata download."""

    source = dict(run.get("run_input") or {})
    source.setdefault("workflow", run.get("workflow"))
    source.setdefault("solver", run.get("solver"))
    source.setdefault("polarization", run.get("polarization"))
    if run.get("custom_grid") and not source.get("custom_grid_config"):
        source["custom_grid_config"] = run["custom_grid"]
    labels = {
        "workflow": "Workflow", "cff": "CFF", "fourier_orders": "Fourier orders",
        "solver": "Solver", "polarization": "Polarization",
        "diffraction_order": "Diffraction order", "grazing_angle_deg": "Grazing angle (deg)",
        "energy_start_ev": "Energy start (eV)", "energy_stop_ev": "Energy stop (eV)",
        "energy_points": "Energy points", "roughness_kind": "Roughness model",
        "max_workers_mode": "Worker mode", "max_workers": "Workers",
    }
    details = []
    for key, value in source.items():
        if value in (None, "") or key == "comment":
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key == "polarization":
            value = _polarization_display_label(value)
        details.append((labels.get(key, key.replace("_", " ").capitalize()), str(value)))
    return details


def _run_comparison_parameter_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten saved run inputs into aligned comparison-table rows."""

    field_labels = {
        "workflow": "Workflow",
        "cff": "CFF",
        "fourier_orders": "Fourier orders",
        "solver": "Solver",
        "polarization": "Polarization",
        "diffraction_order": "Diffraction order",
        "grazing_angle_deg": "Grazing angle (deg)",
        "energy_ev": "Energy (eV)",
        "energy_start_ev": "Energy start (eV)",
        "energy_stop_ev": "Energy stop (eV)",
        "energy_points": "Energy points",
        "angle_start_deg": "Angle start (deg)",
        "angle_stop_deg": "Angle stop (deg)",
        "angle_points": "Angle points",
        "roughness_kind": "Roughness model",
        "x_resolution_nm": "X resolution (nm)",
        "z_resolution_nm": "Z resolution (nm)",
    }
    ignored_grating_fields = {"id", "name", "created_at", "updated_at", "source_path", "profile_path"}
    flattened: list[dict[str, str]] = []

    def add_value(target: dict[str, str], label: str, value: Any) -> None:
        if value is None or (isinstance(value, str) and value == ""):
            return
        is_custom_grid = label == "Custom grid config" or label.startswith("Custom grid config ·")
        if is_custom_grid and isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                decoded = value
            if isinstance(decoded, (dict, list)):
                add_value(target, label, decoded)
                return
        if isinstance(value, dict):
            if not value:
                target[label] = "{}"
                return
            for key in sorted(value, key=str):
                if str(key) in ignored_grating_fields:
                    continue
                child_name = field_labels.get(str(key), str(key).replace("_", " ").capitalize())
                add_value(target, f"{label} · {child_name}" if label else child_name, value[key])
        elif isinstance(value, list):
            if value and any(isinstance(item, (dict, list)) for item in value):
                preview_items = value[:5] if is_custom_grid else value
                for index, item in enumerate(preview_items, start=1):
                    add_value(target, f"{label} [{index}]", item)
                if is_custom_grid and len(value) > len(preview_items):
                    target[f"{label} · additional entries"] = f"{len(value) - len(preview_items)} more"
            else:
                if is_custom_grid:
                    preview_items = value[:5]
                    preview = ", ".join(str(item) for item in preview_items)
                    if len(value) > len(preview_items):
                        preview += f" … ({len(value) - len(preview_items)} more)"
                    target[label] = preview or "[]"
                else:
                    target[label] = ", ".join(str(item) for item in value) if value else "[]"
        else:
            target[label] = str(value)

    for run in runs:
        values: dict[str, str] = {}
        source = dict(run.get("run_input") or {})
        source.setdefault("workflow", run.get("workflow"))
        source.setdefault("solver", run.get("solver"))
        source.setdefault("polarization", run.get("polarization"))
        source.setdefault("diffraction_order", run.get("diffraction_order"))
        if run.get("custom_grid") and not source.get("custom_grid_config"):
            source["custom_grid_config"] = run["custom_grid"]
        for key, value in source.items():
            if key == "comment":
                continue
            if key == "polarization":
                value = _polarization_display_label(value)
            label = field_labels.get(key, key.replace("_", " ").capitalize())
            add_value(values, label, value)

        spec = run.get("grating_spec")
        if isinstance(spec, dict):
            for key in (
                "grating_type", "period_lpermm", "x_resolution_nm", "z_resolution_nm",
                "width_to_period_ratio", "depth_nm", "sinusoidal_depth_nm", "blaze_angle_deg",
                "anti_blaze_angle_deg", "left_wall_angle_deg", "right_wall_angle_deg", "stack",
                "roughness",
            ):
                if key in spec and spec[key] not in (None, ""):
                    label = "Grating stack" if key == "stack" else field_labels.get(
                        key, key.replace("_", " ").capitalize()
                    )
                    add_value(values, label, spec[key])
        flattened.append(values)

    all_labels = sorted({label for values in flattened for label in values}, key=str.casefold)
    rows = []
    for label in all_labels:
        cells = [values.get(label, "Not recorded") for values in flattened]
        rows.append({"label": label, "values": cells, "different": len(set(cells)) > 1})
    return rows


def _linked_runs(data_dir: Path, grating_id: str) -> list[dict[str, Any]]:
    """Return saved runs whose manifests reference one grating ID."""

    return [
        run
        for run in _list_plot_runs(data_dir / "runs")
        if str(run.get("grating_id", "")) == grating_id
    ]
