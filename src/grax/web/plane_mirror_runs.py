"""Background execution helpers for saved plane-mirror scan runs."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from werkzeug.datastructures import MultiDict

from .plane_mirror import MAX_TOTAL_SAMPLES as _MAX_TOTAL_SAMPLES
from .plane_mirror import PLANE_MIRROR_DEFAULTS, compute_plane_mirror, parse_plane_mirror_options

MAX_TOTAL_SAMPLES = _MAX_TOTAL_SAMPLES


SCAN_FIELDS = {
    "scan_mode",
    "fixed_angle_deg",
    "fixed_energy_ev",
    "grating_density_lpermm",
    "cff",
    "diffraction_order",
    "roughness_sigma_nm",
    "grading_percent_per_mm",
    "footprint_length_mm",
    "footprint_points",
    "energy_points",
    "angle_points",
    "map_energy_points",
    "map_angle_points",
    "energy_min",
    "energy_max",
    "angle_min",
    "angle_max",
    "polarization",
    "grading_mode",
}
STACK_FIELDS = {
    "layer_thickness_nm",
    "d_period_nm",
    "gamma",
    "n_bilayers",
    "top_cap_thickness_nm",
    "material_a_density_g_cm3",
    "material_b_density_g_cm3",
    "top_cap_material_density_g_cm3",
    "layer_material_density_g_cm3",
    "substrate_material",
    "material_a",
    "material_b",
    "layer_material",
    "top_material",
    "top_cap_material",
    "substrate_roughness_sigma_nm",
    "material_a_roughness_sigma_nm",
    "material_b_roughness_sigma_nm",
    "layer_roughness_sigma_nm",
    "top_cap_roughness_sigma_nm",
    "substrate_correlation_length_nm",
    "material_a_correlation_length_nm",
    "material_b_correlation_length_nm",
    "layer_correlation_length_nm",
    "top_cap_correlation_length_nm",
}
CUSTOM_STACK_FIELDS = {
    "cl_thickness_nm",
    "cl_roughness_sigma_nm",
    "cl_correlation_length_nm",
    "cl_thickness_b_nm",
    "cl_roughness_b_sigma_nm",
    "cl_correlation_b_length_nm",
    "cl_repeats",
    "cl_density_g_cm3",
    "cl_density_b_g_cm3",
    "cl_material",
    "cl_material_b",
    "cl_kind",
}
INTEGER_FIELDS = {
    "energy_points",
    "angle_points",
    "map_energy_points",
    "map_angle_points",
    "footprint_points",
    "diffraction_order",
    "n_bilayers",
    "cl_repeats",
}


def expand_variations(
    scan: dict[str, Any], stack: dict[str, Any], axes_text: str, coupled_text: str
):
    """Expand independent numeric ranges and row-wise coupled values."""
    axes = json.loads(axes_text or "[]")
    coupled = json.loads(coupled_text or "[]")
    if not isinstance(axes, list) or not isinstance(coupled, list):
        raise ValueError("Variations must be JSON lists.")
    independent = []
    for axis in axes:
        if not isinstance(axis, dict):
            raise ValueError("Each independent variation must be an object.")
        field = str(axis.get("parameter", ""))
        target_info = _variation_target(scan, stack, field)
        if target_info is None:
            raise ValueError(f"Unsupported varied parameter: {field}")
        target, base_field, index = target_info
        if "values" in axis:
            values = axis["values"]
            if not isinstance(values, list) or not values or len(values) > 500:
                raise ValueError("A value variation must contain between 1 and 500 values.")
            choices = [str(value) for value in values]
        else:
            start, stop, count = float(axis["start"]), float(axis["stop"]), int(axis["points"])
            if not np.isfinite(start) or not np.isfinite(stop) or count < 1 or count > 500:
                raise ValueError(
                    "Each numeric variation must have finite bounds and between 1 and 500 points."
                )
            choices = [str(value) for value in np.linspace(start, stop, count)]
            if base_field in INTEGER_FIELDS and any(
                float(value) != int(float(value)) for value in choices
            ):
                raise ValueError(f"{field} must use whole-number range values.")
        independent.append([[(target, base_field, index, field, value)] for value in choices])
    groups = []
    for group in coupled:
        if not isinstance(group, dict) or not group:
            raise ValueError("Each coupled variation must map parameter names to value lists.")
        keys = list(group)
        targets = [_variation_target(scan, stack, key) for key in keys]
        if any(target is None for target in targets):
            raise ValueError("Coupled variations include an unsupported parameter.")
        columns = [group[key] for key in keys]
        if not columns[0] or not all(
            isinstance(column, list) and len(column) == len(columns[0]) for column in columns
        ):
            raise ValueError("Values in a coupled variation must be lists of equal length.")
        groups.append(
            [
                [
                    (
                        targets[index][0],
                        targets[index][1],
                        targets[index][2],
                        keys[index],
                        str(column[row]),
                    )
                    for index, column in enumerate(columns)
                ]
                for row in range(len(columns[0]))
            ]
        )
    combinations = [[]]
    for axis in independent + groups:
        combinations = [left + right for left in combinations for right in axis]
        if len(combinations) > 1000:
            raise ValueError("A run cannot contain more than 1000 variation combinations.")
    cases = []
    for combination in combinations:
        case_scan, case_stack = deepcopy(scan), deepcopy(stack)
        changed = {}
        for target, base_field, index, display_field, value in combination:
            if base_field in INTEGER_FIELDS and float(value) != int(float(value)):
                raise ValueError(f"{display_field} must use whole-number values.")
            target_dict = case_scan if target is scan else case_stack
            if target_dict is case_stack and base_field in {
                "substrate_material", "material_a", "material_b", "top_material",
                "layer_material", "top_cap_material", "cl_material", "cl_material_b",
            } and str(value).strip().casefold() == "none":
                _apply_none_material(case_stack, base_field, index, display_field)
                changed[display_field] = "None"
                continue
            if index is None:
                target_dict[base_field] = value
            else:
                target_dict[base_field][index] = value
            changed[display_field] = value
        _validate_varied_material_properties(case_stack, changed)
        cases.append({"scan": case_scan, "stack": case_stack, "parameters": changed})
    return cases or [{"scan": deepcopy(scan), "stack": deepcopy(stack), "parameters": {}}]


def _variation_target(scan, stack, field):
    if field in SCAN_FIELDS:
        return scan, field, None
    if field in STACK_FIELDS:
        return stack, field, None
    if "[" in field and field.endswith("]"):
        base, raw_index = field[:-1].rsplit("[", 1)
        if base in CUSTOM_STACK_FIELDS and raw_index.isdigit():
            index = int(raw_index)
            values = stack.get(base)
            if isinstance(values, list) and index < len(values):
                return stack, base, index
    return None


def _apply_none_material(stack: dict[str, Any], field: str, index: int | None, label: str) -> None:
    """Resolve the shared None sentinel for plane-mirror material variations."""
    if field == "top_cap_material":
        stack[field] = ""
        return
    if field == "layer_material" and stack.get("stack_type") == "single_layer":
        stack["stack_type"] = "bare"
        return
    if field == "cl_material" and stack.get("stack_type") == "custom" and index is not None:
        kinds = stack.get("cl_kind") or ["single"] * len(stack.get("cl_material", []))
        if (kinds[index] or "single") != "single":
            raise ValueError(f"{label} is part of a multilayer block and cannot be None.")
        for key in CUSTOM_STACK_FIELDS:
            values = stack.get(key)
            if isinstance(values, list) and index < len(values):
                values.pop(index)
        if not stack.get("cl_material"):
            stack["stack_type"] = "bare"
        return
    label_text = label.replace("_", " ").replace("[", " ").replace("]", "")
    raise ValueError(
        f"{label_text.capitalize()} is required by this mirror stack and cannot be None. "
        "Use None for an optional coating or top cap to model an uncoated case."
    )


def _validate_varied_material_properties(stack: dict[str, Any], changed: dict[str, Any]) -> None:
    """Reject density or interface-property scans when the selected material is absent."""
    fields = set(changed)
    for property_field in fields:
        if "density_g_cm3" not in property_field and not any(
            token in property_field for token in ("roughness_sigma_nm", "correlation_length_nm")
        ):
            continue
        material_field = property_field.replace("_density_g_cm3", "").replace(
            "_roughness_sigma_nm", "_material"
        ).replace("_correlation_length_nm", "_material")
        if material_field in {"substrate_material", "material_a", "material_b", "layer_material", "top_material", "top_cap_material"}:
            material = stack.get(material_field)
            if (
                material is None
                or str(material).strip().casefold() in {"", "none"}
                or (material_field == "layer_material" and stack.get("stack_type") == "bare")
            ):
                raise ValueError(f"Choose a {material_field.replace('_', ' ')} before scanning its properties.")
        elif property_field.startswith("cl_"):
            raw_index = property_field.rsplit("[", 1)[-1].rstrip("]") if "[" in property_field else ""
            if raw_index.isdigit():
                index = int(raw_index)
                materials = stack.get("cl_material", [])
                if index >= len(materials) or str(materials[index]).strip().casefold() in {"", "none"}:
                    raise ValueError(f"Choose a custom-layer material before scanning its properties.")


def total_sample_count(scan):
    """Return the number of calculated samples in one scan case."""
    values = {**PLANE_MIRROR_DEFAULTS, **scan}
    mode = str(values["scan_mode"])
    if mode == "map":
        return int(values["map_energy_points"]) * int(values["map_angle_points"])
    return int(
        values["energy_points"]
        if mode in {"energy", "energy_sweep", "monochromator"}
        else values["angle_points"]
    )


def start_worker(app, data_dir: Path, run_id: str, *, resume=False):
    """Start or resume a persisted mirror run in a daemon thread."""
    active = app.extensions.setdefault("grax_mirror_run_events", {})
    if run_id in active:
        return
    stop = threading.Event()
    active[run_id] = stop
    worker = threading.Thread(
        target=execute_worker,
        args=(app, data_dir, run_id, stop, resume),
        daemon=True,
        name=f"grax-plane-mirror-{run_id}",
    )
    worker.start()


def execute_worker(app, data_dir: Path, run_id: str, stop: threading.Event, resume: bool):
    """Compute each saved variation and persist result checkpoints and plots."""
    run_dir = data_dir / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results_path = run_dir / "mirror_results.json"
        results = (
            json.loads(results_path.read_text(encoding="utf-8"))
            if resume and results_path.exists()
            else []
        )
        manifest.update(status="running", error_text="", completed_points=len(results))
        _save(manifest_path, manifest)
        cases = manifest["cases"]
        for index in range(len(results), len(cases)):
            if stop.is_set():
                break
            case = cases[index]
            form = MultiDict()
            for source in (case["stack"], case["scan"]):
                for key, value in source.items():
                    for item in value if isinstance(value, list) else [value]:
                        form.add(key, item)
            from .plane_mirror_views import _plane_mirror_stack

            result = compute_plane_mirror(
                _plane_mirror_stack(form), parse_plane_mirror_options(form)
            )
            results.append({**result, "parameters": case["parameters"], "index": index + 1})
            _save(results_path, results)
            manifest.update(
                completed_points=len(results), status="aborting" if stop.is_set() else "running"
            )
            _save(manifest_path, manifest)
            _plot_results(results, run_dir / "plot.png", manifest["display_name"])
        manifest["status"] = "aborted" if stop.is_set() else "completed"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["artifacts"] = ["mirror_results.json", "plot.png"] if results else []
        _save(manifest_path, manifest)
    except Exception as error:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(status="error", error_text=str(error))
            _save(manifest_path, manifest)
        except Exception:
            app.logger.exception("Could not persist plane-mirror run error")
        app.logger.exception("Plane-mirror run %s failed", run_id)
    finally:
        app.extensions.setdefault("grax_mirror_run_events", {}).pop(run_id, None)


def _save(path: Path, value: Any):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _plot_results(results, path: Path, title: str):
    maps = results[0]["mode"] == "map"
    if maps:
        shown = results[:9]
        columns = min(3, len(shown))
        rows = int(np.ceil(len(shown) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False
        )
        for axis, result in zip(axes.flat, shown, strict=False):
            axis.pcolormesh(result["x"], result["y"], result["z"], shading="auto", vmin=0, vmax=1)
            axis.set_title(_case_title(result["parameters"]))
            axis.set_xlabel("Photon energy (eV)")
            axis.set_ylabel("Grazing angle (deg)")
        for axis in list(axes.flat)[len(shown) :]:
            axis.remove()
    else:
        figure, axis = plt.subplots(figsize=(8, 4.2))
        for result in results:
            axis.plot(result["x"], result["reflectivity"], label=_case_title(result["parameters"]))
        axis.set_xlabel(results[0]["x_label"])
        axis.set_ylabel("Reflectivity")
        axis.set_ylim(0, 1)
        if len(results) > 1:
            axis.legend(fontsize=7)
    if not maps:
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120, facecolor="white")
    plt.close(figure)


def _case_title(parameters):
    return ", ".join(f"{key}={value}" for key, value in parameters.items()) or "Base settings"
