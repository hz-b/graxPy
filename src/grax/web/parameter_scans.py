"""Grouped parameter combinations, each containing a complete configured scan."""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .custom_grid import ParameterDescriptor, expand_custom_grid, parameter_catalog

CFF = ParameterDescriptor("cff", "CFF", "Conditions", "float", "case", minimum=1e-12)


def scan_catalog(spec: dict[str, Any], mode: str) -> list[ParameterDescriptor]:
    """Return parameters compatible with the chosen mounting."""
    excluded = {"energy_ev", "grazing_angle_deg"}
    items = [item for item in parameter_catalog(spec) if item.id not in excluded]
    return items + ([CFF] if mode == "monochromator" else [])


def parse_config(value: Any) -> dict[str, Any]:
    """Read a posted JSON grid or an already normalized configuration."""
    if not value:
        return {}
    config = json.loads(value) if isinstance(value, str) else value
    if not isinstance(config, dict):
        raise ValueError("Parameter variations must be an object.")
    return config


def has_variations(config: dict[str, Any]) -> bool:
    """Report whether any parameter axis is configured."""
    return bool(config.get("independent") or config.get("coupled_groups"))


def sample_values(sweep: dict[str, Any]) -> list[float]:
    """Return the x grid, including legacy energy-based angle runs."""
    axis = sweep.get("sample_axis")
    return axis["values"] if axis else sweep["energy_values"]


def sample_key(result: Any, sweep: dict[str, Any]) -> float:
    return float(result.grazing_angle_deg if sweep.get("sample_axis") else result.energy_ev)


def build_scans(
    config: dict[str, Any],
    spec: dict[str, Any],
    *,
    mode: str,
    energies: np.ndarray,
    defaults: dict[str, Any],
    angle_values: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve combinations before applying the chosen mounting at each energy."""
    from grax import simulation

    if mode not in {"energy_sweep", "angle_sweep", "fixed_angle", "monochromator", "constant_deviation"}:
        raise ValueError("Parameter scans require Energy sweep, Angle sweep, or Monochromator.")
    allowed = {item.id for item in scan_catalog(spec, mode)}
    # Fixed-angle runs inject their sampled grazing-angle values as a resolved
    # scan axis. It is not an optional parameter-axis control in the form.
    if mode in {"angle_sweep", "fixed_angle"} and any(
        axis.get("parameter") == "grazing_angle_deg" for axis in config.get("independent", [])
    ):
        allowed.add("grazing_angle_deg")
    requested = [axis.get("parameter") for axis in config.get("independent", [])]
    requested += [
        parameter
        for group in config.get("coupled_groups", [])
        for parameter in group.get("parameters", [])
    ]
    if any(parameter not in allowed for parameter in requested):
        raise ValueError("A varied parameter is not available for this scan mode.")
    energies = np.sort(np.asarray(energies, dtype=float))
    if energies.size == 0 or np.any(~np.isfinite(energies)) or np.any(energies <= 0):
        raise ValueError("Energy values must be finite and positive.")
    if energies.size > 1 and np.any(np.diff(energies) == 0):
        raise ValueError("Use one energy point when the energy bounds are equal.")
    if angle_values is not None:
        if mode != "angle_sweep" or energies.size != 1:
            raise ValueError("Angular sampling requires one fixed energy in Angle sweep.")
        if not angle_values or any(not math.isfinite(value) or value < 0 for value in angle_values):
            raise ValueError("Angles must be finite and nonnegative.")
        if any(right <= left for left, right in zip(angle_values, angle_values[1:])):
            raise ValueError("Angle values must increase without duplicates.")
    normalized, combinations = expand_custom_grid(
        config,
        spec,
        defaults={"cff": 2.25, **defaults},
        extra_descriptors=[CFF] if mode == "monochromator" else [],
    )
    labels = {item.id: item.label for item in scan_catalog(spec, mode)}
    axes = [
        {
            "id": axis["parameter"],
            "label": labels.get(
                axis["parameter"],
                "Grazing angle, deg"
                if axis["parameter"] == "grazing_angle_deg"
                else axis["parameter"],
            ),
            "parameters": [axis["parameter"]],
            "rows": [{axis["parameter"]: value} for value in axis["values"]],
        }
        for axis in normalized["independent"]
    ]
    axes += [
        {
            "id": f"coupled-{index:03d}",
            "label": group["name"],
            "parameters": group["parameters"],
            "rows": group["rows"],
        }
        for index, group in enumerate(normalized["coupled_groups"])
    ]
    indices = itertools.product(*(range(len(axis["rows"])) for axis in axes)) if axes else [()]
    scans, cases = [], []
    for index, (combination, selection) in enumerate(zip(combinations, indices, strict=True)):
        scan_id = f"scan-{index:06d}"
        parameters = {
            key: value
            for key, value in combination["custom_parameters"].items()
            if key != "variant"
        }
        grating = combination["grating"]
        if "roughness.kind" not in parameters and defaults.get("roughness_kind") is not None:
            kind = str(defaults["roughness_kind"])
            if kind == "none":
                grating.roughness = None
            elif grating.roughness is None:
                from grax import RoughnessSpec

                grating.roughness = RoughnessSpec(kind=kind, sigma_nm=0)
            elif grating.roughness.kind != kind:
                grating.roughness = replace(
                    grating.roughness, kind=kind, num_supercells=1, num_realizations=1
                )
        combination["custom_grating_spec"]["roughness"] = (
            asdict(grating.roughness) if grating.roughness else None
        )
        for key in ("x_resolution_nm", "z_resolution_nm"):
            if key not in parameters and defaults.get(f"run_{key}"):
                value = float(defaults[f"run_{key}"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Simulation resolutions must be finite and positive.")
                combination[key] = value
                combination["custom_grating_spec"][key] = value
                setattr(grating, key, value)
        label = (
            "; ".join(
                f"{labels.get(key, key)}={display_value(value)}"
                for key, value in parameters.items()
            )
            or "Default settings"
        )
        settings = {
            key: combination[key]
            for key in (
                "diffraction_order",
                "fourier_orders",
                "polarization",
                "solver",
                "x_resolution_nm",
                "z_resolution_nm",
            )
        }
        settings["grazing_angle_deg"] = combination["grazing_angle_deg"]
        settings["cff"] = combination["custom_resolved_parameters"].get(
            "cff", defaults.get("cff", 2.25)
        )
        scans.append(
            {
                "id": scan_id,
                "label": label,
                "parameters": parameters,
                "axis_indices": dict(zip((axis["id"] for axis in axes), selection, strict=True)),
                "grating_spec": combination["custom_grating_spec"],
                "settings": settings,
            }
        )
        if mode in {"energy_sweep", "angle_sweep", "fixed_angle"}:
            generated = simulation.fixed_angle_cases(
                grating=combination["grating"],
                energies_ev=energies,
                grazing_angle_deg=settings["grazing_angle_deg"],
                polarization=settings["polarization"],
            )
        elif mode == "monochromator":
            generated = simulation.monochromator_cases(
                grating=combination["grating"],
                energies_ev=energies,
                cff=settings["cff"],
                diffraction_order=settings["diffraction_order"],
                polarization=settings["polarization"],
            )
        else:
            generated = simulation.constant_deviation_cases(
                grating=combination["grating"],
                energies_ev=energies,
                total_deviation_deg=float(defaults["total_deviation_deg"]),
                diffraction_order=settings["diffraction_order"],
                polarization=settings["polarization"],
            )
        if angle_values is not None:
            generated = (
                case
                for angle in angle_values
                for case in simulation.fixed_angle_cases(
                    grating=combination["grating"],
                    energies_ev=energies,
                    grazing_angle_deg=angle,
                    polarization=settings["polarization"],
                )
            )
        for energy_index, case in enumerate(generated):
            for key in (
                "diffraction_order",
                "fourier_orders",
                "polarization",
                "solver",
                "x_resolution_nm",
                "z_resolution_nm",
            ):
                case[key] = settings[key]
            case.update(
                case_id=f"{scan_id}-{'angle' if angle_values is not None else 'energy'}-{energy_index:06d}",
                scan_id=scan_id,
                energy_index=energy_index,
                scan_parameters=parameters,
                scan_label=label,
                scan_grating_spec=combination["custom_grating_spec"],
                scan_settings=settings,
            )
            cases.append(case)
    sweep = {
        "schema_version": 1,
        "mode": mode,
        "grid": normalized,
        "energy_values": energies.tolist(),
        "axes": axes,
        "scans": scans,
    }
    if angle_values is not None:
        sweep["sample_axis"] = {"id": "grazing_angle_deg", "label": "Grazing angle (deg)", "values": angle_values}
    return sweep, cases


def display_value(value: Any) -> str:
    """Format a varied value for a readable series label."""
    if isinstance(value, dict):
        if "name" in value:
            parts = [str(value["name"])]
            if value.get("density_g_cm3") is not None:
                parts.append(f"density={value['density_g_cm3']:g} g/cm³")
            if value.get("roughness_sigma_nm") is not None:
                parts.append(f"σ={value['roughness_sigma_nm']:g} nm")
            return ", ".join(parts)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return f"{value:.10g}" if isinstance(value, float) else str(value)


def result_order(result: Any) -> tuple[Any, ...]:
    """Sort completed cases by scan identity and sampled-point index."""
    return (result.case_data.get("scan_id", ""), int(result.case_data.get("energy_index", result.index)), int(result.index))


def result_series(
    results: list[Any], sweep: dict[str, Any], scan_id: str, order: int
) -> list[float]:
    """Align one calculated order with the full configured energy grid."""
    values = {}
    for result in results:
        if result.case_data.get("scan_id") != scan_id or result.status != "ok":
            continue
        for actual_order, efficiency in zip(result.orders, result.efficiency_all, strict=True):
            if float(actual_order) == float(order):
                values[sample_key(result, sweep)] = float(efficiency)
    return [values.get(float(value), math.nan) for value in sample_values(sweep)]


def load_results(run_dir: Path) -> list[Any]:
    """Load checkpointed results in deterministic scan order."""
    from grax.simulation.serialization import _load_checkpoint_case_results

    path = run_dir / "checkpoints" / "results.jsonl"
    return (
        sorted(_load_checkpoint_case_results(path).values(), key=result_order)
        if path.exists()
        else []
    )


def load_series(run_dir: Path, sweep: dict[str, Any], scan_id: str, order: int) -> list[float]:
    """Load one scan curve while retaining gaps for missing points."""
    return load_all_series(run_dir, sweep, order)[scan_id]


def load_all_series(run_dir: Path, sweep: dict[str, Any], order: int) -> dict[str, list[float]]:
    """Read all curves for one order from checkpoints or canonical CSV."""
    path = run_dir / "all_orders.csv"
    values = {}
    if (run_dir / "checkpoints" / "results.jsonl").exists():
        for result in load_results(run_dir):
            if result.status != "ok":
                continue
            for actual_order, efficiency in zip(result.orders, result.efficiency_all, strict=True):
                if float(actual_order) == float(order):
                    values[(result.case_data["scan_id"], sample_key(result, sweep))] = float(
                        efficiency
                    )
    elif path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if float(row["order"]) == float(order):
                    key = "grazing_angle_deg" if sweep.get("sample_axis") else "energy_ev"
                    values[(row["scan_id"], float(row[key]))] = float(row["efficiency"])
    return {
        scan["id"]: [
            values.get((scan["id"], float(value)), math.nan) for value in sample_values(sweep)
        ]
        for scan in sweep["scans"]
    }


def plot_scans(
    results: list[Any],
    sweep: dict[str, Any],
    output_path: Path,
    *,
    title: str,
    scan_id: str | None = None,
) -> None:
    """Draw separate energy curves for the selected scan or grouped overview."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    try:
        legacy_angles = (
            sweep.get("mode") == "angle_sweep"
            and not sweep.get("sample_axis")
            and any(item.get("id") == "grazing_angle_deg" for item in sweep.get("axes", []))
            and scan_id is None
        )
        if legacy_angles:
            grouped: dict[tuple[tuple[str, int], ...], list[tuple[float, float]]] = defaultdict(list)
            by_scan = {result.case_data.get("scan_id"): result for result in results}
            for scan in sweep["scans"]:
                fixed = tuple(sorted((key, value) for key, value in scan["axis_indices"].items() if key != "grazing_angle_deg"))
                result = by_scan.get(scan["id"])
                efficiency = float(result.selected_efficiency) if result is not None and result.status == "ok" else math.nan
                grouped[fixed].append((float(scan["parameters"]["grazing_angle_deg"]), efficiency))
            for index, (fixed, points) in enumerate(grouped.items()):
                points.sort()
                axis.plot(
                    [angle for angle, _ in points], [value for _, value in points],
                    label="; ".join(f"{key}={value}" for key, value in fixed) or "Angle sweep",
                    linewidth=1.2, linestyle=["-", "--", ":", "-."][index % 4],
                )
        else:
            for index, scan in enumerate(sweep["scans"]):
                if scan_id is not None and scan["id"] != scan_id:
                    continue
                values = result_series(
                    results, sweep, scan["id"], int(scan["settings"]["diffraction_order"])
                )
                axis.plot(
                    sample_values(sweep), values,
                    label=scan["label"], linewidth=1.2,
                    linestyle=["-", "--", ":", "-."][index % 4],
                )
        axis.set(xlabel="Grazing angle (deg)" if legacy_angles else sweep.get("sample_axis", {}).get("label", "Photon Energy (eV)"), ylabel="Selected-order efficiency", title=title)
        axis.grid(True, alpha=0.3)
        if len(sweep["scans"]) <= 12 or scan_id:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(figure)


def write_outputs(
    results: list[Any], sweep: dict[str, Any], run_dir: Path, *, title: str, include_plot: bool
) -> None:
    """Atomically publish sorted tables and render per-scan energy plots."""
    results = sorted(results, key=result_order)
    with (run_dir / "summary.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_id",
                "case_id",
                "energy_ev",
                "grazing_angle_deg",
                "parameters",
                "selected_efficiency",
                "status",
                "error_message",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_data["scan_id"],
                    result.case_id,
                    result.energy_ev,
                    result.grazing_angle_deg,
                    json.dumps(result.case_data["scan_parameters"], sort_keys=True),
                    result.selected_efficiency,
                    result.status,
                    result.error_message or "",
                ]
            )
    (run_dir / "summary.csv.tmp").replace(run_dir / "summary.csv")
    with (run_dir / "all_orders.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_id",
                "case_id",
                "energy_ev",
                "grazing_angle_deg",
                "order",
                "efficiency",
                "diffraction_angle_deg",
            ]
        )
        for result in results:
            if result.status == "ok":
                for order, efficiency, angle in sorted(
                    zip(
                        result.orders,
                        result.efficiency_all,
                        result.diffraction_angle_all,
                        strict=True,
                    )
                ):
                    writer.writerow(
                        [
                            result.case_data["scan_id"],
                            result.case_id,
                            result.energy_ev,
                            result.grazing_angle_deg,
                            order,
                            efficiency,
                            angle,
                        ]
                    )
    (run_dir / "all_orders.csv.tmp").replace(run_dir / "all_orders.csv")
    if include_plot:
        plot_scans(results, sweep, run_dir / "selected_efficiency.png", title=title)
        for scan in sweep["scans"]:
            plot_scans(
                results,
                sweep,
                run_dir / "scans" / scan["id"] / "efficiency.png",
                title=scan["label"],
                scan_id=scan["id"],
            )


def export_tables(
    run_dir: Path,
    sweep: dict[str, Any],
    *,
    orders: list[int],
    axis_ids: list[str],
    formats: set[str],
) -> bytes:
    """Export each requested parameter slice as a wide energy table."""
    import yaml

    axes = {axis["id"]: axis for axis in sweep["axes"]}
    if not axes:
        label = "Angle scan" if sweep.get("sample_axis") else "Energy scan"
        axes = {"scan": {"id": "scan", "label": label, "parameters": [], "rows": [{}]}}
    if not orders or not axis_ids or any(axis not in axes for axis in axis_ids):
        raise ValueError("Choose diffraction orders and valid parameter axes.")
    if not formats or not formats <= {"csv", "dat"}:
        raise ValueError("Choose CSV and/or DAT.")
    buffer = io.BytesIO()
    series_by_order = {order: load_all_series(run_dir, sweep, order) for order in set(orders)}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for axis_index, axis_id in enumerate(dict.fromkeys(axis_ids)):
            groups = defaultdict(list)
            for scan in sweep["scans"]:
                fixed = tuple(
                    (key, value) for key, value in scan["axis_indices"].items() if key != axis_id
                )
                groups[fixed].append(scan)
            for slice_index, (fixed, scans) in enumerate(groups.items()):
                scans.sort(key=lambda scan: scan["axis_indices"].get(axis_id, 0))
                for order in dict.fromkeys(orders):
                    axis_name = "".join(
                        character if character.isalnum() else "_"
                        for character in axes[axis_id]["label"]
                    ).strip("_")[:48]
                    name = (
                        f"{axis_name}_axis-{axis_index:03d}_slice-{slice_index:04d}_order-{order}"
                    )
                    columns = [scan["label"] for scan in scans]
                    matrix = [series_by_order[order][scan["id"]] for scan in scans]
                    for format_name in sorted(formats):
                        text = io.StringIO(newline="")
                        writer = csv.writer(
                            text,
                            delimiter="," if format_name == "csv" else "\t",
                            lineterminator="\n",
                        )
                        writer.writerow([sweep.get("sample_axis", {}).get("id", "energy_ev"), *columns])
                        for index, value in enumerate(sample_values(sweep)):
                            writer.writerow(
                                [
                                    value,
                                    *(
                                        "NaN" if math.isnan(values[index]) else values[index]
                                        for values in matrix
                                    ),
                                ]
                            )
                        archive.writestr(f"{name}.{format_name}", text.getvalue())
                    archive.writestr(
                        f"{name}.yaml",
                        yaml.safe_dump(
                            {
                                "axis": axes[axis_id],
                                "fixed_axis_rows": dict(fixed),
                                "diffraction_order": order,
                                "columns": [
                                    {"column": index + 2, "label": scan["label"], "scan": scan}
                                    for index, scan in enumerate(scans)
                                ],
                                "sample_axis": sweep.get("sample_axis", {"id": "energy_ev", "values": sweep["energy_values"]}),
                            },
                            allow_unicode=True,
                            sort_keys=False,
                        ),
                    )
    return buffer.getvalue()
