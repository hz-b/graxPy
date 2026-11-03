"""Turning submitted run forms into scan configurations and simulation cases."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from . import parameter_scans
from .common import (
    _normalized_polarization,
    _normalized_solver,
)
from .custom_grid import expand_custom_grid


def _parameter_study_range(
    form: Any,
    prefix: str,
    default: float | int,
    *,
    integer: bool = False,
) -> np.ndarray:
    """Parse one parameter-study min/max/count range."""

    min_key = f"study_{prefix}_min"
    max_key = f"study_{prefix}_max"
    points_key = f"study_{prefix}_points"
    if not str(form.get(min_key, "")).strip():
        return np.asarray([default])
    lower = float(form[min_key])
    upper = float(form[max_key])
    points = int(float(form[points_key]))
    if lower <= 0 or upper <= 0:
        raise ValueError(f"Study {prefix.replace('_', ' ')} values must be greater than 0.")
    if upper < lower:
        raise ValueError(f"Study {prefix.replace('_', ' ')} maximum must be at least its minimum.")
    if points < 1:
        raise ValueError("Each parameter-study range must contain at least one value.")
    values = np.linspace(lower, upper, points)
    if integer:
        if not lower.is_integer() or not upper.is_integer():
            raise ValueError(f"Study {prefix.replace('_', ' ')} bounds must be integers.")
        integer_values = np.arange(int(lower), int(upper) + 1, dtype=int)
        if points >= integer_values.size:
            return integer_values
        return np.unique(np.rint(np.linspace(lower, upper, points)).astype(int))
    return values


def _parameter_study_configuration(form: Any, grating: Any) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the selected energy, angle, and numerical study ranges."""

    energy_text = str(form.get("study_energy_ev", "")).strip()
    angle_text = str(form.get("study_grazing_angle_deg", "")).strip()
    # Keep compatibility with older clients/forms that submitted the shared
    # energy and angle fields before the dedicated parameter-study controls.
    energy = float(energy_text or form.get("energy_ev", form.get("energy_start_ev", 100.0)))
    angle = float(angle_text or form.get("grazing_angle_deg", 1.5))
    if energy <= 0:
        raise ValueError("Study energy must be greater than 0 eV.")
    if not np.isfinite(angle):
        raise ValueError("Study grazing angle must be finite.")
    fourier_default = int(float(form.get("fourier_orders", 15)))
    fourier = _parameter_study_range(form, "fourier", fourier_default, integer=True)
    x_default = float(form.get("run_x_resolution_nm") or getattr(grating, "x_resolution_nm", 1.0))
    z_default = float(form.get("run_z_resolution_nm") or getattr(grating, "z_resolution_nm", 1.0))
    x_values = _parameter_study_range(form, "x_resolution", x_default)
    z_values = _parameter_study_range(form, "z_resolution", z_default)
    if np.any(fourier < 1):
        raise ValueError("Fourier orders must be at least 1.")
    return energy, angle, fourier, x_values, z_values


def _parameter_study_total_points(form: Any, grating: Any, energy_count: int) -> int:
    """Return the number of individual simulations in a parameter study."""

    _, _, fourier, x_values, z_values = _parameter_study_configuration(form, grating)
    return int(energy_count * (len(fourier) + len(x_values) + len(z_values)))


def _parameter_study_fixed_configuration(
    form: Any,
    fourier_values: np.ndarray,
    x_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[float, float, int, float, int, float]:
    """Return fixed X/Z/Fourier settings for each individual sweep."""

    def positive_float(field: str, default: float) -> float:
        value = float(form.get(field, default))
        if value <= 0:
            raise ValueError(f"{field.replace('_', ' ').capitalize()} must be greater than 0.")
        return value

    def positive_integer(field: str, default: int) -> int:
        value = float(form.get(field, default))
        if not value.is_integer() or value < 1:
            raise ValueError(f"{field.replace('_', ' ').capitalize()} must be a positive integer.")
        return int(value)

    return (
        positive_float("study_fourier_fixed_x_resolution", float(np.max(x_values))),
        positive_float("study_fourier_fixed_z_resolution", float(np.max(z_values))),
        positive_integer("study_x_fixed_fourier", int(np.min(fourier_values))),
        positive_float("study_x_fixed_z_resolution", float(np.max(z_values))),
        positive_integer("study_z_fixed_fourier", int(np.min(fourier_values))),
        positive_float("study_z_fixed_x_resolution", float(np.max(x_values))),
    )


def _custom_grid_configuration_from_value(value: Any) -> dict[str, Any]:
    """Return a custom-grid configuration submitted as JSON or a mapping."""

    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        raise ValueError("Configure the custom grid before starting the run.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Custom-grid configuration is invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("Custom-grid configuration must be an object.")
    return payload


def _custom_grid_from_form(
    form: Any,
    base_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the submitted custom grid and return its expanded cases."""

    config = _custom_grid_configuration_from_value(form.get("custom_grid_config"))
    defaults = _custom_grid_defaults_from_form(form)
    return expand_custom_grid(config, base_spec, defaults=defaults)


def _custom_grid_defaults_from_form(form: Any) -> dict[str, Any]:
    """Return the direct-simulation values inherited by custom-grid cases."""

    return {
        "energy_ev": float(form.get("custom_default_energy_ev", 100.0) or 100.0),
        "grazing_angle_deg": float(form.get("custom_default_grazing_angle_deg", 1.5) or 1.5),
        "diffraction_order": int(float(form.get("custom_default_diffraction_order", 1) or 1)),
        "fourier_orders": int(float(form.get("custom_default_fourier_orders", 15) or 15)),
        "polarization": str(form.get("custom_default_polarization", "p") or "p"),
        "solver": str(form.get("custom_default_solver", "rcwa") or "rcwa"),
    }


def _prepare_monochromator_cff_config(form: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Drop CFF = 1 from generated ranges and reject it in explicit inputs."""
    prepared = deepcopy(config)
    varied_cff = False

    def is_forbidden(value: Any) -> bool:
        try:
            return bool(np.isclose(float(value), 1.0, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError, OverflowError):
            return False

    for axis in prepared.get("independent") or []:
        if not isinstance(axis, dict):
            raise ValueError("Independent parameter axes must be objects.")
        if axis.get("parameter") != "cff":
            continue
        varied_cff = True
        if str(axis.get("mode", "list")).lower() == "range":
            try:
                minimum = float(axis["minimum"])
                maximum = float(axis["maximum"])
                count = int(axis["count"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError("The CFF linear range needs finite bounds and a positive point count.") from error
            if not np.isfinite(minimum + maximum) or count < 1:
                raise ValueError("The CFF linear range needs finite bounds and a positive point count.")
            values = np.linspace(minimum, maximum, count).tolist()
            values = [value for value in values if not is_forbidden(value)]
            if not values:
                raise ValueError("The CFF linear range has no usable values after CFF = 1 is removed.")
            axis.clear()
            axis.update({"parameter": "cff", "mode": "list", "values": values})
        elif any(is_forbidden(value) for value in axis.get("values") or []):
            raise ValueError("CFF = 1 is not allowed. Remove it from the explicit values.")

    for group in prepared.get("coupled_groups") or []:
        if not isinstance(group, dict):
            raise ValueError("Coupled parameter groups must be objects.")
        parameters = group.get("parameters") or []
        if "cff" not in parameters:
            continue
        varied_cff = True
        range_settings = (group.get("input_settings") or {}).get("cff") or {}
        rows = group.get("rows") or []
        if str(range_settings.get("mode", "manual")).lower() == "range":
            rows = [row for row in rows if not is_forbidden(row.get("cff"))]
            if not rows:
                raise ValueError("The CFF linear range has no usable values after CFF = 1 is removed.")
            group["rows"] = rows
        elif any(is_forbidden(row.get("cff")) for row in rows):
            raise ValueError("CFF = 1 is not allowed. Remove it from the explicit values.")

    if not varied_cff and is_forbidden(form.get("cff", 2.25) or 2.25):
        raise ValueError("CFF = 1 cannot be used for a monochromator run. Choose a different value.")
    return prepared


def _parameter_scans_from_form(
    form: Any, spec: dict[str, Any] | None, energies: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve a posted or checkpointed sweep using its saved grating snapshot."""
    config = parameter_scans.parse_config(form.get("parameter_sweep_config"))
    mode = str(form.get("workflow"))
    if mode == "monochromator":
        config = _prepare_monochromator_cff_config(form, config)
    angle_values = None
    if mode == "angle_sweep" and not any(
        axis.get("parameter") == "grazing_angle_deg" for axis in config.get("independent", [])
    ):
        lower = float(form.get("grazing_angle_start_deg", 1.0))
        upper = float(form.get("grazing_angle_stop_deg", 2.0))
        count = int(float(form.get("grazing_angle_points", 11)))
        if not np.isfinite(lower + upper) or lower < 0 or upper < lower or count < 1:
            raise ValueError("Grazing-angle bounds must be finite, nonnegative, and ordered, with at least one point.")
        if (count == 1) != (lower == upper):
            raise ValueError("A single angle point requires equal bounds; differing bounds require at least two points.")
        angle_values = np.linspace(lower, upper, count).tolist()
    variants = config.get("variants") or []
    base_spec = dict(variants[0]["spec"]) if variants else dict(spec or {})
    if not base_spec:
        raise ValueError("Parameter variations require a saved grating snapshot.")
    defaults = {
        "diffraction_order": int(float(form.get("diffraction_order", 1))),
        "fourier_orders": int(float(form.get("fourier_orders", 15))),
        "grazing_angle_deg": float(form.get("grazing_angle_deg", 1.5)),
        "cff": float(form.get("cff", 2.25)),
        "total_deviation_deg": float(form.get("total_deviation_deg", 174.0)),
        "solver": _normalized_solver(form.get("solver", "rcwa")),
        "polarization": _normalized_polarization(form.get("polarization", "p")),
        "roughness_kind": form.get("roughness_kind"),
        "run_x_resolution_nm": form.get("run_x_resolution_nm"),
        "run_z_resolution_nm": form.get("run_z_resolution_nm"),
    }
    return parameter_scans.build_scans(
        config, base_spec, mode=mode, energies=energies, defaults=defaults,
        angle_values=angle_values,
    )


def _parameter_scan_point_count(form: Any, energy_count: int) -> int:
    """Count points from the canonical, already normalized parameter grid."""
    config = parameter_scans.parse_config(form.get("parameter_sweep_config"))
    legacy_angle_axis = any(axis.get("parameter") == "grazing_angle_deg" for axis in config.get("independent", []))
    total = int(float(form.get("grazing_angle_points", 11))) if form.get("workflow") == "angle_sweep" and not legacy_angle_axis else energy_count
    for axis in config.get("independent", []):
        total *= len(axis["values"])
    for group in config.get("coupled_groups", []):
        total *= len(group["rows"])
    return total


def _snapshot_custom_grid_afm_profiles(config: dict[str, Any], run_dir: Path) -> None:
    """Copy AFM profiles into the run and update variant snapshots in place."""

    profile_dir = run_dir / "custom_grid_profiles"
    copied = False
    for index, variant in enumerate(config.get("variants") or []):
        spec = variant.get("spec") if isinstance(variant, dict) else None
        profile = spec.get("profile") if isinstance(spec, dict) else None
        if not isinstance(spec, dict) or spec.get("grating_type") != "afm" or not isinstance(profile, dict):
            continue
        source = Path(str(profile.get("profile_path", ""))).resolve()
        if not source.is_file():
            raise ValueError(f"AFM profile artifact is missing for variant {variant.get('name', index + 1)}.")
        profile_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(variant.get("id") or index + 1))
        destination = profile_dir / f"{safe_id}{source.suffix or '.csv'}"
        shutil.copyfile(source, destination)
        profile["profile_path"] = str(destination.resolve())
        copied = True
    if not copied and profile_dir.exists() and not any(profile_dir.iterdir()):
        profile_dir.rmdir()


def _custom_grid_case_count_from_form(form: Any) -> int:
    """Return the expanded case count from persisted custom-grid form data."""

    config = _custom_grid_configuration_from_value(form.get("custom_grid_config"))
    variants = config.get("variants") or []
    if not variants or not isinstance(variants[0], dict) or not isinstance(variants[0].get("spec"), dict):
        raise ValueError("Custom-grid configuration has no valid base variant.")
    _, cases = _custom_grid_from_form(form, dict(variants[0]["spec"]))
    return len(cases)


def _worker_settings_from_form(form: Any) -> tuple[str, str | int, int | None]:
    """Return worker-mode metadata and the runner max_workers setting."""

    worker_mode = str(form.get("max_workers_mode", "auto") or "auto").strip().lower()
    if worker_mode == "manual":
        requested_workers = int(float(form.get("max_workers", 1)))
        if requested_workers < 1:
            raise ValueError("Manual worker count must be at least 1.")
        return "manual", requested_workers, requested_workers
    return "auto", "auto", None


def _run_input_from_form(form: Any) -> dict[str, Any]:
    """Return persisted run-input metadata needed for checkpoint resume."""

    input_data: dict[str, Any] = {}
    for key in form.keys():
        values = form.getlist(key)
        input_data[key] = values if len(values) > 1 else form.get(key)
    input_data["polarization"] = _normalized_polarization(input_data.get("polarization", "p"))
    input_data["solver"] = _normalized_solver(input_data.get("solver", "rcwa"))
    return input_data


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
    polarization = _normalized_polarization(form.get("polarization", "p"))
    solver = _normalized_solver(form.get("solver", "rcwa"))

    if workflow == "angle_sweep" or (
        workflow == "fixed_angle" and parameter_scans.has_variations(parameter_scans.parse_config(form.get("parameter_sweep_config")))
    ) or (
        workflow in {"energy_sweep", "monochromator", "constant_deviation"}
        and parameter_scans.has_variations(parameter_scans.parse_config(form.get("parameter_sweep_config")))
    ):
        return _parameter_scans_from_form(form, None, energies)[1]

    if workflow in {"energy_sweep", "fixed_angle"}:
        cases = list(
            simulation.fixed_angle_cases(
                grating=grating,
                energies_ev=energies,
                grazing_angle_deg=float(form["grazing_angle_deg"]),
                polarization=polarization,
            )
        )
    elif workflow == "monochromator":
        cases = list(
            simulation.monochromator_cases(
                grating=grating,
                energies_ev=energies,
                diffraction_order=diffraction_order,
                cff=float(form.get("cff", 2.25) or 2.25),
                polarization=polarization,
            )
        )
    elif workflow == "constant_deviation":
        cases = list(
            simulation.constant_deviation_cases(
                grating=grating,
                energies_ev=energies,
                total_deviation_deg=float(form.get("total_deviation_deg", 174.0)),
                diffraction_order=diffraction_order,
                polarization=polarization,
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
    elif workflow == "custom_grid":
        config = _custom_grid_configuration_from_value(form.get("custom_grid_config"))
        variants = config.get("variants") or []
        if not variants or not isinstance(variants[0], dict):
            raise ValueError("Custom-grid configuration has no structural variants.")
        _, cases = _custom_grid_from_form(form, dict(variants[0].get("spec") or {}))
    else:
        raise ValueError("Unsupported workflow.")
    for index, case in enumerate(cases):
        case["case_id"] = f"{workflow}-{index:08d}"
        case.setdefault("fourier_orders", fourier_orders)
        case.setdefault("polarization", polarization)
        case.setdefault("solver", solver)
        if workflow != "multilayer_theta_search":
            case.setdefault("x_resolution_nm", run_x_resolution_nm)
            case.setdefault("z_resolution_nm", run_z_resolution_nm)
    return cases
