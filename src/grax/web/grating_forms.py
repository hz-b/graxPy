"""Grating and layer-stack form handling: defaults, parsing and live previews."""

from __future__ import annotations

import json
import shutil
import warnings
from copy import copy
from dataclasses import (
    replace,
)
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from grax.materials import material_density_g_cm3

from .grating_roughness import (
    optional_nonnegative,
    realization_grating,
    roughness_from_form,
)
from .persistence import (
    build_grating_from_spec,
    stack_to_spec,
)
from .plane_mirror import (
    DEFAULT_CUSTOM_LAYERS,
    custom_stack_from_form,
)


def _default_form_values() -> dict[str, str]:
    """Return display defaults for the create-grating form."""
    return {
        "name": "",
        "roughness_kind": "none",
        "roughness_seed": "0",
        "roughness_num_supercells": "1",
        "roughness_num_realizations": "1",
        **{prefix + "_correlation_length_nm": "" for prefix in ("substrate", "layer", "material_a", "material_b", "top_cap")},
        "period_lpermm": "400",
        "grating_type": "laminar",
        "x_resolution_nm": "1.0",
        "z_resolution_nm": "1.0",
        "width_to_period_ratio": "0.67",
        "depth_nm": "14.9",
        "sinusoidal_depth_nm": "20.0",
        "left_wall_angle_deg": "15.0",
        "right_wall_angle_deg": "15.0",
        "blaze_angle_deg": "0.75",
        "anti_blaze_angle_deg": "",
        "substrate_material": "Si",
        "substrate_material_density_g_cm3": _default_density_text("Si"),
        "stack_type": "single_layer",
        "layer_material": "Pt",
        "layer_material_density_g_cm3": _default_density_text("Pt"),
        "layer_thickness_nm": "28.77",
        "material_a": "Cr",
        "material_a_density_g_cm3": _default_density_text("Cr"),
        "material_b": "C",
        "material_b_density_g_cm3": _default_density_text("C"),
        "d_period_nm": "6.5",
        "gamma": "0.45",
        "n_bilayers": "40",
        "top_material": "C",
        "top_material_density_g_cm3": _default_density_text("C"),
        "top_cap_material": "",
        "top_cap_material_density_g_cm3": "",
        "top_cap_thickness_nm": "0.0",
        "substrate_roughness_sigma_nm": "",
        "layer_roughness_sigma_nm": "",
        "material_a_roughness_sigma_nm": "",
        "material_b_roughness_sigma_nm": "",
        "top_cap_roughness_sigma_nm": "",
    }


def _default_density_text(material_name: str) -> str:
    """Return the default density string for one built-in material."""

    density = material_density_g_cm3(material_name)
    return "" if density is None else str(density)


def _material_density_placeholders(defaults: dict[str, str]) -> dict[str, str]:
    """Return density placeholders for the currently selected materials."""
    placeholders = {}
    for field_name in (
        "substrate_material",
        "layer_material",
        "material_a",
        "material_b",
        "top_material",
        "top_cap_material",
    ):
        material_name = defaults.get(field_name, "")
        density = material_density_g_cm3(material_name)
        placeholders[field_name] = "" if density is None else str(density)
    return placeholders


def _form_values_from_spec(spec: dict[str, Any]) -> dict[str, str]:
    """Return form defaults populated from a saved grating spec."""
    values = _default_form_values()
    values.update({key: "" if value is None else str(value) for key, value in spec.items()})
    roughness = spec.get("roughness") or {}
    values["roughness_kind"] = roughness.get("kind", "none")
    for field in ("seed", "num_supercells", "num_realizations"):
        if field in roughness:
            values["roughness_" + field] = str(roughness[field])
    if spec.get("grating_type") == "sinusoidal":
        values["sinusoidal_depth_nm"] = str(spec.get("depth_nm", "20.0"))
    if spec.get("grating_type") == "afm":
        profile = spec.get("profile") or {}
        values.update({
            "afm_profile_path": str(profile.get("profile_path") or ""),
            "afm_source_filename": str(profile.get("source_filename") or ""),
            "afm_units": str(profile.get("units") or "nm"),
            "afm_profile_type": str(profile.get("profile_type") or "blazed"),
            "afm_period_nm": str(profile.get("period_nm") or "0"),
        })
        settings = profile.get("import_settings") or {}
        for key, field in (
            ("min_separation_fraction", "afm_min_separation"),
            ("min_prominence_fraction", "afm_min_prominence"),
            ("period_index", "afm_period_index"),
        ):
            if key in settings:
                values[field] = str(settings[key])
        for key, field in (
            ("average", "afm_average"), ("reverse", "afm_reverse"),
            ("zero_baseline", "afm_zero_baseline"),
            ("periodicity_ramp", "afm_periodicity_ramp"),
        ):
            if key in settings:
                values[field] = "1" if settings[key] else "0"
    stack = dict(spec.get("stack", {}))
    values["stack_type"] = str(stack.get("type", "single_layer"))
    for key, value in stack.items():
        if key == "layers_bottom_up":
            continue
        if isinstance(value, dict):
            values[key] = "" if value.get("name") is None else str(value.get("name"))
            density_key = f"{key}_density_g_cm3"
            density_value = value.get("density_g_cm3")
            if density_value in (None, ""):
                values[density_key] = _default_density_text(str(value.get("name", "")))
            else:
                values[density_key] = str(density_value)
            continue
        values[key] = "" if value is None else str(value)
        if key in {
            "substrate_material",
            "layer_material",
            "material_a",
            "material_b",
            "top_material",
            "top_cap_material",
        }:
            density_key = f"{key}_density_g_cm3"
            values[density_key] = _default_density_text(values[key])
    return values


def _grating_custom_layer_rows(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Return saved custom coating layers in form display order."""
    stack = dict(spec.get("stack", {}))
    if "blocks_top_down" in stack:
        return stack["blocks_top_down"]
    rows = []
    for layer in reversed(stack.get("layers_bottom_up", [])):
        material = layer.get("material", {})
        if isinstance(material, dict):
            name = str(material.get("name", ""))
            density = material.get("density_g_cm3")
        else:
            name = str(material)
            density = None
        sigma = layer.get("roughness_sigma_nm")
        rows.append({
            "material": name,
            "density_g_cm3": _default_density_text(name) if density in (None, "") else str(density),
            "thickness_nm": str(layer.get("thickness_nm", "")),
            "roughness_sigma_nm": "" if sigma is None else str(sigma),
            "correlation_length_nm": "" if layer.get("correlation_length_nm") is None else str(layer["correlation_length_nm"]),
        })
    return rows or [{
        "material": "", "density_g_cm3": "", "thickness_nm": "", "roughness_sigma_nm": ""
    }]


def _persist_afm_artifacts(
    spec: dict[str, Any], data_dir: Path, *, previous: dict[str, Any] | None = None,
) -> None:
    """Keep the processed profile and original scan beside the saved grating."""

    profile = spec["profile"]
    source = Path(str(profile["profile_path"])).resolve()
    if not source.is_file() or not source.is_relative_to(data_dir.resolve()):
        raise ValueError("The processed AFM profile is unavailable in this workspace.")
    destination = (data_dir / "saved_gratings" / str(spec["id"])).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    saved_profile = destination / "profile.csv"
    if source != saved_profile:
        shutil.copy2(source, saved_profile)
    source_upload = source.parent / "source_upload"
    if source_upload.is_file() and source_upload != destination / "source_upload":
        shutil.copy2(source_upload, destination / "source_upload")
    metadata_path = source.parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        profile["import_settings"] = metadata.get("settings") or {}
    elif previous and source == Path(str((previous.get("profile") or {}).get("profile_path", ""))).resolve():
        profile["import_settings"] = (previous.get("profile") or {}).get("import_settings") or {}
    profile["profile_path"] = str(saved_profile)


def _spec_from_form(form: Any, *, require_name: bool = True) -> dict[str, Any]:
    """Build a saved grating spec from submitted form data."""
    name = str(form.get("name") or "").strip()
    if require_name and not name:
        raise ValueError("Enter a grating name before saving.")
    grating_type = str(form["grating_type"])
    for key, label in (("period_lpermm", "Period (lines/mm)"),
                       ("x_resolution_nm", "X resolution (nm)"),
                       ("z_resolution_nm", "Z resolution (nm)")):
        try:
            value = float(form[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{label} must be a finite number greater than 0.") from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a finite number greater than 0.")
        if key == "period_lpermm" and not value.is_integer():
            raise ValueError("Period (lines/mm) must be a whole number of at least 1.")
    spec: dict[str, Any] = {
        "name": name or "Preview grating",
        "grating_type": grating_type,
        "period_lpermm": int(float(form["period_lpermm"])),
        "x_resolution_nm": float(form["x_resolution_nm"]),
        "z_resolution_nm": float(form["z_resolution_nm"]),
        "stack": _stack_spec_from_form(form),
        "roughness": roughness_from_form(form),
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
    if grating_type == "afm":
        profile_path = str(form.get("afm_profile_path", "")).strip()
        if not profile_path or not Path(profile_path).is_file():
            raise ValueError("A processed AFM profile is required before saving.")
        spec["profile"] = {
            "profile_path": profile_path,
            "source_filename": str(form.get("afm_source_filename", "")),
            "units": str(form.get("afm_units", "nm")),
            "profile_type": str(form.get("afm_profile_type", "blazed")),
            "period_nm": float(form.get("afm_period_nm", "0")),
        }
        return spec
    if grating_type == "sinusoidal":
        try:
            depth_nm = float(form["sinusoidal_depth_nm"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "Sinusoidal peak-to-valley depth must be a finite number greater than 0 nm."
            ) from None
        if not np.isfinite(depth_nm) or depth_nm <= 0:
            raise ValueError(
                "Sinusoidal peak-to-valley depth must be a finite number greater than 0 nm."
            )
        spec["depth_nm"] = depth_nm
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


def _render_grating_previews(grating: Any, preview_path: Path) -> tuple[Path | None, list[str]]:
    """Keep the three-period nominal plot separate from the simulation realization."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    nominal = copy(grating)
    nominal.roughness = None
    nominal.plot_profile(preview_path)
    rough_path = None
    notices = []
    if grating._random_interface_active():
        sample = realization_grating(grating)
        rough_path = preview_path.with_name(f"{preview_path.stem}-rough.png")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample._warn_if_roughness_underresolved()
            sample.plot_roughness(rough_path)
        notices = [str(item.message) for item in caught if "underresolved" in str(item.message)]
    return rough_path, notices


def _build_grating_preview(
    *,
    data_dir: Path,
    form_data: Any,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a live grating preview payload from form data."""
    try:
        grating = build_grating_from_spec(_spec_from_form(form_data, require_name=False) if spec is None else spec)
        preview_id = uuid4().hex
        preview_path = data_dir / "previews" / "live" / f"{preview_id}.png"
        rough_path, notices = _render_grating_previews(grating, preview_path)
        rough_url = f"/_data/previews/live/{rough_path.name}" if rough_path else None
        return {
            "ok": True, "error": "", "error_category": None, "error_fields": [], "preview_id": preview_id,
            "preview_url": f"/_data/previews/live/{preview_id}.png",
            "roughness_preview_url": rough_url, "warnings": notices,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        message = str(error)
        is_geometry_error = "LaminarGrating walls overlap" in message
        return {
            "ok": False,
            "error": message,
            "error_category": "geometry" if is_geometry_error else "inputs",
            "error_fields": (
                [
                    "depth_nm",
                    "left_wall_angle_deg",
                    "right_wall_angle_deg",
                    "width_to_period_ratio",
                ]
                if is_geometry_error
                else []
            ),
            "preview_id": None,
            "preview_url": None,
            "roughness_preview_url": None,
            "warnings": [],
        }


def _custom_layer_rows(stack_form: dict[str, Any]) -> list[dict[str, str]]:
    """Return the custom-layer table rows (top to bottom) for the page."""
    if "cl_material" not in stack_form:
        return [
            {**layer, "density_g_cm3": _default_density_text(layer["material"])}
            for layer in DEFAULT_CUSTOM_LAYERS
        ]
    names = ("material", "density_g_cm3", "thickness_nm", "roughness_sigma_nm",
             "kind", "material_b", "density_b_g_cm3", "thickness_b_nm",
             "roughness_b_sigma_nm", "repeats", "correlation_length_nm", "correlation_b_length_nm")
    rows = []
    for index in range(len(stack_form["cl_material"])):
        row = {}
        for name in names:
            values = stack_form.get("cl_" + name, [])
            default = "single" if name == "kind" else "1" if name == "repeats" else ""
            row[name] = values[index] if index < len(values) else default
        rows.append(row)
    return rows


def _stack_spec_from_form(form: Any) -> dict[str, Any]:
    spec = _stack_spec_from_form_base(form)
    for prefix in ("substrate", "layer", "material_a", "material_b", "top_cap"):
        key = prefix + "_correlation_length_nm"
        if key in form:
            spec[key] = optional_nonnegative(form.get(key), prefix.replace("_", " ") + " correlation length")
    return spec


def _stack_spec_from_form_base(form: Any) -> dict[str, Any]:
    """Build a stack spec from submitted form data."""
    stack_type = str(form["stack_type"])
    if stack_type == "bare":
        return {
            "type": "bare",
            "substrate_material": _material_spec_from_form(form, "substrate_material"),
            "substrate_roughness_sigma_nm": _optional_float(form, "substrate_roughness_sigma_nm"),
        }
    if stack_type == "custom":
        spec = stack_to_spec(custom_stack_from_form(form))
        spec["blocks_top_down"] = _custom_layer_rows(form.to_dict(flat=False))
        return spec
    top_cap_thickness = float(form.get("top_cap_thickness_nm", 0.0) or 0.0)
    if stack_type == "multilayer":
        return {
            "type": "multilayer",
            "substrate_material": _material_spec_from_form(form, "substrate_material"),
            "material_a": _material_spec_from_form(form, "material_a"),
            "material_b": _material_spec_from_form(form, "material_b"),
            "d_period_nm": float(form["d_period_nm"]),
            "gamma": float(form["gamma"]),
            "n_bilayers": int(float(form["n_bilayers"])),
            "top_material": _material_spec_from_form(form, "top_material"),
            "top_cap_material": _optional_material_spec_from_form(form, "top_cap_material"),
            "top_cap_thickness_nm": top_cap_thickness,
            "substrate_roughness_sigma_nm": _optional_float(form, "substrate_roughness_sigma_nm"),
            "material_a_roughness_sigma_nm": _optional_float(form, "material_a_roughness_sigma_nm"),
            "material_b_roughness_sigma_nm": _optional_float(form, "material_b_roughness_sigma_nm"),
            "top_cap_roughness_sigma_nm": _optional_float(form, "top_cap_roughness_sigma_nm"),
        }
    return {
        "type": "single_layer",
        "substrate_material": _material_spec_from_form(form, "substrate_material"),
        "layer_material": _material_spec_from_form(form, "layer_material"),
        "layer_thickness_nm": float(form["layer_thickness_nm"]),
        "top_cap_material": _optional_material_spec_from_form(form, "top_cap_material"),
        "top_cap_thickness_nm": top_cap_thickness,
        "substrate_roughness_sigma_nm": _optional_float(form, "substrate_roughness_sigma_nm"),
        "layer_roughness_sigma_nm": _optional_float(form, "layer_roughness_sigma_nm"),
        "top_cap_roughness_sigma_nm": _optional_float(form, "top_cap_roughness_sigma_nm"),
    }


def _optional_float(form: Any, field_name: str) -> float | None:
    """Return an optional float from a form field, treating blanks as unset."""
    text = str(form.get(field_name, "")).strip()
    if text == "":
        return None
    return float(text)


def _material_spec_from_form(form: Any, field_name: str) -> dict[str, Any]:
    """Return one serialized material spec from form fields."""
    name = str(form[field_name]).strip()
    density_text = str(form.get(f"{field_name}_density_g_cm3", "")).strip()
    density = None if density_text == "" else float(density_text)
    return {"name": name, "density_g_cm3": density}


def _optional_material_spec_from_form(form: Any, field_name: str) -> dict[str, Any] | None:
    """Return one optional serialized material spec from form fields."""
    name = str(form.get(field_name, "")).strip()
    if name == "":
        return None
    return _material_spec_from_form(form, field_name)


def _attach_roughness(grating: Any, form_data: Any) -> None:
    """Apply a run-only model override, retaining saved interface parameters."""
    if "roughness_kind" not in form_data:
        return
    kind = str(form_data.get("roughness_kind", "none") or "none").strip()
    if kind == "none":
        grating.roughness = None
    else:
        from grax import RoughnessSpec
        saved = grating.roughness
        if saved is not None and saved.kind == kind:
            return
        if saved is None:
            grating.roughness = RoughnessSpec(
                kind=kind, sigma_nm=0.0, seed=0, num_realizations=1,
            )
        else:
            grating.roughness = replace(
                saved, kind=kind, num_supercells=1, num_realizations=1,
            )
