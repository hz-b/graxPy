"""Plane-mirror page helpers: saved mirrors, form defaults and scan summaries."""

from __future__ import annotations

from typing import Any

from .common import (
    _polarization_display_label,
)
from .grating_forms import (
    _default_density_text,
    _default_form_values,
    _stack_spec_from_form,
)
from .persistence import (
    PlaneMirrorStore,
    build_stack_from_spec,
)
from .plane_mirror import (
    PLANE_MIRROR_DEFAULTS,
    PLANE_MIRROR_STACK_DEFAULTS,
    compute_plane_mirror,
    custom_stack_from_form,
    parse_plane_mirror_options,
)


def _load_plane_mirror(store: PlaneMirrorStore, mirror_id: str) -> dict[str, Any]:
    """Load one saved plane mirror or answer 404."""
    from flask import abort

    try:
        return store.load(mirror_id)
    except (FileNotFoundError, ValueError):
        abort(404)


def _plane_mirror_page_defaults(stack_form: dict[str, Any], scan_form: dict[str, str]) -> dict[str, str]:
    """Return template defaults for the plane-mirror page, overlaid with saved values."""
    defaults = _default_form_values()
    defaults.update(PLANE_MIRROR_STACK_DEFAULTS)
    for field_name in ("material_a", "material_b", "top_material"):
        defaults[f"{field_name}_density_g_cm3"] = _default_density_text(defaults[field_name])
    defaults.update(PLANE_MIRROR_DEFAULTS)
    defaults.update({key: value for key, value in stack_form.items() if isinstance(value, str)})
    for field_name in (
        "substrate_material",
        "layer_material",
        "material_a",
        "material_b",
        "top_material",
        "top_cap_material",
    ):
        density_key = f"{field_name}_density_g_cm3"
        if field_name in stack_form and density_key not in stack_form:
            defaults[density_key] = _default_density_text(defaults[field_name])
    defaults.update(scan_form)
    return defaults


def _plane_mirror_result(form: Any) -> dict[str, Any]:
    """Validate the plane-mirror form and compute its reflectivity curves."""
    return compute_plane_mirror(_plane_mirror_stack(form), parse_plane_mirror_options(form))


def _plane_mirror_scan_parameter_details(scan_form: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the saved scan settings in a concise, readable list."""

    values = {**PLANE_MIRROR_DEFAULTS, **scan_form}
    mode = str(values.get("scan_mode", "energy"))
    mode_labels = {
        "energy": "Energy scan at fixed grazing angle",
        "energy_sweep": "Energy sweep",
        "angle": "Angle scan at fixed photon energy",
        "angle_sweep": "Angle sweep",
        "map": "Angle vs energy map",
        "monochromator": "Monochromator",
    }
    details = [("Scan", mode_labels.get(mode, mode))]
    if mode in {"energy", "energy_sweep", "monochromator", "map"}:
        points_key = "map_energy_points" if mode == "map" else "energy_points"
        details.append(("Photon energy", f"{values['energy_min']}–{values['energy_max']} eV · {values[points_key]} points"))
    if mode in {"energy", "energy_sweep"}:
        details.append(("Grazing angle", f"{values['fixed_angle_deg']}°"))
    if mode == "monochromator":
        details.extend([
            ("Coupled grating density", f"{values['grating_density_lpermm']} lines/mm"),
            ("CFF", str(values["cff"])),
            ("Diffraction order", str(values["diffraction_order"])),
        ])
    if mode in {"angle", "angle_sweep", "map"}:
        points_key = "map_angle_points" if mode == "map" else "angle_points"
        details.append(("Grazing angle", f"{values['angle_min']}–{values['angle_max']}° · {values[points_key]} points"))
    if mode in {"angle", "angle_sweep"}:
        details.append(("Photon energy", f"{values['fixed_energy_ev']} eV"))
    polarization = str(values.get("polarization", "p")).lower()
    details.append(("Polarization", _polarization_display_label(polarization)))
    details.append(("Roughness", f"{values['roughness_sigma_nm']} nm"))
    grading = str(values.get("grading_mode", "none"))
    if grading == "linear":
        details.append(("Lateral grading", f"Linear · {values['grading_percent_per_mm']} %/mm"))
    elif grading == "polynomial":
        details.append(("Lateral grading", f"Polynomial · {values['grading_coefficients']}"))
    else:
        details.append(("Lateral grading", "None"))
    if grading != "none":
        details.append(("Footprint", f"{values['footprint_length_mm']} mm · {values['footprint_points']} positions"))
    return details


def _plane_mirror_variation_catalog(
    *, numeric_only: bool = False, custom_layer_count: int = 0
) -> list[dict[str, str]]:
    """Names of scan and stack settings accepted by the variation controls."""

    numeric = {
        "fixed_angle_deg": "Fixed grazing angle (deg)",
        "fixed_energy_ev": "Fixed photon energy (eV)",
        "grating_density_lpermm": "Coupled grating density (lines/mm)",
        "cff": "Monochromator CFF",
        "diffraction_order": "Diffraction order",
        "roughness_sigma_nm": "Scan roughness (nm)",
        "grading_percent_per_mm": "Grading (%/mm)",
        "footprint_length_mm": "Footprint length (mm)",
        "footprint_points": "Footprint sample points",
        "energy_points": "Energy sample points", "angle_points": "Angle sample points",
        "map_energy_points": "Map energy points", "map_angle_points": "Map angle points",
        "energy_min": "Energy start (eV)", "energy_max": "Energy stop (eV)",
        "angle_min": "Angle start (deg)", "angle_max": "Angle stop (deg)",
        "layer_thickness_nm": "Layer thickness (nm)", "d_period_nm": "Bilayer period (nm)",
        "gamma": "Gamma", "n_bilayers": "Bilayers", "top_cap_thickness_nm": "Top cap thickness (nm)",
        "material_a_density_g_cm3": "Material A density (g/cm³)",
        "material_b_density_g_cm3": "Material B density (g/cm³)",
        "layer_material_density_g_cm3": "Layer density (g/cm³)",
        "top_cap_material_density_g_cm3": "Top cap density (g/cm³)",
        "substrate_roughness_sigma_nm": "Substrate roughness (nm)",
        "material_a_roughness_sigma_nm": "Material A roughness (nm)",
        "material_b_roughness_sigma_nm": "Material B roughness (nm)",
        "layer_roughness_sigma_nm": "Layer roughness (nm)",
        "top_cap_roughness_sigma_nm": "Top cap roughness (nm)",
        "substrate_correlation_length_nm": "Substrate correlation length (nm)",
        "material_a_correlation_length_nm": "Material A correlation length (nm)",
        "material_b_correlation_length_nm": "Material B correlation length (nm)",
        "layer_correlation_length_nm": "Layer correlation length (nm)",
        "top_cap_correlation_length_nm": "Top cap correlation length (nm)",
    }
    categorical = {
        "polarization": "Polarization", "material_a": "Material A", "material_b": "Material B",
        "layer_material": "Layer material", "top_material": "Top material",
        "top_cap_material": "Top cap material", "substrate_material": "Substrate material",
        "grading_mode": "Grading mode",
    }
    result = [{"value": key, "label": label} for key, label in numeric.items()]
    custom_numeric = {
        "cl_thickness_nm": "Thickness (nm)", "cl_roughness_sigma_nm": "Roughness (nm)",
        "cl_correlation_length_nm": "Correlation length (nm)",
        "cl_thickness_b_nm": "B thickness (nm)", "cl_roughness_b_sigma_nm": "B roughness (nm)",
        "cl_correlation_b_length_nm": "B correlation length (nm)", "cl_repeats": "Bilayer repeats",
        "cl_density_g_cm3": "Density (g/cm³)", "cl_density_b_g_cm3": "B density (g/cm³)",
    }
    for index in range(custom_layer_count):
        result.extend(
            {"value": f"{field}[{index}]", "label": f"Custom layer {index + 1}: {label}"}
            for field, label in custom_numeric.items()
        )
    if not numeric_only:
        result.extend({"value": key, "label": label} for key, label in categorical.items())
        custom_categorical = {
            "cl_material": "Material", "cl_material_b": "B material", "cl_kind": "Block type",
        }
        for index in range(custom_layer_count):
            result.extend(
                {"value": f"{field}[{index}]", "label": f"Custom layer {index + 1}: {label}"}
                for field, label in custom_categorical.items()
            )
    return result


def _plane_mirror_stack(form: Any) -> Any:
    """Build the single-layer, multilayer or custom-layer stack from the form."""
    if str(form.get("stack_type", "")) == "custom":
        return custom_stack_from_form(form)
    return build_stack_from_spec(_stack_spec_from_form(form))
