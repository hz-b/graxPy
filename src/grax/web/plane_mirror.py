"""Form parsing and computation helpers for the web app's plane-mirror tab."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from grax.materials import MaterialSpec, validate_material_input
from grax.planar_mirror import LateralGrading, footprint_reflectivity, parratt_reflectivity
from grax.stacks import BaseStack, CustomStack, LayerSpec, assemble_custom_stack

MAX_SCAN_POINTS = 5000
MAX_FOOTPRINT_POINTS = 500
MAX_TOTAL_SAMPLES = 2_000_000
MAX_CUSTOM_LAYERS = 500
SCHEMATIC_MAX_LAYERS = 40

PLANE_MIRROR_DEFAULTS: dict[str, str] = {
    "scan_mode": "energy",
    "energy_min": "100",
    "energy_max": "6000",
    "energy_points": "1000",
    "angle_min": "0.1",
    "angle_max": "8",
    "angle_points": "5000",
    "map_energy_points": "1000",
    "map_angle_points": "1000",
    "fixed_angle_deg": "0.4",
    "fixed_energy_ev": "1000",
    "polarization": "s",
    "roughness_sigma_nm": "0",
    "grading_mode": "none",
    "grading_percent_per_mm": "0.05",
    "grading_coefficients": "1, 0.0005",
    "footprint_length_mm": "40",
    "footprint_points": "41",
}

PLANE_MIRROR_STACK_DEFAULTS: dict[str, str] = {
    "stack_type": "multilayer",
    "substrate_material": "Si",
    "material_a": "Ru",
    "material_b": "C",
    "d_period_nm": "7.0",
    "gamma": "0.4",
    "n_bilayers": "20",
    "top_material": "C",
}


# Default custom stack, listed top to bottom as the form shows it.
DEFAULT_CUSTOM_LAYERS: tuple[dict[str, str], ...] = (
    {"material": "Pt", "thickness_nm": "10", "roughness_sigma_nm": ""},
    {"material": "Cr", "thickness_nm": "3", "roughness_sigma_nm": ""},
)


def _material_from_text(name: str, density_text: str, label: str) -> MaterialSpec:
    """Return a validated material from a name and an optional density text."""

    name = name.strip()
    if name == "":
        raise ValueError(f"{label}: material is required.")
    try:
        density = None if density_text.strip() == "" else float(density_text)
    except ValueError as error:
        raise ValueError(f"{label}: density must be a number.") from error
    material = MaterialSpec(name, density)
    validate_material_input(material, field_name=label)
    return material


def _optional_sigma(text: str, label: str) -> float | None:
    """Return an optional non-negative roughness sigma from form text."""

    if text.strip() == "":
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label}: roughness must be a number.") from error
    if value < 0.0:
        raise ValueError(f"{label}: roughness must be >= 0.")
    return value


def custom_stack_from_form(form: Any) -> CustomStack:
    """Build an arbitrary layer stack from the repeated ``cl_*`` form fields.

    The form lists layers top to bottom; the returned stack is bottom-up.

    Raises:
        ValueError: If there are no layers, too many layers, or a field is invalid.
    """

    materials = form.getlist("cl_material")
    densities = form.getlist("cl_density_g_cm3")
    thicknesses = form.getlist("cl_thickness_nm")
    sigmas = form.getlist("cl_roughness_sigma_nm")
    count = len(materials)
    if count == 0:
        raise ValueError("Add at least one layer.")
    if count > MAX_CUSTOM_LAYERS:
        raise ValueError(f"At most {MAX_CUSTOM_LAYERS} layers are supported.")
    if not (len(densities) == len(thicknesses) == len(sigmas) == count):
        raise ValueError("Every layer needs material, density, thickness and roughness fields.")

    layers_top_down: list[LayerSpec] = []
    for index in range(count):
        label = f"Layer {index + 1}"
        try:
            thickness = float(thicknesses[index])
        except ValueError as error:
            raise ValueError(f"{label}: thickness must be a number.") from error
        layers_top_down.append(
            LayerSpec(
                material=_material_from_text(materials[index], densities[index], label),
                thickness_nm=thickness,
                roughness_sigma_nm=_optional_sigma(sigmas[index], label),
            )
        )

    cap_name = str(form.get("top_cap_material", "")).strip()
    cap_thickness_text = str(form.get("top_cap_thickness_nm", "")).strip()
    return assemble_custom_stack(
        substrate_material=_material_from_text(
            str(form.get("substrate_material", "")),
            str(form.get("substrate_material_density_g_cm3", "")),
            "Substrate",
        ),
        layers_bottom_up=layers_top_down[::-1],
        top_cap_material=(
            None
            if cap_name == ""
            else _material_from_text(cap_name, str(form.get("top_cap_material_density_g_cm3", "")), "Top cap")
        ),
        top_cap_thickness_nm=0.0 if cap_thickness_text == "" else float(cap_thickness_text),
        substrate_roughness_sigma_nm=_optional_sigma(str(form.get("substrate_roughness_sigma_nm", "")), "Substrate"),
        top_cap_roughness_sigma_nm=_optional_sigma(str(form.get("top_cap_roughness_sigma_nm", "")), "Top cap"),
    )


def stack_schematic_data_uri(stack: BaseStack) -> str:
    """Render the stack schematic (``BaseStack.plot_schematic``) as a PNG data URI."""

    import base64

    buffer = io.BytesIO()
    stack.plot_schematic(buffer, max_layers=SCHEMATIC_MAX_LAYERS)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True)
class PlaneMirrorOptions:
    """Validated scan, polarization, roughness and grading settings."""

    scan_mode: str
    energy_min: float
    energy_max: float
    energy_points: int
    angle_min: float
    angle_max: float
    angle_points: int
    fixed_angle_deg: float
    fixed_energy_ev: float
    polarization: str
    roughness_sigma_nm: float
    grading: LateralGrading | None
    footprint_length_mm: float
    footprint_points: int


def _too_many_points_message(
    scan_mode: str, energy_points: int, angle_points: int, footprint_points: int, positions: int
) -> str:
    """Explain a scan that exceeds the sample limit and suggest point counts that fit."""

    max_points = MAX_TOTAL_SAMPLES // positions
    per_position = f" per footprint position ({positions} positions)" if positions > 1 else ""
    if scan_mode == "map":
        total = energy_points * angle_points
        factor = (max_points / total) ** 0.5
        suggested_energy = max(1, min(MAX_SCAN_POINTS, int(energy_points * factor)))
        suggested_angle = max(1, min(MAX_SCAN_POINTS, int(angle_points * factor)))
        message = (
            f"The map has {energy_points:,} x {angle_points:,} = {total:,} energy x angle points"
            + (f", times {positions} footprint positions = {total * positions:,} samples" if positions > 1 else "")
            + f". The limit is {MAX_TOTAL_SAMPLES:,} samples, i.e. at most {max_points:,} energy x angle points{per_position}. "
            + f"For example use {suggested_energy:,} energy x {suggested_angle:,} angle points"
        )
        if positions > 1:
            message += ", or reduce the footprint sample points or turn grading off"
        return message + "."
    points = energy_points if scan_mode == "energy" else angle_points
    return (
        f"{points:,} points x {positions} footprint positions = {points * positions:,} samples exceeds the limit of "
        f"{MAX_TOTAL_SAMPLES:,}. With {positions} footprint positions use at most {min(max_points, MAX_SCAN_POINTS):,} points, "
        "or reduce the footprint sample points."
    )


def parse_plane_mirror_options(form: Any) -> PlaneMirrorOptions:
    """Parse and validate plane-mirror settings from submitted form data.

    Raises:
        ValueError: If any value is missing, malformed or out of range.
    """

    def text(name: str) -> str:
        value = str(form.get(name, PLANE_MIRROR_DEFAULTS[name])).strip()
        return value if value != "" else PLANE_MIRROR_DEFAULTS[name]

    def number(name: str) -> float:
        try:
            value = float(text(name))
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    scan_mode = text("scan_mode")
    if scan_mode not in {"energy", "angle", "map"}:
        raise ValueError("scan_mode must be 'energy', 'angle' or 'map'.")

    def axis(prefix: str, label: str, upper: float | None = None) -> tuple[float, float, int]:
        # Maps use their own (smaller) point counts; the ranges are shared.
        points_field = f"map_{prefix}_points" if scan_mode == "map" else f"{prefix}_points"
        low, high = number(f"{prefix}_min"), number(f"{prefix}_max")
        points = int(number(points_field))
        if not 1 <= points <= MAX_SCAN_POINTS:
            kind = "per axis of a map" if scan_mode == "map" else "per scan"
            raise ValueError(
                f"{label.capitalize()} points: {points:,} is out of range; use between 1 and {MAX_SCAN_POINTS:,} {kind}."
            )
        if low <= 0.0 or high < low:
            raise ValueError(f"The {label} range must be positive with max >= min.")
        if upper is not None and high > upper:
            raise ValueError(f"The {label} range is limited to {upper:g}.")
        return low, high, points

    energy_min, energy_max, energy_points = axis("energy", "energy") if scan_mode in {"energy", "map"} else (0.0, 0.0, 1)
    angle_min, angle_max, angle_points = axis("angle", "angle", 90.0) if scan_mode in {"angle", "map"} else (0.0, 0.0, 1)

    grading_mode = text("grading_mode")
    grading: LateralGrading | None
    if grading_mode == "none":
        grading = None
    elif grading_mode == "linear":
        grading = LateralGrading.linear(number("grading_percent_per_mm") / 100.0)
    elif grading_mode == "polynomial":
        try:
            coefficients = tuple(float(part) for part in text("grading_coefficients").replace(";", ",").split(",") if part.strip())
        except ValueError as error:
            raise ValueError("grading_coefficients must be comma-separated numbers.") from error
        grading = LateralGrading(coefficients=coefficients)
    else:
        raise ValueError("grading_mode must be 'none', 'linear' or 'polynomial'.")

    footprint_points = int(number("footprint_points"))
    if not 1 <= footprint_points <= MAX_FOOTPRINT_POINTS:
        raise ValueError(f"footprint_points must be between 1 and {MAX_FOOTPRINT_POINTS}.")
    footprint_length_mm = number("footprint_length_mm")
    if footprint_length_mm < 0.0:
        raise ValueError("footprint_length_mm must be >= 0.")
    positions = footprint_points if grading is not None else 1
    if energy_points * angle_points * positions > MAX_TOTAL_SAMPLES:
        raise ValueError(
            _too_many_points_message(scan_mode, energy_points, angle_points, footprint_points, positions)
        )

    return PlaneMirrorOptions(
        scan_mode=scan_mode,
        energy_min=energy_min,
        energy_max=energy_max,
        energy_points=energy_points,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_points=angle_points,
        fixed_angle_deg=number("fixed_angle_deg"),
        fixed_energy_ev=number("fixed_energy_ev"),
        polarization=text("polarization"),
        roughness_sigma_nm=number("roughness_sigma_nm"),
        grading=grading,
        footprint_length_mm=footprint_length_mm,
        footprint_points=footprint_points,
    )


def compute_plane_mirror(stack: BaseStack, options: PlaneMirrorOptions) -> dict[str, Any]:
    """Return the reflectivity curve or map for one stack.

    Returns:
        For ``energy`` and ``angle`` scans: ``mode``, ``x``, ``x_label``,
        ``reflectivity`` (central position, ungraded) and ``graded``
        (footprint average, or ``None``). For ``map``: ``mode``, ``x``
        (energies), ``y`` (angles), ``z`` (rows are angles, columns energies)
        and ``graded`` (whether ``z`` is the footprint average).
    """

    mode = options.scan_mode
    if mode == "energy":
        energies = np.linspace(options.energy_min, options.energy_max, options.energy_points)
        angles = np.array([options.fixed_angle_deg])
    elif mode == "angle":
        energies = np.array([options.fixed_energy_ev])
        angles = np.linspace(options.angle_min, options.angle_max, options.angle_points)
    else:
        energies = np.linspace(options.energy_min, options.energy_max, options.energy_points)
        angles = np.linspace(options.angle_min, options.angle_max, options.angle_points)

    common = {"polarization": options.polarization, "roughness_sigma_nm": options.roughness_sigma_nm}

    def graded_average() -> np.ndarray:
        half = 0.5 * options.footprint_length_mm
        return footprint_reflectivity(
            stack,
            energies,
            angles,
            grading=options.grading,
            positions_mm=np.linspace(-half, half, options.footprint_points),
            **common,
        )

    if mode == "map":
        grid = graded_average() if options.grading is not None else parratt_reflectivity(stack, energies, angles, **common)
        return {
            "mode": mode,
            "x": energies.tolist(),
            "y": angles.tolist(),
            "z": grid.T.tolist(),
            "graded": options.grading is not None,
        }

    def curve(values: np.ndarray) -> list[float]:
        return (values[:, 0] if mode == "energy" else values[0, :]).tolist()

    return {
        "mode": mode,
        "x": (energies if mode == "energy" else angles).tolist(),
        "x_label": "Photon energy (eV)" if mode == "energy" else "Grazing angle (deg)",
        "reflectivity": curve(parratt_reflectivity(stack, energies, angles, **common)),
        "graded": None if options.grading is None else curve(graded_average()),
    }


def plane_mirror_csv(result: dict[str, Any]) -> str:
    """Serialize a :func:`compute_plane_mirror` result as CSV text."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if result["mode"] == "map":
        writer.writerow(["energy_ev", "grazing_angle_deg", "reflectivity"])
        for angle, row in zip(result["y"], result["z"]):
            for energy, value in zip(result["x"], row):
                writer.writerow([repr(energy), repr(angle), repr(value)])
        return buffer.getvalue()
    header = [result["x_label"], "reflectivity"]
    if result["graded"] is not None:
        header.append("reflectivity_graded")
    writer.writerow(header)
    for index, x_value in enumerate(result["x"]):
        row = [repr(x_value), repr(result["reflectivity"][index])]
        if result["graded"] is not None:
            row.append(repr(result["graded"][index]))
        writer.writerow(row)
    return buffer.getvalue()


# Form fields that are not part of the stack: scan settings and the save controls.
_NON_STACK_FIELDS = frozenset(PLANE_MIRROR_DEFAULTS) | {"mirror_name", "mirror_label", "scan_name", "scan_label"}


def split_plane_mirror_form(form: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Split a submitted page form into stack fields and scan fields.

    Repeated fields (custom layer columns) stay lists; all other values are
    strings. Scan fields are returned for every key of
    :data:`PLANE_MIRROR_DEFAULTS` that was submitted.
    """

    stack_form: dict[str, Any] = {}
    scan_form: dict[str, str] = {}
    for key in form.keys():
        values = form.getlist(key)
        if key in PLANE_MIRROR_DEFAULTS:
            scan_form[key] = str(values[0])
        elif key not in _NON_STACK_FIELDS:
            stack_form[key] = list(values) if key.startswith("cl_") else str(values[0])
    return stack_form, scan_form


def clean_label(value: Any) -> str:
    """Return a stripped optional free-text label (at most 300 characters)."""

    return str(value or "").strip()[:300]


def clean_name(value: Any, label: str) -> str:
    """Return a stripped non-empty name, or raise ``ValueError``."""

    name = str(value or "").strip()
    if name == "":
        raise ValueError(f"{label} is required.")
    return name[:120]


def unique_scan_id(name: str, existing_ids: set[str]) -> str:
    """Return a slug of ``name`` that is not in ``existing_ids``."""

    import re

    base = re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-") or "scan"
    candidate, index = base, 2
    while candidate in existing_ids:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def plane_mirror_summary(spec: dict[str, Any]) -> str:
    """Return a one-line description of a saved plane mirror for list pages."""

    stack = spec.get("stack_form", {})
    stack_type = str(stack.get("stack_type", "single_layer"))
    substrate = str(stack.get("substrate_material", "")) or "?"
    if stack_type == "multilayer":
        what = (
            f"{stack.get('material_a', '?')}/{stack.get('material_b', '?')} multilayer, "
            f"{stack.get('n_bilayers', '?')} bilayers"
        )
    elif stack_type == "custom":
        what = f"{len(stack.get('cl_material', []))} custom layers"
    else:
        what = f"{stack.get('layer_material', '?')} single layer"
    count = len(spec.get("scans", []))
    return f"{what} on {substrate} · {count} scan{'s' if count != 1 else ''}"
