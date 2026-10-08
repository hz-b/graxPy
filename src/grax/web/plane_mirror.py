"""Form parsing and computation helpers for the web app's plane-mirror tab."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from grax.planar_mirror import LateralGrading, footprint_reflectivity, parratt_reflectivity
from grax.stacks import BaseStack

MAX_SCAN_POINTS = 5000
MAX_FOOTPRINT_POINTS = 500
MAX_TOTAL_SAMPLES = 2_000_000

PLANE_MIRROR_DEFAULTS: dict[str, str] = {
    "scan_mode": "energy",
    "scan_min": "150",
    "scan_max": "400",
    "scan_points": "500",
    "fixed_angle_deg": "20",
    "fixed_energy_ev": "259",
    "polarization": "s",
    "roughness_sigma_nm": "0.3",
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


@dataclass(frozen=True)
class PlaneMirrorOptions:
    """Validated scan, polarization, roughness and grading settings."""

    scan_mode: str
    scan_min: float
    scan_max: float
    scan_points: int
    fixed_angle_deg: float
    fixed_energy_ev: float
    polarization: str
    roughness_sigma_nm: float
    grading: LateralGrading | None
    footprint_length_mm: float
    footprint_points: int


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
    if scan_mode not in {"energy", "angle"}:
        raise ValueError("scan_mode must be 'energy' or 'angle'.")
    scan_min, scan_max = number("scan_min"), number("scan_max")
    scan_points = int(number("scan_points"))
    if not 1 <= scan_points <= MAX_SCAN_POINTS:
        raise ValueError(f"scan_points must be between 1 and {MAX_SCAN_POINTS}.")
    if scan_min <= 0.0 or scan_max < scan_min:
        raise ValueError("The scan range must be positive with max >= min.")
    if scan_mode == "angle" and scan_max > 90.0:
        raise ValueError("Angle scans are limited to 90 degrees.")

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
    if scan_points * footprint_points > MAX_TOTAL_SAMPLES:
        raise ValueError("scan_points x footprint_points is too large; reduce one of them.")

    return PlaneMirrorOptions(
        scan_mode=scan_mode,
        scan_min=scan_min,
        scan_max=scan_max,
        scan_points=scan_points,
        fixed_angle_deg=number("fixed_angle_deg"),
        fixed_energy_ev=number("fixed_energy_ev"),
        polarization=text("polarization"),
        roughness_sigma_nm=number("roughness_sigma_nm"),
        grading=grading,
        footprint_length_mm=footprint_length_mm,
        footprint_points=footprint_points,
    )


def compute_plane_mirror(stack: BaseStack, options: PlaneMirrorOptions) -> dict[str, Any]:
    """Return the scan axis and reflectivity curves for one stack.

    Returns:
        Dict with ``x``, ``x_label``, ``reflectivity`` (central position,
        ungraded), and ``graded`` (footprint average, or ``None``).
    """

    axis = np.linspace(options.scan_min, options.scan_max, options.scan_points)
    if options.scan_mode == "energy":
        energies, angles, x_label = axis, np.array([options.fixed_angle_deg]), "Photon energy (eV)"
    else:
        energies, angles, x_label = np.array([options.fixed_energy_ev]), axis, "Grazing angle (deg)"

    common = {"polarization": options.polarization, "roughness_sigma_nm": options.roughness_sigma_nm}
    nominal = parratt_reflectivity(stack, energies, angles, **common)
    nominal_curve = nominal[:, 0] if options.scan_mode == "energy" else nominal[0, :]

    graded_curve = None
    if options.grading is not None:
        half = 0.5 * options.footprint_length_mm
        graded = footprint_reflectivity(
            stack,
            energies,
            angles,
            grading=options.grading,
            positions_mm=np.linspace(-half, half, options.footprint_points),
            **common,
        )
        graded_curve = graded[:, 0] if options.scan_mode == "energy" else graded[0, :]

    return {
        "x": axis.tolist(),
        "x_label": x_label,
        "reflectivity": nominal_curve.tolist(),
        "graded": None if graded_curve is None else graded_curve.tolist(),
    }


def plane_mirror_csv(result: dict[str, Any]) -> str:
    """Serialize a :func:`compute_plane_mirror` result as CSV text."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
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
