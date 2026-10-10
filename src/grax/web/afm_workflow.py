"""Safe web-facing AFM upload and preprocessing helpers."""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from grax.afm_preprocessing import AFMPreprocessing

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 1_000_000
PREVIEW_POINTS = 1600


def parse_afm_upload(raw: bytes, *, filename: str = "", units: str = "nm") -> np.ndarray:
    """Parse a bounded, non-executable two-column AFM text upload."""
    if len(raw) > MAX_BYTES:
        raise ValueError("AFM file is too large (maximum 10 MiB).")
    if units not in {"nm", "um", "m"}:
        raise ValueError("AFM units must be nm, um, or m.")
    if not raw.strip():
        raise ValueError("AFM file is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("AFM file must be UTF-8 text.") from error
    rows: list[tuple[float, float]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [part for part in line.replace(",", " ").replace(";", " ").split() if part]
        if len(fields) < 2:
            if not rows and any(ch.isalpha() for ch in line):
                continue
            raise ValueError(f"AFM file line {line_number} must contain x and z values.")
        try:
            x, z = float(fields[0]), float(fields[1])
        except ValueError:
            if not rows and any(ch.isalpha() for ch in line):
                continue
            raise ValueError(f"AFM file line {line_number} is not numeric.") from None
        rows.append((x, z))
        if len(rows) > MAX_ROWS:
            raise ValueError("AFM file contains too many rows.")
    if len(rows) < 3:
        raise ValueError("AFM file must contain at least three numeric rows.")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("AFM file contains non-finite values.")
    return data


def _preview_curve(x: np.ndarray, z: np.ndarray) -> dict[str, list[float]]:
    """Bound the browser payload while retaining endpoints and sharp extrema."""

    if len(x) <= PREVIEW_POINTS:
        indices = np.arange(len(x))
    else:
        boundaries = np.linspace(1, len(x) - 1, (PREVIEW_POINTS - 2) // 2 + 1, dtype=int)
        selected = [0]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop <= start:
                continue
            low = start + int(np.argmin(z[start:stop]))
            high = start + int(np.argmax(z[start:stop]))
            selected.extend(sorted({low, high}))
        indices = np.asarray([*selected, len(x) - 1], dtype=int)
    return {"x": x[indices].tolist(), "z": z[indices].tolist()}


def estimate_afm_period(x_nm: np.ndarray, z_nm: np.ndarray) -> float | None:
    """Suggest a period from regularly spaced prominent peaks or troughs.

    An irregular or too-short scan has no trustworthy automatic estimate; the
    user can still enter the expected period manually.
    """

    from scipy.signal import find_peaks

    if len(x_nm) < 7 or not (np.all(np.diff(x_nm) > 0) or np.all(np.diff(x_nm) < 0)):
        return None
    height = float(np.ptp(z_nm))
    if not np.isfinite(height) or height <= 0:
        return None
    candidates: list[tuple[float, int, float]] = []
    for signed_z in (z_nm, -z_nm):
        peaks, _ = find_peaks(
            signed_z, prominence=0.25 * height, distance=max(3, len(z_nm) // 100)
        )
        if len(peaks) < 3:
            continue
        spacing = np.abs(np.diff(x_nm[peaks]))
        period = float(np.median(spacing))
        if period <= 0:
            continue
        relative_spread = float(np.median(np.abs(spacing - period)) / period)
        if relative_spread <= 0.25:
            candidates.append((relative_spread, -len(peaks), period))
    return min(candidates)[2] if candidates else None


def preview_afm_upload(
    raw: bytes,
    *,
    units: str,
    profile_type: str,
    period_nm: float | None,
    min_separation_fraction: float = 0.4,
    min_prominence_fraction: float = 0.1,
    period_index: int = 0,
    average: bool = False,
    reverse: bool = False,
    zero_baseline: bool = True,
    periodicity_ramp: bool = False,
) -> dict[str, Any]:
    """Return in-memory traces for each visible AFM preprocessing stage."""

    data = parse_afm_upload(raw, units=units)
    factor = {"nm": 1.0, "um": 1e3, "m": 1e9}[units]
    raw_x = data[:, 0] * factor
    raw_z = data[:, 1] * factor
    if not np.all(np.isfinite(raw_x)) or not np.all(np.isfinite(raw_z)):
        raise ValueError("AFM coordinates exceed the supported range after unit conversion.")
    traces: dict[str, Any] = {"raw": _preview_curve(raw_x, raw_z)}
    estimated_period_nm = estimate_afm_period(raw_x, raw_z) if period_nm is None else None
    # AFM exports in metres often contain ~1e-5 x and ~1e-8 z values.
    # A nanometre interpretation would make the entire scan sub-atomic.
    suggested_units = (
        "m" if units == "nm" and 0 < np.ptp(data[:, 0]) < 1e-2
        and 0 < np.ptp(data[:, 1]) < 1e-3 else None
    )

    x = raw_x.copy()
    z = raw_z.copy()
    if reverse:
        x = x[::-1].copy()
        z = z[::-1].copy()
        x = x[0] - x
        traces["reverse"] = _preview_curve(x, z)
    if zero_baseline:
        z = z - float(np.min(z))
        traces["baseline"] = _preview_curve(x, z)

    if period_nm is None:
        return {
            "traces": traces,
            "estimated_period_nm": estimated_period_nm,
            "suggested_units": suggested_units,
            "stage_message": (
                "Check the estimated period and coordinate units; you can correct the value."
                if estimated_period_nm is not None else
                "Could not estimate a regular period. Enter the expected period manually."
            ),
        }
    try:
        afm = AFMPreprocessing(data, units=units, save_plots=False, show_plots=False)
        afm.normalize_scan(reverse=reverse, zero_baseline=zero_baseline)
        afm.find_troughs(
            period_nm=period_nm,
            profile_type=profile_type,
            min_separation_fraction=min_separation_fraction,
            min_prominence_fraction=min_prominence_fraction,
        )
        assert afm.trough_indices is not None
        trough_curve: dict[str, Any] = _preview_curve(afm.x_nm, afm.z_nm)
        if len(afm.trough_indices):
            trough_curve["markers"] = _preview_curve(
                afm.x_nm[afm.trough_indices], afm.z_nm[afm.trough_indices]
            )
        if not average and len(afm.trough_indices) > period_index + 1 and period_index >= 0:
            trough_curve["selection"] = [
                float(afm.x_nm[afm.trough_indices[period_index]]),
                float(afm.x_nm[afm.trough_indices[period_index + 1]]),
            ]
        traces["troughs"] = trough_curve
        afm.extract_period(period_index=period_index, average=average)
        assert afm.period_x is not None and afm.period_z is not None
        extracted = _preview_curve(afm.period_x * period_nm, afm.period_z)
        if average:
            individuals = []
            for start, stop in zip(afm.trough_indices[:-1], afm.trough_indices[1:]):
                segment_x = afm.x_nm[start : stop + 1]
                span = float(segment_x[-1] - segment_x[0])
                if span > 0:
                    individuals.append(_preview_curve(
                        (segment_x - segment_x[0]) * period_nm / span,
                        afm.z_nm[start : stop + 1],
                    ))
            extracted["individuals"] = individuals[:50]
        traces["average" if average else "period"] = extracted
        if periodicity_ramp:
            afm.apply_periodicity_ramp()
            assert afm.period_x is not None and afm.period_z is not None
            traces["ramp"] = {
                **_preview_curve(afm.period_x * period_nm, afm.period_z),
                "before": extracted,
            }
        afm.rescale_period(period_nm=period_nm)
        final_x, final_z = afm.get_profile()
        traces["rescaled"] = _preview_curve(final_x, final_z)
    except (RuntimeError, ValueError) as error:
        return {"traces": traces, "suggested_units": suggested_units, "stage_message": str(error)}
    return {"traces": traces, "suggested_units": suggested_units, "stage_message": ""}


def process_afm_upload(
    raw: bytes,
    output_dir: Path,
    *,
    filename: str,
    units: str,
    profile_type: str,
    period_nm: float,
    min_separation_fraction: float = 0.4,
    min_prominence_fraction: float = 0.1,
    period_index: int = 0,
    average: bool = False,
    reverse: bool = False,
    zero_baseline: bool = True,
    periodicity_ramp: bool = False,
) -> dict[str, Any]:
    """Run the complete AFM pipeline and write safe profile/diagnostic artifacts."""
    if profile_type not in {"laminar", "blazed"}:
        raise ValueError("AFM profile type must be laminar or blazed.")
    data = parse_afm_upload(raw, filename=filename, units=units)
    if not np.isfinite(period_nm) or period_nm <= 0:
        raise ValueError("Target AFM period must be greater than zero.")
    output_dir.mkdir(parents=True, exist_ok=True)
    afm = AFMPreprocessing(data, units=units, results_folder=output_dir / "diagnostics", save_plots=True, show_plots=False)
    afm.normalize_scan(reverse=reverse, zero_baseline=zero_baseline)
    afm.find_troughs(period_nm=period_nm, profile_type=profile_type, min_separation_fraction=min_separation_fraction, min_prominence_fraction=min_prominence_fraction)
    afm.extract_period(period_index=period_index, average=average)
    if periodicity_ramp:
        afm.apply_periodicity_ramp()
    afm.rescale_period(period_nm=period_nm)
    x_points, z_points = afm.get_profile()
    profile_path = output_dir / "profile.csv"
    np.savetxt(profile_path, np.column_stack((x_points, z_points)), delimiter=",", header="x_nm,z_nm", comments="")
    return {"id": output_dir.name, "source_filename": filename, "units": units, "profile_type": profile_type, "period_nm": period_nm, "profile_path": str(profile_path.name), "diagnostics": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.png")), "points": int(len(x_points))}


def load_profile_artifact(profile_dir: Path, profile_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a previously validated profile artifact."""
    path = (profile_dir / profile_name).resolve()
    if profile_dir.resolve() not in path.parents or path.suffix.lower() != ".csv":
        raise ValueError("Invalid AFM profile artifact.")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("AFM profile artifact is invalid.")
    return data[:, 0], data[:, 1]
