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
