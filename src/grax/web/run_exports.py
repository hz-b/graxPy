"""Run downloads: CSV/DAT tables, YAML metadata and ZIP archives."""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    _grax_version,
)


def _run_export_rows(
    run_dir: Path,
    *,
    workflow: str,
) -> tuple[list[str], list[list[str]]]:
    """Read the canonical data rows that should be offered for one run."""

    if workflow == "parameter_study":
        paths = sorted(run_dir.glob("parameter_study_*.csv"))
    else:
        paths = [run_dir / "all_orders.csv"]
        if not paths[0].exists():
            paths = [run_dir / "summary.csv"]

    header: list[str] = []
    rows: list[list[str]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                current_header = next(reader)
            except StopIteration:
                continue
            if not header:
                header = current_header
            if current_header != header:
                raise ValueError(
                    "Cannot combine exported files with different columns: "
                    f"{path.name}"
                )
            rows.extend(list(reader))
    return header, rows


def _run_export_table(header: list[str], rows: list[list[str]], *, delimiter: str) -> str:
    """Serialize exported rows using the requested delimiter."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _run_export_archive(
    manifest: dict[str, Any],
    run_dir: Path,
    formats: set[str],
) -> bytes:
    """Build a ZIP archive containing the selected run export formats."""

    workflow = str(manifest.get("workflow", ""))
    header: list[str] = []
    rows: list[list[str]] = []
    if formats & {"csv", "dat"}:
        header, rows = _run_export_rows(run_dir, workflow=workflow)
        if not header:
            raise ValueError("No exported data are available for this run yet.")

    run_id = str(manifest["id"])
    archive_buffer = io.BytesIO()
    # Keep the scientific data byte-for-byte uncompressed inside the archive.
    # This makes the selected CSV/DAT/YAML files immediately inspectable and
    # avoids spending CPU time compressing already compact tabular data.
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        if "csv" in formats:
            archive.writestr(f"{run_id}.csv", _run_export_table(header, rows, delimiter=","))
        if "dat" in formats:
            archive.writestr(f"{run_id}.dat", _run_export_table(header, rows, delimiter="\t"))
        if "yaml" in formats:
            archive.writestr(f"{run_id}_metadata.yaml", _run_metadata_yaml(manifest, run_dir))
    return archive_buffer.getvalue()


def _run_metadata_payload(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Return stable, user-facing metadata for a downloadable run export."""

    run_input = dict(manifest.get("run_input") or {})
    grating = dict(manifest.get("grating_spec") or {})
    grating.setdefault("id", manifest.get("grating_id"))
    grating.setdefault("name", manifest.get("grating_name"))
    workflow = str(manifest.get("workflow", "run"))
    data_filename = f"{manifest['id']}.csv"
    metadata_filename = f"{manifest['id']}_metadata.yaml"
    header, _ = _run_export_rows(run_dir, workflow=workflow)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "grax_version": _grax_version(),
        "run": {
            "id": manifest.get("id"),
            "name": manifest.get("display_name"),
            "workflow": workflow,
            "created_at": manifest.get("created_at"),
            "status": manifest.get("status"),
            "comment": manifest.get("comment", ""),
        },
        "grating": grating,
        "simulation": {
            "polarization": manifest.get("polarization", run_input.get("polarization", "p")),
            "solver": manifest.get("solver", run_input.get("solver", "rcwa")),
            "diffraction_order": run_input.get("diffraction_order"),
            "input": run_input,
        },
        "results": {
            "completed_cases": len(manifest.get("cases") or []),
            "total_points": manifest.get("total_points"),
            "columns": header,
        },
        "files": {
            "csv": data_filename,
            "dat": f"{manifest['id']}.dat",
            "metadata": metadata_filename,
        },
    }
    if workflow == "parameter_study":
        payload["study"] = {
            "type": "parameter_study",
            "purpose": "Numerical-stability verification only",
            "configuration": {
                key: value for key, value in run_input.items() if key.startswith("study_")
            },
        }
    elif workflow == "custom_grid":
        payload["custom_grid"] = manifest.get("custom_grid") or run_input.get(
            "custom_grid_config"
        )
    if manifest.get("parameter_sweep"):
        payload["parameter_sweep"] = manifest["parameter_sweep"]
    return payload


def _run_metadata_yaml(manifest: dict[str, Any], run_dir: Path) -> str:
    """Serialize run metadata as human-readable, machine-readable YAML."""

    import yaml

    return yaml.safe_dump(
        _run_metadata_payload(manifest, run_dir),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
