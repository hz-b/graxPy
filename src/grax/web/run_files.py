"""Reading and writing saved run folders: manifests, checkpoints and result tables."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import parameter_scans
from .common import (
    _file_token,
    _is_safe_id,
    _normalized_polarization,
    _normalized_solver,
)
from .custom_grid import (
    plot_custom_grid,
    write_custom_grid_all_orders,
    write_custom_grid_summary,
)
from .runs import RunStore


def _run_result_location(data_dir: Path, run_id: str) -> Path:
    """Return the directory holding one run's saved artifacts."""
    return data_dir / "runs" / run_id


def _load_checkpoint_results(run_dir: Path) -> list[Any]:
    """Load deduplicated checkpointed case results for one run directory."""

    from grax.simulation.serialization import _load_checkpoint_case_results

    checkpoint_path = run_dir / "checkpoints" / "results.jsonl"
    if not checkpoint_path.exists():
        return []
    results = list(_load_checkpoint_case_results(checkpoint_path).values())
    return sorted(results, key=parameter_scans.result_order)


def _checkpoint_counts(run_dir: Path) -> tuple[int, int]:
    """Return completed and total point counts for one run directory."""

    manifest = _load_run_manifest(run_dir.parent.parent, run_dir.name)
    total_points = int((manifest or {}).get("total_points") or 0)
    completed_points = len(_load_checkpoint_results(run_dir))
    return completed_points, total_points


def _parameter_study_checkpoint_count(run_dir: Path) -> int:
    """Count successful, deduplicated parameter-study point checkpoints."""
    path = run_dir / "parameter_study_checkpoints.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                latest[str(record["point_id"])] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return sum(not bool(record.get("error", False)) for record in latest.values())


def _run_completed_points(manifest: dict[str, Any], checkpoint_completed_points: int) -> int:
    """Combine persisted study progress with case-backed run progress."""

    return max(
        checkpoint_completed_points,
        int(manifest.get("completed_points") or 0),
        len(manifest.get("cases") or []),
    )


def _manifest_cases(results: list[Any]) -> list[dict[str, Any]]:
    """Return manifest-ready case summaries for persisted run metadata."""

    if any("scan_id" in result.case_data for result in results):
        results = sorted(results, key=parameter_scans.result_order)
    cases: list[dict[str, Any]] = []
    for result in results:
        case = {
            "case_id": result.case_id,
            "energy_ev": result.energy_ev,
            "grazing_angle_deg": result.grazing_angle_deg,
            "selected_efficiency": result.selected_efficiency,
            "status": result.status,
            "error_message": result.error_message,
        }
        case_data = getattr(result, "case_data", {})
        if "scan_id" in case_data:
            case["scan_id"] = case_data["scan_id"]
            case["scan_parameters"] = case_data.get("scan_parameters", {})
        if "custom_parameters" in case_data:
            case["custom_parameters"] = case_data.get("custom_parameters")
            case["custom_parameter_order"] = case_data.get("custom_parameter_order", [])
            case["resolved_parameters"] = case_data.get("custom_resolved_parameters", {})
            case["resolved_parameter_order"] = case_data.get(
                "custom_resolved_parameter_order", []
            )
            case["structural_variant_id"] = case_data.get("custom_variant_id")
            case["structural_variant_name"] = case_data.get("custom_variant_name")
        cases.append(case)
    return cases


def _run_artifacts_for_results(run_dir: Path, *, include_plot: bool) -> list[str]:
    """Return the artifact filenames currently available for one run."""

    artifact_names = ["summary.csv", "all_orders.csv"]
    if include_plot and (run_dir / "selected_efficiency.png").exists():
        artifact_names.append("selected_efficiency.png")
    if (run_dir / "parameter_study.png").exists():
        artifact_names.append("parameter_study.png")
    artifact_names.extend(str(path.relative_to(run_dir)) for path in sorted((run_dir / "scans").glob("*/efficiency.png")))
    return artifact_names


def _parameter_study_artifacts(run_dir: Path) -> list[str]:
    """Return complete parameter-study artifacts currently present."""

    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_file()
        and (
            path.name in {"parameter_study.png", "error.txt"}
            or (path.name.startswith("parameter_study_") and path.suffix == ".csv")
        )
    )


def _replace_plot_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 5,
    retry_delay: float = 0.05,
) -> bool:
    """Replace a live plot, tolerating brief Windows file-sharing locks."""

    for attempt in range(attempts):
        try:
            source.replace(destination)
            return True
        except PermissionError:
            if attempt + 1 == attempts:
                return False
            time.sleep(retry_delay)
    return False


def _run_error_summary(workflow: str, error: Exception) -> str:
    """Return concise user-facing text for a failed run."""

    if workflow == "parameter_study":
        return f"Parameter study failed: {error}"
    return str(error)


def _persist_run_outputs(
    *,
    data_dir: Path,
    run_id: str,
    results: list[Any],
    diffraction_order: int,
    title: str,
    include_plot: bool,
) -> None:
    """Persist current partial or final run artifacts from collected results."""

    if not results:
        return
    from grax import simulation

    run_dir = data_dir / "runs" / run_id
    manifest = _load_run_manifest(data_dir, run_id) or {}
    if manifest.get("parameter_sweep"):
        parameter_scans.write_outputs(results, manifest["parameter_sweep"], run_dir, title=title, include_plot=include_plot)
        return
    is_custom_grid = any("custom_parameters" in result.case_data for result in results)
    if is_custom_grid:
        write_custom_grid_summary(results, run_dir / "summary.csv")
        write_custom_grid_all_orders(results, run_dir / "all_orders.csv")
    else:
        _write_summary_csv(results, run_dir / "summary.csv")
        simulation.write_all_orders_csv(results, run_dir / "all_orders.csv")
    if include_plot and is_custom_grid:
        plot_custom_grid(results, run_dir / "selected_efficiency.png", title=title)
    elif include_plot:
        simulation.plot_order_subset(
            results,
            run_dir / "selected_efficiency.png",
            diffraction_orders=[diffraction_order],
            title=title,
        )


def _load_run_manifest(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Load one run manifest if it exists."""

    if not _is_safe_id(run_id):
        return None
    manifest_path = data_dir / "runs" / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        # Polling may race an atomic manifest replacement. A transiently
        # unavailable manifest must not turn the status endpoint into a 500.
        return None
    payload.setdefault("id", run_id)
    payload.setdefault("display_name", f"{payload.get('grating_name', 'Run')} · {payload.get('workflow', 'run')}")
    payload.setdefault("polarization", _normalized_polarization(payload.get("run_input", {}).get("polarization", "p")))
    payload.setdefault("solver", _normalized_solver(payload.get("run_input", {}).get("solver", "rcwa")))
    if isinstance(payload.get("run_input"), dict):
        payload["run_input"].setdefault("polarization", payload["polarization"])
    return payload


def _update_manifest_fields(data_dir: Path, run_id: str, **updates: Any) -> dict[str, Any]:
    """Update one run manifest in place and return the saved payload."""

    store = RunStore(data_dir / "runs")
    payload = store.load(run_id)
    payload.update({key: value for key, value in updates.items() if value is not None})
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return store.save(payload)


def _preferred_run_plot_path(data_dir: Path, run_id: str) -> str | None:
    """Return the best available plot file for one run."""

    run_dir = data_dir / "runs" / run_id
    for candidate in ("selected_efficiency.png", "parameter_study.png", "live_progress.png"):
        if (run_dir / candidate).exists():
            return candidate
    return None


def _run_plot_url(*, data_dir: Path, run_id: str, relative_path: str | None, token: str) -> str | None:
    """Return the cache-busted plot URL for one run image."""

    if not relative_path:
        return None
    plot_path = data_dir / "runs" / run_id / relative_path
    if not plot_path.exists():
        return None
    suffix = f"?v={token or _file_token(plot_path)}"
    return f"/_data/runs/{run_id}/{relative_path}{suffix}"


def _write_summary_csv(results: list[Any], output_path: Path) -> None:
    """Write selected-order results to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "energy_ev",
                "grazing_angle_deg",
                "selected_efficiency",
                "selected_diffraction_angle_deg",
                "status",
                "error_message",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.energy_ev,
                    result.grazing_angle_deg,
                    result.selected_efficiency,
                    result.selected_diffraction_angle_deg,
                    result.status,
                    result.error_message or "",
                ]
            )


def _list_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Return persisted run manifests newest first."""
    return RunStore(run_dir).list()


def _available_orders(run_dir: Path) -> list[int]:
    """Return positive diffraction orders available for one run."""
    all_orders_path = run_dir / "all_orders.csv"
    orders: set[int] = set()
    grouped = bool((_load_run_manifest(run_dir.parent.parent, run_dir.name) or {}).get("parameter_sweep"))
    if all_orders_path.exists():
        with all_orders_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    value = int(float(row["order"]))
                    orders.add(value if grouped else abs(value))
                except (KeyError, TypeError, ValueError):
                    continue
    if not orders:
        for result in _load_checkpoint_results(run_dir):
            if getattr(result, "status", "") != "ok":
                continue
            for value in np.asarray(getattr(result, "orders", []), dtype=int).tolist():
                orders.add(int(value) if grouped else abs(int(value)))
    return sorted(orders)


def _load_order_series(
    run_dir: Path, *, order: int, label: str, scan_id: str | None = None
) -> dict[str, Any] | None:
    """Load one order-vs-energy series from a saved run."""
    all_orders_path = run_dir / "all_orders.csv"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sweep = manifest.get("parameter_sweep")
        if sweep:
            scan = next((scan for scan in sweep["scans"] if scan["id"] == scan_id), None)
            if scan is None:
                return None
            return {
                "label": f"{label} · {scan['label']}",
                "order": int(order),
                "scan_id": scan_id,
                "energies": parameter_scans.sample_values(sweep),
                "x_axis_id": sweep.get("sample_axis", {}).get("id", "energy_ev"),
                "efficiencies": parameter_scans.load_series(run_dir, sweep, scan_id, int(order)),
            }

    requested_order = abs(int(order))
    rows: list[dict[str, float]] = []
    if all_orders_path.exists():
        with all_orders_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    row_order = int(float(row["order"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if row_order != requested_order:
                    continue
                try:
                    rows.append(
                        {
                            "energy_ev": float(row["energy_ev"]),
                            "efficiency": float(row["efficiency"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    if not rows:
        for result in _load_checkpoint_results(run_dir):
            if getattr(result, "status", "") != "ok":
                continue
            orders = np.asarray(getattr(result, "orders", []), dtype=int)
            efficiencies = np.asarray(getattr(result, "efficiency_all", []), dtype=float)
            if orders.size == 0 or efficiencies.size != orders.size:
                continue
            for row_order, efficiency in zip(orders.tolist(), efficiencies.tolist()):
                if int(row_order) != requested_order:
                    continue
                rows.append(
                    {
                        "energy_ev": float(result.energy_ev),
                        "efficiency": float(efficiency),
                    }
                )

    if not rows:
        return None

    rows.sort(key=lambda row: row["energy_ev"])
    return {
        "label": label,
        "order": int(order),
        "energies": [row["energy_ev"] for row in rows],
        "efficiencies": [row["efficiency"] for row in rows],
    }
