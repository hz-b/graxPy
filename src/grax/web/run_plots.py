"""Live progress images during a run and plots rebuilt for saved runs."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import parameter_scans
from .custom_grid import plot_custom_grid
from .files import replace_file
from .run_files import (
    _load_checkpoint_results,
    _load_run_manifest,
    _parameter_study_artifacts,
    _run_artifacts_for_results,
    _update_manifest_fields,
)
from .run_state import (
    ActiveRunState,
    _active_runs,
    _active_runs_lock,
    _update_active_run,
)

matplotlib.use("Agg")


def _results_sorted_for_live_plot(results: list[Any], *, x_key: str = "energy_ev") -> list[Any]:
    """Return successful case results sorted by the chosen live-plot x axis."""

    def x_value(result: Any) -> float:
        if x_key == "index":
            return float(result.index + 1)
        if x_key in getattr(result, "case_data", {}):
            return float(result.case_data[x_key])
        if hasattr(result, x_key):
            return float(getattr(result, x_key))
        raise KeyError(f"Unable to sort live plot using x key '{x_key}'.")

    return sorted(
        [result for result in results if result.status == "ok"],
        key=x_value,
    )


def _publish_live_progress_snapshot(
    *,
    state: ActiveRunState,
    results: list[Any],
    output_path: Path,
    diffraction_order: int,
    title: str,
    now: float | None = None,
    min_interval_seconds: float = 1.5,
) -> bool:
    """Render and atomically publish one live progress image when due."""

    from grax import simulation

    publish_time = time.monotonic() if now is None else float(now)
    if not results:
        return False
    if (
        state.last_plot_publish_monotonic is not None
        and publish_time - state.last_plot_publish_monotonic < min_interval_seconds
    ):
        return False

    is_custom_grid = any("custom_parameters" in result.case_data for result in results)
    ordered_results = _results_sorted_for_live_plot(
        results,
        x_key="index" if is_custom_grid else "energy_ev",
    )
    grouped_results = any("scan_id" in result.case_data for result in results)
    if not ordered_results and not grouped_results:
        return False

    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if grouped_results:
        manifest = _load_run_manifest(output_path.parent.parent.parent, output_path.parent.name) or {}
        parameter_scans.plot_scans(results, manifest["parameter_sweep"], temp_path, title=title)
    elif is_custom_grid:
        plot_custom_grid(ordered_results, temp_path, title=title)
    else:
        simulation.plot_order_subset(
            ordered_results,
            temp_path,
            diffraction_orders=[diffraction_order],
            title=title,
        )
    replace_file(temp_path, output_path)
    state.plot_relative_path = output_path.name
    state.plot_token = uuid4().hex
    state.last_plot_publish_monotonic = publish_time
    return True


def _publish_live_progress_if_due(
    *,
    app: Any,
    run_id: str,
    results: list[Any],
    output_path: Path,
    diffraction_order: int,
    title: str,
    min_interval_seconds: float = 1.5,
) -> None:
    """Publish the live progress image on a throttled interval."""

    with _active_runs_lock(app):
        state = _active_runs(app).get(run_id)
    if state is None:
        return
    published = _publish_live_progress_snapshot(
        state=state,
        results=results,
        output_path=output_path,
        diffraction_order=diffraction_order,
        title=title,
        min_interval_seconds=min_interval_seconds,
    )
    if not published:
        return
    _update_active_run(
        app,
        run_id,
        plot_relative_path=output_path.name,
    )


def _repair_saved_run_plot(app: Any, data_dir: Path, run_id: str, manifest: dict[str, Any]) -> None:
    """Recreate a missing completed plot from checkpoints without rerunning cases."""

    if manifest.get("status") not in {"completed", "aborted", "failed", "ok"}:
        return
    run_dir = data_dir / "runs" / run_id
    study = manifest.get("workflow") == "parameter_study"
    destination = run_dir / ("parameter_study.png" if study else "selected_efficiency.png")
    if destination.exists():
        return
    results = [] if study else _load_checkpoint_results(run_dir)
    if not study and not results and not (run_dir / "summary.csv").exists():
        return
    temporary = run_dir / f"{destination.stem}.{uuid4().hex}.tmp.png"
    try:
        sweep = manifest.get("parameter_sweep")
        title = f"{manifest.get('grating_name', run_id)} {manifest.get('workflow', 'run')}"
        if study:
            if not _plot_saved_parameter_study(run_dir, manifest, temporary):
                return
        elif not results:
            _plot_saved_summary(run_dir, manifest, temporary, title=title)
        elif sweep:
            parameter_scans.plot_scans(results, sweep, temporary, title=title)
        elif any("custom_parameters" in result.case_data for result in results):
            plot_custom_grid(results, temporary, title=title)
        else:
            from grax import simulation

            simulation.plot_order_subset(
                results, temporary,
                diffraction_orders=[int(manifest.get("run_input", {}).get("diffraction_order", 1))],
                title=title,
            )
        replace_file(temporary, destination)
        _update_manifest_fields(
            data_dir, run_id,
            artifacts=_parameter_study_artifacts(run_dir) if study else _run_artifacts_for_results(run_dir, include_plot=True),
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        app.logger.exception("Could not rebuild plot for run %s", run_id)


def _plot_saved_summary(run_dir: Path, manifest: dict[str, Any], output_path: Path, *, title: str) -> None:
    """Plot saved CSV rows when a completed run has no checkpoint file."""

    from types import SimpleNamespace

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Saved summary has no result rows.")
    sweep = manifest.get("parameter_sweep")
    if sweep:
        scan_orders = {scan["id"]: int(scan["settings"]["diffraction_order"]) for scan in sweep["scans"]}
        results = []
        for row in rows:
            scan_id = row["scan_id"]
            selected = float(row["selected_efficiency"] or "nan")
            results.append(SimpleNamespace(
                case_data={"scan_id": scan_id},
                energy_ev=float(row["energy_ev"]),
                grazing_angle_deg=float(row["grazing_angle_deg"]),
                status=row["status"],
                selected_efficiency=selected,
                orders=np.asarray([scan_orders[scan_id]]),
                efficiency_all=np.asarray([selected]),
            ))
        parameter_scans.plot_scans(results, sweep, output_path, title=title)
        return
    workflow = manifest.get("workflow")
    x_key = "grazing_angle_deg" if workflow == "angle_sweep" else "energy_ev"
    x_label = "Grazing angle (deg)" if workflow == "angle_sweep" else "Photon Energy (eV)"
    if workflow == "custom_grid":
        x_key, x_label = None, "Case"
    points = [
        (float(row[x_key]) if x_key else float(index + 1),
         float(row["selected_efficiency"] or "nan") if row["status"] == "ok" else np.nan)
        for index, row in enumerate(rows)
    ]
    points.sort(key=lambda item: item[0])
    figure, axis = plt.subplots(figsize=(10, 6))
    try:
        axis.plot([x for x, _ in points], [y for _, y in points], marker=".", linewidth=1.2)
        axis.set(xlabel=x_label, ylabel="Selected-order efficiency", title=title)
        axis.grid(True, alpha=0.3)
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_saved_parameter_study(run_dir: Path, manifest: dict[str, Any], output_path: Path) -> bool:
    """Rebuild the numerical-stability figure from its saved sweep CSV files."""

    from grax import parameter_sweep

    by_energy: dict[float, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("parameter_study_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        parameter = rows[0].get("parameter")
        if parameter not in {"fourier_orders", "x_resolution_nm", "z_resolution_nm"}:
            continue
        energy = float(rows[0]["energy_ev"])
        by_energy.setdefault(energy, {})[parameter] = parameter_sweep.ParameterSweepSeries(
            parameter=parameter,
            values=np.asarray([float(row["value"]) for row in rows]),
            efficiencies=np.asarray([float(row["efficiency"]) for row in rows]),
            errors=np.asarray([row["error"].strip().lower() == "true" for row in rows]),
            error_messages=np.asarray([row.get("error_message", "") for row in rows]),
        )
    required = {"fourier_orders", "x_resolution_nm", "z_resolution_nm"}
    complete = [(energy, sweeps) for energy, sweeps in sorted(by_energy.items()) if required <= sweeps.keys()]
    if not complete:
        return False
    form = manifest.get("run_input", {})
    result = parameter_sweep.ParameterStudyResult(
        energies_ev=np.asarray([energy for energy, _ in complete]),
        grazing_angle_deg=float(form.get("study_grazing_angle_deg", form.get("grazing_angle_deg", 1.5))),
        diffraction_order=int(form.get("diffraction_order", 1)),
        fourier_orders_values=complete[0][1]["fourier_orders"].values,
        x_resolution_values=complete[0][1]["x_resolution_nm"].values,
        z_resolution_values=complete[0][1]["z_resolution_nm"].values,
        results=[
            parameter_sweep.ParameterStudyEnergyResult(
                energy_ev=energy,
                grazing_angle_deg=float(form.get("study_grazing_angle_deg", form.get("grazing_angle_deg", 1.5))),
                sweeps=sweeps,
            )
            for energy, sweeps in complete
        ],
    )
    parameter_sweep.plot_parameter_study(result, output_filename=output_path)
    return True
