"""Adaptive multilayer theta-search sweep workflow."""

from __future__ import annotations

import concurrent.futures
import csv
import importlib
import json
import logging
import multiprocessing as mp
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..gratings import BaseGrating
from .batch import (
    _case_payload,
    _run_payload,
    _available_memory_bytes,
    _calibrate_auto_max_workers_from_result,
    _load_checkpoint_case_results,
    _multiprocessing_start_method,
    _parallel_worker_execute,
    _resolve_max_workers,
    _worker_initializer,
)
from .cases import multilayer_theta_search_cases
from .core import _refresh_interactive_figure, plot_order_subset, write_all_orders_csv
from .models import (
    BatchSimulationResult,
    CaseExecutionResult,
    ErrorPolicy,
    MaxWorkers,
    MultilayerThetaSearchSweepResult,
    SingleSimulationResult,
)
from .serialization import _case_result_from_record, _case_result_to_record, _single_result_from_record
from .theta_search import _format_elapsed_seconds
from ..peak_fitting import PeakSelectionMode

logger = logging.getLogger(__name__)


def _simulation_api():
    """Return the public simulation package for monkeypatch-compatible dispatch."""

    return importlib.import_module("grax.simulation")


def _load_checkpoint_case_results_for_ids(
    checkpoint_path: Path,
    allowed_case_ids: set[str],
) -> dict[str, CaseExecutionResult]:
    """Load checkpoint results only for the requested case IDs."""

    if not checkpoint_path.exists():
        return {}
    loaded: dict[str, CaseExecutionResult] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                case_id = str(record.get("case_id"))
                if case_id not in allowed_case_ids:
                    continue
                case_result = _case_result_from_record(record)
            except Exception:
                logger.warning("Ignoring malformed or incompatible checkpoint record during resume filtering.")
                continue
            loaded[case_result.case_id] = case_result
    return loaded

def _adaptive_scan_half_widths(
    *,
    energy_ev: float,
    initial_rough_half_width_deg: float,
    initial_precise_half_width_deg: float,
    completed_fwhm_by_energy: dict[float, float],
) -> tuple[float, float, str, float | None, float | None]:
    """Resolve rough/precise half-widths from completed lower-energy FWHM data."""

    lower_energy_pairs = [
        (completed_energy, fwhm)
        for completed_energy, fwhm in completed_fwhm_by_energy.items()
        if completed_energy < energy_ev and np.isfinite(fwhm) and fwhm > 0.0
    ]
    if not lower_energy_pairs:
        return (
            float(initial_rough_half_width_deg),
            float(initial_precise_half_width_deg),
            "initial",
            None,
            None,
        )

    source_energy_ev, source_fwhm_deg = max(lower_energy_pairs, key=lambda pair: pair[0])
    precise_half_width_deg = float(
        np.clip(
            4.0 * source_fwhm_deg,
            0.05,
            float(initial_precise_half_width_deg),
        )
    )
    rough_half_width_deg = float(5.0 * precise_half_width_deg)
    return (
        rough_half_width_deg,
        precise_half_width_deg,
        "from_lower_energy",
        float(source_energy_ev),
        float(source_fwhm_deg),
    )


def _update_adaptive_live_plot(
    *,
    figure: plt.Figure | None,
    axis: plt.Axes | None,
    successful_cases: list[CaseExecutionResult],
) -> tuple[plt.Figure, plt.Axes]:
    """Update live plot for adaptive sweep execution path."""

    if figure is None or axis is None or not plt.fignum_exists(figure.number):
        plt.ion()
        figure, axis = plt.subplots(figsize=(10, 6))
    axis.clear()
    if successful_cases:
        ordered_cases = sorted(successful_cases, key=lambda case: case.energy_ev)
        selected_order = ordered_cases[0].case_data.get("diffraction_order", None)
        order_label = (
            f"Selected order: {abs(int(selected_order))}"
            if selected_order is not None
            else "Selected order"
        )
        axis.plot(
            [case.energy_ev for case in ordered_cases],
            [case.selected_efficiency for case in ordered_cases],
            "o-",
            linewidth=1.0,
            markersize=3.0,
            label=order_label,
        )
    axis.set_xlabel("energy_ev")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title("Batch Simulation Progress")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    _refresh_interactive_figure(figure)
    return figure, axis


def run_multilayer_theta_search_sweep(
    *,
    grating: BaseGrating,
    energies_ev: Iterable[float],
    output_dir: str | Path,
    diffraction_order: int = 1,
    case_id_prefix: str = "multilayer-theta-search",
    rough_fourier_orders: int = 3,
    fine_fourier_orders: int = 5,
    final_fourier_orders: int = 25,
    rough_x_resolution_nm: float = 1.0,
    rough_z_resolution_nm: float = 1.0,
    fine_x_resolution_nm: float = 0.5,
    fine_z_resolution_nm: float = 0.5,
    final_x_resolution_nm: float = 0.3,
    final_z_resolution_nm: float = 0.3,
    rough_scan_half_width_deg: float = 0.5,
    rough_scan_points: int = 82,
    precise_scan_half_width_deg: float = 0.1,
    precise_scan_points: int = 81,
    multilayer_bragg_order: int = 1,
    roughness_sigma_nm: float | None = None,
    max_workers: MaxWorkers = None,
    show_progress: bool = True,
    live_plot: bool = False,
    live_theta_scan_plot: bool = False,
    on_error: ErrorPolicy = "fail_fast",
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume: bool = False,
    theta_tracking_mode: Literal["auto", "previous", "bragg"] = "auto",
    max_tracking_energy_step_ev: float | None = None,
    precise_peak_selection_mode: PeakSelectionMode = "max",
    retry_on_selected_efficiency_zero: bool = True,
    retry_selected_efficiency_threshold: float = 1e-4,
    max_zero_efficiency_retries: int = 3,
    theta_retry_jitter_deg: tuple[float, ...] | None = None,
    save_profile_plot: bool = True,
    save_stack_plot: bool = True,
) -> MultilayerThetaSearchSweepResult:
    """Run a multilayer theta-search sweep and persist standard output artifacts.

    Args:
        grating: Grating reused by every energy point.
        energies_ev: Photon energies in electronvolts.
        output_dir: Destination directory for CSV and plot outputs.
        diffraction_order: Positive diffraction order to optimize and report.
        case_id_prefix: Prefix used for stable generated case IDs.
        rough_fourier_orders: Fourier order used during the rough scan.
        fine_fourier_orders: Fourier order used during the precise scan.
        final_fourier_orders: Fourier order used during the final solve.
        rough_x_resolution_nm: X resolution used during the rough scan.
        rough_z_resolution_nm: Z resolution used during the rough scan.
        fine_x_resolution_nm: X resolution used during the precise scan.
        fine_z_resolution_nm: Z resolution used during the precise scan.
        final_x_resolution_nm: X resolution used during the final solve.
        final_z_resolution_nm: Z resolution used during the final solve.
        rough_scan_half_width_deg: Rough scan half-width around the analytical estimate.
        rough_scan_points: Number of rough scan points.
        precise_scan_half_width_deg: Precise scan half-width around the rough maximum.
        precise_scan_points: Number of precise scan points.
        multilayer_bragg_order: Positive Bragg order used for the analytical estimate.
        roughness_sigma_nm: Optional rms roughness in nanometers.
        max_workers: Optional batch worker count. ``"auto"`` calibrates from one
            completed theta-search case and available system memory before
            launching the remaining parallel work.
        show_progress: Whether to show a progress bar during execution.
        live_plot: Whether to update the standard batch live plot.
        live_theta_scan_plot: Whether to show the theta-scan diagnostic window when eligible.
        on_error: Batch error policy.
        checkpoint_dir: Optional checkpoint directory for the internal runner.
        checkpoint_interval: Flush interval for checkpoint writes.
        resume: Whether to resume from an existing checkpoint.
        theta_tracking_mode: How to choose the initial theta center after the first
            successful point. ``auto`` tracks the previous theta for dense energy
            steps and falls back to the Bragg estimate for sparse jumps. ``previous``
            always tracks the previous theta. ``bragg`` always uses the Bragg estimate.
        max_tracking_energy_step_ev: Optional dense-step threshold override used by
            ``auto`` mode. When omitted, the threshold is derived from the median
            positive energy step in the requested sweep.
        precise_peak_selection_mode: Mode used to select the final theta from the
            precise scan. ``max`` uses the sampled maximum, ``gauss`` fits a local
            Gaussian neighborhood, and ``voigt`` fits a local Voigt neighborhood.
        retry_on_selected_efficiency_zero: Whether to retry zero-efficiency selected
            order results in multilayer theta-search cases.
        retry_selected_efficiency_threshold: Retry trigger threshold for selected
            efficiency. Retries are attempted when selected efficiency is less
            than or equal to this value.
        max_zero_efficiency_retries: Maximum number of additional retries.
        theta_retry_jitter_deg: Deterministic jitter offsets for retry attempts.
        save_profile_plot: Whether to save the grating profile plot.
        save_stack_plot: Whether to save the resolved stack schematic when available.

    Returns:
        Typed result object containing collected results and the created output paths.
    """

    energy_list = [float(energy) for energy in energies_ev]
    if theta_tracking_mode not in {"auto", "previous", "bragg"}:
        raise ValueError("theta_tracking_mode must be 'auto', 'previous', or 'bragg'.")
    if precise_peak_selection_mode not in {"max", "gauss", "voigt"}:
        raise ValueError("precise_peak_selection_mode must be 'max', 'gauss', or 'voigt'.")
    if max_tracking_energy_step_ev is not None and (
        not np.isfinite(max_tracking_energy_step_ev) or max_tracking_energy_step_ev < 0.0
    ):
        raise ValueError("max_tracking_energy_step_ev must be finite and >= 0.0 when provided.")
    if not np.isfinite(retry_selected_efficiency_threshold) or retry_selected_efficiency_threshold < 0.0:
        raise ValueError("retry_selected_efficiency_threshold must be finite and >= 0.0.")
    simulation_api = _simulation_api()
    effective_workers = simulation_api._resolve_max_workers(max_workers)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    theta_scan_directory = output_path / "theta_scans"
    theta_scan_directory.mkdir(parents=True, exist_ok=True)

    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir to be specified.")

    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir)
    checkpoint_file = None if checkpoint_path is None else checkpoint_path / "results.jsonl"
    checkpoint_metadata_file = None if checkpoint_path is None else checkpoint_path / "metadata.json"
    checkpoint_handle = None
    completed_since_flush = 0
    resumed_case_results: dict[str, CaseExecutionResult] = {}
    current_run_started_at_iso = datetime.now().isoformat()
    run_started_monotonic = time.perf_counter()
    previous_elapsed_seconds = 0.0
    current_run_elapsed_seconds = 0.0
    total_elapsed_seconds = 0.0
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        existing_metadata: dict[str, object] = {}
        if checkpoint_metadata_file is not None and checkpoint_metadata_file.exists():
            try:
                with checkpoint_metadata_file.open("r", encoding="utf-8") as handle:
                    loaded_metadata = json.load(handle)
                if isinstance(loaded_metadata, dict):
                    existing_metadata = loaded_metadata
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring malformed adaptive metadata file during elapsed-time restore.")
        if resume:
            previous_elapsed_seconds = float(existing_metadata.get("cumulative_elapsed_seconds", 0.0) or 0.0)
        metadata_payload = {
            "created": existing_metadata.get("created", current_run_started_at_iso) if resume else current_run_started_at_iso,
            "workflow": "multilayer_theta_search",
            "requested_max_workers": max_workers,
            "resolved_max_workers": effective_workers,
            "theta_tracking_mode": theta_tracking_mode,
            "max_tracking_energy_step_ev": max_tracking_energy_step_ev,
            "resume": bool(resume),
            "current_run_started": current_run_started_at_iso,
            "last_updated": current_run_started_at_iso,
            "cumulative_elapsed_seconds": previous_elapsed_seconds,
            "last_run_elapsed_seconds": 0.0,
        }
        if checkpoint_metadata_file is not None:
            with checkpoint_metadata_file.open("w", encoding="utf-8") as handle:
                json.dump(metadata_payload, handle, indent=2)
        checkpoint_handle = checkpoint_file.open("a", encoding="utf-8") if checkpoint_file is not None else None

    base_cases = list(
        multilayer_theta_search_cases(
            grating=grating,
            energies_ev=energy_list,
            case_id_prefix=case_id_prefix,
            diffraction_order=diffraction_order,
            rough_fourier_orders=rough_fourier_orders,
            fine_fourier_orders=fine_fourier_orders,
            final_fourier_orders=final_fourier_orders,
            rough_x_resolution_nm=rough_x_resolution_nm,
            rough_z_resolution_nm=rough_z_resolution_nm,
            fine_x_resolution_nm=fine_x_resolution_nm,
            fine_z_resolution_nm=fine_z_resolution_nm,
            final_x_resolution_nm=final_x_resolution_nm,
            final_z_resolution_nm=final_z_resolution_nm,
            rough_scan_half_width_deg=rough_scan_half_width_deg,
            rough_scan_points=rough_scan_points,
            precise_scan_half_width_deg=precise_scan_half_width_deg,
            precise_scan_points=precise_scan_points,
            multilayer_bragg_order=multilayer_bragg_order,
            precise_peak_selection_mode=precise_peak_selection_mode,
            roughness_sigma_nm=roughness_sigma_nm,
        )
    )
    pending_by_case_id: dict[str, dict[str, object]] = {}
    pending_order: list[tuple[float, str]] = []
    for index, case in enumerate(base_cases):
        case_copy = dict(case)
        case_id = str(case_copy.get("case_id") or f"multilayer-theta-search-{index:08d}")
        case_copy["case_id"] = case_id
        pending_by_case_id[case_id] = case_copy
        pending_order.append((float(case_copy["energy_ev"]), case_id))
    pending_order.sort(key=lambda pair: pair[0])
    target_case_ids = {case_id for _, case_id in pending_order}
    if resume and checkpoint_file is not None:
        resumed_case_results = _load_checkpoint_case_results_for_ids(checkpoint_file, target_case_ids)
    positive_steps = np.diff(np.asarray(sorted(energy_list), dtype=float))
    positive_steps = positive_steps[positive_steps > 0.0]
    nominal_energy_step_ev = None if positive_steps.size == 0 else float(np.median(positive_steps))
    auto_tracking_threshold_ev = (
        float(max_tracking_energy_step_ev)
        if max_tracking_energy_step_ev is not None
        else (
            None
            if nominal_energy_step_ev is None
            else float(2.0 * nominal_energy_step_ev)
        )
    )
    resumed_case_ids = {
        case_id
        for case_id, case_result in resumed_case_results.items()
        if case_result.status in {"ok", "error"}
    }
    pending_order = [(energy_ev, case_id) for energy_ev, case_id in pending_order if case_id not in resumed_case_ids]
    resumed_cases_sorted = sorted(resumed_case_results.values(), key=lambda case: (case.energy_ev, case.index))
    completed_fwhm_by_energy: dict[float, float] = {}
    cases_result: list[CaseExecutionResult] = list(resumed_cases_sorted)
    for resumed_case in resumed_cases_sorted:
        diagnostics = resumed_case.theta_search_diagnostics
        if resumed_case.status == "ok" and diagnostics is not None and diagnostics.precise_fwhm_deg is not None:
            completed_fwhm_by_energy[float(resumed_case.energy_ev)] = float(diagnostics.precise_fwhm_deg)
    progress_bar = (
        simulation_api.tqdm(total=len(energy_list), desc="RCWA batch", unit="point")
        if show_progress
        else None
    )
    live_figure: plt.Figure | None = None
    live_axis: plt.Axes | None = None
    settings = {
        "default_diffraction_order": diffraction_order,
        "default_fourier_orders": final_fourier_orders,
        "max_fourier_orders": max(rough_fourier_orders, fine_fourier_orders, final_fourier_orders),
        "validate_physical_results": True,
        "max_reflected_efficiency": 1.05,
        "min_efficiency": -1e-8,
        "max_total_reflected_efficiency": 1.05,
    }
    retry_jitter_values = (theta_retry_jitter_deg or (0.002, -0.002, 0.005))[: max(0, int(max_zero_efficiency_retries))]
    theta_continuity_tolerance_deg = 0.02
    def _tracking_metadata_from_case(case: dict[str, object]) -> dict[str, object]:
        """Extract tracking metadata from a case dictionary."""

        return {
            "theta_tracking_center_mode": str(case.get("theta_tracking_center_mode", "bragg")),
            "theta_tracking_auto_classification": str(case.get("theta_tracking_auto_classification", "initial")),
            "theta_tracking_previous_energy_ev": (
                None if case.get("theta_tracking_previous_energy_ev") is None else float(case["theta_tracking_previous_energy_ev"])
            ),
            "theta_tracking_previous_grazing_angle_deg": (
                None
                if case.get("theta_tracking_previous_grazing_angle_deg") is None
                else float(case["theta_tracking_previous_grazing_angle_deg"])
            ),
            "theta_tracking_used_previous_theta": bool(case.get("theta_tracking_used_previous_theta", False)),
            "theta_tracking_bragg_fallback_triggered": bool(
                case.get("theta_tracking_bragg_fallback_triggered", False)
            ),
            "theta_tracking_continuity_rejected": bool(
                case.get("theta_tracking_continuity_rejected", False)
            ),
        }

    def _apply_tracking_metadata(
        single: SingleSimulationResult,
        *,
        center_mode: str,
        auto_classification: str,
        previous_energy_ev: float | None,
        previous_grazing_angle_deg: float | None,
        used_previous_theta: bool,
        bragg_fallback_triggered: bool,
        continuity_rejected: bool,
    ) -> SingleSimulationResult:
        """Attach continuity-tracking metadata to a single result."""

        single.theta_tracking_center_mode = center_mode
        single.theta_tracking_auto_classification = auto_classification
        single.theta_tracking_previous_energy_ev = previous_energy_ev
        single.theta_tracking_previous_grazing_angle_deg = previous_grazing_angle_deg
        single.theta_tracking_used_previous_theta = used_previous_theta
        single.theta_tracking_bragg_fallback_triggered = bragg_fallback_triggered
        single.theta_tracking_continuity_rejected = continuity_rejected
        if single.theta_search_diagnostics is not None:
            single.theta_search_diagnostics.theta_tracking_center_mode = center_mode
            single.theta_search_diagnostics.theta_tracking_auto_classification = auto_classification
            single.theta_search_diagnostics.theta_tracking_previous_energy_ev = previous_energy_ev
            single.theta_search_diagnostics.theta_tracking_previous_grazing_angle_deg = previous_grazing_angle_deg
            single.theta_search_diagnostics.theta_tracking_used_previous_theta = used_previous_theta
            single.theta_search_diagnostics.theta_tracking_bragg_fallback_triggered = bragg_fallback_triggered
            single.theta_search_diagnostics.theta_tracking_continuity_rejected = continuity_rejected
        return single

    def _copy_tracking_fields(single: SingleSimulationResult) -> dict[str, object]:
        """Return result tracking fields for case construction."""

        return {
            "theta_tracking_center_mode": single.theta_tracking_center_mode,
            "theta_tracking_auto_classification": single.theta_tracking_auto_classification,
            "theta_tracking_previous_energy_ev": single.theta_tracking_previous_energy_ev,
            "theta_tracking_previous_grazing_angle_deg": single.theta_tracking_previous_grazing_angle_deg,
            "theta_tracking_used_previous_theta": single.theta_tracking_used_previous_theta,
            "theta_tracking_bragg_fallback_triggered": single.theta_tracking_bragg_fallback_triggered,
            "theta_tracking_continuity_rejected": single.theta_tracking_continuity_rejected,
        }

    def _set_progress_postfix(*, active: int, queued: int, completed: int) -> None:
        """Update the adaptive sweep progress postfix."""

        if progress_bar is None:
            return
        progress_bar.set_postfix_str(f"active={active} queued={queued} done={completed}")

    def _append_checkpoint_case_result(case_result: CaseExecutionResult) -> None:
        """Append one newly completed adaptive result to the checkpoint."""

        nonlocal completed_since_flush
        if checkpoint_handle is None:
            return
        checkpoint_handle.write(json.dumps(_case_result_to_record(case_result)) + "\n")
        completed_since_flush += 1
        if completed_since_flush >= checkpoint_interval:
            checkpoint_handle.flush()
            completed_since_flush = 0

    def _continuity_violation(
        *,
        selected_theta_deg: float,
        previous_theta_deg: float | None,
    ) -> bool:
        """Return whether the candidate violates dense-step continuity."""

        if previous_theta_deg is None:
            return False
        return bool(selected_theta_deg > (previous_theta_deg + theta_continuity_tolerance_deg))

    def _choose_tracking_mode(
        *,
        energy_ev: float,
        previous_successful_case: CaseExecutionResult | None,
    ) -> tuple[str, str]:
        """Choose the primary theta center mode for one energy."""

        if previous_successful_case is None:
            return "bragg", "initial"
        if theta_tracking_mode == "bragg":
            return "bragg", "manual_bragg"
        if theta_tracking_mode == "previous":
            return "tracked_previous", "manual_previous"
        delta_energy_ev = float(energy_ev - previous_successful_case.energy_ev)
        if auto_tracking_threshold_ev is None:
            return "bragg", "auto_sparse"
        if delta_energy_ev <= auto_tracking_threshold_ev:
            return "tracked_previous", "auto_dense"
        return "bragg", "auto_sparse"

    def _prepare_case_for_tracking(
        case: dict[str, object],
        *,
        previous_successful_case: CaseExecutionResult | None,
    ) -> tuple[dict[str, object], str, str]:
        """Return a copied case prepared with continuity-tracking metadata."""

        prepared_case = dict(case)
        center_mode, auto_classification = _choose_tracking_mode(
            energy_ev=float(case["energy_ev"]),
            previous_successful_case=previous_successful_case,
        )
        prepared_case["theta_tracking_center_mode"] = center_mode
        prepared_case["theta_tracking_auto_classification"] = auto_classification
        if previous_successful_case is None:
            prepared_case["theta_tracking_previous_energy_ev"] = None
            prepared_case["theta_tracking_previous_grazing_angle_deg"] = None
            prepared_case["theta_tracking_used_previous_theta"] = False
            prepared_case.pop("initial_grazing_angle_deg", None)
        else:
            prepared_case["theta_tracking_previous_energy_ev"] = float(previous_successful_case.energy_ev)
            prepared_case["theta_tracking_previous_grazing_angle_deg"] = float(previous_successful_case.grazing_angle_deg)
            prepared_case["theta_tracking_used_previous_theta"] = center_mode == "tracked_previous"
            if center_mode == "tracked_previous":
                prepared_case["initial_grazing_angle_deg"] = float(previous_successful_case.grazing_angle_deg)
            else:
                prepared_case.pop("initial_grazing_angle_deg", None)
        prepared_case["theta_tracking_bragg_fallback_triggered"] = False
        prepared_case["theta_tracking_continuity_rejected"] = False
        return prepared_case, center_mode, auto_classification

    def _apply_zero_retry(case: dict[str, object], single: SingleSimulationResult) -> SingleSimulationResult:
        """Retry an exact-zero selected efficiency using deterministic theta jitter."""

        retry_triggered = False
        retry_attempts = 0
        retry_status = "not_needed"
        selected_efficiency_is_exact_zero = bool(single.selected_efficiency == 0.0)
        selected_efficiency_below_retry_threshold = bool(
            single.selected_efficiency <= retry_selected_efficiency_threshold
        )
        if retry_on_selected_efficiency_zero and selected_efficiency_below_retry_threshold:
            retry_triggered = True
            base_initial = case.get("initial_grazing_angle_deg")
            if base_initial is None and single.theta_search_diagnostics is not None:
                base_initial = float(single.theta_search_diagnostics.estimated_grazing_angle_deg)
            if base_initial is None:
                base_initial = float(single.grazing_angle_deg)
            for jitter in retry_jitter_values:
                retry_attempts += 1
                retry_case = dict(case)
                retry_case["initial_grazing_angle_deg"] = float(base_initial) + float(jitter)
                retry_payload = _case_payload(retry_case, settings)
                single = _run_payload(retry_payload)
                selected_efficiency_is_exact_zero = bool(single.selected_efficiency == 0.0)
                selected_efficiency_below_retry_threshold = bool(
                    single.selected_efficiency <= retry_selected_efficiency_threshold
                )
                if not selected_efficiency_below_retry_threshold:
                    retry_status = "recovered"
                    break
            else:
                retry_status = "retry_exhausted"
        single.retry_triggered = retry_triggered
        single.retry_attempts = retry_attempts
        single.retry_status = retry_status
        single.selected_efficiency_is_exact_zero = selected_efficiency_is_exact_zero
        single.selected_efficiency_below_retry_threshold = selected_efficiency_below_retry_threshold
        return single

    def _prepare_payload_with_adaptive_width(case: dict[str, object]) -> dict[str, object]:
        rough_half_width_deg, precise_half_width_deg, source, source_energy, source_fwhm = _adaptive_scan_half_widths(
            energy_ev=float(case["energy_ev"]),
            initial_rough_half_width_deg=float(rough_scan_half_width_deg),
            initial_precise_half_width_deg=float(precise_scan_half_width_deg),
            completed_fwhm_by_energy=completed_fwhm_by_energy,
        )
        case["rough_scan_half_width_deg"] = rough_half_width_deg
        case["precise_scan_half_width_deg"] = precise_half_width_deg
        if source == "initial":
            logger.info(
                "Energy %.2f eV: using initial scan widths rough=%.6f deg precise=%.6f deg",
                float(case["energy_ev"]),
                rough_half_width_deg,
                precise_half_width_deg,
            )
        else:
            logger.info(
                "Energy %.2f eV: widths from lower energy %.2f eV (FWHM=%.6f deg) -> rough=%.6f deg precise=%.6f deg",
                float(case["energy_ev"]),
                source_energy,
                source_fwhm,
                rough_half_width_deg,
                precise_half_width_deg,
            )
        return _case_payload(case, settings)

    def _run_with_tracking(
        *,
        case: dict[str, object],
        previous_successful_case: CaseExecutionResult | None,
    ) -> tuple[dict[str, object], SingleSimulationResult]:
        """Run one energy point with continuity-aware center selection."""

        prepared_case, center_mode, auto_classification = _prepare_case_for_tracking(
            case,
            previous_successful_case=previous_successful_case,
        )
        primary_payload = _prepare_payload_with_adaptive_width(prepared_case)
        primary_single = _run_payload(primary_payload)
        return _finalize_tracked_result(
            prepared_case=prepared_case,
            center_mode=center_mode,
            auto_classification=auto_classification,
            primary_single=primary_single,
        )

    def _finalize_tracked_result(
        *,
        prepared_case: dict[str, object],
        center_mode: str,
        auto_classification: str,
        primary_single: SingleSimulationResult,
    ) -> tuple[dict[str, object], SingleSimulationResult]:
        """Finalize one tracked result, optionally evaluating a Bragg fallback."""

        tracking_previous_energy_ev = (
            None
            if prepared_case.get("theta_tracking_previous_energy_ev") is None
            else float(prepared_case["theta_tracking_previous_energy_ev"])
        )
        tracking_previous_theta_deg = (
            None
            if prepared_case.get("theta_tracking_previous_grazing_angle_deg") is None
            else float(prepared_case["theta_tracking_previous_grazing_angle_deg"])
        )
        continuity_rejected = False
        bragg_fallback_triggered = False
        chosen_case = prepared_case
        chosen_single = primary_single

        should_try_bragg_fallback = center_mode == "tracked_previous" and (
            primary_single.selected_efficiency <= retry_selected_efficiency_threshold
            or _continuity_violation(
                selected_theta_deg=primary_single.grazing_angle_deg,
                previous_theta_deg=tracking_previous_theta_deg,
            )
        )
        if should_try_bragg_fallback:
            bragg_fallback_triggered = True
            fallback_case = dict(prepared_case)
            fallback_case["theta_tracking_center_mode"] = "bragg"
            fallback_case["theta_tracking_bragg_fallback_triggered"] = True
            fallback_case.pop("initial_grazing_angle_deg", None)
            fallback_payload = _prepare_payload_with_adaptive_width(fallback_case)
            fallback_single = _run_payload(fallback_payload)
            continuity_rejected = _continuity_violation(
                selected_theta_deg=primary_single.grazing_angle_deg,
                previous_theta_deg=tracking_previous_theta_deg,
            )

            primary_efficiency = float(primary_single.selected_efficiency)
            fallback_efficiency = float(fallback_single.selected_efficiency)
            efficiency_gap = abs(primary_efficiency - fallback_efficiency)
            efficiency_tolerance = max(1e-4, 0.05 * max(primary_efficiency, fallback_efficiency, 1e-12))
            if continuity_rejected:
                chosen_case = fallback_case
                chosen_single = fallback_single
            elif efficiency_gap > efficiency_tolerance:
                chosen_case = fallback_case if fallback_efficiency > primary_efficiency else prepared_case
                chosen_single = fallback_single if fallback_efficiency > primary_efficiency else primary_single
            elif tracking_previous_theta_deg is not None:
                primary_distance = abs(float(primary_single.grazing_angle_deg) - tracking_previous_theta_deg)
                fallback_distance = abs(float(fallback_single.grazing_angle_deg) - tracking_previous_theta_deg)
                chosen_case = fallback_case if fallback_distance < primary_distance else prepared_case
                chosen_single = fallback_single if fallback_distance < primary_distance else primary_single
            else:
                chosen_case = fallback_case if fallback_efficiency > primary_efficiency else prepared_case
                chosen_single = fallback_single if fallback_efficiency > primary_efficiency else primary_single

        chosen_case["theta_tracking_bragg_fallback_triggered"] = bragg_fallback_triggered
        chosen_case["theta_tracking_continuity_rejected"] = continuity_rejected
        chosen_single = _apply_tracking_metadata(
            chosen_single,
            center_mode=str(chosen_case.get("theta_tracking_center_mode", center_mode)),
            auto_classification=auto_classification,
            previous_energy_ev=tracking_previous_energy_ev,
            previous_grazing_angle_deg=tracking_previous_theta_deg,
            used_previous_theta=bool(chosen_case.get("theta_tracking_used_previous_theta", False)),
            bragg_fallback_triggered=bragg_fallback_triggered,
            continuity_rejected=continuity_rejected,
        )
        chosen_single = _apply_zero_retry(chosen_case, chosen_single)
        return chosen_case, chosen_single

    def _write_single_theta_scan_artifacts(case: CaseExecutionResult, *, skip_if_exists: bool = False) -> None:
        diagnostics = case.theta_search_diagnostics
        if diagnostics is None:
            return
        energy_tag = f"{int(round(case.energy_ev))}eV"
        scan_csv_path = theta_scan_directory / f"theta_scan_{energy_tag}.csv"
        scan_plot_path = theta_scan_directory / f"theta_scan_{energy_tag}.png"
        if skip_if_exists and scan_csv_path.exists() and scan_plot_path.exists():
            return
        with scan_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scan_type", "grazing_angle_deg", "selected_efficiency", "is_selected_peak"])
            for grazing_angle_deg, efficiency in zip(
                diagnostics.rough_grazing_angles_deg,
                diagnostics.rough_efficiencies,
            ):
                writer.writerow(
                    [
                        "rough",
                        float(grazing_angle_deg),
                        float(efficiency),
                        0,
                    ]
                )
            for grazing_angle_deg, efficiency in zip(
                diagnostics.precise_grazing_angles_deg,
                diagnostics.precise_efficiencies,
            ):
                writer.writerow(
                    [
                        "precise",
                        float(grazing_angle_deg),
                        float(efficiency),
                        int(np.isclose(grazing_angle_deg, diagnostics.selected_grazing_angle_deg)),
                    ]
                )

        # Build saved diagnostic figures with interactive mode disabled so they
        # do not flash on screen while the main live plot remains interactive.
        was_interactive = plt.isinteractive()
        plt.ioff()
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(
            diagnostics.rough_grazing_angles_deg,
            diagnostics.rough_efficiencies,
            "o-",
            linewidth=1.0,
            markersize=3.0,
            label="Rough scan",
        )
        axis.plot(
            diagnostics.precise_grazing_angles_deg,
            diagnostics.precise_efficiencies,
            "s-",
            linewidth=1.0,
            markersize=3.0,
            label="Precise scan",
        )
        axis.plot(
            diagnostics.selected_grazing_angle_deg,
            diagnostics.selected_efficiency,
            "r*",
            markersize=10.0,
            label="Selected peak",
        )
        if (
            diagnostics.precise_peak_fitted_theta_deg is not None
            and diagnostics.precise_peak_fitted_efficiencies is not None
            and diagnostics.precise_peak_selection_mode_used in {"gauss", "voigt"}
        ):
            axis.plot(
                diagnostics.precise_peak_fitted_theta_deg,
                diagnostics.precise_peak_fitted_efficiencies,
                "--",
                linewidth=1.5,
                color="tab:red",
                label=f"{diagnostics.precise_peak_selection_mode_used.capitalize()} fit",
            )
        axis.set_xlabel("Grazing Angle (deg)")
        axis.set_ylabel("Selected-Order Efficiency")
        axis.set_title(f"Final Theta Scan at {case.energy_ev:.0f} eV")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(scan_plot_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        if was_interactive:
            plt.ion()

    logger.info(
        "Multilayer theta-search sweep: requested max_workers=%r, initial resolved workers=%d, "
        "theta_tracking_mode=%s, resumed=%d",
        max_workers,
        effective_workers,
        theta_tracking_mode,
        len(resumed_case_ids),
    )
    if previous_elapsed_seconds > 0.0:
        logger.info(
            "Restored accumulated multilayer theta-search runtime: %s (%.1f s)",
            _format_elapsed_seconds(previous_elapsed_seconds),
            previous_elapsed_seconds,
        )

    try:
        resumed_artifacts_written = 0
        resumed_artifacts_skipped = 0
        for resumed_index, resumed_case in enumerate(resumed_cases_sorted, start=1):
            if resumed_case.status == "ok":
                before_csv = theta_scan_directory / f"theta_scan_{int(round(resumed_case.energy_ev))}eV.csv"
                before_png = theta_scan_directory / f"theta_scan_{int(round(resumed_case.energy_ev))}eV.png"
                existed = before_csv.exists() and before_png.exists()
                _write_single_theta_scan_artifacts(resumed_case, skip_if_exists=True)
                if existed:
                    resumed_artifacts_skipped += 1
                else:
                    resumed_artifacts_written += 1
            if resumed_index % 200 == 0:
                logger.info(
                    "Resume artifact sync progress: processed=%d/%d written=%d skipped=%d",
                    resumed_index,
                    len(resumed_cases_sorted),
                    resumed_artifacts_written,
                    resumed_artifacts_skipped,
                )
        if resumed_cases_sorted:
            logger.info(
                "Resume artifact sync complete: processed=%d written=%d skipped=%d",
                len(resumed_cases_sorted),
                resumed_artifacts_written,
                resumed_artifacts_skipped,
            )
        if progress_bar is not None and resumed_case_ids:
            progress_bar.update(len(resumed_case_ids))
            _set_progress_postfix(active=0, queued=len(pending_order), completed=len(cases_result))
        if live_plot and resumed_cases_sorted:
            live_figure, live_axis = _update_adaptive_live_plot(
                figure=live_figure,
                axis=live_axis,
                successful_cases=[case for case in cases_result if case.status == "ok"],
            )
        if max_workers == "auto" and pending_order:
            calibration_energy_ev, calibration_case_id = pending_order[0]
            resumed_successful_cases = [case for case in resumed_cases_sorted if case.status == "ok"]
            previous_successful_case = (
                max(resumed_successful_cases, key=lambda case: case.energy_ev)
                if resumed_successful_cases
                else None
            )
            calibration_case = pending_by_case_id[calibration_case_id]
            tracked_case, single = _run_with_tracking(
                case=calibration_case,
                previous_successful_case=previous_successful_case,
            )
            diagnostics = single.theta_search_diagnostics
            if diagnostics is not None and diagnostics.precise_fwhm_deg is not None:
                completed_fwhm_by_energy[calibration_energy_ev] = float(diagnostics.precise_fwhm_deg)
            calibration_result = CaseExecutionResult(
                case_id=calibration_case_id,
                index=0,
                label=None if calibration_case.get("label") is None else str(calibration_case.get("label")),
                energy_ev=single.energy_ev,
                grazing_angle_deg=single.grazing_angle_deg,
                orders=single.orders,
                selected_efficiency=single.selected_efficiency,
                selected_diffraction_angle_deg=single.selected_diffraction_angle_deg,
                efficiency_all=single.efficiency_all,
                diffraction_angle_all=single.diffraction_angle_all,
                status="ok",
                case_data={key: value for key, value in tracked_case.items() if key != "grating"},
                theta_search_diagnostics=single.theta_search_diagnostics,
                retry_triggered=single.retry_triggered,
                retry_attempts=single.retry_attempts,
                retry_status=single.retry_status,
                selected_efficiency_is_exact_zero=single.selected_efficiency_is_exact_zero,
                selected_efficiency_below_retry_threshold=single.selected_efficiency_below_retry_threshold,
                **_copy_tracking_fields(single),
            )
            cases_result.append(calibration_result)
            _append_checkpoint_case_result(calibration_result)
            _write_single_theta_scan_artifacts(calibration_result)
            if progress_bar is not None:
                progress_bar.update(1)
            effective_workers = simulation_api._calibrate_auto_max_workers_from_result(
                pending_case_count=len(pending_order),
                available_memory_bytes=simulation_api._available_memory_bytes(),
            )
            pending_order = pending_order[1:]
            logger.info(
                "Multilayer theta-search sweep: calibrated auto workers to %d after %.2f eV case; execution=%s",
                effective_workers,
                calibration_energy_ev,
                "parallel" if effective_workers > 1 else "serial",
            )
            _set_progress_postfix(active=0, queued=len(pending_order), completed=len(cases_result))
            if live_plot:
                live_figure, live_axis = _update_adaptive_live_plot(
                    figure=live_figure,
                    axis=live_axis,
                    successful_cases=[case for case in cases_result if case.status == "ok"],
                )
        if effective_workers == 1:
            resumed_successful_cases = [case for case in resumed_cases_sorted if case.status == "ok"]
            previous_successful_case: CaseExecutionResult | None = (
                max(resumed_successful_cases, key=lambda case: case.energy_ev)
                if resumed_successful_cases
                else None
            )
            for serial_index, (energy_ev, case_id) in enumerate(pending_order):
                _set_progress_postfix(active=1, queued=max(len(pending_order) - serial_index - 1, 0), completed=len(cases_result))
                case = pending_by_case_id[case_id]
                tracked_case, single = _run_with_tracking(
                    case=case,
                    previous_successful_case=previous_successful_case,
                )
                diagnostics = single.theta_search_diagnostics
                if diagnostics is not None and diagnostics.precise_fwhm_deg is not None:
                    completed_fwhm_by_energy[energy_ev] = float(diagnostics.precise_fwhm_deg)
                case_result = CaseExecutionResult(
                    case_id=case_id,
                    index=serial_index,
                    label=None if case.get("label") is None else str(case.get("label")),
                    energy_ev=single.energy_ev,
                    grazing_angle_deg=single.grazing_angle_deg,
                    orders=single.orders,
                    selected_efficiency=single.selected_efficiency,
                    selected_diffraction_angle_deg=single.selected_diffraction_angle_deg,
                    efficiency_all=single.efficiency_all,
                    diffraction_angle_all=single.diffraction_angle_all,
                    status="ok",
                    case_data={key: value for key, value in tracked_case.items() if key != "grating"},
                    theta_search_diagnostics=single.theta_search_diagnostics,
                    retry_triggered=single.retry_triggered,
                    retry_attempts=single.retry_attempts,
                    retry_status=single.retry_status,
                    selected_efficiency_is_exact_zero=single.selected_efficiency_is_exact_zero,
                    selected_efficiency_below_retry_threshold=single.selected_efficiency_below_retry_threshold,
                    **_copy_tracking_fields(single),
                )
                cases_result.append(case_result)
                previous_successful_case = case_result
                _append_checkpoint_case_result(case_result)
                _write_single_theta_scan_artifacts(case_result)
                if progress_bar is not None:
                    progress_bar.update(1)
                _set_progress_postfix(active=0, queued=max(len(pending_order) - serial_index - 1, 0), completed=len(cases_result))
                if live_plot:
                    live_figure, live_axis = _update_adaptive_live_plot(
                        figure=live_figure,
                        axis=live_axis,
                        successful_cases=[case for case in cases_result if case.status == "ok"],
                    )
        else:
            context = mp.get_context(simulation_api._multiprocessing_start_method())
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=context,
                initializer=_worker_initializer,
            ) as executor:
                futures: dict[
                    concurrent.futures.Future[dict[str, object]],
                    tuple[int, str, float, dict[str, object], str, str],
                ] = {}
                pending_cursor = 0

                def _latest_available_lower_success(energy_ev: float) -> CaseExecutionResult | None:
                    """Return the latest completed successful lower-energy result."""

                    candidates = [
                        existing
                        for existing in cases_result
                        if existing.status == "ok" and existing.energy_ev < energy_ev
                    ]
                    if not candidates:
                        return None
                    return max(candidates, key=lambda existing: existing.energy_ev)

                while pending_cursor < len(pending_order) or futures:
                    while pending_cursor < len(pending_order) and len(futures) < effective_workers:
                        energy_ev, case_id = pending_order[pending_cursor]
                        case = pending_by_case_id[case_id]
                        previous_successful_case = _latest_available_lower_success(energy_ev)
                        prepared_case, center_mode, auto_classification = _prepare_case_for_tracking(
                            case,
                            previous_successful_case=previous_successful_case,
                        )
                        payload = _prepare_payload_with_adaptive_width(prepared_case)
                        future = executor.submit(_parallel_worker_execute, payload)
                        futures[future] = (
                            pending_cursor,
                            case_id,
                            energy_ev,
                            prepared_case,
                            center_mode,
                            auto_classification,
                        )
                        pending_cursor += 1
                        _set_progress_postfix(
                            active=len(futures),
                            queued=len(pending_order) - pending_cursor,
                            completed=len(cases_result),
                        )

                    if not futures:
                        continue
                    completed_future = next(concurrent.futures.as_completed(futures))
                    index, case_id, energy_ev, prepared_case, center_mode, auto_classification = futures.pop(
                        completed_future
                    )
                    message = completed_future.result()
                    if not message["success"]:
                        if on_error == "fail_fast":
                            for future in futures:
                                future.cancel()
                            raise RuntimeError(str(message["error"]))
                        cases_result.append(
                            CaseExecutionResult(
                                case_id=case_id,
                                index=index,
                                label=None if case.get("label") is None else str(case.get("label")),
                                energy_ev=energy_ev,
                                grazing_angle_deg=float("nan"),
                                orders=np.asarray([], dtype=int),
                                selected_efficiency=float("nan"),
                                selected_diffraction_angle_deg=float("nan"),
                                efficiency_all=np.asarray([], dtype=float),
                                diffraction_angle_all=np.asarray([], dtype=float),
                                status="error",
                                error_message=str(message["error"]),
                                case_data={key: value for key, value in prepared_case.items() if key != "grating"},
                            )
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)
                        _append_checkpoint_case_result(cases_result[-1])
                        _set_progress_postfix(
                            active=len(futures),
                            queued=len(pending_order) - pending_cursor,
                            completed=len(cases_result),
                        )
                        continue

                    primary_single = _single_result_from_record(message["result"])  # type: ignore[arg-type]
                    tracked_case, single = _finalize_tracked_result(
                        prepared_case=prepared_case,
                        center_mode=center_mode,
                        auto_classification=auto_classification,
                        primary_single=primary_single,
                    )
                    diagnostics = single.theta_search_diagnostics
                    if diagnostics is not None and diagnostics.precise_fwhm_deg is not None:
                        completed_fwhm_by_energy[energy_ev] = float(diagnostics.precise_fwhm_deg)
                    case_result = CaseExecutionResult(
                        case_id=case_id,
                        index=index,
                        label=None if tracked_case.get("label") is None else str(tracked_case.get("label")),
                        energy_ev=single.energy_ev,
                        grazing_angle_deg=single.grazing_angle_deg,
                        orders=single.orders,
                        selected_efficiency=single.selected_efficiency,
                        selected_diffraction_angle_deg=single.selected_diffraction_angle_deg,
                        efficiency_all=single.efficiency_all,
                        diffraction_angle_all=single.diffraction_angle_all,
                        status="ok",
                        case_data={key: value for key, value in tracked_case.items() if key != "grating"},
                        theta_search_diagnostics=single.theta_search_diagnostics,
                        retry_triggered=single.retry_triggered,
                        retry_attempts=single.retry_attempts,
                        retry_status=single.retry_status,
                        selected_efficiency_is_exact_zero=single.selected_efficiency_is_exact_zero,
                        selected_efficiency_below_retry_threshold=single.selected_efficiency_below_retry_threshold,
                        **_copy_tracking_fields(single),
                    )
                    cases_result.append(case_result)
                    _append_checkpoint_case_result(case_result)
                    _write_single_theta_scan_artifacts(case_result)
                    if progress_bar is not None:
                        progress_bar.update(1)
                    _set_progress_postfix(
                        active=len(futures),
                        queued=len(pending_order) - pending_cursor,
                        completed=len(cases_result),
                    )
                    if live_plot:
                        live_figure, live_axis = _update_adaptive_live_plot(
                            figure=live_figure,
                            axis=live_axis,
                            successful_cases=[case for case in cases_result if case.status == "ok"],
                        )
    finally:
        current_run_elapsed_seconds = float(time.perf_counter() - run_started_monotonic)
        total_elapsed_seconds = float(previous_elapsed_seconds + current_run_elapsed_seconds)
        if checkpoint_metadata_file is not None:
            metadata_payload = {
                "created": current_run_started_at_iso if not resume else None,
                "workflow": "multilayer_theta_search",
                "requested_max_workers": max_workers,
                "resolved_max_workers": effective_workers,
                "theta_tracking_mode": theta_tracking_mode,
                "max_tracking_energy_step_ev": max_tracking_energy_step_ev,
                "resume": bool(resume),
                "current_run_started": current_run_started_at_iso,
                "last_updated": datetime.now().isoformat(),
                "cumulative_elapsed_seconds": total_elapsed_seconds,
                "last_run_elapsed_seconds": current_run_elapsed_seconds,
            }
            if checkpoint_metadata_file.exists():
                try:
                    with checkpoint_metadata_file.open("r", encoding="utf-8") as handle:
                        existing_metadata = json.load(handle)
                    if isinstance(existing_metadata, dict) and existing_metadata.get("created") is not None:
                        metadata_payload["created"] = existing_metadata["created"]
                except (OSError, json.JSONDecodeError):
                    pass
            if metadata_payload["created"] is None:
                metadata_payload["created"] = current_run_started_at_iso
            with checkpoint_metadata_file.open("w", encoding="utf-8") as handle:
                json.dump(metadata_payload, handle, indent=2)
        if checkpoint_handle is not None:
            checkpoint_handle.flush()
            checkpoint_handle.close()
        if progress_bar is not None:
            progress_bar.close()
    batch_result = BatchSimulationResult(cases=cases_result)
    successful_cases = sorted(batch_result.successful_cases, key=lambda case: case.energy_ev)

    summary_csv_path = output_path / "multilayer_theta_search_summary.csv"
    all_orders_csv_path = output_path / "multilayer_theta_search_all_orders.csv"
    energy_efficiency_plot_path = output_path / "multilayer_theta_search_energy_vs_efficiency.png"
    workflow_plot_path = output_path / "multilayer_theta_search_workflow.png"
    profile_plot_path = output_path / "blazed_multilayer_profile.png" if save_profile_plot else None
    stack_plot_path = output_path / "multilayer_stack_schematic.png" if save_stack_plot else None

    with summary_csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "energy_ev,selected_grazing_angle_deg,selected_efficiency,precise_fwhm_deg,"
            "retry_triggered,retry_attempts,retry_status,selected_efficiency_is_exact_zero,"
            "selected_efficiency_below_retry_threshold,theta_tracking_center_mode,"
            "theta_tracking_auto_classification,theta_tracking_previous_energy_ev,"
            "theta_tracking_previous_grazing_angle_deg,theta_tracking_used_previous_theta,"
            "theta_tracking_bragg_fallback_triggered,theta_tracking_continuity_rejected,"
            "precise_peak_selection_mode_requested,precise_peak_selection_mode_used,"
            "precise_peak_fit_fallback_used,precise_peak_fitted_center_deg,precise_peak_fitted_fwhm_deg\n"
        )
        for case in successful_cases:
            fwhm_deg = (
                case.theta_search_diagnostics.precise_fwhm_deg
                if case.theta_search_diagnostics is not None
                else None
            )
            handle.write(
                f"{case.energy_ev:.6f},{case.grazing_angle_deg:.6f},{case.selected_efficiency:.8f},"
                f"{'' if fwhm_deg is None else f'{fwhm_deg:.8f}'},"
                f"{int(case.retry_triggered)},{case.retry_attempts},"
                f"{case.retry_status},{int(case.selected_efficiency_is_exact_zero)},"
                f"{int(case.selected_efficiency_below_retry_threshold)},"
                f"{case.theta_tracking_center_mode},{case.theta_tracking_auto_classification},"
                f"{'' if case.theta_tracking_previous_energy_ev is None else f'{case.theta_tracking_previous_energy_ev:.6f}'},"
                f"{'' if case.theta_tracking_previous_grazing_angle_deg is None else f'{case.theta_tracking_previous_grazing_angle_deg:.6f}'},"
                f"{int(case.theta_tracking_used_previous_theta)},"
                f"{int(case.theta_tracking_bragg_fallback_triggered)},"
                f"{int(case.theta_tracking_continuity_rejected)},"
                f"{case.theta_search_diagnostics.precise_peak_selection_mode_requested if case.theta_search_diagnostics is not None else 'max'},"
                f"{case.theta_search_diagnostics.precise_peak_selection_mode_used if case.theta_search_diagnostics is not None else 'max'},"
                f"{int(case.theta_search_diagnostics.precise_peak_fit_fallback_used) if case.theta_search_diagnostics is not None else 0},"
                f"{'' if case.theta_search_diagnostics is None or case.theta_search_diagnostics.precise_peak_fitted_center_deg is None else f'{case.theta_search_diagnostics.precise_peak_fitted_center_deg:.8f}'},"
                f"{'' if case.theta_search_diagnostics is None or case.theta_search_diagnostics.precise_peak_fitted_fwhm_deg is None else f'{case.theta_search_diagnostics.precise_peak_fitted_fwhm_deg:.8f}'}\n"
            )

    with all_orders_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_ev",
                "selected_grazing_angle_deg",
                "order",
                "efficiency",
                "diffraction_angle_deg",
                "retry_triggered",
                "retry_attempts",
                "retry_status",
                "selected_efficiency_is_exact_zero",
                "selected_efficiency_below_retry_threshold",
                "theta_tracking_center_mode",
                "theta_tracking_auto_classification",
                "theta_tracking_previous_energy_ev",
                "theta_tracking_previous_grazing_angle_deg",
                "theta_tracking_used_previous_theta",
                "theta_tracking_bragg_fallback_triggered",
                "theta_tracking_continuity_rejected",
                "precise_peak_selection_mode_requested",
                "precise_peak_selection_mode_used",
                "precise_peak_fit_fallback_used",
                "precise_peak_fitted_center_deg",
                "precise_peak_fitted_fwhm_deg",
            ]
        )
        for case in successful_cases:
            for order, efficiency, angle in zip(
                np.asarray(case.orders, dtype=int),
                np.asarray(case.efficiency_all, dtype=float),
                np.asarray(case.diffraction_angle_all, dtype=float),
            ):
                writer.writerow(
                    [
                        float(case.energy_ev),
                        float(case.grazing_angle_deg),
                        int(order),
                        float(efficiency),
                        float(angle),
                        int(case.retry_triggered),
                        int(case.retry_attempts),
                        case.retry_status,
                        int(case.selected_efficiency_is_exact_zero),
                        int(case.selected_efficiency_below_retry_threshold),
                        case.theta_tracking_center_mode,
                        case.theta_tracking_auto_classification,
                        None if case.theta_tracking_previous_energy_ev is None else float(case.theta_tracking_previous_energy_ev),
                        (
                            None
                            if case.theta_tracking_previous_grazing_angle_deg is None
                            else float(case.theta_tracking_previous_grazing_angle_deg)
                        ),
                        int(case.theta_tracking_used_previous_theta),
                        int(case.theta_tracking_bragg_fallback_triggered),
                        int(case.theta_tracking_continuity_rejected),
                        (
                            case.theta_search_diagnostics.precise_peak_selection_mode_requested
                            if case.theta_search_diagnostics is not None
                            else "max"
                        ),
                        (
                            case.theta_search_diagnostics.precise_peak_selection_mode_used
                            if case.theta_search_diagnostics is not None
                            else "max"
                        ),
                        (
                            int(case.theta_search_diagnostics.precise_peak_fit_fallback_used)
                            if case.theta_search_diagnostics is not None
                            else 0
                        ),
                        (
                            None
                            if case.theta_search_diagnostics is None
                            else case.theta_search_diagnostics.precise_peak_fitted_center_deg
                        ),
                        (
                            None
                            if case.theta_search_diagnostics is None
                            else case.theta_search_diagnostics.precise_peak_fitted_fwhm_deg
                        ),
                    ]
                )

    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            [case.energy_ev for case in successful_cases],
            [case.selected_efficiency for case in successful_cases],
            "o-",
            linewidth=1.0,
            markersize=3.0,
        )
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title("Blazed Multilayer Theta Search: Final Efficiency")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(energy_efficiency_plot_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        first_diagnostics = None if not successful_cases else successful_cases[0].theta_search_diagnostics
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        axes[0].plot(
            [case.energy_ev for case in successful_cases],
            [case.selected_efficiency for case in successful_cases],
            "o-",
            linewidth=1.0,
            markersize=3.0,
        )
        axes[0].set_xlabel("Energy (eV)")
        axes[0].set_ylabel("Diffraction Efficiency")
        axes[0].set_title("Final Theta-Search Results")
        axes[0].grid(True, alpha=0.3)
        if first_diagnostics is not None:
            axes[1].plot(
                first_diagnostics.rough_grazing_angles_deg,
                first_diagnostics.rough_efficiencies,
                "o-",
                linewidth=1.0,
                markersize=3.0,
                label="Rough scan",
            )
            axes[1].plot(
                first_diagnostics.precise_grazing_angles_deg,
                first_diagnostics.precise_efficiencies,
                "s-",
                linewidth=1.0,
                markersize=3.0,
                label="Precise scan",
            )
            axes[1].plot(
                first_diagnostics.selected_grazing_angle_deg,
                first_diagnostics.selected_efficiency,
                "r*",
                markersize=10.0,
                label="Selected peak",
            )
            axes[1].set_title(f"First-Energy Diagnostics ({successful_cases[0].energy_ev:.0f} eV)")
            axes[1].legend(loc="best")
        axes[1].set_xlabel("Grazing Angle (deg)")
        axes[1].set_ylabel("Diffraction Efficiency")
        axes[1].grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(workflow_plot_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        if profile_plot_path is not None:
            grating.plot_profile(profile_plot_path)
        if stack_plot_path is not None:
            resolved_stack = grating.resolved_stack()
            if hasattr(resolved_stack, "plot_schematic"):
                resolved_stack.plot_schematic(stack_plot_path)
    finally:
        if was_interactive:
            plt.ion()

    logger.info(
        "Multilayer theta-search sweep completed in %s for this run, %s total accumulated.",
        _format_elapsed_seconds(current_run_elapsed_seconds),
        _format_elapsed_seconds(total_elapsed_seconds),
    )

    return MultilayerThetaSearchSweepResult(
        batch_result=batch_result,
        summary_csv_path=summary_csv_path,
        all_orders_csv_path=all_orders_csv_path,
        energy_efficiency_plot_path=energy_efficiency_plot_path,
        workflow_plot_path=workflow_plot_path,
        theta_scan_directory=theta_scan_directory,
        profile_plot_path=profile_plot_path,
        stack_plot_path=stack_plot_path,
        total_elapsed_seconds=total_elapsed_seconds,
        current_run_elapsed_seconds=current_run_elapsed_seconds,
    )
