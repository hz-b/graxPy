"""Generic batch execution and checkpoint orchestration."""

from __future__ import annotations

import concurrent.futures
import ctypes
import importlib
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..gratings import BaseGrating
from .core import (
    _clone_grating_with_overrides,
    _refresh_interactive_figure,
    _warn_if_numpy_backend_requested,
    efficiency_for_order,
    run_simulation,
)
from ._profiling import SolverProfiler
from .models import (
    AUTO_WORKER_MEMORY_RESERVE_BYTES,
    AUTO_WORKER_MEMORY_SAFETY_FACTOR,
    BatchSimulationResult,
    CaseExecutionResult,
    ExecutionMode,
    ErrorPolicy,
    MaxWorkers,
    SingleSimulationResult,
    ThetaSearchDiagnostics,
)
from .serialization import (
    _case_result_from_record,
    _case_result_to_record,
    _completed_case_ids,
    _load_checkpoint_case_results,
    _single_result_from_record,
    _single_result_to_record,
)
from .theta_search import run_multilayer_theta_search

logger = logging.getLogger(__name__)

_BATCH_MAX_TOTAL_REFLECTED_EFFICIENCY = 1.05
_BATCH_THETA_RETRY_JITTER_DEG = (0.002, -0.002, 0.005)


def _simulation_api():
    """Return the public simulation package for monkeypatch-compatible dispatch."""

    return importlib.import_module("grax.simulation")


def _case_memory_mode(case: dict[str, object]) -> str:
    """Return the internal memory mode for one batch case."""

    memory_mode = str(case.get("_memory_mode", case.get("memory_mode", "low_memory")))
    if memory_mode not in {"low_memory", "legacy_dense"}:
        raise ValueError("batch case memory_mode must be 'low_memory' or 'legacy_dense'.")
    return memory_mode


def _case_payload(case: dict[str, object], runner_settings: dict[str, object]) -> dict[str, object]:
    """Build the serializable payload used by inline and subprocess execution."""

    grating = case["grating"]
    if not isinstance(grating, BaseGrating):
        raise TypeError("Each case must provide a 'grating' derived from BaseGrating.")
    fourier_orders = int(case.get("fourier_orders", runner_settings["default_fourier_orders"]))
    max_fourier_orders = int(runner_settings["max_fourier_orders"])
    if fourier_orders > max_fourier_orders:
        logger.warning(
            "Case %s: fourier_orders=%s exceeds max=%s, reducing",
            case["case_id"],
            fourier_orders,
            max_fourier_orders,
        )
        fourier_orders = max_fourier_orders

    if case.get("workflow") == "multilayer_theta_search":
        rough_fourier_orders = int(case.get("rough_fourier_orders", 3))
        fine_fourier_orders = int(case.get("fine_fourier_orders", 5))
        final_fourier_orders = int(case.get("final_fourier_orders", fourier_orders))
        if rough_fourier_orders > max_fourier_orders:
            logger.warning(
                "Case %s: rough_fourier_orders=%s exceeds max=%s, reducing",
                case["case_id"],
                rough_fourier_orders,
                max_fourier_orders,
            )
            rough_fourier_orders = max_fourier_orders
        if fine_fourier_orders > max_fourier_orders:
            logger.warning(
                "Case %s: fine_fourier_orders=%s exceeds max=%s, reducing",
                case["case_id"],
                fine_fourier_orders,
                max_fourier_orders,
            )
            fine_fourier_orders = max_fourier_orders
        if final_fourier_orders > max_fourier_orders:
            logger.warning(
                "Case %s: final_fourier_orders=%s exceeds max=%s, reducing",
                case["case_id"],
                final_fourier_orders,
                max_fourier_orders,
            )
            final_fourier_orders = max_fourier_orders
        return {
            "workflow": "multilayer_theta_search",
            "grating": grating,
            "energy_ev": float(case["energy_ev"]),
            "diffraction_order": int(case.get("diffraction_order", runner_settings["default_diffraction_order"])),
            "initial_grazing_angle_deg": case.get("initial_grazing_angle_deg"),
            "multilayer_bragg_order": int(case.get("multilayer_bragg_order", 1)),
            "rough_scan_half_width_deg": float(case.get("rough_scan_half_width_deg", 0.5)),
            "rough_scan_points": int(case.get("rough_scan_points", 41)),
            "rough_fourier_orders": rough_fourier_orders,
            "rough_x_resolution_nm": case.get("rough_x_resolution_nm", 1.0),
            "rough_z_resolution_nm": case.get("rough_z_resolution_nm", 1.0),
            "fine_scan_half_width_deg": float(case.get("fine_scan_half_width_deg", 0.1)),
            "fine_scan_points": int(case.get("fine_scan_points", 81)),
            "fine_fourier_orders": fine_fourier_orders,
            "fine_x_resolution_nm": case.get("fine_x_resolution_nm", 0.5),
            "fine_z_resolution_nm": case.get("fine_z_resolution_nm", 0.5),
            "final_fourier_orders": final_fourier_orders,
            "final_x_resolution_nm": case.get("final_x_resolution_nm", 0.3),
            "final_z_resolution_nm": case.get("final_z_resolution_nm", 0.3),
            "_memory_mode": _case_memory_mode(case),
            "profile_memory": bool(case.get("profile_memory", False)),
            "roughness_sigma_nm": case.get("roughness_sigma_nm"),
            "validate_physical_results": bool(runner_settings["validate_physical_results"]),
            "max_reflected_efficiency": float(runner_settings["max_reflected_efficiency"]),
            "min_efficiency": float(runner_settings["min_reflected_efficiency"]),
            "max_total_reflected_efficiency": _BATCH_MAX_TOTAL_REFLECTED_EFFICIENCY,
            "precise_peak_selection_mode": str(case.get("precise_peak_selection_mode", "max")),
            "backend": runner_settings["backend"],
        }

    return {
        "grating": _clone_grating_with_overrides(
            grating,
            x_resolution_nm=case.get("x_resolution_nm"),
            z_resolution_nm=case.get("z_resolution_nm"),
        ),
        "energy_ev": float(case["energy_ev"]),
        "grazing_angle_deg": float(case["grazing_angle_deg"]),
        "diffraction_order": int(case.get("diffraction_order", runner_settings["default_diffraction_order"])),
        "fourier_orders": fourier_orders,
        "_memory_mode": _case_memory_mode(case),
        "profile_memory": bool(case.get("profile_memory", False)),
        "roughness_sigma_nm": case.get("roughness_sigma_nm"),
        "polarization": str(case.get("polarization", runner_settings["default_polarization"])),
        "validate_physical_results": bool(runner_settings["validate_physical_results"]),
        "max_reflected_efficiency": float(runner_settings["max_reflected_efficiency"]),
        "min_efficiency": float(runner_settings["min_reflected_efficiency"]),
        "max_total_reflected_efficiency": _BATCH_MAX_TOTAL_REFLECTED_EFFICIENCY,
        "backend": runner_settings["backend"],
    }


def _run_case_payload(
    payload: dict[str, object],
    *,
    diagnostic_callback: Callable[[ThetaSearchDiagnostics, float], None] | None = None,
) -> SingleSimulationResult:
    """Execute one prepared case payload with an optional diagnostics callback."""

    if payload.get("workflow") == "multilayer_theta_search":
        theta_payload = dict(payload)
        theta_payload.pop("workflow", None)
        theta_payload.pop("_memory_mode", None)
        theta_payload.pop("profile_memory", None)
        return _simulation_api().run_multilayer_theta_search(  # type: ignore[arg-type]
            **theta_payload,
            diagnostic_callback=diagnostic_callback,
        )
    run_payload = dict(payload)
    run_payload.pop("profile_memory", None)
    return _simulation_api().run_simulation(**run_payload)  # type: ignore[arg-type]


def _run_case_payload_with_optional_memory_profile(
    payload: dict[str, object],
) -> tuple[SingleSimulationResult, int | None, float | None]:
    """Execute one prepared case payload and optionally measure its peak memory."""

    if not bool(payload.get("profile_memory", False)):
        return _run_case_payload(payload), None, None

    profiler = SolverProfiler()
    profiler.enable_memory_tracking()
    try:
        result = _run_case_payload(payload)
    finally:
        profiler.finalize()
    summary = profiler.summary_dict()
    peak_memory_bytes = summary.get("peak_memory_bytes")
    wall_seconds = summary.get("total_wall_seconds")
    return (
        result,
        None if peak_memory_bytes is None else int(peak_memory_bytes),
        None if wall_seconds is None else float(wall_seconds),
    )


def _run_payload(payload: dict[str, object]) -> SingleSimulationResult:
    """Execute one prepared case payload."""

    return _run_case_payload(payload)


def _worker_identity() -> str:
    """Return a compact worker identity for execution-time logging."""

    process = mp.current_process()
    thread = threading.current_thread()
    return f"pid={os.getpid()} proc={process.name} thread={thread.name}"


def _log_case_execution_start(
    *,
    case_id: str,
    case: dict[str, object],
    index: int,
    location: str,
) -> None:
    """Log one case only when execution actually starts."""

    workflow = str(case.get("workflow", "single"))
    energy_ev = float(case["energy_ev"])
    angle = case.get("grazing_angle_deg")
    if angle is None:
        logger.info(
            "[%s] starting case=%s idx=%d workflow=%s energy=%.6f eV %s",
            location,
            case_id,
            index,
            workflow,
            energy_ev,
            _worker_identity(),
        )
        return
    logger.info(
        "[%s] starting case=%s idx=%d workflow=%s energy=%.6f eV grazing=%.6f deg %s",
        location,
        case_id,
        index,
        workflow,
        energy_ev,
        float(angle),
        _worker_identity(),
    )


def _subprocess_worker(payload: dict[str, object], result_queue: mp.Queue) -> None:
    """Run one case payload in a child process and send back a result record."""

    try:
        result, peak_memory_bytes, wall_seconds = _run_case_payload_with_optional_memory_profile(payload)
        message: dict[str, object] = {"success": True, "result": _single_result_to_record(result)}
        if peak_memory_bytes is not None:
            message["peak_memory_bytes"] = peak_memory_bytes
        if wall_seconds is not None:
            message["wall_seconds"] = wall_seconds
        result_queue.put(message)
    except Exception as error:  # pragma: no cover - exercised by parent error path
        result_queue.put({"success": False, "error": str(error)})


def _run_payload_in_subprocess(
    payload: dict[str, object],
    *,
    timeout: float,
) -> tuple[SingleSimulationResult, int | None]:
    """Execute one prepared case payload in a spawned subprocess."""

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(target=_subprocess_worker, args=(payload, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Timeout after {timeout} seconds")
    if process.exitcode not in (0, None) and result_queue.empty():
        raise RuntimeError(f"Subprocess exited with code {process.exitcode}")
    try:
        message = result_queue.get(timeout=1)
    except queue_module.Empty as error:
        raise RuntimeError("Subprocess produced no result") from error
    if not message["success"]:
        raise RuntimeError(str(message["error"]))
    peak_memory_bytes = message.get("peak_memory_bytes")
    wall_seconds = message.get("wall_seconds")
    return (
        _single_result_from_record(message["result"]),
        None if peak_memory_bytes is None else int(peak_memory_bytes),
        None if wall_seconds is None else float(wall_seconds),
    )



def _json_safe_case_data(case_data: dict[str, object]) -> dict[str, object]:
    """Return case metadata with non-JSON values represented as strings."""

    safe: dict[str, object] = {}
    for key, value in case_data.items():
        if key == "grating":
            continue
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = repr(value)
    return safe


def _resolve_max_workers(max_workers: MaxWorkers) -> int:
    """Return the effective worker count for batch execution."""

    if max_workers is None:
        return 1
    if isinstance(max_workers, str):
        cpu_count = os.cpu_count() or 1
        if max_workers == "all":
            return max(cpu_count, 1)
        if max_workers == "auto":
            return max(cpu_count - 2, 1)
        raise ValueError("max_workers must be None, a positive integer, 'all', or 'auto'.")
    if isinstance(max_workers, int):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided.")
        return max_workers
    raise ValueError("max_workers must be None, a positive integer, 'all', or 'auto'.")


def _available_memory_bytes() -> int | None:
    """Return currently available system memory in bytes when detectable."""

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    if sys.platform.startswith("win"):
        try:
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, ValueError, OSError):
        return None


def _multiprocessing_start_method() -> str:
    """Return the platform-appropriate multiprocessing start method."""

    return "spawn" if sys.platform.startswith("win") else "fork"


def _worker_initializer() -> None:
    """Limit worker-local BLAS/OpenMP thread counts to avoid oversubscription."""

    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _current_process_memory_bytes() -> int | None:
    """Return current RSS memory usage for the active process in bytes."""

    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    if sys.platform.startswith("win"):
        try:
            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process_handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.WorkingSetSize)
        except Exception:
            pass

    return None


def _calibrate_auto_max_workers_from_result(
    *,
    pending_case_count: int,
    available_memory_bytes: int | None,
) -> int:
    """Return an ``auto`` worker count from one already-completed calibration case."""

    cpu_limited_workers = _resolve_max_workers("auto")
    if pending_case_count <= 1 or available_memory_bytes is None:
        return cpu_limited_workers

    measured_memory = _current_process_memory_bytes()
    if measured_memory is None:
        return cpu_limited_workers

    per_worker_memory = max(int(measured_memory * AUTO_WORKER_MEMORY_SAFETY_FACTOR), 1)
    usable_memory = max(available_memory_bytes - AUTO_WORKER_MEMORY_RESERVE_BYTES, 0)
    if usable_memory <= 0:
        return 1
    memory_limited_workers = max(usable_memory // per_worker_memory, 1)
    return max(min(cpu_limited_workers, memory_limited_workers), 1)


def _parallel_worker_execute(payload: dict[str, object]) -> dict[str, object]:
    """Execute one payload in a worker and return a serializable envelope."""

    try:
        logger.info(
            "[parallel-worker] running workflow=%s energy=%.6f eV %s",
            str(payload.get("workflow", "single")),
            float(payload["energy_ev"]),
            _worker_identity(),
        )
        result, peak_memory_bytes, wall_seconds = _run_case_payload_with_optional_memory_profile(payload)
        message: dict[str, object] = {"success": True, "result": _single_result_to_record(result)}
        if peak_memory_bytes is not None:
            message["peak_memory_bytes"] = peak_memory_bytes
        if wall_seconds is not None:
            message["wall_seconds"] = wall_seconds
        return message
    except Exception as error:  # pragma: no cover - exercised in parent tests
        return {"success": False, "error": str(error)}


def _completed_case_ids(checkpoint_path: Path) -> set[str]:
    """Load completed case IDs from an append-only JSONL checkpoint."""

    if not checkpoint_path.exists():
        return set()
    completed: set[str] = set()
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(str(record["case_id"]))
    return completed


def _load_checkpoint_case_results(checkpoint_path: Path) -> dict[str, CaseExecutionResult]:
    """Load deduplicated checkpoint case results keyed by case ID.

    Malformed lines or records that cannot be converted are ignored so the caller
    can recompute those cases during resume.
    """

    if not checkpoint_path.exists():
        return {}
    loaded: dict[str, CaseExecutionResult] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                case_result = _case_result_from_record(record)
            except Exception:
                logger.warning("Ignoring malformed checkpoint record during resume.")
                continue
            loaded[case_result.case_id] = case_result
    return loaded


class BatchSimulationRunner:
    """Stream RCWA simulations for arbitrary case iterables.

    Args:
        default_diffraction_order: Default selected diffraction order.
        default_fourier_orders: Default Fourier truncation order.
        max_fourier_orders: Maximum allowed Fourier orders.
        backend: Fourier coefficient backend selector. ``"numba"`` is the
            default backend. ``"numpy"`` remains available temporarily for
            compatibility but is deprecated and will be removed in a future
            version.
        checkpoint_dir: Directory for ``results.jsonl`` and ``metadata.json``.
        checkpoint_interval: Flush checkpoint file every N completed cases.
        resume: Whether to skip case IDs already present in the checkpoint.
        stop_event: Optional cooperative stop signal checked between case submissions.
        on_worker_pids_changed: Optional callback receiving current worker PIDs.
        live_plot: Whether to update a live plot during execution.
        live_plot_x_key: Case/result field used for the live-plot x axis.
        live_plot_order_count: Number of positive diffraction orders to plot.
        live_plot_reference_data: Optional two-column reference data to overlay.
        validate_physical_results: Whether to validate reflected efficiencies by checking
            the minimum reflected efficiency, maximum reflected efficiency, and
            maximum total propagating reflected efficiency.
        max_reflected_efficiency: Maximum allowed single-order reflected efficiency.
        min_reflected_efficiency: Minimum allowed reflected efficiency.
        retry_on_selected_efficiency_zero: Whether to retry multilayer theta-search
            cases in inline execution when selected efficiency is less than or equal
            to ``retry_selected_efficiency_threshold``.
        retry_selected_efficiency_threshold: Retry trigger threshold for selected
            efficiency. Retries are attempted when selected efficiency is less
            than or equal to this value.
        max_zero_efficiency_retries: Maximum number of additional retries for
            zero-efficiency multilayer theta-search cases.
        execution_mode: ``inline`` or ``subprocess`` execution.
        max_workers: Optional case-level multiprocessing worker count.
        timeout: Subprocess timeout in seconds.
        show_progress: Whether to show a progress bar.
        on_error: Error policy for per-case execution failures such as solver
            exceptions, validation failures, timeout failures, or worker/subprocess
            failures. ``continue`` converts a failed case into a result with
            ``status="error"`` and keeps processing later cases; ``fail_fast``
            raises immediately and stops the batch. Use ``continue`` for exploratory
            or long sweeps where partial results are still useful, and ``fail_fast``
            when a failing case likely means the configuration is wrong and should be
            corrected before continuing.
    """

    def __init__(
        self,
        *,
        default_diffraction_order: int = 1,
        default_fourier_orders: int = 25,
        execution_mode: ExecutionMode = "inline",
        max_workers: MaxWorkers = None,
        timeout: float = 3600,
        show_progress: bool = False,
        live_plot: bool = False,
        live_plot_x_key: str = "index",
        live_plot_order_count: int = 1,
        live_plot_reference_data: np.ndarray | None = None,
        on_error: ErrorPolicy = "continue",
        max_fourier_orders: int = 100,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 1,
        resume: bool = False,
        stop_event: threading.Event | None = None,
        on_worker_pids_changed: Callable[[set[int]], None] | None = None,
        validate_physical_results: bool = True,
        max_reflected_efficiency: float = 1.05,
        min_reflected_efficiency: float = -1e-8,
        retry_on_selected_efficiency_zero: bool = True,
        retry_selected_efficiency_threshold: float = 1e-4,
        max_zero_efficiency_retries: int = 3,
        backend: str = "numba",
        default_polarization: str = "s",
    ) -> None:
        """Initialize a streaming batch simulation runner.

        Configures batch execution parameters for RCWA simulations. Supports inline
        single-threaded execution and multiprocessing via subprocess workers with
        automatic memory calibration.

        Args:
            default_diffraction_order: Default diffraction order for efficiency selection
                when case does not specify. Must be positive integer.
            default_fourier_orders: Default Fourier truncation orders when case does
                not specify. Higher values improve accuracy but increase computation.
            execution_mode: Execution strategy. ``inline`` runs in current process,
                ``subprocess`` spawns separate worker processes.
            max_workers: Worker count for parallel execution. ``"auto"`` calibrates
                from available memory and single-case profile. Integer specifies exact count.
            timeout: Maximum seconds per case before considering it failed.
            show_progress: Display progress bar during execution.
            live_plot: Enable real-time efficiency plotting during execution.
            live_plot_x_key: Case field to use for x-axis. One of "index", "energy_ev", "grazing_angle_deg".
            live_plot_order_count: Number of diffraction orders to plot.
            live_plot_reference_data: Optional experimental data array with shape (N, 3) for comparison.
            on_error: Error policy for per-case execution failures such as solver
                exceptions, validation failures, timeout failures, or
                worker/subprocess failures. ``continue`` converts a failed case into
                a result with ``status="error"`` and keeps processing later cases;
                ``fail_fast`` raises immediately and stops the batch. Use
                ``continue`` for exploratory or long sweeps where partial results are
                still useful, and ``fail_fast`` when a failing case likely means the
                configuration is wrong and should be corrected before continuing.
            max_fourier_orders: Maximum Fourier orders for validation warnings.
            checkpoint_dir: Directory for checkpoint persistence. Enables resume capability.
            checkpoint_interval: Number of cases between checkpoint writes.
            resume: Restore previous results from checkpoint directory.
            stop_event: Optional cooperative stop signal checked before launching
                new work. In-flight work is allowed to finish.
            on_worker_pids_changed: Optional callback receiving the current set of
                worker subprocess IDs after executor changes.
            validate_physical_results: Enforce physical constraints on reflected efficiencies
                by checking the minimum reflected efficiency, maximum reflected efficiency,
                and maximum total propagating reflected efficiency.
            max_reflected_efficiency: Maximum allowed reflected efficiency for validation.
            min_reflected_efficiency: Minimum allowed reflected efficiency (may be slightly
                negative due to numerics).
            retry_on_selected_efficiency_zero: Retry inline multilayer theta-search simulations
                when selected efficiency is less than or equal to
                ``retry_selected_efficiency_threshold``.
            retry_selected_efficiency_threshold: Threshold that triggers retry when
                ``selected_efficiency <= retry_selected_efficiency_threshold``.
            max_zero_efficiency_retries: Maximum retry attempts for threshold-triggered cases.
            backend: Fourier coefficient backend selector. ``"numba"`` is the
                default backend. ``"numpy"`` remains available temporarily for
                compatibility but is deprecated and will be removed in a future
                version.
            default_polarization: Default polarization used when a case omits the
                value. Must be ``"s"`` or ``"p"``.

        Example:
            >>> runner = BatchSimulationRunner(
            ...     default_diffraction_order=1,
            ...     default_fourier_orders=25,
            ...     max_workers="auto",
            ...     checkpoint_dir="results",
            ...     resume=True
            ... )
            >>> for result in runner.run_cases(cases):
            ...     print(f"E={result.energy_ev:.1f} eV, eff={result.selected_efficiency:.3f}")
        """

        if execution_mode not in {"inline", "subprocess"}:
            raise ValueError("execution_mode must be 'inline' or 'subprocess'.")
        if on_error not in {"continue", "fail_fast"}:
            raise ValueError("on_error must be 'continue' or 'fail_fast'.")
        _warn_if_numpy_backend_requested(backend, stacklevel=2)
        if not np.isfinite(retry_selected_efficiency_threshold) or retry_selected_efficiency_threshold < 0.0:
            raise ValueError("retry_selected_efficiency_threshold must be finite and >= 0.0.")
        resolved_max_workers = _resolve_max_workers(max_workers)
        if resolved_max_workers > 1 and execution_mode == "subprocess":
            raise ValueError("max_workers > 1 cannot be combined with execution_mode='subprocess'.")
        if resume and checkpoint_dir is None:
            raise ValueError(
                "resume=True requires checkpoint_dir to be specified. "
                "Please provide a checkpoint_dir path to enable resumption from checkpoint."
            )
        self.default_diffraction_order = default_diffraction_order
        self.default_fourier_orders = default_fourier_orders
        self.execution_mode = execution_mode
        self.max_workers = max_workers
        self.resolved_max_workers = resolved_max_workers
        self.timeout = timeout
        self.show_progress = show_progress
        self.live_plot = live_plot
        self.live_plot_x_key = live_plot_x_key
        self.live_plot_order_count = live_plot_order_count
        self.live_plot_reference_data = live_plot_reference_data
        self.on_error = on_error
        self.max_fourier_orders = max_fourier_orders
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.resume = resume
        self.stop_event = stop_event
        self.on_worker_pids_changed = on_worker_pids_changed
        self.validate_physical_results = validate_physical_results
        self.max_reflected_efficiency = max_reflected_efficiency
        self.min_reflected_efficiency = min_reflected_efficiency
        self.retry_on_selected_efficiency_zero = retry_on_selected_efficiency_zero
        self.retry_selected_efficiency_threshold = float(retry_selected_efficiency_threshold)
        self.max_zero_efficiency_retries = max(0, int(max_zero_efficiency_retries))
        self.backend = backend
        self.default_polarization = default_polarization
        if self.default_polarization not in {"s", "p"}:
            raise ValueError("default_polarization must be 's' or 'p'.")
        self._live_figure: plt.Figure | None = None
        self._live_axis: plt.Axes | None = None
        self._live_x_values: list[float] = []
        self._live_y_values: dict[int, list[float]] = {
            order: [] for order in range(1, live_plot_order_count + 1)
        }
        self.stopped_early = False

    @property
    def checkpoint_path(self) -> Path | None:
        """Return the JSONL checkpoint path, if checkpointing is enabled.

        Returns:
            Path to results.jsonl in checkpoint directory, or None if checkpointing disabled.
        """

        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "results.jsonl"

    def run_cases(
        self,
        cases: Iterable[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> Iterator[CaseExecutionResult]:
        """Yield simulation results for an arbitrary iterable of cases.

        Executes RCWA simulations for all provided cases with support for
        single-threaded inline execution or multiprocessing via subprocess workers,
        real-time progress bars and live plotting, checkpoint persistence and
        resume capability, automatic worker calibration for memory-efficient
        parallel execution, retry logic for failed or low-efficiency multilayer
        theta-search cases, and reflected-efficiency validation.

        Each case must provide ``grating`` and ``energy_ev``. Optional fields
        include ``case_id``, ``diffraction_order``, ``fourier_orders``,
        ``x_resolution_nm``, ``z_resolution_nm``, ``grazing_angle_deg``,
        and ``polarization``.
        Workflow-specific cases such as multilayer theta search include
        additional fields that trigger adaptive three-stage scanning internally.

        Args:
            cases: Iterable of case dictionaries. Each case must include
                ``grating`` and ``energy_ev``. ``case_id`` is optional; when
                omitted, a deterministic ID is generated from workflow/index.
                Fixed-angle cases also provide ``grazing_angle_deg``;
                workflow-tagged cases may resolve the angle internally.
                Extra serializable metadata such as ``label`` or ``depth_nm``
                is preserved in ``case_data`` on the yielded results.
            metadata: Optional run metadata saved next to checkpoints.
                Saved to metadata.json in checkpoint directory.

        Yields:
            Per-case execution results as each case completes. Each result
            includes basic identifiers and selected-order outputs, full
            efficiency and diffraction-angle arrays, status and retry metadata,
            and theta-search diagnostics or tracking metadata when applicable.

        Example:
            >>> runner = BatchSimulationRunner(max_workers="auto", checkpoint_dir="results")
            >>> cases = fixed_angle_cases(grating, energies_ev=[500, 600, 700], grazing_angle_deg=5.0)
            >>> for result in runner.run_cases(cases):
            ...     if result.status == "ok":
            ...         print(f"E={result.energy_ev:.1f} eV, eff={result.selected_efficiency:.4f}")
        """

        checkpoint_path = self.checkpoint_path
        completed_ids: set[str] = set()
        checkpoint_handle = None
        completed_since_flush = 0
        progress = None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata(metadata or {})
            if checkpoint_path is not None and self.resume:
                completed_ids = _completed_case_ids(checkpoint_path)
            checkpoint_handle = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None

        iterable: Iterable[dict[str, object]] = cases
        try:
            pending_cases = self._prepare_pending_cases(iterable, completed_ids)
            effective_total_cases = len(completed_ids) + len(pending_cases) if self.resume else len(pending_cases)
            if self.show_progress:
                progress = _simulation_api().tqdm(total=effective_total_cases, desc="RCWA batch", unit="case")
                if self.resume and completed_ids:
                    progress.update(len(completed_ids))
            calibrated_result = None
            if self.max_workers == "auto" and pending_cases:
                calibrated_result = self._execute_case(*pending_cases[0])
                pending_cases = pending_cases[1:]
                self.resolved_max_workers = _calibrate_auto_max_workers_from_result(
                    pending_case_count=len(pending_cases) + 1,
                    available_memory_bytes=_available_memory_bytes(),
                )
            if self.resolved_max_workers == 1:
                result_iterator = self._run_serial_cases(pending_cases)
            else:
                result_iterator = self._run_parallel_cases(pending_cases)
            if calibrated_result is not None:
                if checkpoint_handle is not None:
                    checkpoint_handle.write(json.dumps(_case_result_to_record(calibrated_result)) + "\n")
                    completed_since_flush += 1
                    if completed_since_flush >= self.checkpoint_interval:
                        checkpoint_handle.flush()
                        completed_since_flush = 0
                if calibrated_result.status == "ok":
                    self._update_live_plot(calibrated_result)
                if progress is not None:
                    progress.update(1)
                yield calibrated_result
            for result in result_iterator:
                if checkpoint_handle is not None:
                    checkpoint_handle.write(json.dumps(_case_result_to_record(result)) + "\n")
                    completed_since_flush += 1
                    if completed_since_flush >= self.checkpoint_interval:
                        checkpoint_handle.flush()
                        completed_since_flush = 0
                if result.status == "ok":
                    self._update_live_plot(result)
                if progress is not None:
                    progress.update(1)
                yield result
        finally:
            if checkpoint_handle is not None:
                checkpoint_handle.flush()
                checkpoint_handle.close()
            if progress is not None:
                progress.close()
            if self.live_plot:
                self.close_live_plot()

    def _settings(self) -> dict[str, object]:
        """Return runner settings used to prepare each case payload.

        Constructs the configuration dictionary passed to each RCWA simulation
        execution. Contains all global settings that affect simulation behavior
        and validation.

        Returns:
            Dictionary with settings for: default_diffraction_order,
            default_fourier_orders, max_fourier_orders, validate_physical_results,
            max_reflected_efficiency, min_reflected_efficiency,
            backend.
        """

        return {
            "default_diffraction_order": self.default_diffraction_order,
            "default_fourier_orders": self.default_fourier_orders,
            "max_fourier_orders": self.max_fourier_orders,
            "validate_physical_results": self.validate_physical_results,
            "max_reflected_efficiency": self.max_reflected_efficiency,
            "min_reflected_efficiency": self.min_reflected_efficiency,
            "backend": self.backend,
            "default_polarization": self.default_polarization,
        }

    def _prepare_pending_cases(
        self,
        iterable: Iterable[dict[str, object]],
        completed_ids: set[str],
    ) -> list[tuple[int, str, dict[str, object]]]:
        """Return runnable cases after resume filtering.

        Filters out cases that have already been completed (found in completed_ids).
        Assigns sequential indices and generated case IDs to pending cases.

        Args:
            iterable: Raw case iterable from the user.
            completed_ids: Set of case IDs that have already been completed.

        Returns:
            List of (index, case_id, case) tuples ready for execution.
        """

        pending_cases: list[tuple[int, str, dict[str, object]]] = []
        for index, case in enumerate(iterable):
            case_id = self._resolve_case_id(case, index)
            if case_id in completed_ids:
                continue
            pending_cases.append((index, case_id, case))
        return pending_cases

    def _stop_requested(self) -> bool:
        """Return whether cooperative stopping has been requested."""

        return self.stop_event is not None and self.stop_event.is_set()

    def _emit_worker_pids(self, worker_pids: set[int]) -> None:
        """Notify listeners about the current worker subprocess IDs."""

        if self.on_worker_pids_changed is None:
            return
        self.on_worker_pids_changed(set(worker_pids))

    def _run_serial_cases(
        self,
        pending_cases: Sequence[tuple[int, str, dict[str, object]]],
    ) -> Iterator[CaseExecutionResult]:
        """Yield serially executed case results.

        Executes cases one at a time in the current process. Used when max_workers=1
        or execution_mode="subprocess". Includes logging for each case execution.

        Args:
            pending_cases: List of (index, case_id, case) tuples to execute.

        Yields:
            CaseExecutionResult for each completed case.
        """

        for index, case_id, case in pending_cases:
            if self._stop_requested():
                self.stopped_early = True
                return
            _log_case_execution_start(case_id=case_id, case=case, index=index, location="serial")
            yield self._execute_case(index, case_id, case)

    def _run_parallel_cases(
        self,
        pending_cases: Sequence[tuple[int, str, dict[str, object]]],
    ) -> Iterator[CaseExecutionResult]:
        """Yield multiprocessing case results in completion order.

        Executes cases in parallel using subprocess workers. Implements a
        work queue pattern where workers pick up cases as they become available.
        Results are yielded as soon as each case completes.

        Args:
            pending_cases: List of (index, case_id, case) tuples to execute.

        Yields:
            CaseExecutionResult for each completed case, in completion order
            (not necessarily the same as input order).

        Note:
            Uses multiprocessing with spawn start method for cross-platform
            compatibility. Workers are automatically calibrated based on
            available memory.
        """

        context = mp.get_context(_multiprocessing_start_method())
        max_workers = self.resolved_max_workers
        settings = self._settings()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
            initializer=_worker_initializer,
        ) as executor:
            futures: dict[concurrent.futures.Future[dict[str, object]], tuple[int, str, dict[str, object], dict[str, object]]] = {}
            pending_iter = iter(pending_cases)

            def submit_next_case() -> bool:
                """Submit one additional case when available."""

                if self._stop_requested():
                    self.stopped_early = True
                    return False
                try:
                    index, case_id, case = next(pending_iter)
                except StopIteration:
                    return False
                _log_case_execution_start(case_id=case_id, case=case, index=index, location="parallel-submit")
                payload = _case_payload(case, settings)
                futures[executor.submit(_parallel_worker_execute, payload)] = (
                    index,
                    case_id,
                    case,
                    {key: value for key, value in case.items() if key != "grating"},
                )
                worker_pids = {process.pid for process in executor._processes.values() if process.pid is not None}
                self._emit_worker_pids(worker_pids)
                return True

            for _ in range(max_workers):
                if not submit_next_case():
                    break

            try:
                while futures:
                    completed, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed:
                        index, case_id, case, case_data = futures.pop(future)
                        label = case.get("label")
                        energy_ev = float(case["energy_ev"])
                        grazing_angle_deg = self._case_energy_and_angle(case)[1]
                        message = future.result()
                        if message["success"]:
                            single = _single_result_from_record(message["result"])  # type: ignore[arg-type]
                            peak_memory_bytes = message.get("peak_memory_bytes")
                            wall_seconds = message.get("wall_seconds")
                            yield CaseExecutionResult(
                                case_id=case_id,
                                index=index,
                                label=None if label is None else str(label),
                                energy_ev=single.energy_ev,
                                grazing_angle_deg=single.grazing_angle_deg,
                                orders=single.orders,
                                selected_efficiency=single.selected_efficiency,
                                selected_diffraction_angle_deg=single.selected_diffraction_angle_deg,
                                efficiency_all=single.efficiency_all,
                                diffraction_angle_all=single.diffraction_angle_all,
                                status="ok",
                                case_data=case_data,
                                polarization=single.polarization,
                                theta_search_diagnostics=single.theta_search_diagnostics,
                                retry_triggered=single.retry_triggered,
                                retry_attempts=single.retry_attempts,
                                retry_status=single.retry_status,
                                selected_efficiency_is_exact_zero=single.selected_efficiency_is_exact_zero,
                                selected_efficiency_below_retry_threshold=single.selected_efficiency_below_retry_threshold,
                                peak_memory_bytes=(
                                    None if peak_memory_bytes is None else int(peak_memory_bytes)
                                ),
                                wall_seconds=None if wall_seconds is None else float(wall_seconds),
                            )
                        else:
                            if self.on_error == "fail_fast":
                                for pending in futures:
                                    pending.cancel()
                                raise RuntimeError(str(message["error"]))
                            yield CaseExecutionResult(
                                case_id=case_id,
                                index=index,
                                label=None if label is None else str(label),
                                energy_ev=energy_ev,
                                grazing_angle_deg=grazing_angle_deg,
                                orders=np.asarray([], dtype=int),
                                selected_efficiency=float("nan"),
                                selected_diffraction_angle_deg=float("nan"),
                                efficiency_all=np.asarray([], dtype=float),
                                diffraction_angle_all=np.asarray([], dtype=float),
                                status="error",
                                error_message=str(message["error"]),
                                case_data=case_data,
                                polarization=str(case.get("polarization", self.default_polarization)),
                                peak_memory_bytes=None,
                                wall_seconds=None,
                            )
                        if submit_next_case():
                            continue
                    worker_pids = {process.pid for process in executor._processes.values() if process.pid is not None}
                    self._emit_worker_pids(worker_pids)
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                self._emit_worker_pids(set())

    def _execute_case(
        self,
        index: int,
        case_id: str,
        case: dict[str, object],
    ) -> CaseExecutionResult:
        """Execute one batch case and return a case result.

        Runs a single RCWA simulation with full retry and validation logic.
        Handles both inline execution and subprocess execution modes.

        Workflow for multilayer theta search cases:
        1. Execute primary scan with configured resolution
        2. Check if retry needed (low efficiency or zero)
        3. If retry triggered, re-execute with theta jitter
        4. Repeat until efficiency exceeds threshold or retries exhausted

        For other cases, executes once and returns result directly.

        Args:
            index: Sequential index of this case in the batch.
            case_id: Unique identifier for this case.
            case: Case dictionary with grating, energy_ev, and other parameters.

        Returns:
            CaseExecutionResult with simulation results and status.

        Retry behavior:
        - Only triggered for multilayer_theta_search workflow
        - Only in inline execution mode (diagnostic access needed)
        - Uses configurable jitter values for theta perturbation
        - Stops when efficiency exceeds retry threshold
        """

        label = case.get("label")
        energy_ev, grazing_angle_deg = self._case_energy_and_angle(case)
        case_data = {key: value for key, value in case.items() if key != "grating"}
        try:
            payload = _case_payload(case, self._settings())
            if self.execution_mode == "inline":
                single, peak_memory_bytes, wall_seconds = _run_case_payload_with_optional_memory_profile(payload)
            else:
                single, peak_memory_bytes, wall_seconds = _run_payload_in_subprocess(payload, timeout=self.timeout)
            retry_triggered = False
            retry_attempts = 0
            retry_status = "not_needed"
            selected_efficiency_is_exact_zero = bool(single.selected_efficiency == 0.0)
            selected_efficiency_below_retry_threshold = bool(
                single.selected_efficiency <= self.retry_selected_efficiency_threshold
            )
            if (
                self.retry_on_selected_efficiency_zero
                and case.get("workflow") == "multilayer_theta_search"
                and self.execution_mode == "inline"
                and selected_efficiency_below_retry_threshold
            ):
                retry_triggered = True
                base_initial = case.get("initial_grazing_angle_deg")
                if base_initial is None and single.theta_search_diagnostics is not None:
                    base_initial = float(single.theta_search_diagnostics.estimated_grazing_angle_deg)
                if base_initial is None:
                    base_initial = float(single.grazing_angle_deg)
                jitter_values = _BATCH_THETA_RETRY_JITTER_DEG[: self.max_zero_efficiency_retries]
                for jitter in jitter_values:
                    retry_attempts += 1
                    retry_case = dict(case)
                    retry_case["initial_grazing_angle_deg"] = float(base_initial) + float(jitter)
                    retry_payload = _case_payload(retry_case, self._settings())
                    retry_single, retry_peak_memory_bytes, retry_wall_seconds = _run_case_payload_with_optional_memory_profile(
                        retry_payload
                    )
                    single = retry_single
                    if retry_peak_memory_bytes is not None:
                        peak_memory_bytes = (
                            retry_peak_memory_bytes
                            if peak_memory_bytes is None
                            else max(int(peak_memory_bytes), int(retry_peak_memory_bytes))
                        )
                    if retry_wall_seconds is not None:
                        wall_seconds = (
                            float(retry_wall_seconds)
                            if wall_seconds is None
                            else float(wall_seconds) + float(retry_wall_seconds)
                        )
                    selected_efficiency_is_exact_zero = bool(single.selected_efficiency == 0.0)
                    selected_efficiency_below_retry_threshold = bool(
                        single.selected_efficiency <= self.retry_selected_efficiency_threshold
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
            return CaseExecutionResult(
                case_id=case_id,
                index=index,
                label=None if label is None else str(label),
                energy_ev=single.energy_ev,
                grazing_angle_deg=single.grazing_angle_deg,
                orders=single.orders,
                selected_efficiency=single.selected_efficiency,
                selected_diffraction_angle_deg=single.selected_diffraction_angle_deg,
                efficiency_all=single.efficiency_all,
                diffraction_angle_all=single.diffraction_angle_all,
                status="ok",
                case_data=case_data,
                polarization=single.polarization,
                theta_search_diagnostics=single.theta_search_diagnostics,
                retry_triggered=single.retry_triggered,
                retry_attempts=single.retry_attempts,
                retry_status=single.retry_status,
                selected_efficiency_is_exact_zero=single.selected_efficiency_is_exact_zero,
                selected_efficiency_below_retry_threshold=single.selected_efficiency_below_retry_threshold,
                peak_memory_bytes=peak_memory_bytes,
                wall_seconds=wall_seconds,
            )
        except Exception as error:
            if self.on_error == "fail_fast":
                raise
            return CaseExecutionResult(
                case_id=case_id,
                index=index,
                label=None if label is None else str(label),
                energy_ev=energy_ev,
                grazing_angle_deg=grazing_angle_deg,
                orders=np.asarray([], dtype=int),
                selected_efficiency=float("nan"),
                selected_diffraction_angle_deg=float("nan"),
                efficiency_all=np.asarray([], dtype=float),
                diffraction_angle_all=np.asarray([], dtype=float),
                status="error",
                error_message=str(error),
                case_data=case_data,
                polarization=str(case.get("polarization", self.default_polarization)),
                peak_memory_bytes=None,
                wall_seconds=None,
            )

    def _resolve_case_id(self, case: dict[str, object], index: int) -> str:
        """Return a stable case ID, generating one deterministically when missing.

        Resolves case ID from case dictionary, or generates one if not present.
        Generated IDs use workflow name and zero-padded index for stability.

        Args:
            case: Case dictionary with optional case_id field.
            index: Sequential index for ID generation.

        Returns:
            Case ID string (from case or generated as "workflow-NNNNNNNN").

        Example:
            >>> runner = BatchSimulationRunner()
            >>> case = {"energy_ev": 500, "grating": grating}
            >>> runner._resolve_case_id(case, 42)
            'batch-00000042'
            >>> case_with_id = {"case_id": "my-case", "energy_ev": 500, "grating": grating}
            >>> runner._resolve_case_id(case_with_id, 42)
            'my-case'
        """

        if "case_id" in case and case["case_id"] is not None:
            return str(case["case_id"])
        workflow = str(case.get("workflow", "batch"))
        return f"{workflow}-{index:08d}"

    def _write_metadata(self, metadata: dict[str, object]) -> None:
        """Write small run metadata next to the append-only checkpoint.

        Saves run metadata to metadata.json in the checkpoint directory.
        Contains user-provided metadata plus execution configuration.

        Args:
            metadata: Dictionary with arbitrary metadata to persist.

        File format:
            JSON file with keys from user metadata plus runner configuration.
            Saved to {checkpoint_dir}/metadata.json
        """

        if self.checkpoint_dir is None:
            return
        payload = dict(metadata)
        payload.setdefault("created", datetime.now().isoformat())
        payload.update(
            {
                "execution_mode": self.execution_mode,
                "max_workers": self.max_workers,
                "resolved_max_workers": self.resolved_max_workers,
                "default_diffraction_order": self.default_diffraction_order,
                "default_fourier_orders": self.default_fourier_orders,
            }
        )
        with (self.checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _case_energy_and_angle(self, case: dict[str, object]) -> tuple[float, float]:
        """Return the nominal energy and angle placeholders for one case.

        Extracts energy and grazing angle from case dictionary for live plotting.
        Uses NaN for missing grazing angles (workflow cases resolve internally).

        Args:
            case: Case dictionary with energy_ev and optional grazing_angle_deg.

        Returns:
            Tuple of (energy_ev, grazing_angle_deg). Grazing angle is NaN if not
            present in case (expected for workflow cases that resolve angle internally).

        Note:
            Live plotting uses these nominal values; actual results may differ.
        """

        energy_ev = float(case["energy_ev"])
        grazing_angle_deg = float(case["grazing_angle_deg"]) if "grazing_angle_deg" in case else float("nan")
        return energy_ev, grazing_angle_deg

    def _extract_live_plot_x(self, case: CaseExecutionResult) -> float:
        """Return the x value for one streamed case in the live plot."""

        if self.live_plot_x_key == "index":
            return float(case.index + 1)
        if self.live_plot_x_key in case.case_data:
            return float(case.case_data[self.live_plot_x_key])
        if hasattr(case, self.live_plot_x_key):
            return float(getattr(case, self.live_plot_x_key))
        raise KeyError(f"Unable to extract live-plot x axis from key '{self.live_plot_x_key}'.")

    def close_live_plot(self) -> None:
        """Close and reset the live plot figure, if one exists."""

        if self._live_figure is not None and plt.fignum_exists(self._live_figure.number):
            plt.close(self._live_figure)
        self._live_figure = None
        self._live_axis = None
        self._live_x_values = []
        self._live_y_values = {order: [] for order in range(1, self.live_plot_order_count + 1)}

    def _update_live_plot(self, case: CaseExecutionResult) -> None:
        """Update the live plot incrementally from one successful result.

        Adds one point to the live efficiency plot. Supports multiple diffraction
        orders and optional experimental reference data for comparison.

        Args:
            case: CaseExecutionResult with simulation results to plot.

        Plot configuration:
        - X-axis: Controlled by live_plot_x_key (index, energy_ev, or grazing_angle_deg)
        - Y-axis: Selected order efficiency
        - Multiple orders: Displayed with different markers
        - Reference data: Overlaid as gray line if provided

        Note:
            Only updates plot for successful ("ok") results. Skips if plot is
            disabled or figure has been closed.
        """

        if not self.live_plot:
            return
        if self._live_figure is None or self._live_axis is None or not plt.fignum_exists(self._live_figure.number):
            plt.ion()
            self._live_figure, self._live_axis = plt.subplots(figsize=(10, 6))

        x_value = self._extract_live_plot_x(case)
        self._live_x_values.append(x_value)
        for order in range(1, self.live_plot_order_count + 1):
            self._live_y_values.setdefault(order, []).append(
                efficiency_for_order(case.orders, case.efficiency_all, diffraction_order=order)
            )

        sorted_indices = sorted(range(len(self._live_x_values)), key=self._live_x_values.__getitem__)
        sorted_x_values = [self._live_x_values[index] for index in sorted_indices]
        axis = self._live_axis
        axis.clear()
        markers = ["o", "s", "^", "d", "v", "x"]
        for order in range(1, self.live_plot_order_count + 1):
            sorted_order_values = [self._live_y_values[order][index] for index in sorted_indices]
            axis.plot(
                sorted_x_values,
                sorted_order_values,
                f"{markers[(order - 1) % len(markers)]}-",
                linewidth=1.0,
                markersize=3.0,
                label=f"Order {order}",
            )
        if self.live_plot_reference_data is not None:
            axis.plot(
                self.live_plot_reference_data[:, 0],
                self.live_plot_reference_data[:, 1],
                "k--",
                linewidth=1.0,
                label="Reference",
            )
        axis.set_xlabel(self.live_plot_x_key)
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title("Batch Simulation Progress")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        self._live_figure.tight_layout()
        self._live_figure.canvas.draw()
        self._live_figure.canvas.flush_events()
        _refresh_interactive_figure(self._live_figure)
