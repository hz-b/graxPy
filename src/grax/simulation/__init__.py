"""Streaming RCWA simulation helpers."""

import concurrent
import os

from tqdm import tqdm

from . import batch as _batch
from .batch import (
    BatchSimulationRunner,
    _run_case_payload,
    _available_memory_bytes,
    _multiprocessing_start_method,
    _parallel_worker_execute,
    _worker_initializer,
)
from .cases import (
    energy_angle_cases,
    fixed_angle_cases,
    monochromator_cases,
    monochromator_grazing_angles_deg,
    multilayer_theta_search_cases,
)
from .core import (
    GratingSimulation,
    efficiency_for_order,
    load_experimental_csv,
    plot_order_subset,
    run_simulation,
    write_all_orders_csv,
)
from .models import (
    AUTO_WORKER_MEMORY_RESERVE_BYTES,
    AUTO_WORKER_MEMORY_SAFETY_FACTOR,
    BatchSimulationResult,
    CaseExecutionResult,
    MultilayerThetaSearchSweepResult,
    SimulationResult,
    SingleSimulationResult,
    ThetaSearchDiagnostics,
)
from .serialization import (
    _case_result_from_record,
    _case_result_to_record,
    _single_result_from_record,
    _single_result_to_record,
)
from .theta_search import _precise_scan_fwhm_deg, estimate_multilayer_bragg_angle_deg, run_multilayer_theta_search
from .theta_search_sweep import _adaptive_scan_half_widths, run_multilayer_theta_search_sweep


def _resolve_max_workers(max_workers):
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


def _current_process_memory_bytes():
    """Return current RSS memory usage for the active process in bytes."""

    return _batch._current_process_memory_bytes()


def _peak_process_memory_bytes():
    """Return the process's lifetime peak RSS (high-water mark) in bytes."""

    return _batch._peak_process_memory_bytes()


def _calibrate_auto_max_workers_from_result(*, pending_case_count, available_memory_bytes):
    """Return an ``auto`` worker count from one already-completed calibration case."""

    cpu_limited_workers = _resolve_max_workers("auto")
    if pending_case_count <= 1 or available_memory_bytes is None:
        return cpu_limited_workers
    # Size from the per-solve peak RSS when available; steady-state RSS
    # understates a solve's transient high-water mark, which grows with
    # supercell count. Mirrors ``batch._calibrate_auto_max_workers_from_result``.
    candidate_bytes = [
        value
        for value in (_current_process_memory_bytes(), _peak_process_memory_bytes())
        if value is not None
    ]
    if not candidate_bytes:
        return cpu_limited_workers
    measured_memory = max(candidate_bytes)
    per_worker_memory = max(int(measured_memory * AUTO_WORKER_MEMORY_SAFETY_FACTOR), 1)
    usable_memory = max(available_memory_bytes - AUTO_WORKER_MEMORY_RESERVE_BYTES, 0)
    if usable_memory <= 0:
        return 1
    memory_limited_workers = max(usable_memory // per_worker_memory, 1)
    return max(min(cpu_limited_workers, memory_limited_workers), 1)

__all__ = [
    "AUTO_WORKER_MEMORY_RESERVE_BYTES",
    "AUTO_WORKER_MEMORY_SAFETY_FACTOR",
    "BatchSimulationRunner",
    "BatchSimulationResult",
    "CaseExecutionResult",
    "MultilayerThetaSearchSweepResult",
    "GratingSimulation",
    "SimulationResult",
    "SingleSimulationResult",
    "ThetaSearchDiagnostics",
    "efficiency_for_order",
    "energy_angle_cases",
    "estimate_multilayer_bragg_angle_deg",
    "fixed_angle_cases",
    "load_experimental_csv",
    "multilayer_theta_search_cases",
    "monochromator_cases",
    "monochromator_grazing_angles_deg",
    "plot_order_subset",
    "run_multilayer_theta_search",
    "run_multilayer_theta_search_sweep",
    "run_simulation",
    "write_all_orders_csv",
]

for _name in [
    "_adaptive_scan_half_widths",
    "_available_memory_bytes",
    "_calibrate_auto_max_workers_from_result",
    "_case_result_from_record",
    "_case_result_to_record",
    "_multiprocessing_start_method",
    "_parallel_worker_execute",
    "_precise_scan_fwhm_deg",
    "_resolve_max_workers",
    "_run_case_payload",
    "_single_result_from_record",
    "_single_result_to_record",
    "_worker_initializer",
]:
    globals()[_name].__module__ = __name__
