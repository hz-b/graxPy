"""Checkpoint and record serialization helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import CaseExecutionResult, SingleSimulationResult, ThetaSearchDiagnostics

logger = logging.getLogger(__name__)

def _single_result_to_record(result: SingleSimulationResult) -> dict[str, object]:
    """Convert a single result to a JSON-serializable record."""

    diagnostics_record = None
    if result.theta_search_diagnostics is not None:
        diagnostics_record = {
            "estimated_grazing_angle_deg": result.theta_search_diagnostics.estimated_grazing_angle_deg,
            "rough_grazing_angles_deg": result.theta_search_diagnostics.rough_grazing_angles_deg.tolist(),
            "rough_efficiencies": result.theta_search_diagnostics.rough_efficiencies.tolist(),
            "precise_grazing_angles_deg": result.theta_search_diagnostics.precise_grazing_angles_deg.tolist(),
            "precise_efficiencies": result.theta_search_diagnostics.precise_efficiencies.tolist(),
            "selected_grazing_angle_deg": result.theta_search_diagnostics.selected_grazing_angle_deg,
            "selected_efficiency": result.theta_search_diagnostics.selected_efficiency,
            "precise_fwhm_deg": result.theta_search_diagnostics.precise_fwhm_deg,
            "theta_tracking_center_mode": result.theta_search_diagnostics.theta_tracking_center_mode,
            "theta_tracking_auto_classification": result.theta_search_diagnostics.theta_tracking_auto_classification,
            "theta_tracking_previous_energy_ev": result.theta_search_diagnostics.theta_tracking_previous_energy_ev,
            "theta_tracking_previous_grazing_angle_deg": (
                result.theta_search_diagnostics.theta_tracking_previous_grazing_angle_deg
            ),
            "theta_tracking_used_previous_theta": result.theta_search_diagnostics.theta_tracking_used_previous_theta,
            "theta_tracking_bragg_fallback_triggered": (
                result.theta_search_diagnostics.theta_tracking_bragg_fallback_triggered
            ),
            "theta_tracking_continuity_rejected": result.theta_search_diagnostics.theta_tracking_continuity_rejected,
            "precise_peak_selection_mode_requested": (
                result.theta_search_diagnostics.precise_peak_selection_mode_requested
            ),
            "precise_peak_selection_mode_used": result.theta_search_diagnostics.precise_peak_selection_mode_used,
            "precise_peak_fit_fallback_used": result.theta_search_diagnostics.precise_peak_fit_fallback_used,
            "precise_peak_fitted_center_deg": result.theta_search_diagnostics.precise_peak_fitted_center_deg,
            "precise_peak_fitted_fwhm_deg": result.theta_search_diagnostics.precise_peak_fitted_fwhm_deg,
            "precise_peak_fitted_theta_deg": (
                None
                if result.theta_search_diagnostics.precise_peak_fitted_theta_deg is None
                else result.theta_search_diagnostics.precise_peak_fitted_theta_deg.tolist()
            ),
            "precise_peak_fitted_efficiencies": (
                None
                if result.theta_search_diagnostics.precise_peak_fitted_efficiencies is None
                else result.theta_search_diagnostics.precise_peak_fitted_efficiencies.tolist()
            ),
        }

    return {
        "energy_ev": result.energy_ev,
        "grazing_angle_deg": result.grazing_angle_deg,
        "orders": result.orders.tolist(),
        "selected_efficiency": result.selected_efficiency,
        "selected_diffraction_angle_deg": result.selected_diffraction_angle_deg,
        "efficiency_all": result.efficiency_all.tolist(),
        "diffraction_angle_all": result.diffraction_angle_all.tolist(),
        "diffraction_order": result.diffraction_order,
        "fourier_orders": result.fourier_orders,
        "roughness_sigma_nm": result.roughness_sigma_nm,
        "polarization": result.polarization,
        "solver": result.solver,
        "solver_options": result.solver_options,
        "theta_search_diagnostics": diagnostics_record,
        "retry_triggered": result.retry_triggered,
        "retry_attempts": result.retry_attempts,
        "retry_status": result.retry_status,
        "selected_efficiency_is_exact_zero": result.selected_efficiency_is_exact_zero,
        "selected_efficiency_below_retry_threshold": result.selected_efficiency_below_retry_threshold,
        "theta_tracking_center_mode": result.theta_tracking_center_mode,
        "theta_tracking_auto_classification": result.theta_tracking_auto_classification,
        "theta_tracking_previous_energy_ev": result.theta_tracking_previous_energy_ev,
        "theta_tracking_previous_grazing_angle_deg": result.theta_tracking_previous_grazing_angle_deg,
        "theta_tracking_used_previous_theta": result.theta_tracking_used_previous_theta,
        "theta_tracking_bragg_fallback_triggered": result.theta_tracking_bragg_fallback_triggered,
        "theta_tracking_continuity_rejected": result.theta_tracking_continuity_rejected,
    }


def _single_result_from_record(record: dict[str, object]) -> SingleSimulationResult:
    """Convert a JSON-compatible record to a single result."""

    diagnostics = None
    diagnostics_record = record.get("theta_search_diagnostics")
    if isinstance(diagnostics_record, dict):
        diagnostics = ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=float(diagnostics_record["estimated_grazing_angle_deg"]),
            rough_grazing_angles_deg=np.asarray(diagnostics_record["rough_grazing_angles_deg"], dtype=float),
            rough_efficiencies=np.asarray(diagnostics_record["rough_efficiencies"], dtype=float),
            precise_grazing_angles_deg=np.asarray(diagnostics_record["precise_grazing_angles_deg"], dtype=float),
            precise_efficiencies=np.asarray(diagnostics_record["precise_efficiencies"], dtype=float),
            selected_grazing_angle_deg=float(diagnostics_record["selected_grazing_angle_deg"]),
            selected_efficiency=float(diagnostics_record["selected_efficiency"]),
            precise_fwhm_deg=(
                None
                if diagnostics_record.get("precise_fwhm_deg") is None
                else float(diagnostics_record["precise_fwhm_deg"])
            ),
            theta_tracking_center_mode=str(diagnostics_record.get("theta_tracking_center_mode", "bragg")),
            theta_tracking_auto_classification=str(
                diagnostics_record.get("theta_tracking_auto_classification", "initial")
            ),
            theta_tracking_previous_energy_ev=(
                None
                if diagnostics_record.get("theta_tracking_previous_energy_ev") is None
                else float(diagnostics_record["theta_tracking_previous_energy_ev"])
            ),
            theta_tracking_previous_grazing_angle_deg=(
                None
                if diagnostics_record.get("theta_tracking_previous_grazing_angle_deg") is None
                else float(diagnostics_record["theta_tracking_previous_grazing_angle_deg"])
            ),
            theta_tracking_used_previous_theta=bool(
                diagnostics_record.get("theta_tracking_used_previous_theta", False)
            ),
            theta_tracking_bragg_fallback_triggered=bool(
                diagnostics_record.get("theta_tracking_bragg_fallback_triggered", False)
            ),
            theta_tracking_continuity_rejected=bool(
                diagnostics_record.get("theta_tracking_continuity_rejected", False)
            ),
            precise_peak_selection_mode_requested=str(
                diagnostics_record.get("precise_peak_selection_mode_requested", "max")
            ),
            precise_peak_selection_mode_used=str(diagnostics_record.get("precise_peak_selection_mode_used", "max")),
            precise_peak_fit_fallback_used=bool(diagnostics_record.get("precise_peak_fit_fallback_used", False)),
            precise_peak_fitted_center_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_center_deg") is None
                else float(diagnostics_record["precise_peak_fitted_center_deg"])
            ),
            precise_peak_fitted_fwhm_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_fwhm_deg") is None
                else float(diagnostics_record["precise_peak_fitted_fwhm_deg"])
            ),
            precise_peak_fitted_theta_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_theta_deg") is None
                else np.asarray(diagnostics_record["precise_peak_fitted_theta_deg"], dtype=float)
            ),
            precise_peak_fitted_efficiencies=(
                None
                if diagnostics_record.get("precise_peak_fitted_efficiencies") is None
                else np.asarray(diagnostics_record["precise_peak_fitted_efficiencies"], dtype=float)
            ),
        )

    return SingleSimulationResult(
        energy_ev=float(record["energy_ev"]),
        grazing_angle_deg=float(record["grazing_angle_deg"]),
        orders=np.asarray(record["orders"], dtype=float),
        selected_efficiency=float(record["selected_efficiency"]),
        selected_diffraction_angle_deg=float(record["selected_diffraction_angle_deg"]),
        efficiency_all=np.asarray(record["efficiency_all"], dtype=float),
        diffraction_angle_all=np.asarray(record["diffraction_angle_all"], dtype=float),
        diffraction_order=int(record["diffraction_order"]),
        fourier_orders=int(record["fourier_orders"]),
        roughness_sigma_nm=(
            None if record.get("roughness_sigma_nm") is None else float(record["roughness_sigma_nm"])
        ),
        polarization=str(record.get("polarization", "s")),
        solver=str(record.get("solver", "rcwa")),
        solver_options=record.get("solver_options"),
        theta_search_diagnostics=diagnostics,
        retry_triggered=bool(record.get("retry_triggered", False)),
        retry_attempts=int(record.get("retry_attempts", 0)),
        retry_status=str(record.get("retry_status", "not_needed")),
        selected_efficiency_is_exact_zero=bool(record.get("selected_efficiency_is_exact_zero", False)),
        selected_efficiency_below_retry_threshold=bool(
            record.get("selected_efficiency_below_retry_threshold", False)
        ),
        theta_tracking_center_mode=str(record.get("theta_tracking_center_mode", "bragg")),
        theta_tracking_auto_classification=str(record.get("theta_tracking_auto_classification", "initial")),
        theta_tracking_previous_energy_ev=(
            None
            if record.get("theta_tracking_previous_energy_ev") is None
            else float(record["theta_tracking_previous_energy_ev"])
        ),
        theta_tracking_previous_grazing_angle_deg=(
            None
            if record.get("theta_tracking_previous_grazing_angle_deg") is None
            else float(record["theta_tracking_previous_grazing_angle_deg"])
        ),
        theta_tracking_used_previous_theta=bool(record.get("theta_tracking_used_previous_theta", False)),
        theta_tracking_bragg_fallback_triggered=bool(
            record.get("theta_tracking_bragg_fallback_triggered", False)
        ),
        theta_tracking_continuity_rejected=bool(record.get("theta_tracking_continuity_rejected", False)),
    )


def _case_result_to_record(result: CaseExecutionResult) -> dict[str, object]:
    """Convert a case execution result to a JSON-serializable record."""

    diagnostics_record = None
    if result.theta_search_diagnostics is not None:
        diagnostics_record = {
            "estimated_grazing_angle_deg": result.theta_search_diagnostics.estimated_grazing_angle_deg,
            "rough_grazing_angles_deg": result.theta_search_diagnostics.rough_grazing_angles_deg.tolist(),
            "rough_efficiencies": result.theta_search_diagnostics.rough_efficiencies.tolist(),
            "precise_grazing_angles_deg": result.theta_search_diagnostics.precise_grazing_angles_deg.tolist(),
            "precise_efficiencies": result.theta_search_diagnostics.precise_efficiencies.tolist(),
            "selected_grazing_angle_deg": result.theta_search_diagnostics.selected_grazing_angle_deg,
            "selected_efficiency": result.theta_search_diagnostics.selected_efficiency,
            "precise_fwhm_deg": result.theta_search_diagnostics.precise_fwhm_deg,
            "theta_tracking_center_mode": result.theta_search_diagnostics.theta_tracking_center_mode,
            "theta_tracking_auto_classification": result.theta_search_diagnostics.theta_tracking_auto_classification,
            "theta_tracking_previous_energy_ev": result.theta_search_diagnostics.theta_tracking_previous_energy_ev,
            "theta_tracking_previous_grazing_angle_deg": (
                result.theta_search_diagnostics.theta_tracking_previous_grazing_angle_deg
            ),
            "theta_tracking_used_previous_theta": result.theta_search_diagnostics.theta_tracking_used_previous_theta,
            "theta_tracking_bragg_fallback_triggered": (
                result.theta_search_diagnostics.theta_tracking_bragg_fallback_triggered
            ),
            "theta_tracking_continuity_rejected": result.theta_search_diagnostics.theta_tracking_continuity_rejected,
            "precise_peak_selection_mode_requested": (
                result.theta_search_diagnostics.precise_peak_selection_mode_requested
            ),
            "precise_peak_selection_mode_used": result.theta_search_diagnostics.precise_peak_selection_mode_used,
            "precise_peak_fit_fallback_used": result.theta_search_diagnostics.precise_peak_fit_fallback_used,
            "precise_peak_fitted_center_deg": result.theta_search_diagnostics.precise_peak_fitted_center_deg,
            "precise_peak_fitted_fwhm_deg": result.theta_search_diagnostics.precise_peak_fitted_fwhm_deg,
            "precise_peak_fitted_theta_deg": (
                None
                if result.theta_search_diagnostics.precise_peak_fitted_theta_deg is None
                else result.theta_search_diagnostics.precise_peak_fitted_theta_deg.tolist()
            ),
            "precise_peak_fitted_efficiencies": (
                None
                if result.theta_search_diagnostics.precise_peak_fitted_efficiencies is None
                else result.theta_search_diagnostics.precise_peak_fitted_efficiencies.tolist()
            ),
        }

    return {
        "case_id": result.case_id,
        "index": result.index,
        "label": result.label,
        "energy_ev": result.energy_ev,
        "grazing_angle_deg": result.grazing_angle_deg,
        "orders": result.orders.tolist(),
        "selected_efficiency": result.selected_efficiency,
        "selected_diffraction_angle_deg": result.selected_diffraction_angle_deg,
        "efficiency_all": result.efficiency_all.tolist(),
        "diffraction_angle_all": result.diffraction_angle_all.tolist(),
        "status": result.status,
        "error_message": result.error_message,
        "peak_memory_bytes": result.peak_memory_bytes,
        "wall_seconds": result.wall_seconds,
        "case_data": _json_safe_case_data(result.case_data),
        "polarization": result.polarization,
        "solver": result.solver,
        "solver_options": result.solver_options,
        "theta_search_diagnostics": diagnostics_record,
        "retry_triggered": result.retry_triggered,
        "retry_attempts": result.retry_attempts,
        "retry_status": result.retry_status,
        "selected_efficiency_is_exact_zero": result.selected_efficiency_is_exact_zero,
        "selected_efficiency_below_retry_threshold": result.selected_efficiency_below_retry_threshold,
        "theta_tracking_center_mode": result.theta_tracking_center_mode,
        "theta_tracking_auto_classification": result.theta_tracking_auto_classification,
        "theta_tracking_previous_energy_ev": result.theta_tracking_previous_energy_ev,
        "theta_tracking_previous_grazing_angle_deg": result.theta_tracking_previous_grazing_angle_deg,
        "theta_tracking_used_previous_theta": result.theta_tracking_used_previous_theta,
        "theta_tracking_bragg_fallback_triggered": result.theta_tracking_bragg_fallback_triggered,
        "theta_tracking_continuity_rejected": result.theta_tracking_continuity_rejected,
        "timestamp": datetime.now().isoformat(),
    }


def _case_result_from_record(record: dict[str, object]) -> CaseExecutionResult:
    """Convert a JSON record to a case execution result."""

    diagnostics = None
    diagnostics_record = record.get("theta_search_diagnostics")
    if isinstance(diagnostics_record, dict):
        diagnostics = ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=float(diagnostics_record["estimated_grazing_angle_deg"]),
            rough_grazing_angles_deg=np.asarray(diagnostics_record["rough_grazing_angles_deg"], dtype=float),
            rough_efficiencies=np.asarray(diagnostics_record["rough_efficiencies"], dtype=float),
            precise_grazing_angles_deg=np.asarray(diagnostics_record["precise_grazing_angles_deg"], dtype=float),
            precise_efficiencies=np.asarray(diagnostics_record["precise_efficiencies"], dtype=float),
            selected_grazing_angle_deg=float(diagnostics_record["selected_grazing_angle_deg"]),
            selected_efficiency=float(diagnostics_record["selected_efficiency"]),
            precise_fwhm_deg=(
                None
                if diagnostics_record.get("precise_fwhm_deg") is None
                else float(diagnostics_record["precise_fwhm_deg"])
            ),
            theta_tracking_center_mode=str(diagnostics_record.get("theta_tracking_center_mode", "bragg")),
            theta_tracking_auto_classification=str(
                diagnostics_record.get("theta_tracking_auto_classification", "initial")
            ),
            theta_tracking_previous_energy_ev=(
                None
                if diagnostics_record.get("theta_tracking_previous_energy_ev") is None
                else float(diagnostics_record["theta_tracking_previous_energy_ev"])
            ),
            theta_tracking_previous_grazing_angle_deg=(
                None
                if diagnostics_record.get("theta_tracking_previous_grazing_angle_deg") is None
                else float(diagnostics_record["theta_tracking_previous_grazing_angle_deg"])
            ),
            theta_tracking_used_previous_theta=bool(
                diagnostics_record.get("theta_tracking_used_previous_theta", False)
            ),
            theta_tracking_bragg_fallback_triggered=bool(
                diagnostics_record.get("theta_tracking_bragg_fallback_triggered", False)
            ),
            theta_tracking_continuity_rejected=bool(
                diagnostics_record.get("theta_tracking_continuity_rejected", False)
            ),
            precise_peak_selection_mode_requested=str(
                diagnostics_record.get("precise_peak_selection_mode_requested", "max")
            ),
            precise_peak_selection_mode_used=str(diagnostics_record.get("precise_peak_selection_mode_used", "max")),
            precise_peak_fit_fallback_used=bool(diagnostics_record.get("precise_peak_fit_fallback_used", False)),
            precise_peak_fitted_center_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_center_deg") is None
                else float(diagnostics_record["precise_peak_fitted_center_deg"])
            ),
            precise_peak_fitted_fwhm_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_fwhm_deg") is None
                else float(diagnostics_record["precise_peak_fitted_fwhm_deg"])
            ),
            precise_peak_fitted_theta_deg=(
                None
                if diagnostics_record.get("precise_peak_fitted_theta_deg") is None
                else np.asarray(diagnostics_record["precise_peak_fitted_theta_deg"], dtype=float)
            ),
            precise_peak_fitted_efficiencies=(
                None
                if diagnostics_record.get("precise_peak_fitted_efficiencies") is None
                else np.asarray(diagnostics_record["precise_peak_fitted_efficiencies"], dtype=float)
            ),
        )

    return CaseExecutionResult(
        case_id=str(record["case_id"]),
        index=int(record["index"]),
        label=None if record.get("label") is None else str(record["label"]),
        energy_ev=float(record["energy_ev"]),
        grazing_angle_deg=float(record["grazing_angle_deg"]),
        orders=np.asarray(record.get("orders", []), dtype=float),
        selected_efficiency=float(record["selected_efficiency"]),
        selected_diffraction_angle_deg=float(record["selected_diffraction_angle_deg"]),
        efficiency_all=np.asarray(record.get("efficiency_all", []), dtype=float),
        diffraction_angle_all=np.asarray(record.get("diffraction_angle_all", []), dtype=float),
        status=record["status"],  # type: ignore[arg-type]
        error_message=None if record.get("error_message") is None else str(record["error_message"]),
        peak_memory_bytes=(
            None if record.get("peak_memory_bytes") is None else int(record["peak_memory_bytes"])
        ),
        wall_seconds=None if record.get("wall_seconds") is None else float(record["wall_seconds"]),
        case_data=dict(record.get("case_data", {})),
        polarization=str(record.get("polarization", "s")),
        solver=str(record.get("solver", "rcwa")),
        solver_options=record.get("solver_options"),
        theta_search_diagnostics=diagnostics,
        retry_triggered=bool(record.get("retry_triggered", False)),
        retry_attempts=int(record.get("retry_attempts", 0)),
        retry_status=str(record.get("retry_status", "not_needed")),
        selected_efficiency_is_exact_zero=bool(record.get("selected_efficiency_is_exact_zero", False)),
        selected_efficiency_below_retry_threshold=bool(
            record.get("selected_efficiency_below_retry_threshold", False)
        ),
        theta_tracking_center_mode=str(record.get("theta_tracking_center_mode", "bragg")),
        theta_tracking_auto_classification=str(record.get("theta_tracking_auto_classification", "initial")),
        theta_tracking_previous_energy_ev=(
            None
            if record.get("theta_tracking_previous_energy_ev") is None
            else float(record["theta_tracking_previous_energy_ev"])
        ),
        theta_tracking_previous_grazing_angle_deg=(
            None
            if record.get("theta_tracking_previous_grazing_angle_deg") is None
            else float(record["theta_tracking_previous_grazing_angle_deg"])
        ),
        theta_tracking_used_previous_theta=bool(record.get("theta_tracking_used_previous_theta", False)),
        theta_tracking_bragg_fallback_triggered=bool(
            record.get("theta_tracking_bragg_fallback_triggered", False)
        ),
        theta_tracking_continuity_rejected=bool(record.get("theta_tracking_continuity_rejected", False)),
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
        return {"success": True, "result": _single_result_to_record(_run_payload(payload))}
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
