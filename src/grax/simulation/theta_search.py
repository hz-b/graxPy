"""Single-energy multilayer theta-search helpers."""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import threading
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from ..gratings import BaseGrating
from ..materials import resolve_refractive_index
from ..peak_fitting import PeakSelectionMode, select_peak_theta_from_scan
from ..stacks import MultilayerStack
from .core import _clone_grating_with_overrides, run_simulation
from .models import SingleSimulationResult, ThetaSearchDiagnostics

logger = logging.getLogger(__name__)


def _worker_identity() -> str:
    """Return a compact worker identity for theta-search logs."""

    process = mp.current_process()
    thread = threading.current_thread()
    return f"pid={os.getpid()} proc={process.name} thread={thread.name}"


def _simulation_api():
    """Return the public simulation package for monkeypatch-compatible dispatch."""

    return importlib.import_module("grax.simulation")

def estimate_multilayer_bragg_angle_deg(
    *,
    grating: BaseGrating,
    energy_ev: float,
    multilayer_bragg_order: int = 1,
) -> float:
    """Estimate the grazing angle from the nominal multilayer Bragg condition.

    Computes an initial guess for the grazing angle using the multilayer Bragg
    equation:

        m * lambda = 2 * d * sin(theta) * sqrt(1 - 2*delta)

    where delta is the refractive index decrement. This provides a reasonable
    starting point for the adaptive theta search.

    Args:
        grating: Grating carrying a multilayer coating stack.
        energy_ev: Photon energy in electronvolts.
        multilayer_bragg_order: Positive multilayer Bragg order used for the estimate.

    Returns:
        Estimated grazing angle in degrees.

    Raises:
        TypeError: If the grating does not resolve to a multilayer stack.
        ValueError: If the requested Bragg order is invalid or no real angle exists.

    Note:
        The estimate uses the real part of the refractive index decrement for
        stability. Returns an angle in the range (0, 90) degrees.
    """

    if multilayer_bragg_order < 1:
        raise ValueError("multilayer_bragg_order must be >= 1.")

    stack = grating.resolved_stack()
    if not isinstance(stack, MultilayerStack):
        raise TypeError("estimate_multilayer_bragg_angle_deg requires a MultilayerStack coating.")

    wavelength_nm = 1239.8 / float(energy_ev)
    n_a = resolve_refractive_index(stack.material_a, float(energy_ev))
    n_b = resolve_refractive_index(stack.material_b, float(energy_ev))
    gamma = float(stack.gamma)
    average_real_index = gamma * float(np.real(n_a)) + (1.0 - gamma) * float(np.real(n_b))
    delta_average = max(1.0 - average_real_index, 0.0)
    sine_squared = ((multilayer_bragg_order * wavelength_nm) / (2.0 * stack.d_period_nm)) ** 2 + (
        2.0 * delta_average
    )
    if sine_squared <= 0.0:
        raise ValueError("Unable to estimate a positive multilayer Bragg angle.")
    sine_value = np.sqrt(sine_squared)
    if sine_value >= 1.0:
        raise ValueError("Multilayer Bragg estimate is outside the physical grazing-angle range.")
    return float(np.rad2deg(np.arcsin(sine_value)))


def _safe_theta_scan_half_width_deg(
    *,
    center_deg: float,
    requested_half_width_deg: float,
) -> float:
    """Return a non-negative-theta-safe half width around ``center_deg``.

    Constrains the scan half-width so the lower bound of the scan remains
    above 0 degrees. Prevents physically invalid negative grazing angles.

    Args:
        center_deg: Center angle of the scan range.
        requested_half_width_deg: Requested half-width of the scan.

    Returns:
        Half-width limited so center - half_width > 0. If the requested
        half-width would violate this constraint, returns 95% of the center
        angle (leaving 5% margin).

    Note:
        This is a safety mechanism for very small center angles where the
        requested scan range would extend into negative angles.
    """

    if center_deg <= 0.0:
        raise ValueError("Theta scan center must be > 0 deg.")
    max_safe_half_width = 0.95 * center_deg
    if requested_half_width_deg > max_safe_half_width:
        logger.warning(
            "Requested theta half-width %.6f deg around center %.6f deg reaches near/into 0 deg. "
            "Reducing to %.6f deg.",
            requested_half_width_deg,
            center_deg,
            max_safe_half_width,
        )
    return float(min(requested_half_width_deg, max_safe_half_width))


def _theta_scan_grid(
    *,
    center_deg: float,
    half_width_deg: float,
    point_count: int,
) -> np.ndarray:
    """Return an evenly spaced theta scan grid."""

    if half_width_deg <= 0.0:
        raise ValueError("Theta-scan half widths must be > 0.")
    if point_count < 2:
        raise ValueError("Theta-scan point counts must be >= 2.")
    safe_half_width_deg = _safe_theta_scan_half_width_deg(
        center_deg=float(center_deg),
        requested_half_width_deg=float(half_width_deg),
    )
    grid = np.linspace(
        center_deg - safe_half_width_deg,
        center_deg + safe_half_width_deg,
        point_count,
        dtype=float,
    )
    return grid


def _run_theta_scan(
    *,
    grating: BaseGrating,
    energy_ev: float,
    diffraction_order: int,
    theta_grid_deg: np.ndarray,
    fourier_orders: int,
    roughness_sigma_nm: float | None,
    validate_physical_results: bool,
    max_reflected_efficiency: float,
    min_efficiency: float,
    max_total_reflected_efficiency: float,
    backend: str,
) -> tuple[np.ndarray, list[SingleSimulationResult]]:
    """Run one theta scan and return selected efficiencies plus full results."""

    scan_results = [
        _simulation_api().run_simulation(
            grating=grating,
            energy_ev=energy_ev,
            grazing_angle_deg=float(theta_deg),
            diffraction_order=diffraction_order,
            fourier_orders=fourier_orders,
            roughness_sigma_nm=roughness_sigma_nm,
            validate_physical_results=validate_physical_results,
            max_reflected_efficiency=max_reflected_efficiency,
            min_efficiency=min_efficiency,
            max_total_reflected_efficiency=max_total_reflected_efficiency,
            backend=backend,
        )
        for theta_deg in theta_grid_deg
    ]
    efficiencies = np.asarray([result.selected_efficiency for result in scan_results], dtype=float)
    return efficiencies, scan_results


def _interpolate_half_max_crossing(
    *,
    x_left: float,
    y_left: float,
    x_right: float,
    y_right: float,
    half_max: float,
) -> float:
    """Return linear-interpolated x position where y crosses ``half_max``."""

    if np.isclose(y_right, y_left):
        return float(0.5 * (x_left + x_right))
    fraction = (half_max - y_left) / (y_right - y_left)
    return float(x_left + fraction * (x_right - x_left))


def _precise_scan_fwhm_deg(theta_grid_deg: np.ndarray, efficiencies: np.ndarray) -> float | None:
    """Estimate FWHM in degrees from one precise theta scan.

    Returns ``None`` when the half-maximum crossings are not bracketed on both
    sides of the selected peak.
    """

    if theta_grid_deg.size < 3 or efficiencies.size != theta_grid_deg.size:
        return None
    if not np.all(np.isfinite(theta_grid_deg)) or not np.all(np.isfinite(efficiencies)):
        return None

    peak_index = int(np.nanargmax(efficiencies))
    peak_value = float(efficiencies[peak_index])
    if peak_value <= 0.0:
        return None
    half_max = 0.5 * peak_value

    left_crossing = None
    for index in range(peak_index - 1, -1, -1):
        y_left = float(efficiencies[index])
        y_right = float(efficiencies[index + 1])
        if (y_left - half_max) * (y_right - half_max) <= 0.0:
            left_crossing = _interpolate_half_max_crossing(
                x_left=float(theta_grid_deg[index]),
                y_left=y_left,
                x_right=float(theta_grid_deg[index + 1]),
                y_right=y_right,
                half_max=half_max,
            )
            break

    right_crossing = None
    for index in range(peak_index, theta_grid_deg.size - 1):
        y_left = float(efficiencies[index])
        y_right = float(efficiencies[index + 1])
        if (y_left - half_max) * (y_right - half_max) <= 0.0:
            right_crossing = _interpolate_half_max_crossing(
                x_left=float(theta_grid_deg[index]),
                y_left=y_left,
                x_right=float(theta_grid_deg[index + 1]),
                y_right=y_right,
                half_max=half_max,
            )
            break

    if left_crossing is None or right_crossing is None:
        return None
    width = right_crossing - left_crossing
    if width <= 0.0 or not np.isfinite(width):
        return None
    return float(width)


def _peak_capture_status(efficiencies: np.ndarray, peak_index: int, edge_margin: int = 2) -> tuple[bool, str]:
    """Classify whether the sampled maximum is interior or edge-clipped."""

    sample_count = int(efficiencies.size)
    if sample_count < 5:
        return True, "ok"
    margin = max(1, min(edge_margin, sample_count // 4))
    if peak_index <= margin:
        return False, "left_edge"
    if peak_index >= sample_count - 1 - margin:
        return False, "right_edge"
    return True, "ok"


def _report_peak_selection_warning(message: str) -> None:
    """Emit a peak-selection warning without breaking tqdm output."""

    tqdm.write(message)
    logger.warning(message)


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    """Format elapsed seconds for terminal-facing status output."""

    total_seconds = max(0, int(round(float(elapsed_seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def run_multilayer_theta_search(
    *,
    grating: BaseGrating,
    energy_ev: float,
    diffraction_order: int = 1,
    initial_grazing_angle_deg: float | None = None,
    multilayer_bragg_order: int = 1,
    rough_scan_half_width_deg: float = 0.5,
    rough_scan_points: int = 82,
    rough_fourier_orders: int = 3,
    rough_x_resolution_nm: float | None = 1.0,
    rough_z_resolution_nm: float | None = 1.0,
    fine_scan_half_width_deg: float = 0.1,
    fine_scan_points: int = 81,
    fine_fourier_orders: int = 5,
    fine_x_resolution_nm: float | None = 0.5,
    fine_z_resolution_nm: float | None = 0.5,
    final_fourier_orders: int = 25,
    final_x_resolution_nm: float | None = 0.3,
    final_z_resolution_nm: float | None = 0.3,
    roughness_sigma_nm: float | None = None,
    validate_physical_results: bool = True,
    max_reflected_efficiency: float = 1.05,
    min_efficiency: float = -1e-8,
    max_total_reflected_efficiency: float = 1.05,
    precise_peak_selection_mode: PeakSelectionMode = "max",
    backend: str = "numba",
    diagnostic_callback: Callable[[ThetaSearchDiagnostics, float], None] | None = None,
) -> SingleSimulationResult:
    """Run one energy point with an internal rough/precise grazing-angle search.

    Implements a three-stage adaptive angular scan for multilayer gratings.
    The rough scan searches a wide angular range with reduced Fourier orders and
    coarse resolution to find an approximate peak location. The precise scan
    narrows the range around the rough maximum with moderate Fourier orders and
    finer resolution for accurate peak characterization. The final solve runs a
    single high-resolution calculation at the selected angle with the final
    Fourier-order and resolution settings.

    The refinement path supports Gaussian or Voigt sub-grid peak localization,
    records scan diagnostics including grating angles, efficiencies, FWHM, and
    fitted curves, and optionally validates reflected-efficiency bounds.

    Args:
        grating: Grating profile and material stack.
        energy_ev: Photon energy in electronvolts.
        diffraction_order: Positive diffraction order to optimize.
        initial_grazing_angle_deg: Optional explicit center for the rough scan.
            Uses multilayer Bragg estimate if not provided.
        multilayer_bragg_order: Positive multilayer Bragg order used when estimating
            the initial grazing angle. Should match the desired diffraction order
            for best accuracy.
        rough_scan_half_width_deg: Half-width of the rough scan around the estimate.
            Larger values provide more conservative search at increased computation.
        rough_scan_points: Number of angular points in the rough scan. Higher values
            give better angular resolution in the initial search.
        rough_fourier_orders: Fourier order used during the rough scan. Lower values
            enable fast initial search.
        rough_x_resolution_nm: Optional x resolution override during the rough scan.
            Lower resolution accelerates initial search.
        rough_z_resolution_nm: Optional z resolution override during the rough scan.
        fine_scan_half_width_deg: Half-width of the precise scan around the rough
            maximum. Should be narrow enough to focus on the peak region.
        fine_scan_points: Number of angular points in the precise scan. Must be
            odd for symmetric sampling around the peak.
        fine_fourier_orders: Fourier order used during the precise scan. Should be
            sufficient for accurate peak characterization.
        fine_x_resolution_nm: Optional x resolution override during the precise scan.
        fine_z_resolution_nm: Optional z resolution override during the precise scan.
        final_fourier_orders: Fourier order used during the final solve. Use higher
            values for final convergence and publication-quality results.
        final_x_resolution_nm: Optional x resolution override during the final solve.
        final_z_resolution_nm: Optional z resolution override during the final solve.
        roughness_sigma_nm: Optional rms roughness in nanometers applied to all
            interfaces.
        validate_physical_results: Whether to validate reflected efficiencies against
            physical constraints, including minimum reflected efficiency, maximum
            reflected efficiency, and maximum total propagating reflected efficiency.
        max_reflected_efficiency: Maximum allowed single-order reflected efficiency
            during validation.
        min_efficiency: Minimum allowed efficiency (slightly negative values allowed
            for numerical noise).
        max_total_reflected_efficiency: Maximum allowed sum of propagating reflected
            efficiencies.
        precise_peak_selection_mode: Mode used to select the final theta from the
            precise scan. ``"max"`` uses the sampled maximum, ``"gauss"`` fits a
            local Gaussian neighborhood, and ``"voigt"`` fits a local Voigt profile.
        backend: Fourier coefficient backend selector. Options: ``"numpy"`` (pure Python,
            default), ``"numba"`` (JIT-compiled, requires numba package).
        diagnostic_callback: Optional callback invoked with scan diagnostics after
            the peak search. Receives ThetaSearchDiagnostics and energy_ev.

    Returns:
        Single-case RCWA result at the selected grazing angle. The returned
        object includes the selected grazing angle, selected efficiency,
        selected diffraction angle, and full theta-search diagnostics.

    Example:
        >>> result = run_multilayer_theta_search(
        ...     grating=grating,
        ...     energy_ev=500,
        ...     diffraction_order=1,
        ...     precise_peak_selection_mode="gauss"
        ... )
        >>> print(f"Optimal theta: {result.grazing_angle_deg:.3f}°")
        >>> print(f"Selected efficiency: {result.selected_efficiency:.4f}")
    """
    if precise_peak_selection_mode not in {"max", "gauss", "voigt"}:
        raise ValueError("precise_peak_selection_mode must be 'max', 'gauss', or 'voigt'.")

    estimated_grazing_angle_deg = (
        float(initial_grazing_angle_deg)
        if initial_grazing_angle_deg is not None
        else estimate_multilayer_bragg_angle_deg(
            grating=grating,
            energy_ev=energy_ev,
            multilayer_bragg_order=multilayer_bragg_order,
        )
    )
    rough_grating = _clone_grating_with_overrides(
        grating,
        x_resolution_nm=rough_x_resolution_nm,
        z_resolution_nm=rough_z_resolution_nm,
    )
    fine_grating = _clone_grating_with_overrides(
        grating,
        x_resolution_nm=fine_x_resolution_nm,
        z_resolution_nm=fine_z_resolution_nm,
    )
    final_grating = _clone_grating_with_overrides(
        grating,
        x_resolution_nm=final_x_resolution_nm,
        z_resolution_nm=final_z_resolution_nm,
    )

    rough_center_deg = estimated_grazing_angle_deg
    rough_half_width_deg = float(rough_scan_half_width_deg)
    rough_grazing_angles_deg = np.asarray([], dtype=float)
    rough_efficiencies = np.asarray([], dtype=float)
    rough_peak_angle_deg = estimated_grazing_angle_deg
    for _attempt in range(4):
        attempt = _attempt + 1
        logger.info(
            "[theta-search][rough] start energy=%.6f eV attempt=%d center=%.6f half_width=%.6f points=%d fourier=%d %s",
            float(energy_ev),
            attempt,
            float(rough_center_deg),
            float(rough_half_width_deg),
            int(rough_scan_points),
            int(rough_fourier_orders),
            _worker_identity(),
        )
        rough_grazing_angles_deg = _theta_scan_grid(
            center_deg=rough_center_deg,
            half_width_deg=rough_half_width_deg,
            point_count=rough_scan_points,
        )
        rough_efficiencies, _ = _run_theta_scan(
            grating=rough_grating,
            energy_ev=float(energy_ev),
            diffraction_order=diffraction_order,
            theta_grid_deg=rough_grazing_angles_deg,
            fourier_orders=int(rough_fourier_orders),
            roughness_sigma_nm=roughness_sigma_nm,
            validate_physical_results=validate_physical_results,
            max_reflected_efficiency=max_reflected_efficiency,
            min_efficiency=min_efficiency,
            max_total_reflected_efficiency=max_total_reflected_efficiency,
            backend=backend,
        )
        rough_peak_index = int(np.nanargmax(rough_efficiencies))
        rough_peak_angle_deg = float(rough_grazing_angles_deg[rough_peak_index])
        rough_ok, rough_status = _peak_capture_status(rough_efficiencies, rough_peak_index, edge_margin=2)
        logger.info(
            "[theta-search][rough] done energy=%.6f eV attempt=%d peak_theta=%.6f peak_eff=%.6e status=%s",
            float(energy_ev),
            attempt,
            float(rough_peak_angle_deg),
            float(rough_efficiencies[rough_peak_index]),
            rough_status,
        )
        if rough_ok:
            break
        shift = 0.7 * rough_half_width_deg
        if rough_status == "left_edge":
            rough_center_deg = max(0.01, rough_center_deg - shift)
        else:
            rough_center_deg = rough_center_deg + shift
        rough_half_width_deg *= 2.0

    precise_center_deg = rough_peak_angle_deg
    precise_half_width_deg = float(fine_scan_half_width_deg)
    precise_grazing_angles_deg = np.asarray([], dtype=float)
    precise_efficiencies = np.asarray([], dtype=float)
    precise_results: list[SingleSimulationResult] = []
    for _attempt in range(4):
        attempt = _attempt + 1
        logger.info(
            "[theta-search][fine] start energy=%.6f eV attempt=%d center=%.6f half_width=%.6f points=%d fourier=%d %s",
            float(energy_ev),
            attempt,
            float(precise_center_deg),
            float(precise_half_width_deg),
            int(fine_scan_points),
            int(fine_fourier_orders),
            _worker_identity(),
        )
        precise_grazing_angles_deg = _theta_scan_grid(
            center_deg=precise_center_deg,
            half_width_deg=precise_half_width_deg,
            point_count=fine_scan_points,
        )
        precise_efficiencies, precise_results = _run_theta_scan(
            grating=fine_grating,
            energy_ev=float(energy_ev),
            diffraction_order=diffraction_order,
            theta_grid_deg=precise_grazing_angles_deg,
            fourier_orders=int(fine_fourier_orders),
            roughness_sigma_nm=roughness_sigma_nm,
            validate_physical_results=validate_physical_results,
            max_reflected_efficiency=max_reflected_efficiency,
            min_efficiency=min_efficiency,
            max_total_reflected_efficiency=max_total_reflected_efficiency,
            backend=backend,
        )
        precise_peak_index = int(np.nanargmax(precise_efficiencies))
        precise_ok, precise_status = _peak_capture_status(precise_efficiencies, precise_peak_index, edge_margin=2)
        logger.info(
            "[theta-search][fine] done energy=%.6f eV attempt=%d peak_theta=%.6f peak_eff=%.6e status=%s",
            float(energy_ev),
            attempt,
            float(precise_grazing_angles_deg[precise_peak_index]),
            float(precise_efficiencies[precise_peak_index]),
            precise_status,
        )
        if precise_ok:
            break
        shift = 0.7 * precise_half_width_deg
        if precise_status == "left_edge":
            precise_center_deg = max(0.01, precise_center_deg - shift)
        else:
            precise_center_deg = precise_center_deg + shift
        precise_half_width_deg *= 1.1
    precise_fwhm_deg = _precise_scan_fwhm_deg(precise_grazing_angles_deg, precise_efficiencies)
    peak_selection = select_peak_theta_from_scan(
        precise_grazing_angles_deg,
        precise_efficiencies,
        requested_mode=precise_peak_selection_mode,
    )
    selected_theta_deg = float(peak_selection.selected_theta_deg)
    if peak_selection.warning_message is not None:
        _report_peak_selection_warning(f"Energy {float(energy_ev):.6f} eV: {peak_selection.warning_message}")
    logger.info(
        "[theta-search][final-scan] start energy=%.6f eV selected_theta=%.6f mode=%s fourier=%d %s",
        float(energy_ev),
        float(selected_theta_deg),
        str(peak_selection.used_mode),
        int(final_fourier_orders),
        _worker_identity(),
    )
    selected_result = _simulation_api().run_simulation(
        grating=final_grating,
        energy_ev=float(energy_ev),
        grazing_angle_deg=selected_theta_deg,
        diffraction_order=diffraction_order,
        fourier_orders=int(final_fourier_orders),
        roughness_sigma_nm=roughness_sigma_nm,
        validate_physical_results=validate_physical_results,
        max_reflected_efficiency=max_reflected_efficiency,
        min_efficiency=min_efficiency,
        max_total_reflected_efficiency=max_total_reflected_efficiency,
        backend=backend,
    )
    logger.info(
        "[theta-search][final-scan] done energy=%.6f eV selected_theta=%.6f selected_eff=%.6e",
        float(energy_ev),
        float(selected_result.grazing_angle_deg),
        float(selected_result.selected_efficiency),
    )
    diagnostics = ThetaSearchDiagnostics(
        estimated_grazing_angle_deg=estimated_grazing_angle_deg,
        rough_grazing_angles_deg=rough_grazing_angles_deg,
        rough_efficiencies=rough_efficiencies,
        precise_grazing_angles_deg=precise_grazing_angles_deg,
        precise_efficiencies=precise_efficiencies,
        selected_grazing_angle_deg=selected_result.grazing_angle_deg,
        selected_efficiency=selected_result.selected_efficiency,
        precise_fwhm_deg=precise_fwhm_deg,
        precise_peak_selection_mode_requested=peak_selection.requested_mode,
        precise_peak_selection_mode_used=peak_selection.used_mode,
        precise_peak_fit_fallback_used=peak_selection.fit_fallback_used,
        precise_peak_fitted_center_deg=peak_selection.fitted_center_deg,
        precise_peak_fitted_fwhm_deg=peak_selection.fitted_fwhm_deg,
        precise_peak_fitted_theta_deg=peak_selection.fitted_theta_deg,
        precise_peak_fitted_efficiencies=peak_selection.fitted_efficiencies,
    )
    selected_result.theta_search_diagnostics = diagnostics
    if diagnostic_callback is not None:
        diagnostic_callback(diagnostics, float(energy_ev))
    return selected_result
