"""Reusable peak-selection helpers for sampled scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

PeakSelectionMode = Literal["max", "gauss", "voigt"]

_GAUSSIAN_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
_VOIGT_FWHM_COEFF = 0.5346
_VOIGT_FWHM_ROOT = 0.2166


@dataclass(frozen=True)
class PeakSelectionResult:
    """Selected peak metadata for a sampled scan.

    Attributes:
        requested_mode: User-requested peak-selection mode.
        used_mode: Mode that actually supplied the final center.
        selected_theta_deg: Chosen peak center in degrees.
        sampled_peak_index: Index of the discrete sampled maximum.
        fit_window_start_index: Inclusive fit-window start index.
        fit_window_end_index: Exclusive fit-window end index.
        fit_fallback_used: Whether the requested mode fell back to a different mode.
        warning_message: Optional human-readable fallback warning.
        fitted_center_deg: Fitted peak center, if a fit supplied the result.
        fitted_fwhm_deg: Fitted FWHM-like width, if available.
        fitted_theta_deg: Theta samples used to draw the fitted profile.
        fitted_efficiencies: Fitted profile values at ``fitted_theta_deg``.
    """

    requested_mode: PeakSelectionMode
    used_mode: PeakSelectionMode
    selected_theta_deg: float
    sampled_peak_index: int
    fit_window_start_index: int
    fit_window_end_index: int
    fit_fallback_used: bool = False
    warning_message: str | None = None
    fitted_center_deg: float | None = None
    fitted_fwhm_deg: float | None = None
    fitted_theta_deg: np.ndarray | None = None
    fitted_efficiencies: np.ndarray | None = None


def _gaussian_with_baseline(
    theta_deg: np.ndarray,
    amplitude: float,
    center_deg: float,
    sigma_deg: float,
    baseline: float,
) -> np.ndarray:
    """Evaluate a Gaussian plus constant baseline."""

    return baseline + amplitude * np.exp(-0.5 * ((theta_deg - center_deg) / sigma_deg) ** 2)


def _voigt_with_baseline(
    theta_deg: np.ndarray,
    amplitude: float,
    center_deg: float,
    sigma_deg: float,
    gamma_deg: float,
    baseline: float,
) -> np.ndarray:
    """Evaluate a Voigt profile plus constant baseline."""

    return baseline + amplitude * voigt_profile(theta_deg - center_deg, sigma_deg, gamma_deg)


def _local_fit_window(
    theta_deg: np.ndarray,
    efficiencies: np.ndarray,
    peak_index: int,
    *,
    half_max_margin: int = 2,
    fallback_window_size: int = 9,
) -> tuple[int, int]:
    """Return a local fit window around the sampled maximum."""

    sample_count = int(theta_deg.size)
    fallback_size = max(5, int(fallback_window_size))
    if fallback_size % 2 == 0:
        fallback_size += 1
    if sample_count <= fallback_size:
        return 0, sample_count

    peak_value = float(efficiencies[peak_index])
    local_min = float(np.nanmin(efficiencies))
    half_level = local_min + 0.5 * (peak_value - local_min)
    left_index = peak_index
    while left_index > 0 and float(efficiencies[left_index]) >= half_level:
        left_index -= 1
    right_index = peak_index
    while right_index < sample_count - 1 and float(efficiencies[right_index]) >= half_level:
        right_index += 1

    half_max_bracketed = (
        left_index < peak_index
        and right_index > peak_index
        and float(efficiencies[left_index]) < half_level
        and float(efficiencies[right_index]) < half_level
    )
    if half_max_bracketed:
        start_index = max(0, left_index - half_max_margin)
        end_index = min(sample_count, right_index + half_max_margin + 1)
        if end_index - start_index >= 5:
            return start_index, end_index

    half_window = fallback_size // 2
    start_index = max(0, peak_index - half_window)
    end_index = min(sample_count, peak_index + half_window + 1)
    if end_index - start_index < fallback_size:
        if start_index == 0:
            end_index = min(sample_count, fallback_size)
        else:
            start_index = max(0, sample_count - fallback_size)
    return start_index, end_index


def _sampled_fwhm_deg(theta_deg: np.ndarray, efficiencies: np.ndarray, peak_index: int) -> float | None:
    """Estimate FWHM from sampled values around a local peak."""

    peak_value = float(efficiencies[peak_index])
    if not np.isfinite(peak_value):
        return None
    baseline = float(np.nanmin(efficiencies))
    if peak_value <= baseline:
        return None
    half_level = baseline + 0.5 * (peak_value - baseline)

    left_crossing = None
    for index in range(peak_index, 0, -1):
        y0 = float(efficiencies[index - 1])
        y1 = float(efficiencies[index])
        if (y0 - half_level) * (y1 - half_level) <= 0.0 and y0 != y1:
            x0 = float(theta_deg[index - 1])
            x1 = float(theta_deg[index])
            left_crossing = float(np.interp(half_level, [y0, y1], [x0, x1]))
            break

    right_crossing = None
    for index in range(peak_index, efficiencies.size - 1):
        y0 = float(efficiencies[index])
        y1 = float(efficiencies[index + 1])
        if (y0 - half_level) * (y1 - half_level) <= 0.0 and y0 != y1:
            x0 = float(theta_deg[index])
            x1 = float(theta_deg[index + 1])
            right_crossing = float(np.interp(half_level, [y0, y1], [x0, x1]))
            break

    if left_crossing is None or right_crossing is None:
        return None
    width_deg = right_crossing - left_crossing
    if not np.isfinite(width_deg) or width_deg <= 0.0:
        return None
    return float(width_deg)


def _fit_initial_guesses(
    theta_window_deg: np.ndarray,
    efficiency_window: np.ndarray,
) -> tuple[float, float, float, float]:
    """Build robust automatic initial fit parameters."""

    local_peak_index = int(np.argmax(efficiency_window))
    center_deg = float(theta_window_deg[local_peak_index])
    baseline = float(np.nanmin(efficiency_window))
    peak_value = float(efficiency_window[local_peak_index])
    amplitude = max(peak_value - baseline, np.finfo(float).eps)
    fwhm_deg = _sampled_fwhm_deg(theta_window_deg, efficiency_window, local_peak_index)
    if fwhm_deg is not None:
        sigma_deg = max(fwhm_deg / _GAUSSIAN_TO_FWHM, np.finfo(float).eps)
    else:
        theta_span = float(theta_window_deg[-1] - theta_window_deg[0])
        sigma_deg = max(theta_span / 6.0, np.finfo(float).eps)
        fwhm_deg = sigma_deg * _GAUSSIAN_TO_FWHM
    gamma_deg = max(0.5 * sigma_deg, np.finfo(float).eps)
    return amplitude, center_deg, baseline, max(float(fwhm_deg), np.finfo(float).eps)


def _validate_fit_center(
    center_deg: float,
    theta_window_deg: np.ndarray,
    theta_all_deg: np.ndarray,
) -> bool:
    """Check that a fitted center stays inside the sampled region."""

    return bool(
        np.isfinite(center_deg)
        and float(theta_window_deg[0]) <= center_deg <= float(theta_window_deg[-1])
        and float(theta_all_deg[0]) <= center_deg <= float(theta_all_deg[-1])
    )


def _fit_gaussian(
    theta_window_deg: np.ndarray,
    efficiency_window: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """Fit a Gaussian peak and return center/FWHM."""

    amplitude, center_deg, baseline, fwhm_guess_deg = _fit_initial_guesses(
        theta_window_deg,
        efficiency_window,
    )
    sigma_guess_deg = max(fwhm_guess_deg / _GAUSSIAN_TO_FWHM, np.finfo(float).eps)
    theta_span = float(theta_window_deg[-1] - theta_window_deg[0])
    bounds = (
        [0.0, float(theta_window_deg[0]), np.finfo(float).eps, baseline - abs(amplitude)],
        [
            max(10.0 * amplitude, amplitude + abs(baseline) + 1.0),
            float(theta_window_deg[-1]),
            max(theta_span, sigma_guess_deg) * 2.0,
            float(np.nanmax(efficiency_window)) + abs(amplitude),
        ],
    )
    params, _ = curve_fit(
        _gaussian_with_baseline,
        theta_window_deg,
        efficiency_window,
        p0=[amplitude, center_deg, sigma_guess_deg, baseline],
        bounds=bounds,
        maxfev=20000,
    )
    fitted_amplitude, fitted_center_deg, fitted_sigma_deg, fitted_baseline = [float(value) for value in params]
    if (
        not np.isfinite(fitted_amplitude)
        or not np.isfinite(fitted_sigma_deg)
        or not np.isfinite(fitted_baseline)
        or fitted_amplitude <= 0.0
        or fitted_sigma_deg <= 0.0
    ):
        return None
    fitted_curve = _gaussian_with_baseline(
        theta_window_deg,
        fitted_amplitude,
        fitted_center_deg,
        fitted_sigma_deg,
        fitted_baseline,
    )
    return fitted_center_deg, float(_GAUSSIAN_TO_FWHM * fitted_sigma_deg), fitted_curve


def _fit_voigt(
    theta_window_deg: np.ndarray,
    efficiency_window: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """Fit a Voigt peak and return center/FWHM."""

    amplitude, center_deg, baseline, fwhm_guess_deg = _fit_initial_guesses(
        theta_window_deg,
        efficiency_window,
    )
    sigma_guess_deg = max(fwhm_guess_deg / _GAUSSIAN_TO_FWHM, np.finfo(float).eps)
    gamma_guess_deg = max(0.5 * sigma_guess_deg, np.finfo(float).eps)
    normalized_height = max(float(voigt_profile(0.0, sigma_guess_deg, gamma_guess_deg)), np.finfo(float).eps)
    amplitude_guess = amplitude / normalized_height
    theta_span = float(theta_window_deg[-1] - theta_window_deg[0])
    bounds = (
        [0.0, float(theta_window_deg[0]), np.finfo(float).eps, np.finfo(float).eps, baseline - abs(amplitude)],
        [
            max(100.0 * amplitude_guess, amplitude_guess + abs(baseline) + 1.0),
            float(theta_window_deg[-1]),
            max(theta_span, sigma_guess_deg) * 2.0,
            max(theta_span, gamma_guess_deg) * 2.0,
            float(np.nanmax(efficiency_window)) + abs(amplitude),
        ],
    )
    params, _ = curve_fit(
        _voigt_with_baseline,
        theta_window_deg,
        efficiency_window,
        p0=[amplitude_guess, center_deg, sigma_guess_deg, gamma_guess_deg, baseline],
        bounds=bounds,
        maxfev=30000,
    )
    fitted_amplitude, fitted_center_deg, fitted_sigma_deg, fitted_gamma_deg, fitted_baseline = [
        float(value) for value in params
    ]
    if (
        not np.isfinite(fitted_amplitude)
        or not np.isfinite(fitted_sigma_deg)
        or not np.isfinite(fitted_gamma_deg)
        or not np.isfinite(fitted_baseline)
        or fitted_amplitude <= 0.0
        or fitted_sigma_deg <= 0.0
        or fitted_gamma_deg <= 0.0
    ):
        return None
    gaussian_fwhm_deg = _GAUSSIAN_TO_FWHM * fitted_sigma_deg
    lorentzian_fwhm_deg = 2.0 * fitted_gamma_deg
    fitted_fwhm_deg = _VOIGT_FWHM_COEFF * lorentzian_fwhm_deg + np.sqrt(
        _VOIGT_FWHM_ROOT * lorentzian_fwhm_deg**2 + gaussian_fwhm_deg**2
    )
    fitted_curve = _voigt_with_baseline(
        theta_window_deg,
        fitted_amplitude,
        fitted_center_deg,
        fitted_sigma_deg,
        fitted_gamma_deg,
        fitted_baseline,
    )
    return fitted_center_deg, float(fitted_fwhm_deg), fitted_curve


def select_peak_theta_from_scan(
    theta_deg: np.ndarray,
    efficiencies: np.ndarray,
    *,
    requested_mode: PeakSelectionMode,
) -> PeakSelectionResult:
    """Select a continuous peak position from a sampled scan.

    Args:
        theta_deg: Sampled theta positions in degrees.
        efficiencies: Sampled efficiencies at ``theta_deg``.
        requested_mode: Requested peak-selection mode.

    Returns:
        Peak selection metadata including the chosen theta.
    """

    if theta_deg.ndim != 1 or efficiencies.ndim != 1 or theta_deg.size != efficiencies.size:
        raise ValueError("theta_deg and efficiencies must be one-dimensional arrays of equal length.")
    if theta_deg.size == 0:
        raise ValueError("theta_deg and efficiencies must not be empty.")
    if requested_mode not in {"max", "gauss", "voigt"}:
        raise ValueError("requested_mode must be 'max', 'gauss', or 'voigt'.")

    sampled_peak_index = int(np.nanargmax(efficiencies))
    sampled_peak_theta_deg = float(theta_deg[sampled_peak_index])
    fit_window_start_index, fit_window_end_index = _local_fit_window(theta_deg, efficiencies, sampled_peak_index)
    theta_window_deg = theta_deg[fit_window_start_index:fit_window_end_index]
    efficiency_window = efficiencies[fit_window_start_index:fit_window_end_index]

    if requested_mode == "max":
        return PeakSelectionResult(
            requested_mode=requested_mode,
            used_mode="max",
            selected_theta_deg=sampled_peak_theta_deg,
            sampled_peak_index=sampled_peak_index,
            fit_window_start_index=fit_window_start_index,
            fit_window_end_index=fit_window_end_index,
        )

    fit_sequence: tuple[PeakSelectionMode, ...]
    if requested_mode == "gauss":
        fit_sequence = ("gauss", "voigt")
    else:
        fit_sequence = ("voigt", "gauss")

    for index, fit_mode in enumerate(fit_sequence):
        try:
            fit_result = _fit_gaussian(theta_window_deg, efficiency_window) if fit_mode == "gauss" else _fit_voigt(
                theta_window_deg,
                efficiency_window,
            )
        except Exception:
            fit_result = None
        if fit_result is None:
            continue
        fitted_center_deg, fitted_fwhm_deg, fitted_curve = fit_result
        if not _validate_fit_center(fitted_center_deg, theta_window_deg, theta_deg):
            continue
        return PeakSelectionResult(
            requested_mode=requested_mode,
            used_mode=fit_mode,
            selected_theta_deg=float(fitted_center_deg),
            sampled_peak_index=sampled_peak_index,
            fit_window_start_index=fit_window_start_index,
            fit_window_end_index=fit_window_end_index,
            fit_fallback_used=index > 0,
            warning_message=(
                f"Precise peak selection requested '{requested_mode}' but fell back to '{fit_mode}'."
                if index > 0
                else None
            ),
            fitted_center_deg=float(fitted_center_deg),
            fitted_fwhm_deg=float(fitted_fwhm_deg),
            fitted_theta_deg=np.asarray(theta_window_deg, dtype=float).copy(),
            fitted_efficiencies=np.asarray(fitted_curve, dtype=float).copy(),
        )

    fallback_message = (
        f"Precise peak selection requested '{requested_mode}' but both fit models failed; using sampled max."
    )
    return PeakSelectionResult(
        requested_mode=requested_mode,
        used_mode="max",
        selected_theta_deg=sampled_peak_theta_deg,
        sampled_peak_index=sampled_peak_index,
        fit_window_start_index=fit_window_start_index,
        fit_window_end_index=fit_window_end_index,
        fit_fallback_used=True,
        warning_message=fallback_message,
    )
