"""Planar-multilayer peak-reflectivity curves from the XRT dynamical-diffraction engine.

This module wraps :mod:`xrt.backends.raycing.materials` to compute the peak
reflectivity of a periodic bilayer stack near its Bragg angle as a function of
photon energy. It is the reflectivity engine used by the d-spacing and gamma
stages of :mod:`grax.multilayer_optimization`.

``xrt`` is imported lazily, inside :func:`_xrt_materials`, so ``import grax`` (and
the re-import each spawned batch worker performs) does not pay the XRT import
cost. :mod:`matplotlib.pyplot` is likewise imported only when per-energy
diagnostic plots are requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["MultilayerReflectivity"]

HC_EV_ANGSTROM = 12398.419843320025


def _xrt_materials() -> Any:
    """Import and return :mod:`xrt.backends.raycing.materials`.

    The import is deferred to first use so that importing :mod:`grax` stays free
    of the XRT dependency. Tests monkeypatch this function to inject a stub.

    Returns:
        The ``xrt.backends.raycing.materials`` module.
    """

    import xrt.backends.raycing.materials as rm

    return rm


def _material_name_density(material: Any) -> tuple[str, float]:
    """Normalise a material specification to a ``(name, density)`` pair.

    Args:
        material: Either a ``(name, density_g_cm3)`` pair or an object exposing
            ``name`` and ``density_g_cm3`` attributes (such as
            :class:`grax.MaterialSpec`).

    Returns:
        The material name and its mass density in g/cm^3.

    Raises:
        ValueError: If ``material`` is neither shape nor carries a density.
    """

    if hasattr(material, "name") and hasattr(material, "density_g_cm3"):
        name = str(material.name)
        density = material.density_g_cm3
    elif isinstance(material, (tuple, list)) and len(material) == 2:
        name, density = str(material[0]), material[1]
    else:
        raise ValueError(
            "material must be a (name, density_g_cm3) pair or expose name / "
            f"density_g_cm3 attributes, got {material!r}"
        )
    if density is None or not np.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError(f"material density must be finite and positive, got {density!r}")
    return name, float(density)


def _smooth_curve(y_values: np.ndarray, window: int = 11) -> np.ndarray:
    """Return ``y_values`` smoothed with a centered moving average.

    Args:
        y_values: Samples to smooth.
        window: Averaging window in samples. Values below 2, or wider than the
            input, return the input unchanged.

    Returns:
        The smoothed samples, same length as the input.
    """

    if window <= 1 or len(y_values) < window:
        return y_values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_values, kernel, mode="same")


def _compute_peak_fwhm_crossings(
    x_values: np.ndarray, y_values: np.ndarray, peak_index: int
) -> tuple[float, float, float, float]:
    """Locate the half-maximum crossings on either side of a sampled peak.

    Args:
        x_values: Monotonic sample positions.
        y_values: Sample values.
        peak_index: Index of the peak sample.

    Returns:
        A tuple ``(fwhm, left_crossing, right_crossing, half_max)``. The first
        three entries are ``nan`` when a bracketing half-maximum crossing cannot
        be found on both sides.
    """

    peak_value = float(y_values[peak_index])
    if peak_value <= 0:
        return np.nan, np.nan, np.nan, np.nan
    half_max = peak_value / 2.0

    left_cross = np.nan
    for idx in range(peak_index, 0, -1):
        y_left = float(y_values[idx - 1])
        y_right = float(y_values[idx])
        if (y_left <= half_max <= y_right) or (y_left >= half_max >= y_right):
            if y_right == y_left:
                left_cross = float(x_values[idx])
            else:
                left_cross = float(
                    np.interp(half_max, [y_left, y_right], [x_values[idx - 1], x_values[idx]])
                )
            break

    right_cross = np.nan
    for idx in range(peak_index, len(x_values) - 1):
        y_left = float(y_values[idx])
        y_right = float(y_values[idx + 1])
        if (y_left >= half_max >= y_right) or (y_left <= half_max <= y_right):
            if y_right == y_left:
                right_cross = float(x_values[idx])
            else:
                right_cross = float(
                    np.interp(half_max, [y_left, y_right], [x_values[idx], x_values[idx + 1]])
                )
            break

    if np.isnan(left_cross) or np.isnan(right_cross) or right_cross < left_cross:
        return np.nan, np.nan, np.nan, half_max
    return right_cross - left_cross, left_cross, right_cross, half_max


def _energy_from_bragg_angle(theta_deg: float, period_angstrom: float, order: int) -> float:
    """Convert a Bragg grazing angle to photon energy for a bilayer period.

    Args:
        theta_deg: Grazing angle in degrees.
        period_angstrom: Bilayer period in Angstrom.
        order: Bragg reflection order.

    Returns:
        Photon energy in eV, or ``nan`` for a non-physical angle.
    """

    sin_theta = np.sin(np.deg2rad(theta_deg))
    if sin_theta <= 0:
        return np.nan
    wavelength_angstrom = 2.0 * period_angstrom * sin_theta / order
    if wavelength_angstrom <= 0:
        return np.nan
    return HC_EV_ANGSTROM / wavelength_angstrom


def _compute_energy_fwhm_from_crossings(
    left_theta_deg: float, right_theta_deg: float, period_angstrom: float, order: int
) -> float:
    """Return the energy-domain FWHM implied by two half-maximum grazing angles.

    Args:
        left_theta_deg: Low-angle half-maximum crossing in degrees.
        right_theta_deg: High-angle half-maximum crossing in degrees.
        period_angstrom: Bilayer period in Angstrom.
        order: Bragg reflection order.

    Returns:
        The absolute energy bandwidth in eV, or ``nan`` when either crossing is
        missing or non-physical.
    """

    if np.isnan(left_theta_deg) or np.isnan(right_theta_deg) or right_theta_deg <= left_theta_deg:
        return np.nan
    if left_theta_deg <= 0 or right_theta_deg <= 0:
        return np.nan
    energy_low = _energy_from_bragg_angle(left_theta_deg, period_angstrom, order)
    energy_high = _energy_from_bragg_angle(right_theta_deg, period_angstrom, order)
    if np.isnan(energy_low) or np.isnan(energy_high):
        return np.nan
    return abs(energy_high - energy_low)


def _estimate_peak_from_prescan(
    stack: Any,
    energy_ev: float,
    bragg_deg: float,
    angle_points: int,
    min_angle_deg: float = 0.0,
) -> tuple[float, float]:
    """Pre-scan the s-polarised reflectivity to locate the Bragg peak.

    Args:
        stack: An XRT ``Multilayer`` instance.
        energy_ev: Photon energy in eV.
        bragg_deg: Kinematic Bragg grazing angle in degrees.
        angle_points: Angular sample count of the caller's main scan; the
            pre-scan uses about half as many.
        min_angle_deg: Lower clip for the pre-scan window.

    Returns:
        The estimated peak grazing angle and an estimated FWHM, both in degrees.
    """

    prescan_min = max(min_angle_deg, bragg_deg - 0.15)
    prescan_max = bragg_deg + max(1.2, bragg_deg * 0.4)
    prescan_points = max(1201, angle_points // 2)

    theta = np.linspace(prescan_min, prescan_max, prescan_points)
    sin_theta = np.sin(np.deg2rad(theta))
    rs = stack.get_amplitude(energy_ev, sin_theta)[0]
    rs2_smooth = _smooth_curve(np.abs(rs) ** 2, window=15)

    right_indices = np.where(theta >= bragg_deg)[0]
    if len(right_indices) == 0:
        return bragg_deg, 0.2

    peak_idx = int(right_indices[np.argmax(rs2_smooth[right_indices])])
    peak_theta = float(theta[peak_idx])
    peak_fwhm_deg, _, _, _ = _compute_peak_fwhm_crossings(theta, rs2_smooth, peak_idx)
    if np.isnan(peak_fwhm_deg) or peak_fwhm_deg <= 0:
        peak_fwhm_deg = max(0.2, peak_theta - bragg_deg)
    return peak_theta, peak_fwhm_deg


def _build_window_from_peak_estimate(
    bragg_deg: float,
    peak_theta: float,
    peak_fwhm_deg: float,
    min_angle_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """Build the main scan window and peak-search mask from a peak estimate.

    Args:
        bragg_deg: Kinematic Bragg grazing angle in degrees.
        peak_theta: Estimated peak grazing angle in degrees.
        peak_fwhm_deg: Estimated peak FWHM in degrees.
        min_angle_deg: Lower clip for the returned angles.

    Returns:
        A tuple ``(angle_min, angle_max, mask_min, mask_max)`` in degrees.
    """

    left_span = max(0.10, 1.0 * peak_fwhm_deg)
    right_span = max(0.35, 3.0 * peak_fwhm_deg)
    angle_min = max(min_angle_deg, bragg_deg - left_span)
    angle_max = peak_theta + right_span
    mask_min = max(min_angle_deg, peak_theta - 1.2 * peak_fwhm_deg)
    mask_max = peak_theta + 1.2 * peak_fwhm_deg
    return angle_min, angle_max, mask_min, mask_max


def _select_bragg_peak_index(
    theta: np.ndarray, rs2: np.ndarray, mask: np.ndarray, bragg_deg: float
) -> int | None:
    """Pick the reflectivity peak sample inside the search mask.

    Prefers a local maximum that lies at or above the Bragg angle and reaches at
    least half the masked global maximum; otherwise falls back to the strongest
    sample right of the Bragg angle, then to the masked maximum.

    Args:
        theta: Grazing-angle samples in degrees.
        rs2: Reflectivity samples aligned with ``theta``.
        mask: Boolean mask selecting the search window.
        bragg_deg: Kinematic Bragg grazing angle in degrees.

    Returns:
        The chosen sample index, or ``None`` when the mask is empty.
    """

    masked_indices = np.where(mask)[0]
    if len(masked_indices) == 0:
        return None

    global_peak_value = float(np.max(rs2[masked_indices]))
    candidate_indices = []
    for idx in masked_indices:
        left_value = rs2[idx - 1] if idx > 0 else -np.inf
        right_value = rs2[idx + 1] if idx < len(rs2) - 1 else -np.inf
        is_local_max = rs2[idx] >= left_value and rs2[idx] >= right_value
        is_right_of_bragg = theta[idx] >= bragg_deg
        is_strong_enough = rs2[idx] >= 0.5 * global_peak_value
        if is_local_max and is_right_of_bragg and is_strong_enough:
            candidate_indices.append(int(idx))

    if candidate_indices:
        return max(candidate_indices, key=lambda idx: rs2[idx])

    right_side_indices = masked_indices[theta[masked_indices] >= bragg_deg]
    if len(right_side_indices) > 0:
        return int(right_side_indices[np.argmax(rs2[right_side_indices])])
    return int(masked_indices[np.argmax(rs2[masked_indices])])


_COLUMNS = (
    "energy_ev",
    "peak_rs",
    "peak_rp",
    "peak_angle_deg",
    "bragg_angle_deg",
    "scan_min_angle_deg",
    "scan_max_angle_deg",
    "fwhm_deg",
    "fwhm_ev",
    "left_half_max_angle_deg",
    "right_half_max_angle_deg",
)


class MultilayerReflectivity:
    """Peak Bragg reflectivity of a periodic bilayer stack versus photon energy.

    The stack is ``material_a`` (incident side) on ``material_b``, repeated
    ``n_bilayers`` times on a ``material_a`` substrate, matching the historical
    Ru/B4C convention. Reflectivity amplitudes come from XRT's dynamical
    diffraction solver.
    """

    def __init__(
        self,
        material_a: Any,
        thickness_a_nm: float,
        material_b: Any,
        thickness_b_nm: float,
        n_bilayers: int,
        *,
        save_recap: str | Path | None = None,
        individuals: bool = False,
    ) -> None:
        """Configure the bilayer stack.

        Args:
            material_a: Incident-side material as a ``(name, density_g_cm3)`` pair
                or a :class:`grax.MaterialSpec`.
            thickness_a_nm: Thickness of ``material_a`` per bilayer, in nm.
            material_b: Second material, same accepted forms as ``material_a``.
            thickness_b_nm: Thickness of ``material_b`` per bilayer, in nm.
            n_bilayers: Number of bilayer periods.
            save_recap: Optional directory for ``results.csv`` and, when
                ``individuals`` is set, per-energy diagnostic plots.
            individuals: Write one reflectivity plot per energy under
                ``save_recap/individuals``.
        """

        self.material_a_name, self.density_a = _material_name_density(material_a)
        self.material_b_name, self.density_b = _material_name_density(material_b)
        self.thickness_a_angstrom = float(thickness_a_nm) * 10.0
        self.thickness_b_angstrom = float(thickness_b_nm) * 10.0
        if self.thickness_a_angstrom <= 0.0 or self.thickness_b_angstrom <= 0.0:
            raise ValueError("bilayer thicknesses must be positive")
        self.n_bilayers = int(n_bilayers)
        if self.n_bilayers < 1:
            raise ValueError("n_bilayers must be at least 1")
        self.period_angstrom = self.thickness_a_angstrom + self.thickness_b_angstrom
        self.save_recap = None if save_recap is None else Path(save_recap)
        self.individuals = bool(individuals)
        self.individuals_dir = None if self.save_recap is None else self.save_recap / "individuals"
        self._stack: Any = None
        self.results: pd.DataFrame | None = None
        if self.save_recap is not None:
            self.save_recap.mkdir(parents=True, exist_ok=True)
        if self.individuals_dir is not None and self.individuals:
            self.individuals_dir.mkdir(parents=True, exist_ok=True)

    def _build_stack(self) -> Any:
        """Instantiate and cache the XRT ``Multilayer`` stack."""

        rm = _xrt_materials()
        material_a = rm.Material(self.material_a_name, rho=self.density_a)
        material_b = rm.Material(self.material_b_name, rho=self.density_b)
        self._stack = rm.Multilayer(
            tLayer=material_a,
            tThickness=self.thickness_a_angstrom,
            bLayer=material_b,
            bThickness=self.thickness_b_angstrom,
            nPairs=self.n_bilayers,
            substrate=material_a,
        )
        return self._stack

    def reflectivity_vs_energy(
        self,
        energies_ev: Iterable[float] | Sequence[float],
        *,
        bragg_order: int = 1,
        window_deg: float = 0.1,
        angle_range: tuple[float, float] | None = None,
        angle_points: int = 2001,
        bragg_margin_deg: float = 0.3,
        min_angle_deg: float = 0.0,
    ) -> pd.DataFrame:
        """Compute peak reflectivity near the Bragg angle for each energy.

        Args:
            energies_ev: Photon energies in eV.
            bragg_order: Bragg reflection order to analyse.
            window_deg: Half-width of the peak-search mask when ``angle_range``
                is given explicitly.
            angle_range: Fixed ``(min, max)`` grazing-angle scan in degrees. When
                omitted, a window is built around the Bragg angle per energy.
            angle_points: Angular samples per energy scan.
            bragg_margin_deg: Retained for signature compatibility; unused when a
                dynamic window is built.
            min_angle_deg: Lower clip when building a dynamic angle window.

        Returns:
            A :class:`pandas.DataFrame` with columns ``energy_ev``, ``peak_rs``,
            ``peak_rp``, ``peak_angle_deg``, ``bragg_angle_deg``,
            ``scan_min_angle_deg``, ``scan_max_angle_deg``, ``fwhm_deg``,
            ``fwhm_ev``, ``left_half_max_angle_deg`` and
            ``right_half_max_angle_deg``. Energies whose peak search fails are
            dropped.
        """

        order = int(bragg_order)
        stack = self._build_stack()
        rows: list[dict[str, float]] = []
        for energy_ev in (float(value) for value in energies_ev):
            bragg_deg = float(np.rad2deg(stack.get_Bragg_angle(energy_ev, order)))
            if angle_range is None:
                peak_theta_est, peak_fwhm_est = _estimate_peak_from_prescan(
                    stack, energy_ev, bragg_deg, angle_points, min_angle_deg=min_angle_deg
                )
                angle_min, angle_max, mask_min, mask_max = _build_window_from_peak_estimate(
                    bragg_deg, peak_theta_est, peak_fwhm_est, min_angle_deg=min_angle_deg
                )
            else:
                angle_min, angle_max = angle_range
                mask_min = bragg_deg - window_deg
                mask_max = bragg_deg + window_deg

            theta = np.linspace(angle_min, angle_max, angle_points)
            sin_theta = np.sin(np.deg2rad(theta))
            rs, rp = stack.get_amplitude(energy_ev, sin_theta)[0:2]
            rs2 = np.abs(rs) ** 2
            rp2 = np.abs(rp) ** 2

            mask = (theta >= mask_min) & (theta <= mask_max)
            if not np.any(mask):
                continue
            peak_idx = _select_bragg_peak_index(theta, rs2, mask, bragg_deg)
            if peak_idx is None:
                continue

            peak_rs = float(rs2[peak_idx])
            peak_rp = float(rp2[peak_idx])
            peak_theta = float(theta[peak_idx])
            fwhm_deg, left_deg, right_deg, half_max = _compute_peak_fwhm_crossings(
                theta, rs2, peak_idx
            )
            fwhm_ev = _compute_energy_fwhm_from_crossings(
                left_deg, right_deg, self.period_angstrom, order
            )
            rows.append(
                {
                    "energy_ev": energy_ev,
                    "peak_rs": peak_rs,
                    "peak_rp": peak_rp,
                    "peak_angle_deg": peak_theta,
                    "bragg_angle_deg": bragg_deg,
                    "scan_min_angle_deg": float(angle_min),
                    "scan_max_angle_deg": float(angle_max),
                    "fwhm_deg": fwhm_deg,
                    "fwhm_ev": fwhm_ev,
                    "left_half_max_angle_deg": left_deg,
                    "right_half_max_angle_deg": right_deg,
                }
            )
            if self.individuals and self.individuals_dir is not None:
                self._save_individual_plot(
                    energy_ev, order, theta, rs2, peak_theta, peak_rs, bragg_deg,
                    left_deg, right_deg, half_max, fwhm_deg, fwhm_ev,
                )

        frame = pd.DataFrame(rows, columns=list(_COLUMNS))
        self.results = frame
        if self.save_recap is not None and not frame.empty:
            frame.to_csv(self.save_recap / "results.csv", index=False)
        return frame

    def _save_individual_plot(
        self,
        energy_ev: float,
        order: int,
        theta: np.ndarray,
        rs2: np.ndarray,
        peak_theta: float,
        peak_rs: float,
        bragg_deg: float,
        left_deg: float,
        right_deg: float,
        half_max: float,
        fwhm_deg: float,
        fwhm_ev: float,
    ) -> None:
        """Write one per-energy reflectivity diagnostic plot."""

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(theta, rs2, color="red", label="|rs|^2")
        axis.plot(peak_theta, peak_rs, "ro", label="peak in window")
        axis.axvline(
            bragg_deg, color="black", linestyle="--", label=f"Bragg {order}: {bragg_deg:.3f} deg"
        )
        if not np.isnan(left_deg):
            axis.axvline(left_deg, color="tab:blue", linestyle=":", label="half-max crossings")
        if not np.isnan(right_deg):
            axis.axvline(right_deg, color="tab:blue", linestyle=":")
        if not np.isnan(half_max):
            axis.axhline(half_max, color="tab:gray", linestyle="--", linewidth=1)
        axis.set_xlabel("Grazing angle (deg)")
        axis.set_ylabel("Reflectivity |rs|^2")
        axis.set_title(f"{energy_ev / 1000:.1f} keV reflectivity")
        axis.text(
            0.98,
            0.02,
            f"Bragg = {bragg_deg:.4f} deg\npeak = {peak_theta:.4f} deg\n"
            f"FWHM = {fwhm_deg:.5f} deg\nbandpass = {fwhm_ev:.2f} eV",
            transform=axis.transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        assert self.individuals_dir is not None
        figure.savefig(self.individuals_dir / f"{int(round(energy_ev))}_eV.png", dpi=150)
        plt.close(figure)
