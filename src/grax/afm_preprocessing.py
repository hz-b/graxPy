"""AFM line-scan preprocessing utilities for profile-based gratings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

_UNIT_TO_NM: dict[str, float] = {
    "m": 1e9,
    "um": 1e3,
    "nm": 1.0,
}


class AFMPreprocessing:
    """Preprocess one AFM line scan into a single periodic profile.

    Args:
        data: Two-column input table where column 0 is x and column 1 is z.
        units: Units of the input coordinates. Supported values are ``"m"``,
            ``"um"``, and ``"nm"``. Internal storage always uses nanometers.
        results_folder: Optional directory used to save step-by-step diagnostic
            plots. When omitted and ``save_plots=True``, defaults to
            ``results/afm_preprocessing`` under the current working directory.
        save_plots: Whether diagnostic plots are saved to disk.
        show_plots: Whether to display diagnostic plots interactively.

    Attributes:
        x_nm: Full-scan x coordinates in nanometers.
        z_nm: Full-scan z coordinates in nanometers.
        trough_indices: Trough indices found by :meth:`find_troughs`.
        period_x: Extracted period x coordinates normalized to ``[0, 1]``.
        period_z: Extracted period z coordinates in nanometers.
        period_x_nm: Extracted period x coordinates in nanometers after
            :meth:`rescale_period`.
        period_z_nm: Alias for the extracted period z coordinates in nanometers.
    """

    def __init__(
        self,
        data: "pd.DataFrame | np.ndarray",
        *,
        units: str = "nm",
        results_folder: Path | str | None = None,
        save_plots: bool = True,
        show_plots: bool = True,
    ) -> None:
        if units not in _UNIT_TO_NM:
            raise ValueError("units must be one of: 'm', 'um', 'nm'.")
        factor = _UNIT_TO_NM[units]

        raw = np.asarray(data, dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("data must be a two-column array-like object (x, z).")
        if raw.shape[0] < 3:
            raise ValueError("data must contain at least three rows.")

        self.x_nm = raw[:, 0] * factor
        self.z_nm = raw[:, 1] * factor
        if not np.all(np.isfinite(self.x_nm)) or not np.all(np.isfinite(self.z_nm)):
            raise ValueError("data contains non-finite values.")

        self.save_plots = save_plots
        if self.save_plots:
            self.results_folder = (
                Path(results_folder)
                if results_folder is not None
                else Path.cwd() / "results" / "afm_preprocessing"
            )
        else:
            self.results_folder = None
        self.show_plots = show_plots

        self.trough_indices: np.ndarray | None = None
        self.period_x: np.ndarray | None = None
        self.period_z: np.ndarray | None = None
        self.period_x_nm: np.ndarray | None = None
        self.period_z_nm: np.ndarray | None = None

    def _save_or_show(self, fig: "matplotlib.figure.Figure", stem: str) -> None:
        """Persist or display one diagnostic figure."""

        import matplotlib.pyplot as plt

        if self.save_plots and self.results_folder is not None:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.results_folder / f"{stem}.png", dpi=150, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def normalize_scan(self, *, reverse: bool = False, zero_baseline: bool = True) -> None:
        """Normalize scan orientation and baseline.

        Args:
            reverse: Reverse sample order and keep x increasing from zero.
            zero_baseline: Shift z so the global minimum is zero.
        """

        import matplotlib.pyplot as plt

        x_before = self.x_nm.copy()
        z_before = self.z_nm.copy()

        if reverse:
            self.x_nm = self.x_nm[::-1].copy()
            self.z_nm = self.z_nm[::-1].copy()
            self.x_nm = self.x_nm[0] - self.x_nm
        if zero_baseline:
            self.z_nm = self.z_nm - float(np.min(self.z_nm))

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        y_max = float(max(np.max(z_before), np.max(self.z_nm)))
        axes[0].plot(x_before * 1e-3, z_before, color="tab:blue", lw=0.8)
        axes[0].set_title("Before normalize_scan")
        axes[0].set_xlabel("x (um)")
        axes[0].set_ylabel("z (nm)")
        axes[0].set_ylim(-2.0, y_max)
        axes[0].axhline(0.0, color="#9e9e9e", lw=1.0, ls=":")
        axes[1].plot(self.x_nm * 1e-3, self.z_nm, color="tab:orange", lw=0.8)
        axes[1].set_title("After normalize_scan")
        axes[1].set_xlabel("x (um)")
        axes[1].set_ylabel("z (nm)")
        axes[1].set_ylim(-2.0, y_max)
        axes[1].axhline(0.0, color="#9e9e9e", lw=1.0, ls=":")
        fig.tight_layout()
        self._save_or_show(fig, "01_normalize_scan")

    def find_troughs(self, *, period_nm: float, min_separation_fraction: float = 0.4) -> None:
        """Detect trough locations in the full scan.

        Args:
            period_nm: Expected physical grating period in nanometers.
            min_separation_fraction: Minimum trough spacing as a period fraction.
        """

        from scipy.signal import find_peaks
        import matplotlib.pyplot as plt

        if period_nm <= 0.0:
            raise ValueError("period_nm must be > 0.")
        if min_separation_fraction <= 0.0:
            raise ValueError("min_separation_fraction must be > 0.")

        dx_nm = float(np.mean(np.abs(np.diff(self.x_nm))))
        if dx_nm <= 0.0:
            raise ValueError("x coordinates must span non-zero distance.")
        min_distance_samples = max(1, int((period_nm * min_separation_fraction) / dx_nm))
        trough_indices, _ = find_peaks(-self.z_nm, distance=min_distance_samples)
        self.trough_indices = trough_indices

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.x_nm * 1e-3, self.z_nm, color="tab:blue", lw=0.8, label="scan")
        for index, trough_idx in enumerate(trough_indices):
            x_pos_um = self.x_nm[trough_idx] * 1e-3
            ax.axvline(x_pos_um, color="tab:red", lw=0.9, ls="--", label="troughs" if index == 0 else None)
        ax.set_title(f"find_troughs: {len(trough_indices)} trough(s)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (nm)")
        ax.legend()
        fig.tight_layout()
        self._save_or_show(fig, "02_find_troughs")

    def extract_period(self, *, period_index: int = 0, average: bool = False) -> None:
        """Extract one normalized period from trough-to-trough segments.

        Args:
            period_index: Trough segment index used when ``average=False``.
            average: Average all valid trough-to-trough segments onto one grid.
        """

        if self.trough_indices is None:
            raise RuntimeError("Call find_troughs() before extract_period().")
        if len(self.trough_indices) < 2:
            raise RuntimeError("Need at least two troughs to extract one period.")

        if average:
            self._extract_averaged_period()
        else:
            self._extract_single_period(period_index=period_index)
        self.period_x_nm = None
        self.period_z_nm = None

    def _extract_single_period(self, *, period_index: int) -> None:
        """Extract one trough-to-trough segment."""

        import matplotlib.pyplot as plt

        assert self.trough_indices is not None
        n_segments = len(self.trough_indices) - 1
        if period_index < 0 or period_index >= n_segments:
            raise ValueError(f"period_index must be in [0, {n_segments - 1}].")

        start = int(self.trough_indices[period_index])
        stop = int(self.trough_indices[period_index + 1])
        x_segment = self.x_nm[start : stop + 1]
        z_segment = self.z_nm[start : stop + 1]
        span = float(x_segment[-1] - x_segment[0])
        if span <= 0.0:
            raise RuntimeError("Extracted period has non-positive x span.")

        self.period_x = (x_segment - x_segment[0]) / span
        self.period_z = z_segment.copy()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.x_nm * 1e-3, self.z_nm, color="tab:blue", lw=0.8, label="full scan")
        ax.axvspan(self.x_nm[start] * 1e-3, self.x_nm[stop] * 1e-3, color="tab:orange", alpha=0.2)
        ax.set_title(f"extract_period: segment {period_index}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (nm)")
        ax.legend()
        fig.tight_layout()
        self._save_or_show(fig, f"03_extract_period_{period_index}")

    def _extract_averaged_period(self) -> None:
        """Average all trough-to-trough segments onto one normalized grid."""

        import matplotlib.pyplot as plt

        assert self.trough_indices is not None
        x_norm_segments: list[np.ndarray] = []
        z_segments: list[np.ndarray] = []
        for segment_idx in range(len(self.trough_indices) - 1):
            start = int(self.trough_indices[segment_idx])
            stop = int(self.trough_indices[segment_idx + 1])
            x_segment = self.x_nm[start : stop + 1]
            z_segment = self.z_nm[start : stop + 1]
            span = float(x_segment[-1] - x_segment[0])
            if span <= 0.0:
                continue
            x_norm_segments.append((x_segment - x_segment[0]) / span)
            z_segments.append(z_segment.copy())
        if not x_norm_segments:
            raise RuntimeError("No valid trough-to-trough segments available for averaging.")

        common_points = max(128, max(segment.size for segment in x_norm_segments))
        common_x = np.linspace(0.0, 1.0, common_points)
        stacked = np.vstack(
            [np.interp(common_x, x_norm, z) for x_norm, z in zip(x_norm_segments, z_segments, strict=True)]
        )
        self.period_x = common_x
        self.period_z = np.mean(stacked, axis=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, stacked.shape[0], endpoint=False))
        for segment_index, z_interp in enumerate(stacked):
            ax.plot(
                common_x,
                z_interp,
                color=colors[segment_index],
                alpha=0.9,
                lw=1.1,
                ls="--",
                label="individual periods" if segment_index == 0 else None,
            )
        ax.plot(common_x, self.period_z, color="tab:red", lw=1.1, ls="-", label="average profile")
        ax.set_title(f"extract_period: averaged ({stacked.shape[0]} segments)")
        ax.set_xlabel("normalized x")
        ax.set_ylabel("z (nm)")
        ax.legend()
        fig.tight_layout()
        self._save_or_show(fig, "03_extract_period_averaged")

    def apply_periodicity_ramp(self) -> None:
        """Apply a linear correction so the extracted period endpoints match."""

        import matplotlib.pyplot as plt

        if self.period_x is None or self.period_z is None:
            raise RuntimeError("Call extract_period() before apply_periodicity_ramp().")

        before = self.period_z.copy()
        delta = float(self.period_z[-1] - self.period_z[0])
        correction = np.linspace(0.0, delta, self.period_z.size)
        self.period_z = self.period_z - correction
        self.period_x_nm = None
        self.period_z_nm = None

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.period_x, before, color="tab:blue", lw=1.0, label="before")
        ax.plot(self.period_x, self.period_z, color="tab:orange", lw=1.2, label="after")
        ax.axhline(0.0, color="#9e9e9e", lw=1.0, ls=":")
        ax.set_title("apply_periodicity_ramp")
        ax.set_xlabel("normalized x")
        ax.set_ylabel("z (nm)")
        ax.legend()
        fig.tight_layout()
        self._save_or_show(fig, "04_periodicity_ramp")

    def rescale_period(self, *, period_nm: float) -> None:
        """Convert normalized extracted x coordinates into nanometers.

        Args:
            period_nm: Target physical period in nanometers.
        """

        if period_nm <= 0.0:
            raise ValueError("period_nm must be > 0.")
        if self.period_x is None or self.period_z is None:
            raise RuntimeError("Call extract_period() before rescale_period().")

        self.period_x_nm = self.period_x * period_nm
        self.period_z_nm = self.period_z.copy()

    def get_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the processed period profile in nanometers.

        Returns:
            Tuple ``(x_points_nm, z_points_nm)``.
        """

        if self.period_x_nm is None or self.period_z_nm is None:
            raise RuntimeError("Call rescale_period() before get_profile().")
        return self.period_x_nm.copy(), self.period_z_nm.copy()
