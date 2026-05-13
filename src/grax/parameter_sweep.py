"""Parameter-study helpers for RCWA convergence checks."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from .gratings import BaseGrating
from .simulation import RCWASimulation

__all__ = [
    "ParameterStudyEnergyResult",
    "ParameterStudyResult",
    "ParameterSweepSeries",
    "get_default_parameter_study_ranges",
    "plot_parameter_study",
    "run_parameter_study",
]

LOG_TICK_THRESHOLDS = [10, 1, 0.1, 0.01, 0.001]
LOG_TICK_FORMATS = ["{:.0f}", "{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:.4f}"]


@dataclass
class ParameterSweepSeries:
    """Sweep result for one parameter at one energy.

    Attributes:
        parameter: Swept parameter name.
        values: Tested parameter values.
        efficiencies: Selected-order efficiencies aligned with ``values``.
        errors: Boolean failure mask aligned with ``values``.
    """

    parameter: str
    values: np.ndarray
    efficiencies: np.ndarray
    errors: np.ndarray


@dataclass
class ParameterStudyEnergyResult:
    """Parameter-study result for one photon energy.

    Attributes:
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Fixed grazing angle used for the study.
        sweeps: Sweep results keyed by parameter name.
    """

    energy_ev: float
    grazing_angle_deg: float
    sweeps: dict[str, ParameterSweepSeries]


@dataclass
class ParameterStudyResult:
    """Container returned by :func:`run_parameter_study`.

    Attributes:
        energies_ev: Energies included in the study.
        grazing_angle_deg: Fixed grazing angle used for all energies.
        diffraction_order: Positive diffraction order used for the metric.
        fourier_orders_values: Fourier-order sweep values.
        x_resolution_values: Horizontal discretization sweep values.
        z_resolution_values: Vertical discretization sweep values.
        results: Ordered per-energy study results.
    """

    energies_ev: np.ndarray
    grazing_angle_deg: float
    diffraction_order: int
    fourier_orders_values: np.ndarray
    x_resolution_values: np.ndarray
    z_resolution_values: np.ndarray
    results: list[ParameterStudyEnergyResult]


def get_default_parameter_study_ranges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return default sweep ranges for the public parameter study.

    Returns:
        Tuple of ``(fourier_orders, x_resolution_nm, z_resolution_nm)`` arrays.
    """

    return (
        np.arange(5, 26, 2, dtype=int),
        np.geomspace(10.0, 0.1, 10, dtype=float),
        np.geomspace(10.0, 0.1, 10, dtype=float),
    )


def run_parameter_study(
    grating: BaseGrating,
    energies_ev: Sequence[float],
    grazing_angle_deg: float,
    *,
    diffraction_order: int = 1,
    fourier_orders_values: Sequence[int] | None = None,
    x_resolution_values: Sequence[float] | None = None,
    z_resolution_values: Sequence[float] | None = None,
    max_retries: int = 2,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    show_progress: bool = True,
) -> ParameterStudyResult:
    """Run a convergence study across Fourier orders and x/z discretization.

    Args:
        grating: Grating used for all simulations.
        energies_ev: Photon energies to study.
        grazing_angle_deg: Fixed grazing angle in degrees.
        diffraction_order: Positive diffraction order used for the metric.
        fourier_orders_values: Optional Fourier sweep values.
        x_resolution_values: Optional x-resolution sweep values in nanometers.
        z_resolution_values: Optional z-resolution sweep values in nanometers.
        max_retries: Maximum retry attempts for a failed point.
        output_dir: Optional directory for CSV exports.
        save_csv: Whether to export per-energy sweep CSV files.
        show_progress: Whether to display progress bars.

    Returns:
        Structured parameter-study result.
    """

    default_fourier, default_x, default_z = get_default_parameter_study_ranges()
    fourier_values = np.asarray(
        default_fourier if fourier_orders_values is None else fourier_orders_values,
        dtype=int,
    )
    x_values = np.asarray(
        default_x if x_resolution_values is None else x_resolution_values,
        dtype=float,
    )
    z_values = np.asarray(
        default_z if z_resolution_values is None else z_resolution_values,
        dtype=float,
    )
    energies = np.asarray(energies_ev, dtype=float)
    results: list[ParameterStudyEnergyResult] = []
    csv_output_dir = Path(output_dir) if output_dir is not None else None
    fixed_fourier_orders = int(np.max(fourier_values))

    if save_csv and csv_output_dir is not None:
        csv_output_dir.mkdir(parents=True, exist_ok=True)

    progress_total = len(energies) * 3
    progress_bar = tqdm(
        total=progress_total,
        desc="Parameter study",
        disable=not show_progress,
    )

    for energy_ev in energies:
        sweeps = {
            "fourier_orders": _run_single_parameter_sweep(
                grating=grating,
                parameter="fourier_orders",
                values=fourier_values,
                energy_ev=float(energy_ev),
                grazing_angle_deg=grazing_angle_deg,
                diffraction_order=diffraction_order,
                max_retries=max_retries,
                fixed_fourier_orders=fixed_fourier_orders,
            ),
            "x_resolution_nm": _run_single_parameter_sweep(
                grating=grating,
                parameter="x_resolution_nm",
                values=x_values,
                energy_ev=float(energy_ev),
                grazing_angle_deg=grazing_angle_deg,
                diffraction_order=diffraction_order,
                max_retries=max_retries,
                fixed_fourier_orders=fixed_fourier_orders,
            ),
            "z_resolution_nm": _run_single_parameter_sweep(
                grating=grating,
                parameter="z_resolution_nm",
                values=z_values,
                energy_ev=float(energy_ev),
                grazing_angle_deg=grazing_angle_deg,
                diffraction_order=diffraction_order,
                max_retries=max_retries,
                fixed_fourier_orders=fixed_fourier_orders,
            ),
        }
        results.append(
            ParameterStudyEnergyResult(
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(grazing_angle_deg),
                sweeps=sweeps,
            )
        )
        if save_csv and csv_output_dir is not None:
            for parameter, sweep in sweeps.items():
                _write_parameter_study_csv(
                    output_dir=csv_output_dir,
                    energy_ev=float(energy_ev),
                    grazing_angle_deg=float(grazing_angle_deg),
                    sweep=sweep,
                )
        progress_bar.update(3)

    progress_bar.close()

    return ParameterStudyResult(
        energies_ev=energies,
        grazing_angle_deg=float(grazing_angle_deg),
        diffraction_order=int(diffraction_order),
        fourier_orders_values=fourier_values,
        x_resolution_values=x_values,
        z_resolution_values=z_values,
        results=results,
    )


def plot_parameter_study(
    result: ParameterStudyResult,
    *,
    output_filename: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure | None:
    """Plot a parameter study as a grid of energies and swept parameters.

    Args:
        result: Study result returned by :func:`run_parameter_study`.
        output_filename: Optional output image path.
        title: Optional figure title.

    Returns:
        The created figure, or ``None`` when saved directly to disk.
    """

    parameter_order = ["fourier_orders", "x_resolution_nm", "z_resolution_nm"]
    column_titles = {
        "fourier_orders": "Fourier Orders",
        "x_resolution_nm": "x Resolution (nm)",
        "z_resolution_nm": "z Resolution (nm)",
    }

    n_rows = len(result.results)
    figure, axes = plt.subplots(
        n_rows,
        len(parameter_order),
        figsize=(5.5 * len(parameter_order), 3.8 * max(n_rows, 1)),
        squeeze=False,
    )

    if title is None:
        title = (
            "Parameter Study: Selected-Order Efficiency "
            f"(grazing={result.grazing_angle_deg:.3f} deg)"
        )
    figure.suptitle(title, fontsize=15, y=0.995)

    for row_index, energy_result in enumerate(result.results):
        for col_index, parameter in enumerate(parameter_order):
            axis = axes[row_index, col_index]
            sweep = energy_result.sweeps[parameter]
            successful_mask = ~sweep.errors
            x_values = sweep.values[successful_mask]
            y_values = sweep.efficiencies[successful_mask]

            if parameter == "fourier_orders":
                axis.plot(x_values, y_values, "o-", linewidth=1.3, markersize=4.0)
            else:
                axis.semilogx(x_values, y_values, "o-", linewidth=1.3, markersize=4.0)
                axis.xaxis.set_major_formatter(
                    _get_log_formatter(float(np.nanmin(sweep.values)), float(np.nanmax(sweep.values)))
                )
                axis.invert_xaxis()

            if np.any(sweep.errors):
                axis.scatter(
                    sweep.values[sweep.errors],
                    np.zeros(np.count_nonzero(sweep.errors), dtype=float),
                    color="red",
                    marker="x",
                    s=40,
                    linewidths=1.5,
                    label="Failed" if row_index == 0 and col_index == 0 else None,
                )

            if row_index == 0:
                axis.set_title(column_titles[parameter])
            if col_index == 0:
                axis.set_ylabel("Efficiency")
            axis.set_xlabel(column_titles[parameter])
            axis.grid(True, alpha=0.3)
            axis.text(
                0.02,
                0.96,
                f"{energy_result.energy_ev:.1f} eV",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.65},
            )

    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    if output_filename is not None:
        figure.savefig(output_filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return None

    return figure


def _run_single_parameter_sweep(
    *,
    grating: BaseGrating,
    parameter: str,
    values: np.ndarray,
    energy_ev: float,
    grazing_angle_deg: float,
    diffraction_order: int,
    max_retries: int,
    fixed_fourier_orders: int,
) -> ParameterSweepSeries:
    """Run one parameter sweep for one energy."""

    efficiencies = np.full(values.shape, np.nan, dtype=float)
    errors = np.zeros(values.shape, dtype=bool)

    for index, value in enumerate(values):
        success = False
        retries = 0
        while not success and retries <= max_retries:
            try:
                if parameter == "fourier_orders":
                    case_grating = grating
                    fourier_orders = int(value)
                else:
                    case_grating = _clone_grating_with_override(grating, parameter, float(value))
                    fourier_orders = fixed_fourier_orders

                result = RCWASimulation(
                    grating=case_grating,
                    diffraction_order=diffraction_order,
                    fourier_orders=fourier_orders,
                    grazing_angle_deg=grazing_angle_deg,
                    backend="numba",
                ).run_single(energy_ev)
                efficiencies[index] = float(result["efficiency"])
                success = True
            except Exception:
                retries += 1
                if retries > max_retries:
                    errors[index] = True

    return ParameterSweepSeries(
        parameter=parameter,
        values=np.asarray(values),
        efficiencies=efficiencies,
        errors=errors,
    )


def _clone_grating_with_override(
    grating: BaseGrating,
    parameter: str,
    value: float,
) -> BaseGrating:
    """Return a shallow-copied grating with one discretization override."""

    cloned_grating = copy(grating)
    setattr(cloned_grating, parameter, value)
    return cloned_grating


def _write_parameter_study_csv(
    *,
    output_dir: Path,
    energy_ev: float,
    grazing_angle_deg: float,
    sweep: ParameterSweepSeries,
) -> None:
    """Write one sweep CSV for one energy."""

    csv_path = output_dir / f"parameter_study_{sweep.parameter}_E{energy_ev:.1f}eV.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["parameter", "value", "efficiency", "error", "energy_ev", "grazing_angle_deg"]
        )
        for value, efficiency, error in zip(sweep.values, sweep.efficiencies, sweep.errors):
            writer.writerow(
                [
                    sweep.parameter,
                    value,
                    "" if error else efficiency,
                    bool(error),
                    energy_ev,
                    grazing_angle_deg,
                ]
            )


def _get_log_formatter(x_min: float, x_max: float) -> FuncFormatter:
    """Create a formatter for log axes that preserves decimal labels."""

    def formatter(x: float, _: int) -> str:
        if x < min(x_min, x_max) or x > max(x_min, x_max):
            return ""
        for index, threshold in enumerate(LOG_TICK_THRESHOLDS):
            if x >= threshold:
                return LOG_TICK_FORMATS[index].format(x)
        return LOG_TICK_FORMATS[-1].format(x)

    return FuncFormatter(formatter)
