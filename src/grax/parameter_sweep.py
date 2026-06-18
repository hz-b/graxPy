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
        error_messages: Final failure message for each value. Successful points
            contain empty strings.
    """

    parameter: str
    values: np.ndarray
    efficiencies: np.ndarray
    errors: np.ndarray
    error_messages: np.ndarray


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

    Generates logarithmically-spaced discretization values appropriate for
    convergence studies. The ranges cover a wide spectrum from coarse to fine
    discretization to identify the minimum settings needed for convergence.

    Returns:
        Tuple of ``(fourier_orders, x_resolution_nm, z_resolution_nm)`` arrays:
        - fourier_orders: Odd integers from 5 to 25 for Fourier truncation study
        - x_resolution_nm: 10 values from 10.0 to 0.1 nm for x-discretization study
        - z_resolution_nm: 10 values from 10.0 to 0.1 nm for z-discretization study

    Example:
        >>> fourier_vals, x_res, z_res = get_default_parameter_study_ranges()
        >>> print(f"Fourier orders: {fourier_vals}")
        >>> print(f"X resolution range: {x_res[0]:.1f} to {x_res[-1]:.2f} nm")
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
    polarization: str = "s",
    fourier_orders_values: Sequence[int] | None = None,
    x_resolution_values: Sequence[float] | None = None,
    z_resolution_values: Sequence[float] | None = None,
    max_retries: int = 2,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    show_progress: bool = True,
) -> ParameterStudyResult:
    """Run a convergence study across Fourier orders and x/z discretization.

    Executes three independent parameter sweeps for each energy point:
    1. Fourier order convergence study (discretization in periodic direction)
    2. X-resolution study (discretization along grating profile)
    3. Z-resolution study (discretization along layer stack)

    For Fourier order sweeps, the grating is used directly. For x/z resolution
    sweeps, a shallow copy of the grating is created with the specified resolution
    override. All sweeps use a fixed high Fourier order (max of sweep values) to
    isolate discretization effects.

    Args:
        grating: Grating used for all simulations.
        energies_ev: Photon energies to study.
        grazing_angle_deg: Fixed grazing angle in degrees.
        diffraction_order: Positive diffraction order used for the metric.
        polarization: Incident polarization. ``"s"`` selects TE (default);
            ``"p"`` selects TM.
        fourier_orders_values: Optional Fourier sweep values. Uses defaults if None.
        x_resolution_values: Optional x-resolution sweep values in nanometers.
            Uses defaults if None.
        z_resolution_values: Optional z-resolution sweep values in nanometers.
            Uses defaults if None.
        max_retries: Maximum retry attempts for a failed simulation point.
        output_dir: Optional directory for CSV exports. Creates subdirectory if needed.
        save_csv: Whether to export per-energy sweep CSV files to output_dir.
        show_progress: Whether to display progress bars during execution.

    Returns:
        Structured parameter-study result containing all sweep data and metadata.
        Use :func:`plot_parameter_study` to visualize the results.

    Example:
        >>> grating = grax.LaminarGrating(period_nm=500, duty=0.5, height_nm=100)
        >>> result = run_parameter_study(
        ...     grating=grating,
        ...     energies_ev=[500, 600, 700],
        ...     grazing_angle_deg=5.0,
        ...     diffraction_order=1
        ... )
        >>> plot_parameter_study(result)
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
                polarization=polarization,
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
                polarization=polarization,
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
                polarization=polarization,
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

    Creates a publication-ready figure with a 2D grid showing:
    - Rows: Different photon energies
    - Columns: Different swept parameters (Fourier orders, x-resolution, z-resolution)
    - Color: Efficiency values with logarithmic scaling
    - Crosses: Failed simulation points

    The x-axis uses logarithmic scaling for discretization parameters to clearly
    show convergence behavior across orders of magnitude.

    Args:
        result: Study result returned by :func:`run_parameter_study`.
        output_filename: Optional output image path. Saves to file if provided.
        title: Optional figure title.

    Returns:
        The created matplotlib Figure, or ``None`` when saved directly to disk.

    Example:
        >>> result = run_parameter_study(grating, [500, 600, 700], 5.0)
        >>> plot_parameter_study(result, title="Convergence Study")
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
                failed_y = np.full(np.count_nonzero(sweep.errors), np.nan, dtype=float)
                if y_values.size > 0:
                    y_min = float(np.nanmin(y_values))
                    y_max = float(np.nanmax(y_values))
                    offset = max((y_max - y_min) * 0.08, max(abs(y_max), 1.0) * 0.02, 1e-6)
                    failed_y.fill(y_min - offset)
                else:
                    failed_y.fill(-0.02)
                axis.scatter(
                    sweep.values[sweep.errors],
                    failed_y,
                    color="red",
                    marker="x",
                    s=40,
                    linewidths=1.5,
                    label="Simulation failed" if row_index == 0 and col_index == 0 else None,
                )
                if y_values.size > 0:
                    axis.set_ylim(bottom=min(float(np.nanmin(failed_y)) * 1.05, float(np.nanmin(y_values))))

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
    polarization: str,
    max_retries: int,
    fixed_fourier_orders: int,
) -> ParameterSweepSeries:
    """Run one parameter sweep for one energy.

    Executes a sequence of RCWA simulations varying one discretization parameter
    while holding all others fixed. Implements retry logic for failed points with
    exponential backoff behavior (same parameters retried up to max_retries).

    Args:
        grating: Grating used for all simulations in the sweep.
        parameter: Parameter name to sweep. One of "fourier_orders", "x_resolution_nm",
            or "z_resolution_nm".
        values: Array of parameter values to test.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Fixed grazing incidence angle in degrees.
        diffraction_order: Positive diffraction order to select.
        max_retries: Maximum retry attempts for failed points.
        fixed_fourier_orders: Fourier orders used when sweeping x/z resolution
            (to isolate discretization effects).

    Returns:
        ParameterSweepSeries containing parameter values, efficiencies, and
        error flags aligned by index.

    Note:
        - Fourier order sweeps use the input grating directly.
        - X/Z resolution sweeps clone the grating with the specified override.
        - Failed points (after all retries) receive NaN efficiency and error=True.
    """

    efficiencies = np.full(values.shape, np.nan, dtype=float)
    errors = np.zeros(values.shape, dtype=bool)
    error_messages = np.full(values.shape, "", dtype=object)

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
                    polarization=polarization,
                    backend="numba",
                ).run_single(energy_ev)
                efficiencies[index] = float(result["efficiency"])
                success = True
            except Exception as error:
                retries += 1
                if retries > max_retries:
                    errors[index] = True
                    error_messages[index] = str(error)

    return ParameterSweepSeries(
        parameter=parameter,
        values=np.asarray(values),
        efficiencies=efficiencies,
        errors=errors,
        error_messages=error_messages,
    )


def _clone_grating_with_override(
    grating: BaseGrating,
    parameter: str,
    value: float,
) -> BaseGrating:
    """Return a shallow-copied grating with one discretization override.

    Creates a copy of the grating with a single attribute replaced. Used during
    parameter studies to vary x/z resolution without modifying the original grating.

    Args:
        grating: Original grating to clone.
        parameter: Attribute name to override ("x_resolution_nm" or "z_resolution_nm").
        value: New value for the specified attribute.

    Returns:
        Shallow copy of grating with the specified attribute updated.
    """

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
    """Write one sweep CSV for one energy.

    Exports sweep results to a CSV file with columns: parameter, value,
        efficiency, error, error_message, energy_ev, grazing_angle_deg.
        Failed points (errors=True) have empty efficiency fields.

    Args:
        output_dir: Directory to write the CSV file.
        energy_ev: Photon energy in electronvolts (saved in file).
        grazing_angle_deg: Grazing incidence angle in degrees (saved in file).
        sweep: ParameterSweepSeries with values, efficiencies, and error flags.

    File format:
        parameter,value,efficiency,error,error_message,energy_ev,grazing_angle_deg
        fourier_orders,5,0.823,False,,500.0,5.0
        fourier_orders,7,0.845,False,,500.0,5.0
        fourier_orders,9,,True,synthetic failure,500.0,5.0

    Note:
        Filename format: ``parameter_study_{parameter}_E{energy_ev:.1f}eV.csv``
    """

    csv_path = output_dir / f"parameter_study_{sweep.parameter}_E{energy_ev:.1f}eV.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parameter",
                "value",
                "efficiency",
                "error",
                "error_message",
                "energy_ev",
                "grazing_angle_deg",
            ]
        )
        for value, efficiency, error, error_message in zip(
            sweep.values,
            sweep.efficiencies,
            sweep.errors,
            sweep.error_messages,
        ):
            writer.writerow(
                [
                    sweep.parameter,
                    value,
                    "" if error else efficiency,
                    bool(error),
                    "" if not error else str(error_message),
                    energy_ev,
                    grazing_angle_deg,
                ]
            )


def _get_log_formatter(x_min: float, x_max: float) -> FuncFormatter:
    """Create a formatter for log axes that preserves decimal labels.

    Constructs a matplotlib FuncFormatter for logarithmic x-axis tick labels
    with custom formatting rules. Uses different precision thresholds to
    display meaningful decimal values while avoiding label clutter.

    Args:
        x_min: Minimum value in the axis range.
        x_max: Maximum value in the axis range.

    Returns:
        FuncFormatter configured with threshold-based formatting rules.
        Uses LOG_TICK_THRESHOLDS and LOG_TICK_FORMATS for label formatting.

    Format thresholds:
        Values >= 10: Show as integers
        Values >= 1: Show with one decimal
        Values >= 0.1: Show with two decimals
        Values >= 0.01: Show with three decimals
        Values < 0.01: Show with four decimals
    """

    def formatter(x: float, _: int) -> str:
        if x < min(x_min, x_max) or x > max(x_min, x_max):
            return ""
        for index, threshold in enumerate(LOG_TICK_THRESHOLDS):
            if x >= threshold:
                return LOG_TICK_FORMATS[index].format(x)
        return LOG_TICK_FORMATS[-1].format(x)

    return FuncFormatter(formatter)
