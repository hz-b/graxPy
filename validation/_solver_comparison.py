"""Shared helpers for running validation sweeps with either solver.

Every validation sweep script accepts ``--solver`` and ``--stride``. With no
flags a script behaves exactly as before and writes to its historical output
paths, so the checked-in RCWA artifacts stay reproducible. ``--solver neviere``
writes to ``*_neviere.*`` siblings instead, which lets both solvers' results sit
next to each other and be compared per diffraction order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOLVER_CHOICES = ("rcwa", "neviere")

#: Human-readable labels used in comparison plots and tables.
SOLVER_LABELS = {"rcwa": "RCWA (modal)", "neviere": "Nevière (differential)"}


def add_solver_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared solver-selection arguments to one script's parser.

    Args:
        parser: Parser to extend.

    Returns:
        The same parser, for chaining.
    """

    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="rcwa",
        help="Electromagnetic solver to run (default: rcwa, matching the checked-in results).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth sweep point. Use to subsample an expensive sweep.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Extra suffix appended to every output path, on top of the solver "
            "suffix. Use to write a fresh run alongside the checked-in artifacts "
            "instead of overwriting them, e.g. --tag rerun."
        ),
    )
    return parser


def solver_suffix(solver: str) -> str:
    """Return the output-path suffix for one solver.

    The RCWA suffix is empty so existing artifact paths are untouched.

    Args:
        solver: Solver name.

    Returns:
        Suffix to insert before a file extension.
    """

    if solver not in SOLVER_CHOICES:
        raise ValueError(f"solver must be one of {SOLVER_CHOICES}, got {solver!r}.")
    return "" if solver == "rcwa" else f"_{solver}"


def solver_output_path(path: Path, solver: str, tag: str = "") -> Path:
    """Return one output path rewritten for the given solver.

    Args:
        path: Historical (RCWA) output path.
        solver: Solver name.
        tag: Optional extra suffix, so a fresh run can be written alongside the
            checked-in artifacts rather than over them.

    Returns:
        Path with the solver suffix and tag inserted before the extension.
    """

    suffix = solver_suffix(solver)
    if tag:
        suffix = f"{suffix}_{tag.strip('_')}"
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def apply_stride(values, stride: int):
    """Return every ``stride``-th element of a sweep axis.

    Args:
        values: Sequence or array of sweep points.
        stride: Keep every Nth point. Must be >= 1.

    Returns:
        The subsampled sequence, of the same type where practical.
    """

    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if stride == 1:
        return values
    if isinstance(values, np.ndarray):
        return values[::stride]
    return list(values)[::stride]


def load_all_orders_csv(path: Path) -> pd.DataFrame:
    """Load one ``write_all_orders_csv`` output.

    Args:
        path: CSV path.

    Returns:
        DataFrame with energy, order, efficiency, and diffraction angle.
    """

    frame = pd.read_csv(path)
    frame["order"] = frame["order"].astype(float)
    return frame


def compare_all_orders(
    rcwa_csv: Path,
    neviere_csv: Path,
    *,
    orders: list[int],
) -> pd.DataFrame:
    """Return a per-order deviation table between two solver runs.

    Only energies present in both runs are compared, so a subsampled
    differential-method run can still be checked against a full RCWA sweep.

    Args:
        rcwa_csv: All-orders CSV from the RCWA run.
        neviere_csv: All-orders CSV from the differential-method run.
        orders: Positive diffraction orders to compare. The solver reports the
            reflected orders with a negative sign, matching
            :func:`grax.efficiency_for_order`.

    Returns:
        One row per requested order with the shared point count and the maximum,
        mean, and RMS absolute deviation, plus the maximum deviation relative to
        the peak RCWA efficiency of that order.
    """

    rcwa = load_all_orders_csv(rcwa_csv)
    neviere = load_all_orders_csv(neviere_csv)

    rows = []
    for order in orders:
        rcwa_order = rcwa[np.isclose(rcwa["order"], -order)][["energy_ev", "efficiency"]]
        neviere_order = neviere[np.isclose(neviere["order"], -order)][["energy_ev", "efficiency"]]
        merged = rcwa_order.merge(
            neviere_order,
            on="energy_ev",
            suffixes=("_rcwa", "_neviere"),
        )
        if merged.empty:
            rows.append(
                {
                    "order": order,
                    "shared_points": 0,
                    "max_abs_deviation": float("nan"),
                    "mean_abs_deviation": float("nan"),
                    "rms_deviation": float("nan"),
                    "peak_rcwa_efficiency": float("nan"),
                    "max_deviation_over_peak": float("nan"),
                }
            )
            continue

        deviation = np.abs(
            merged["efficiency_rcwa"].to_numpy() - merged["efficiency_neviere"].to_numpy()
        )
        peak = float(np.max(np.abs(merged["efficiency_rcwa"].to_numpy())))
        rows.append(
            {
                "order": order,
                "shared_points": int(len(merged)),
                "max_abs_deviation": float(np.max(deviation)),
                "mean_abs_deviation": float(np.mean(deviation)),
                "rms_deviation": float(np.sqrt(np.mean(deviation**2))),
                "peak_rcwa_efficiency": peak,
                "max_deviation_over_peak": (
                    float(np.max(deviation) / peak) if peak > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_solver_comparison(
    rcwa_csv: Path,
    neviere_csv: Path,
    output_path: Path,
    *,
    orders: list[int],
    title: str,
) -> None:
    """Save a side-by-side solver comparison with a deviation panel.

    Args:
        rcwa_csv: All-orders CSV from the RCWA run.
        neviere_csv: All-orders CSV from the differential-method run.
        output_path: Output image path.
        orders: Positive diffraction orders to overlay.
        title: Plot title.
    """

    import matplotlib.pyplot as plt

    rcwa = load_all_orders_csv(rcwa_csv)
    neviere = load_all_orders_csv(neviere_csv)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    efficiency_axis, deviation_axis = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, order in enumerate(orders):
        color = colors[index % len(colors)]
        rcwa_order = rcwa[np.isclose(rcwa["order"], -order)].sort_values("energy_ev")
        neviere_order = neviere[np.isclose(neviere["order"], -order)].sort_values("energy_ev")
        efficiency_axis.plot(
            rcwa_order["energy_ev"],
            rcwa_order["efficiency"],
            "-",
            color=color,
            linewidth=1.4,
            label=f"order {order} - {SOLVER_LABELS['rcwa']}",
        )
        efficiency_axis.plot(
            neviere_order["energy_ev"],
            neviere_order["efficiency"],
            "--",
            color=color,
            linewidth=1.4,
            dashes=(4, 3),
            label=f"order {order} - {SOLVER_LABELS['neviere']}",
        )

        merged = rcwa_order[["energy_ev", "efficiency"]].merge(
            neviere_order[["energy_ev", "efficiency"]],
            on="energy_ev",
            suffixes=("_rcwa", "_neviere"),
        )
        if merged.empty:
            continue
        deviation_axis.semilogy(
            merged["energy_ev"],
            np.abs(merged["efficiency_rcwa"] - merged["efficiency_neviere"]),
            "-",
            color=color,
            linewidth=1.0,
            label=f"order {order}",
        )

    efficiency_axis.set_ylabel("Diffraction efficiency")
    efficiency_axis.set_title(title)
    efficiency_axis.grid(True, alpha=0.3)
    efficiency_axis.legend(loc="best", fontsize=8)

    deviation_axis.set_xlabel("Photon energy (eV)")
    deviation_axis.set_ylabel("|RCWA - Nevière|")
    deviation_axis.grid(True, alpha=0.3, which="both")
    deviation_axis.legend(loc="best", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_solver_comparison(
    *,
    rcwa_csv: Path,
    neviere_csv: Path,
    summary_csv: Path,
    plot_path: Path,
    orders: list[int],
    title: str,
) -> pd.DataFrame:
    """Write the comparison table and plot for one validation case.

    Args:
        rcwa_csv: All-orders CSV from the RCWA run.
        neviere_csv: All-orders CSV from the differential-method run.
        summary_csv: Destination for the per-order deviation table.
        plot_path: Destination for the comparison figure.
        orders: Positive diffraction orders to compare.
        title: Plot title.

    Returns:
        The per-order deviation table.
    """

    for path in (rcwa_csv, neviere_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run the sweep for both solvers before comparing:\n"
                f"  python <sweep script>\n"
                f"  python <sweep script> --solver neviere"
            )

    summary = compare_all_orders(rcwa_csv, neviere_csv, orders=orders)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    plot_solver_comparison(rcwa_csv, neviere_csv, plot_path, orders=orders, title=title)
    return summary


def print_solver_comparison(summary: pd.DataFrame, *, title: str) -> None:
    """Print one comparison table in a readable fixed-width form.

    Args:
        summary: Table returned by :func:`compare_all_orders`.
        title: Heading printed above the table.
    """

    print(f"\n{title}")
    print(f"{'order':>6} {'points':>7} {'max|dE|':>12} {'mean|dE|':>12} "
          f"{'rms':>12} {'peak E':>12} {'max|dE|/peak':>14}")
    for _, row in summary.iterrows():
        print(
            f"{int(row['order']):>6} {int(row['shared_points']):>7} "
            f"{row['max_abs_deviation']:>12.3e} {row['mean_abs_deviation']:>12.3e} "
            f"{row['rms_deviation']:>12.3e} {row['peak_rcwa_efficiency']:>12.3e} "
            f"{row['max_deviation_over_peak']:>14.3e}"
        )


def validation_root_on_path() -> Path:
    """Add the validation root to ``sys.path`` and return it.

    Sweep scripts run as plain files rather than as a package, so they import
    this module by first putting its directory on the path.
    """

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
