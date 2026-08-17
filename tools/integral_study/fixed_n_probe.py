"""Stage 1, cheap form: how does the error grow with ``d / lambda`` at fixed N?

Finding the smallest panel count that meets a tolerance costs a whole ladder of
solves per ``d / lambda``, and the expensive rungs dominate. The gate does not
actually need the threshold value -- it needs the *exponent*. Fixing N and
watching the deviation grow gives that in one solve per ratio, and the two are
directly related: with an ``O(h^p)`` discretization,

    deviation(N, r) ~ C(r) / N^p     =>     N_required(r) ~ (C(r) / tol)^(1/p)

so if the deviation at fixed N grows like ``r^q``, then
``N_required ~ r^(q/p)``. With the measured ``p = 2`` for piecewise-constant
panels, a deviation growing like ``r^2`` means ``N_required ~ r``, and a flat
deviation means ``N_required`` saturates.

The script also reports the deviation at two panel counts so ``p`` is measured
rather than assumed.

Usage::

    python tools/integral_study/fixed_n_probe.py --ratios 5 10 25 50 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cases import (  # noqa: E402
    build_sinusoid_case,
    max_deviation,
    run_integral,
    run_reference,
)

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: A deviation this large is not an error estimate; efficiencies are bounded by
#: one, so the coarse solve has simply not begun to converge.
_MAX_TRUSTED_DEVIATION = 0.05
#: Piecewise-constant panels converge at order two. An apparent order far from
#: that means the two solves straddle the onset of convergence.
_MIN_TRUSTED_ORDER = 1.4
_MAX_TRUSTED_ORDER = 3.0


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 25.0, 50.0, 100.0],
        help="d/lambda values to probe (default: 5 10 25 50 100).",
    )
    parser.add_argument(
        "--panels",
        type=int,
        nargs=2,
        default=[96, 192],
        help="Two panel counts, to measure the convergence order (default: 96 192).",
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=2,
        help="Panel Gauss-Legendre order (default: 2, measured to be accuracy-neutral).",
    )
    parser.add_argument(
        "--depth-nm", type=float, default=20.0, help="Sinusoid depth (default: 20)."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance used to project N_required (default: 1e-4).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the probe and return a process exit code."""

    args = parse_arguments()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    low, high = int(args.panels[0]), int(args.panels[1])
    energy_ev = 100.0
    wavelength_nm = 1239.8 / energy_ev

    print(f"panels {low} and {high}, quadrature order {args.quadrature_order}, "
          f"depth {args.depth_nm} nm, 4 deg grazing, TM")
    print(f"{'d/lam':>7} {'h/d':>7} {'eta_0':>9} {'dev@' + str(low):>10} "
          f"{'dev@' + str(high):>10} {'order p':>8} {'N_req':>8} {'t@' + str(high):>8}")
    print("-" * 82)

    rows: list[dict[str, object]] = []
    for ratio in args.ratios:
        period_nm = ratio * wavelength_nm
        case = build_sinusoid_case(
            period_nm=period_nm, depth_nm=args.depth_nm, energy_ev=energy_ev
        )
        reference = run_reference(case, "rcwa")
        coarse = run_integral(
            case, boundary_points=low, quadrature_order=args.quadrature_order
        )
        fine = run_integral(
            case, boundary_points=high, quadrature_order=args.quadrature_order
        )
        deviation_low = max_deviation(coarse, reference)
        deviation_high = max_deviation(fine, reference)

        order = _convergence_order(deviation_low, deviation_high, low, high)
        projected = _projected_panels(deviation_high, high, order, args.tolerance)

        print(
            f"{ratio:>7.0f} {args.depth_nm / period_nm:>7.4f} {reference.order(0):>9.5f} "
            f"{deviation_low:>10.2e} {deviation_high:>10.2e} {order:>8.2f} "
            f"{projected:>8} {fine.seconds:>7.1f}s"
        )
        rows.append(
            {
                "period_over_wavelength": ratio,
                "depth_over_period": args.depth_nm / period_nm,
                "eta_zero": reference.order(0),
                "deviation_low": deviation_low,
                "deviation_high": deviation_high,
                "convergence_order": order,
                "projected_n_required": projected,
                "seconds_high": fine.seconds,
            }
        )

    output = _RESULTS_DIR / "fixed_n_probe.csv"
    _write_csv(output, rows)
    print(f"\nWrote {output}")
    _report_verdict(rows, high)
    return 0


def _convergence_order(low_error: float, high_error: float, low: int, high: int) -> float:
    """Return the measured convergence order between two panel counts.

    Args:
        low_error: Deviation at the coarse count.
        high_error: Deviation at the fine count.
        low: Coarse panel count.
        high: Fine panel count.

    Returns:
        The exponent ``p`` in ``error ~ N^-p``, or ``nan`` if it cannot be formed.
    """

    if low_error <= 0.0 or high_error <= 0.0 or high == low:
        return float("nan")
    return float(np.log(low_error / high_error) / np.log(high / low))


def _projected_panels(error: float, panels: int, order: float, tolerance: float) -> int:
    """Return the panel count projected to reach the tolerance.

    Args:
        error: Deviation at ``panels``.
        panels: The panel count that produced ``error``.
        order: Measured convergence order.
        tolerance: Target deviation.

    Returns:
        Projected panel count, or ``-1`` when the projection is not meaningful.
    """

    if not np.isfinite(order) or order <= 0.0:
        return -1
    if error <= tolerance:
        return panels
    return int(np.ceil(panels * (error / tolerance) ** (1.0 / order)))


def _report_verdict(rows: list[dict[str, object]], panels: int) -> None:
    """Print the growth exponents and the Stage 1 verdict.

    Only rows whose two solves are *both* inside the asymptotic convergence
    regime are fitted. Outside it the ``error ~ N^-p`` model does not hold, and
    the projection inverts a meaningless ``p``: a coarse solve that has not
    converged at all can report an absurdly high apparent order, project a tiny
    ``N_required`` from it, and drag the fitted exponent toward zero -- turning a
    linear trend into a false "saturating" verdict. Two filters catch that.

    Args:
        rows: Collected probe rows.
        panels: The fine panel count the deviations were measured at.
    """

    if len(rows) < 2:
        print("Not enough points for a verdict.")
        return

    trusted = [row for row in rows if _is_asymptotic(row)]
    rejected = [row for row in rows if not _is_asymptotic(row)]
    for row in rejected:
        print(
            f"\nExcluded d/lambda={float(row['period_over_wavelength']):.0f}: "
            f"not in the asymptotic regime "
            f"(dev@coarse={float(row['deviation_low']):.2e}, "
            f"measured order p={float(row['convergence_order']):.2f}). "
            "Re-run this ratio with larger --panels."
        )
    if len(trusted) < 2:
        print("\nToo few trustworthy points for a verdict. Raise --panels and re-run.")
        return

    ratios = np.array([float(row["period_over_wavelength"]) for row in trusted])
    deviations = np.array([float(row["deviation_high"]) for row in trusted])
    projected = np.array([float(row["projected_n_required"]) for row in trusted])

    deviation_exponent = float(np.polyfit(np.log(ratios), np.log(deviations), 1)[0])
    panel_exponent = float(np.polyfit(np.log(ratios), np.log(projected), 1)[0])

    print(f"\nfitted over d/lambda = {ratios[0]:.0f} to {ratios[-1]:.0f} "
          f"({len(trusted)} trustworthy points)")
    print(f"deviation at N={panels} ~ (d/lambda)^{deviation_exponent:.2f}")
    print(f"projected N_required ~ (d/lambda)^{panel_exponent:.2f}")

    if panel_exponent < 0.35:
        verdict = "SATURATING - formulation sound; remaining problem is a constant factor"
    elif panel_exponent < 0.8:
        verdict = "SUBLINEAR - workable; the discretization upgrade is what buys the margin"
    elif panel_exponent < 1.25:
        verdict = "LINEAR - panel discretization must be replaced before anything else"
    else:
        verdict = "SUPERLINEAR - contradicts the literature; suspect a formulation error"
    print(f"Verdict: {verdict}")


def _is_asymptotic(row: dict[str, object]) -> bool:
    """Return whether one row's two solves are both converging as expected.

    Two independent checks, either of which is enough to reject the row:

    - The coarse deviation must be physically small. Efficiencies are bounded by
      one, so a deviation near or above that is not an error estimate at all,
      it is a solve that has not started converging.
    - The measured convergence order must be near the value the discretization
      actually has. Piecewise-constant panels give ``p = 2``; an apparent order
      far from that means the two solves straddle the onset of convergence
      rather than sitting inside it.

    Args:
        row: One probe row.

    Returns:
        Whether the row may be fitted.
    """

    deviation_low = float(row["deviation_low"])
    order = float(row["convergence_order"])
    projected = float(row["projected_n_required"])
    if projected <= 0:
        return False
    if deviation_low > _MAX_TRUSTED_DEVIATION:
        return False
    return _MIN_TRUSTED_ORDER <= order <= _MAX_TRUSTED_ORDER


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV, doing nothing when there are none.

    Args:
        path: Output path.
        rows: Rows to write.
    """

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
