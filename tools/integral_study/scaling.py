"""Stage 1 of the integral-method study: measure how N scales with ``d / lambda``.

This is the decisive gate. The earlier attempt at an integral solver was
abandoned on the assumption that the boundary unknown count grows in proportion
to ``d / lambda``, from the standard rule of ~10 collocation points per
wavelength. That rule holds when the scatterer excites the full angular
spectrum. It should *not* hold for shallow gratings, where only a narrow band of
orders is excited -- which is why RCWA converges on the same cases with 20-35
Fourier orders, and why Goray reports ~50 collocation points per period at
``lambda / d = 1e-3``.

The assumption was never measured. This script measures it.

Two quantities are produced and must be read separately:

``N_required(d / lambda)``
    The *physics* question. For each ``d / lambda``, the smallest panel count
    reaching the tolerance against RCWA. If this saturates, the formulation is
    sound and the remaining problem is a constant factor.

``time(N, d / lambda)``
    The *engineering* question, which the current slow kernel dominates and which
    Stage 3 addresses. Cost per matrix entry grows with ``d / lambda`` because
    the Ewald spectral sum lengthens, independently of N.

Keeping them apart matters: a slow measurement at large ``d / lambda`` says
nothing about whether N itself is growing.

Usage::

    python tools/integral_study/scaling.py --max-ratio 100
    python tools/integral_study/scaling.py --sweep quadrature
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cases import (  # noqa: E402
    build_sinusoid_case,
    max_deviation,
    run_integral,
    run_reference,
)

#: Geometric ladder of panel counts. Each step is a factor of ~1.41, so the
#: bisection lands within 40% of the true threshold.
_PANEL_LADDER = (48, 68, 96, 136, 192, 272, 384, 544, 768, 1088, 1536)

#: Periods giving the listed d/lambda at 100 eV (lambda = 12.398 nm).
_DEFAULT_RATIOS = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep",
        choices=("ratio", "quadrature", "splitting"),
        default="ratio",
        help="Which parameter to sweep (default: ratio, the main study).",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=100.0,
        help="Stop the d/lambda ladder here (default: 100).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Target max absolute efficiency deviation vs RCWA (default: 1e-4).",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=180.0,
        help="Abandon a single solve above this wall-clock cost (default: 180).",
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=2,
        help=(
            "Panel Gauss-Legendre order (default: 2). Measured to have no effect on "
            "accuracy -- order 2 and order 16 agree to 4 significant figures -- while "
            "costing linearly in time, because the error is set by the piecewise-constant "
            "density and not by the panel quadrature."
        ),
    )
    parser.add_argument(
        "--depth-nm",
        type=float,
        default=20.0,
        help="Sinusoid depth in nm, held fixed across the sweep (default: 20).",
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Also verify the RCWA reference is itself converged in fourier_orders.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested sweep and return a process exit code."""

    args = parse_arguments()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep == "ratio":
        return _sweep_ratio(args)
    return _sweep_numerical_parameter(args)


def _sweep_ratio(args: argparse.Namespace) -> int:
    """Measure ``N_required`` and runtime across the ``d / lambda`` ladder.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """

    energy_ev = 100.0
    wavelength_nm = 1239.8 / energy_ev
    ratios = [ratio for ratio in _DEFAULT_RATIOS if ratio <= args.max_ratio]

    output = _RESULTS_DIR / "n_required_vs_ratio.csv"
    rows: list[dict[str, object]] = []

    print(f"{'d/lam':>7} {'period':>9} {'h/d':>8} {'N_req':>7} {'dev':>10} "
          f"{'time@N':>9} {'rcwa_t':>8} {'orders':>7}")
    print("-" * 78)

    for ratio in ratios:
        period_nm = ratio * wavelength_nm
        case = build_sinusoid_case(
            period_nm=period_nm,
            depth_nm=args.depth_nm,
            energy_ev=energy_ev,
        )
        reference = run_reference(case, "rcwa")
        if args.check_reference and not _reference_is_converged(case, reference, args.tolerance):
            print(f"{ratio:>7.0f}  RCWA reference not converged in fourier_orders; skipping")
            continue

        found = _smallest_panel_count(
            case,
            reference=reference,
            tolerance=args.tolerance,
            budget_seconds=args.budget_seconds,
            quadrature_order=args.quadrature_order,
        )
        if found is None:
            print(
                f"{ratio:>7.0f} {period_nm:>9.1f} {args.depth_nm / period_nm:>8.4f} "
                f"{'>ladder':>7} {'-':>10} {'-':>9} {reference.seconds:>7.2f}s "
                f"{len(reference.orders):>7}"
            )
            rows.append(
                {
                    "period_over_wavelength": ratio,
                    "period_nm": period_nm,
                    "depth_over_period": args.depth_nm / period_nm,
                    "n_required": "",
                    "deviation": "",
                    "seconds": "",
                    "rcwa_seconds": reference.seconds,
                }
            )
            continue

        panels, deviation, seconds = found
        print(
            f"{ratio:>7.0f} {period_nm:>9.1f} {args.depth_nm / period_nm:>8.4f} "
            f"{panels:>7} {deviation:>10.2e} {seconds:>8.1f}s {reference.seconds:>7.2f}s "
            f"{len(reference.orders):>7}"
        )
        rows.append(
            {
                "period_over_wavelength": ratio,
                "period_nm": period_nm,
                "depth_over_period": args.depth_nm / period_nm,
                "n_required": panels,
                "deviation": deviation,
                "seconds": seconds,
                "rcwa_seconds": reference.seconds,
            }
        )

    _write_csv(output, rows)
    print(f"\nWrote {output}")
    _report_trend(rows)
    return 0


def _smallest_panel_count(
    case,
    *,
    reference,
    tolerance: float,
    budget_seconds: float,
    quadrature_order: int,
) -> tuple[int, float, float] | None:
    """Return the first ladder rung meeting the tolerance.

    Walks the ladder upward rather than bisecting, because the deviation is not
    guaranteed monotonic and the cheap rungs are nearly free compared with the
    expensive ones.

    Args:
        case: The case to solve.
        reference: The RCWA result to compare against.
        tolerance: Target maximum absolute efficiency deviation.
        budget_seconds: Abandon once a single solve exceeds this.
        quadrature_order: Panel Gauss-Legendre order.

    Returns:
        ``(panels, deviation, seconds)``, or ``None`` if the ladder was exhausted
        or the time budget was hit first.
    """

    for panels in _PANEL_LADDER:
        start = time.perf_counter()
        try:
            result = run_integral(
                case,
                boundary_points=panels,
                quadrature_order=quadrature_order,
            )
        except Exception as error:  # noqa: BLE001 - a study script reports and moves on
            print(f"    N={panels}: solve failed ({type(error).__name__}: {error})")
            return None
        deviation = max_deviation(result, reference)
        if deviation <= tolerance:
            return panels, deviation, result.seconds
        if time.perf_counter() - start > budget_seconds:
            print(f"    N={panels}: {deviation:.2e}, over budget, stopping this ratio")
            return None
    return None


def _reference_is_converged(case, reference, tolerance: float) -> bool:
    """Return whether the RCWA reference is converged in ``fourier_orders``.

    A reference that is itself unconverged would make the integral solver look
    wrong for the wrong reason, so this is checked before it is trusted.

    Args:
        case: The case.
        reference: Its RCWA result at the case's ``fourier_orders``.
        tolerance: The same tolerance the study gates on.

    Returns:
        Whether raising the truncation by half changes the answer by less than
        the tolerance.
    """

    from dataclasses import replace as _replace

    richer = run_reference(
        _replace(case, fourier_orders=int(case.fourier_orders * 1.5)), "rcwa"
    )
    return max_deviation(reference, richer) <= tolerance


def _sweep_numerical_parameter(args: argparse.Namespace) -> int:
    """Sweep one numerical knob at a fixed geometry, to separate its effect.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """

    case = build_sinusoid_case(period_nm=25.0 * (1239.8 / 100.0), depth_nm=args.depth_nm)
    reference = run_reference(case, "rcwa")
    panels = 272

    if args.sweep == "quadrature":
        label = "quadrature_order"
        values: tuple[object, ...] = (2, 4, 6, 8, 12, 16)
        make = lambda value: {"quadrature_order": int(value)}  # noqa: E731
    else:
        label = "ewald_splitting"
        wavenumber = 2.0 * np.pi / case.wavelength_nm
        # Expressed as multiples of |k|/2, which is what default_ewald_splitting
        # floors at. The interesting direction is *below* that floor: a smaller
        # splitting lengthens the lattice series but shortens the spectral one,
        # and the spectral sum is the half that grows with d/lambda. The floor
        # only guarantees the lattice series converges, it does not minimise
        # total work, so scales under 1.0 are the ones worth measuring. Below
        # about 0.2 the lattice series needs more terms than the solver allows
        # and raises.
        values = tuple(
            round(scale * wavenumber / 2.0, 8) for scale in (0.25, 0.5, 1.0, 2.0, 4.0)
        )
        make = lambda value: {"ewald_splitting": float(value)}  # noqa: E731

    print(f"case d/lambda={case.period_over_wavelength:.0f}, N={panels}")
    print(f"{label:>18} {'dev vs rcwa':>13} {'time':>9}")
    print("-" * 44)
    rows: list[dict[str, object]] = []
    for value in values:
        result = run_integral(case, boundary_points=panels, **make(value))
        deviation = max_deviation(result, reference)
        print(f"{value!s:>18} {deviation:>13.3e} {result.seconds:>8.2f}s")
        rows.append({label: value, "deviation": deviation, "seconds": result.seconds})

    output = _RESULTS_DIR / f"sweep_{args.sweep}.csv"
    _write_csv(output, rows)
    print(f"\nWrote {output}")
    return 0


def _report_trend(rows: list[dict[str, object]]) -> None:
    """Print the growth exponent of ``N_required`` against ``d / lambda``.

    The gate is stated in exponents: saturation (near 0) means the formulation is
    sound; linear growth (near 1) means the discretization must be replaced
    before anything else; faster than linear contradicts the literature and
    points at a formulation error.

    Args:
        rows: Collected sweep rows.
    """

    usable = [row for row in rows if row["n_required"] != ""]
    if len(usable) < 2:
        print("Not enough points to estimate a trend.")
        return

    ratios = np.array([float(row["period_over_wavelength"]) for row in usable])
    counts = np.array([float(row["n_required"]) for row in usable])
    exponent = float(np.polyfit(np.log(ratios), np.log(counts), 1)[0])

    print(f"\nN_required ~ (d/lambda)^{exponent:.2f} over "
          f"{ratios[0]:.0f} to {ratios[-1]:.0f}")
    if exponent < 0.35:
        verdict = "SATURATING - formulation is sound, remaining problem is a constant factor"
    elif exponent < 0.8:
        verdict = "SUBLINEAR - workable, but the discretization upgrade matters"
    elif exponent < 1.25:
        verdict = "LINEAR - the panel discretization must be replaced before Stage 3"
    else:
        verdict = "SUPERLINEAR - contradicts the literature; suspect a formulation error"
    print(f"Verdict: {verdict}")


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
