"""Stage 0 of the integral-method study: validate against RCWA and Nevière.

Runs every benchmark case through all three solvers at the ``res2`` level, using
the same grating object for each, and reports the largest efficiency deviation
per case. The flat case is additionally checked against the analytic Fresnel
reflectance, which is the only closed-form answer available.

The integral solver is not wired into ``grax.run_simulation`` and this script
does not wire it in; it imports ``res2_im`` directly.

Usage::

    python tools/integral_study/compare.py
    python tools/integral_study/compare.py --case B2_sinusoid --boundary-points 384
    python tools/integral_study/compare.py --tolerance 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cases import (  # noqa: E402
    build_cases,
    fresnel_reflectance,
    max_deviation,
    run_integral,
    run_reference,
)

from grax.materials import resolve_refractive_index  # noqa: E402

# Cases with corners are expected to converge more slowly; Stage 0 only gates the
# smooth ones, since fixing corner accuracy is Stage 2's job.
_SMOOTH_CASES = ("B1_flat", "B2_sinusoid", "B6_coated")


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default=None, help="Run only this case.")
    parser.add_argument(
        "--boundary-points",
        default="auto",
        help="Panels per interface, or 'auto' (default).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Pass threshold on the maximum efficiency deviation (default: 1e-4).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the comparison and return a process exit code."""

    args = parse_arguments()
    boundary_points = (
        "auto" if args.boundary_points == "auto" else int(args.boundary_points)
    )

    cases = build_cases()
    if args.case is not None:
        if args.case not in cases:
            print(f"Unknown case {args.case!r}. Available: {', '.join(sorted(cases))}")
            return 2
        cases = {args.case: cases[args.case]}

    print(f"{'case':<20} {'d/lam':>7} {'if':>3} {'pol':>4} {'N':>6} "
          f"{'vs rcwa':>10} {'vs nev':>10} {'balance':>8} {'time':>8}")
    print("-" * 92)

    gated_failures: list[str] = []
    for name, case in cases.items():
        rcwa = run_reference(case, "rcwa")
        neviere = run_reference(case, "neviere")
        integral = run_integral(case, boundary_points=boundary_points)

        deviation_rcwa = max_deviation(integral, rcwa)
        deviation_neviere = max_deviation(integral, neviere)
        polarization = "TE" if case.polarization == 1 else "TM"

        print(
            f"{name:<20} {case.period_over_wavelength:>7.1f} {case.interfaces:>3} "
            f"{polarization:>4} {integral.unknowns:>6} "
            f"{deviation_rcwa:>10.2e} {deviation_neviere:>10.2e} "
            f"{integral.energy_balance:>8.4f} {integral.seconds:>7.1f}s"
        )

        if name in _SMOOTH_CASES and deviation_rcwa > args.tolerance:
            gated_failures.append(f"{name}: {deviation_rcwa:.2e} > {args.tolerance:.0e}")

        if name == "B1_flat":
            _report_fresnel(case, integral)

    print()
    if args.case is None:
        print("Note: cases with corners (B3, B4, B5) are not gated at Stage 0.")
    if gated_failures:
        print("GATE FAILED on the smooth cases:")
        for failure in gated_failures:
            print(f"  {failure}")
        return 1
    print(f"GATE PASSED: every smooth case within {args.tolerance:.0e} of RCWA.")
    return 0


def _report_fresnel(case, integral) -> None:
    """Print the analytic Fresnel comparison for the flat case.

    Args:
        case: The flat-interface case.
        integral: Its integral-solver result.
    """

    n_below = complex(resolve_refractive_index(case.grating.substrate_material, case.energy_ev))
    exact = fresnel_reflectance(
        n_above=1.0 + 0.0j,
        n_below=n_below,
        beta0=case.beta0,
        polarization=case.polarization,
    )
    got = integral.order(0)
    print(
        f"{'':<20} analytic Fresnel: exact={exact:.9f} integral={got:.9f} "
        f"rel={abs(got - exact) / max(exact, 1e-30):.2e}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
