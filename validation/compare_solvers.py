"""Compare RCWA and Nevière differential-method results for the validation cases.

Run each sweep with both solvers first, then this script. It writes a per-order
deviation table and a side-by-side plot with a difference panel for every case
that has results from both solvers.

    python validation/laminar/fixed_angle_sweep.py --solver rcwa --tag rerun
    python validation/laminar/fixed_angle_sweep.py --solver neviere
    python validation/compare_solvers.py

Both sides must come from the same code revision. The checked-in RCWA artifacts
predate several solver changes, so ``--tag rerun`` is used to produce a current
RCWA baseline alongside them rather than comparing against a stale one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _solver_comparison import (  # noqa: E402
    print_solver_comparison,
    write_solver_comparison,
)

VALIDATION_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ValidationCase:
    """One validation sweep that can be run with either solver.

    Attributes:
        name: Short case identifier used on the command line.
        title: Human-readable title for the plot.
        results_dir: Directory holding the sweep's result artifacts.
        stem: Base filename of the all-orders CSV, without suffix or extension.
        orders: Positive diffraction orders to compare.
        rcwa_tag: Tag used when the current RCWA baseline was produced.
    """

    name: str
    title: str
    results_dir: Path
    stem: str
    orders: tuple[int, ...]
    rcwa_tag: str = "rerun"

    @property
    def rcwa_csv(self) -> Path:
        """Return the path of the current-code RCWA all-orders CSV."""

        return self.results_dir / f"{self.stem}_{self.rcwa_tag}.csv"

    @property
    def neviere_csv(self) -> Path:
        """Return the path of the differential-method all-orders CSV."""

        return self.results_dir / f"{self.stem}_neviere.csv"

    @property
    def committed_rcwa_csv(self) -> Path:
        """Return the path of the checked-in RCWA all-orders CSV."""

        return self.results_dir / f"{self.stem}.csv"


CASES = (
    ValidationCase(
        name="laminar",
        title="Laminar 400 l/mm, fixed 4 deg grazing, p polarization",
        results_dir=VALIDATION_ROOT / "laminar" / "results",
        stem="laminar_fixed_angle_all_orders",
        orders=(1, 2, 3),
    ),
    ValidationCase(
        name="blazed",
        title="Blazed 600 l/mm monochromator, cff 2.25, p polarization",
        results_dir=VALIDATION_ROOT / "blazed" / "results",
        stem="blazed_comparison_monochromator_orders_1_3",
        orders=(1, 2, 3),
    ),
    ValidationCase(
        name="laminar_150lmm",
        title="Laminar 150 l/mm monochromator, cff 1.45, p polarization",
        results_dir=VALIDATION_ROOT / "laminar_150lmm" / "results",
        stem="laminar_150lmm_monochromator_all_orders",
        orders=(1, 2, 3),
    ),
    ValidationCase(
        name="blazed_multilayer",
        title="Blazed 2400 l/mm on Cr/C multilayer, energy-angle pairs, p polarization",
        results_dir=VALIDATION_ROOT / "blazed_multilayer" / "results",
        stem="blazed_multilayer_all_orders",
        orders=(1, 2, 3),
    ),
)


def main() -> int:
    """Write comparison artifacts for every case with results from both solvers."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES],
        action="append",
        help="Limit to one case. Repeatable. Defaults to every case.",
    )
    arguments = parser.parse_args()
    selected = [case for case in CASES if not arguments.case or case.name in arguments.case]

    missing: list[str] = []
    for case in selected:
        if not case.rcwa_csv.exists() or not case.neviere_csv.exists():
            absent = [
                str(path.relative_to(VALIDATION_ROOT))
                for path in (case.rcwa_csv, case.neviere_csv)
                if not path.exists()
            ]
            missing.append(f"{case.name}: missing {', '.join(absent)}")
            continue

        summary = write_solver_comparison(
            rcwa_csv=case.rcwa_csv,
            neviere_csv=case.neviere_csv,
            summary_csv=case.results_dir / f"{case.stem}_solver_comparison.csv",
            plot_path=case.results_dir / f"{case.stem}_solver_comparison.png",
            orders=list(case.orders),
            title=case.title,
        )
        print_solver_comparison(summary, title=case.title)

        if case.committed_rcwa_csv.exists():
            from _solver_comparison import compare_all_orders

            drift = compare_all_orders(
                case.committed_rcwa_csv,
                case.rcwa_csv,
                orders=list(case.orders),
            )
            print_solver_comparison(
                drift,
                title="  (for reference: checked-in RCWA artifact vs current-code RCWA)",
            )

    if missing:
        print("\nSkipped:")
        for line in missing:
            print(f"  {line}")
        print("Run each sweep with --solver rcwa --tag rerun and --solver neviere first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
