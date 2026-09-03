"""Run the blazed 600 l/mm monochromator sweep with the RCWA solver.

Writes ``results/blazed_comparison_monochromator_orders_1_3_rcwa.csv`` plus an
orders 1-3 plot. The grating and the sweep grid come from
``grating_definition.py``, which ``run_neviere.py`` also uses, so the two runs
stay comparable.

```bash
python validation/blazed/run_rcwa.py
python validation/blazed/run_rcwa.py --quick
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Blazed 600 l/mm monochromator sweep (RCWA)")
parser.add_argument("--quick", action="store_true", help="Run a few coarse energy points")
parser.add_argument("--stride", type=int, default=1, help="Keep every Nth energy point")
parser.add_argument("--live-plot", action="store_true", help="Show the sweep while it runs")
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "TkAgg" if args.live_plot else "Agg")

import grax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grating_definition as case  # noqa: E402

SOLVER = "rcwa"
SOLVER_TITLE = "RCWA"


def main() -> None:
    """Run the sweep and write this solver's artifacts."""

    grax.setup_logging(level="INFO", run_id=f"blazed_sweep_{SOLVER}")
    paths = case.output_paths(SOLVER)

    runner = grax.BatchSimulationRunner(
        diffraction_order=case.DIFFRACTION_ORDER,
        fourier_orders=case.QUICK_FOURIER_ORDERS if args.quick else case.FOURIER_ORDERS,
        show_progress=True,
        live_plot=args.live_plot,
        live_plot_x_key="energy_ev",
        live_plot_order_count=3,
        checkpoint_dir=paths["checkpoint_dir"],
        resume=False,
        backend="numba",
        solver=SOLVER,
    )

    case.build_grating(quick=args.quick).plot_profile(paths["profile_plot"])
    results = list(
        runner.run_cases(
            case.build_cases(quick=args.quick, stride=args.stride),
            metadata=case.sweep_metadata(SOLVER),
        )
    )

    grax.write_all_orders_csv(results, paths["all_orders_csv"])
    grax.plot_order_subset(
        results,
        paths["orders_plot"],
        diffraction_orders=case.PLOT_ORDERS,
        title=(
            f"Blazed Grating Monochromator Sweep ({case.PERIOD_LPERMM} l/mm, "
            f"BA={case.BLAZE_ANGLE_DEG} deg), {SOLVER_TITLE}: Orders 1-3"
        ),
    )

    successful = sum(result.status == "ok" for result in results)
    print(f"Computed {successful} monochromator points with the {SOLVER} solver.")
    print(f"All-orders CSV saved to: {paths['all_orders_csv']}")
    print(f"Orders plot saved to: {paths['orders_plot']}")
    print(f"Grating profile saved to: {paths['profile_plot']}")


# Spawned multiprocessing workers re-import this file by path; without the guard
# each worker would re-run the whole sweep and recursively spawn more.
if __name__ == "__main__":
    main()
