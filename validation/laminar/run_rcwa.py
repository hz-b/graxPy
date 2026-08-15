"""Run the laminar 400 l/mm fixed-angle sweep with the RCWA solver.

Writes ``results/laminar_fixed_angle_all_orders_rcwa.csv`` plus a plot against
the measured curve. The grating and the sweep grid come from
``grating_definition.py``, which ``run_neviere.py`` also uses, so the two runs
stay comparable.

```bash
python validation/laminar/run_rcwa.py
python validation/laminar/run_rcwa.py --quick
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Laminar 400 l/mm fixed-angle sweep (RCWA)")
parser.add_argument("--quick", action="store_true", help="Run a few coarse energy points")
parser.add_argument("--stride", type=int, default=1, help="Keep every Nth energy point")
parser.add_argument("--live-plot", action="store_true", help="Show the sweep while it runs")
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "TkAgg" if args.live_plot else "Agg")

import matplotlib.pyplot as plt  # noqa: E402

import grax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grating_definition as case  # noqa: E402

SOLVER = "rcwa"


def main() -> None:
    """Run the sweep and write this solver's artifacts."""

    paths = case.output_paths(SOLVER)
    experimental = grax.load_experimental_csv(case.MEASUREMENT_FILE)

    runner = grax.BatchSimulationRunner(
        default_diffraction_order=case.DIFFRACTION_ORDER,
        default_fourier_orders=case.QUICK_FOURIER_ORDERS if args.quick else case.FOURIER_ORDERS,
        show_progress=True,
        live_plot=args.live_plot,
        live_plot_x_key="energy_ev",
        live_plot_order_count=1,
        live_plot_reference_data=experimental,
        on_error="fail_fast",
        resume=False,
        max_workers="auto",
        default_solver=SOLVER,
    )

    case.build_grating(quick=args.quick).plot_profile(paths["profile_plot"])
    results = list(runner.run_cases(case.build_cases(quick=args.quick, stride=args.stride)))
    successful = [result for result in results if result.status == "ok"]

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        [result.energy_ev for result in successful],
        [result.selected_efficiency for result in successful],
        "b-o",
        linewidth=0.8,
        markersize=2.0,
        label=f"simulation ({SOLVER})",
    )
    axis.plot(
        experimental[:, 0],
        experimental[:, 1],
        "r-s",
        linewidth=0.8,
        markersize=2.0,
        label="measurement",
    )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title("RCWA Simulation vs Experimental Data")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(paths["measurement_plot"], dpi=150, bbox_inches="tight")
    plt.close(figure)

    grax.write_all_orders_csv(results, paths["all_orders_csv"])

    print(f"Computed {len(successful)} energy points with the {SOLVER} solver.")
    print(f"All-orders CSV saved to: {paths['all_orders_csv']}")
    print(f"Measurement comparison saved to: {paths['measurement_plot']}")
    print(f"Grating profile saved to: {paths['profile_plot']}")


# Spawned multiprocessing workers re-import this file by path; without the guard
# each worker would re-run the whole sweep and recursively spawn more.
if __name__ == "__main__":
    main()
