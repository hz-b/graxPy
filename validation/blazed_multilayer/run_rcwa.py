"""Run the blazed 2400 l/mm multilayer energy-angle sweep with the RCWA solver.

Writes ``results/blazed_multilayer_all_orders_rcwa.csv`` plus a second-order plot
against the DiffractMod reference. The grating and the sweep grid come from
``grating_definition.py``, which ``run_neviere.py`` also uses, so the two runs
stay comparable.

This is the most expensive validation case: at full resolution each point solves
a 60-bilayer stack sliced at 0.01 nm. Use ``--stride`` to subsample the 1727
reference points.

```bash
python validation/blazed_multilayer/run_rcwa.py --stride 10
python validation/blazed_multilayer/run_rcwa.py --quick
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Blazed multilayer energy-angle sweep (RCWA)")
parser.add_argument("--quick", action="store_true", help="Run a few coarse points")
parser.add_argument("--stride", type=int, default=1, help="Keep every Nth reference point")
parser.add_argument("--live-plot", action="store_true", help="Show the sweep while it runs")
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "TkAgg" if args.live_plot else "Agg")

import matplotlib.pyplot as plt  # noqa: E402

import grax  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grating_definition as case  # noqa: E402

SOLVER = "rcwa"
SOLVER_TITLE = "RCWA"


def main() -> None:
    """Run the sweep and write this solver's artifacts."""

    grax.setup_logging(level="INFO", run_id=f"blazed_multilayer_sweep_{SOLVER}")
    paths = case.output_paths(SOLVER)
    reference = case.build_reference_grid(quick=args.quick, stride=args.stride)

    runner = grax.BatchSimulationRunner(
        diffraction_order=case.DIFFRACTION_ORDER,
        fourier_orders=case.QUICK_FOURIER_ORDERS if args.quick else case.FOURIER_ORDERS,
        show_progress=True,
        live_plot=args.live_plot,
        live_plot_x_key="energy_ev",
        live_plot_order_count=2,
        live_plot_reference_data=reference[["Energy", "Efficiency(GR)"]].to_numpy(dtype=float),
        on_error="fail_fast",
        max_workers="auto",
        checkpoint_dir=paths["checkpoint_dir"],
        resume=False,
        backend="numba",
        solver=SOLVER,
    )

    results = list(runner.run_cases(case.build_cases(quick=args.quick, stride=args.stride)))
    grax.write_all_orders_csv(results, paths["all_orders_csv"])
    case.build_grating(quick=args.quick).plot_profile(paths["profile_plot"])
    case.build_multilayer_stack().plot_schematic(paths["stack_plot"])

    successful = [result for result in results if result.status == "ok"]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        [result.energy_ev for result in successful],
        [result.selected_efficiency for result in successful],
        "o-",
        linewidth=1.0,
        markersize=2.0,
        label=f"grax ({SOLVER})",
    )
    axis.plot(
        reference["Energy"],
        reference["Efficiency(GR)"],
        "s-",
        linewidth=1.0,
        markersize=2.0,
        label="DiffraMod",
    )
    axis.set_xlabel("Energy (eV)")
    axis.set_ylabel("Efficiency (2nd order)")
    axis.set_title(f"Blazed Multilayer Energy-Angle Sweep, {SOLVER_TITLE}: 2nd Order")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(paths["selected_order_plot"], dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Computed {len(successful)} energy-angle points with the {SOLVER} solver.")
    print(f"All-orders CSV saved to: {paths['all_orders_csv']}")
    print(f"Selected-order plot saved to: {paths['selected_order_plot']}")
    print(f"Profile plot saved to: {paths['profile_plot']}")
    print(f"Stack schematic saved to: {paths['stack_plot']}")


# Spawned multiprocessing workers re-import this file by path; without the guard
# each worker would re-run the whole sweep and recursively spawn more.
if __name__ == "__main__":
    main()
