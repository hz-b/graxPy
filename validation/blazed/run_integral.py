"""Run the blazed 600 l/mm monochromator sweep with the boundary-integral solver.

Writes ``results/blazed_comparison_monochromator_orders_1_3_integral.csv`` in the
same layout as ``run_rcwa.py`` and ``run_neviere.py``, so
``comparison_blazed_monochromator_sweep.py`` picks it up and overlays it against
the other solvers and the external reference codes. The grating and the sweep
grid come from ``grating_definition.py``, which all three runners share, so the
geometry is identical by construction.

Unlike the other two, this one does not go through ``grax.BatchSimulationRunner``:
the integral solver is not reachable from ``grax.run_simulation`` yet, so the
sweep is driven here and ``res2_im`` is called directly. Only the integral method
is computed -- run ``run_rcwa.py`` separately for the curve to compare against.

Why this case. It is the cheapest real validation geometry for this solver: two
interfaces rather than three, a three-point profile with two corners that is
still a single-valued graph, and the lowest period-to-wavelength ratio in the
suite -- 67 at 50 eV, against 538 for the 150 l/mm case. Cost grows with both the
interface count and ``d / lambda``, so the low-energy end is much cheaper than
the high-energy end. Start there.

```bash
python validation/blazed/run_integral.py --live-plot
python validation/blazed/run_integral.py --max-energy 400 --boundary-points 256
python validation/blazed/run_integral.py --quick
```
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Blazed 600 l/mm monochromator sweep (boundary integral)"
)
parser.add_argument("--quick", action="store_true", help="Run a few coarse energy points")
parser.add_argument("--stride", type=int, default=2, help="Keep every Nth energy point")
parser.add_argument(
    "--min-energy", type=float, default=50.0, help="First photon energy in eV (default: 50)"
)
parser.add_argument(
    "--max-energy",
    type=float,
    default=200.0,
    help="Last photon energy in eV (default: 200). Cost grows with energy, because "
    "d/lambda does; widen once you know what a point costs.",
)
parser.add_argument(
    "--boundary-points",
    type=int,
    default=192,
    help="Collocation nodes per interface (default: 192)",
)
parser.add_argument("--live-plot", action="store_true", help="Show the sweep while it runs")
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "TkAgg" if args.live_plot else "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grax  # noqa: E402
from grax.solvers.integral import IntegralOptions, res2_im  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grating_definition as case  # noqa: E402

SOLVER = "integral"
SOLVER_TITLE = "Boundary Integral"


def solve_point(grating, *, energy_ev: float, grazing_angle_deg: float, nodes: int):
    """Solve one sweep point with the boundary-integral method.

    Args:
        grating: The blazed grating.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        nodes: Collocation nodes per interface.

    Returns:
        Triple of the signed order array, the reflected efficiencies, and the
        diffraction angles in degrees, matching what the batch runner records.
    """

    wavelength_nm = 1239.8 / float(energy_ev)
    reach = case.QUICK_FOURIER_ORDERS if args.quick else case.FOURIER_ORDERS
    result = res2_im(
        grating=grating,
        wavelength_nm=wavelength_nm,
        period_nm=grating.period_nm,
        orders=np.arange(-reach, reach + 1, dtype=float),
        beta0=float(np.sin(np.deg2rad(90.0 - grazing_angle_deg))),
        polarization=-1 if case.POLARIZATION == "p" else 1,
        photon_energy_ev=float(energy_ev),
        options=IntegralOptions(boundary_points=nodes, discretization="nystrom"),
    )
    reflected = result.inc_top_reflected
    return (
        np.asarray(reflected.order, dtype=float),
        np.real(np.asarray(reflected.efficiency, dtype=complex)),
        90.0 - np.asarray(reflected.theta, dtype=float),
    )


def efficiency_for_order(orders: np.ndarray, efficiency: np.ndarray, order: int) -> float:
    """Return one physical diffraction order's efficiency.

    graxPy indexes reflected orders negatively, so physical order ``m`` is the
    entry at ``-m``; this mirrors ``grax.run_simulation``.

    Args:
        orders: Signed order array.
        efficiency: Efficiencies aligned with ``orders``.
        order: Positive physical diffraction order.

    Returns:
        Reflected efficiency, or ``nan`` when the order is outside the range.
    """

    index = np.nonzero(np.isclose(orders, -float(order)))[0]
    if index.size != 1:
        return float("nan")
    return float(efficiency[int(index[0])])


def main() -> None:
    """Run the sweep and write this solver's artifacts."""

    paths = case.output_paths(SOLVER)
    grating = case.build_grating(quick=args.quick)

    energies = case.build_energies_ev(quick=args.quick, stride=args.stride)
    energies = energies[(energies >= args.min_energy) & (energies <= args.max_energy)]
    if energies.size == 0:
        print("No energies selected; widen --min-energy/--max-energy.")
        return

    angles = {
        float(entry["energy_ev"]): float(entry["grazing_angle_deg"])
        for entry in grax.monochromator_cases(
            grating=grating,
            energies_ev=energies,
            period_lpermm=case.PERIOD_LPERMM,
            diffraction_order=case.DIFFRACTION_ORDER,
            cff=case.CFF,
            polarization=case.POLARIZATION,
        )
    }

    grating.plot_profile(paths["profile_plot"])

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title(
        f"Blazed Grating Monochromator Sweep ({case.PERIOD_LPERMM} l/mm, "
        f"BA={case.BLAZE_ANGLE_DEG} deg), {SOLVER_TITLE}: Orders 1-3"
    )
    axis.grid(True, alpha=0.3)
    if args.live_plot:
        plt.ion()
        plt.show(block=False)

    print(
        f"blazed {case.PERIOD_LPERMM} l/mm, d={grating.period_nm:.0f} nm, "
        f"Si + {case.LAYER_THICKNESS_NM:.0f} nm Au (2 interfaces), "
        f"{case.POLARIZATION} polarization, cff={case.CFF}"
    )
    print(
        f"{energies.size} points from {energies[0]:.0f} to {energies[-1]:.0f} eV, "
        f"{args.boundary_points} nodes per interface"
    )
    header = " ".join(f"{'order ' + str(m):>11}" for m in case.PLOT_ORDERS)
    print(f"{'E (eV)':>8} {'d/lam':>7} {'graz':>7} " + header + f"{'secs':>9}")
    print("-" * 78)

    rows: list[dict[str, object]] = []
    tracks: dict[int, list[tuple[float, float]]] = {m: [] for m in case.PLOT_ORDERS}
    total_seconds = 0.0

    for index, energy in enumerate(energies):
        grazing = angles[float(energy)]
        started = time.perf_counter()
        try:
            orders, efficiency, angles_deg = solve_point(
                grating,
                energy_ev=float(energy),
                grazing_angle_deg=grazing,
                nodes=args.boundary_points,
            )
        except Exception as error:  # noqa: BLE001 - a sweep reports and continues
            print(f"{energy:>8.0f}  failed: {type(error).__name__}: {str(error)[:60]}")
            continue
        seconds = time.perf_counter() - started
        total_seconds += seconds

        for order, value, angle in zip(orders, efficiency, angles_deg, strict=True):
            rows.append(
                {
                    "case_id": f"mono-{index:08d}",
                    "energy_ev": float(energy),
                    "grazing_angle_deg": grazing,
                    "order": int(order),
                    "efficiency": float(value),
                    "diffraction_angle_deg": float(angle),
                }
            )

        selected = [efficiency_for_order(orders, efficiency, m) for m in case.PLOT_ORDERS]
        for order, value in zip(case.PLOT_ORDERS, selected, strict=True):
            tracks[order].append((float(energy), value))
        print(
            f"{energy:>8.0f} {grating.period_nm / (1239.8 / float(energy)):>7.0f} "
            f"{grazing:>7.3f} " + " ".join(f"{v:>11.6f}" for v in selected)
            + f"{seconds:>8.1f}s",
            flush=True,
        )

        axis.clear()
        axis.set_xlabel("Photon Energy (eV)")
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title(
            f"Blazed Grating Monochromator Sweep ({case.PERIOD_LPERMM} l/mm, "
            f"BA={case.BLAZE_ANGLE_DEG} deg), {SOLVER_TITLE}: Orders 1-3"
        )
        axis.grid(True, alpha=0.3)
        for order in case.PLOT_ORDERS:
            if not tracks[order]:
                continue
            x = [point[0] for point in tracks[order]]
            y = [point[1] for point in tracks[order]]
            axis.plot(x, y, "-o", ms=4, lw=1.2, label=f"order {order}")
        axis.legend(loc="best")
        figure.tight_layout()
        if args.live_plot:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.01)

    if not rows:
        print("No points completed.")
        return

    with paths["all_orders_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure.savefig(paths["orders_plot"], dpi=150, bbox_inches="tight")

    points = len(tracks[case.PLOT_ORDERS[0]])
    print(f"\nComputed {points} monochromator points with the {SOLVER} solver.")
    print(f"{total_seconds:.1f}s total, {total_seconds / max(points, 1):.1f}s per point")
    print(f"All-orders CSV saved to: {paths['all_orders_csv']}")
    print(f"Orders plot saved to: {paths['orders_plot']}")
    print(f"Grating profile saved to: {paths['profile_plot']}")
    print(
        "\nRun `python validation/blazed/run_rcwa.py` and then "
        "`python validation/blazed/comparison_blazed_monochromator_sweep.py` "
        "to overlay this against the other solvers and the reference codes."
    )
    if args.live_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
