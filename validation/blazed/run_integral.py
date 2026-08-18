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

How many nodes, and what that costs
-----------------------------------
The boundary densities carry ``exp(i alpha_0 x)``, which at grazing incidence
oscillates ``(d / lambda) cos(theta_g)`` times across one period -- 67 at 50 eV
and 282 at 210 eV on this grating. The scheme has to resolve that, and the
threshold is sharp. Measured against RCWA at 50 eV, on order zero:

    nodes   per oscillation   deviation
    128     1.9               3.1e-1
    192     2.9               1.6e-2
    256     3.8               5.5e-4
    384     5.7               4.3e-4   <- RCWA's own truncation floor

Below about three nodes per oscillation the answer is not merely inaccurate, it
is unrelated to the right one and does not improve monotonically with the node
count, so a two-point convergence check inside that regime reports nonsense.
``"auto"`` asks for six, which is why it is the default here.

That fixes the reachable range. Assembly is quadratic in the node count and
linear in the Ewald spectral reach, which itself grows with ``d / lambda``, so
cost goes as roughly ``(d / lambda)^3``: seconds per point at 50 eV, minutes by
150 eV, and out of reach well before the 1500 eV end of the sweep. Use
``--max-energy`` deliberately rather than letting it run.

Corner grading is off by default here, against the ``IntegralOptions`` default of
2.0. Grading clusters nodes at the profile's corners, where the density is
singular; on a 0.729 degree blaze the corners are nearly flat and the singularity
is negligible, while the clustering doubles the node spacing mid-facet -- exactly
where the carrier lives. Measured, it costs a factor of three in node count for
the same accuracy. Raise it for steep profiles.

```bash
python validation/blazed/run_integral.py --live-plot --max-energy 70
python validation/blazed/run_integral.py --max-energy 150 --boundary-points 512
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
    default=0,
    help="Collocation nodes per interface. 0 (default) sizes them from "
    "d/lambda, which is what sets the requirement; see the module docstring.",
)
parser.add_argument(
    "--corner-grading",
    type=float,
    default=1.0,
    help="Node clustering toward the profile corners. 1.0 (default here) spaces "
    "them uniformly, which is measurably better on this shallow blaze.",
)
parser.add_argument(
    "--energy-balance-tolerance",
    type=float,
    default=1.5,
    help="Refuse to record a point whose propagating efficiency sums above this. "
    "An under-resolved solve on this geometry overshoots by orders of magnitude, "
    "so this is what stops nonsense reaching the CSV.",
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
        options=IntegralOptions(
            boundary_points=nodes if nodes > 0 else "auto",
            discretization="nystrom",
            corner_grading=float(args.corner_grading),
            energy_balance_tolerance=float(args.energy_balance_tolerance),
        ),
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


def write_artifacts(rows: list[dict[str, object]], figure, paths) -> None:
    """Rewrite the CSV and the plot from the points completed so far.

    Called after every point rather than once at the end, because a point at the
    top of the reachable range costs minutes and the whole sweep can be several
    times longer than anyone wants to sit in front of. Interrupting the run then
    keeps everything already computed instead of discarding it.

    Rewriting the file whole each time, rather than appending, keeps the header
    and the row set consistent if the run is killed mid-write: the cost is
    negligible next to one solve.

    Args:
        rows: All rows recorded so far.
        figure: The sweep figure, already redrawn for this point.
        paths: Output paths from ``grating_definition.output_paths``.
    """

    if not rows:
        return
    with paths["all_orders_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure.savefig(paths["orders_plot"], dpi=150, bbox_inches="tight")


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
    sizing = (
        f"{args.boundary_points} nodes per interface"
        if args.boundary_points > 0
        else "node count sized from d/lambda"
    )
    print(
        f"{energies.size} points from {energies[0]:.0f} to {energies[-1]:.0f} eV, "
        f"{sizing}, corner grading {args.corner_grading}"
    )
    header = " ".join(f"{'order ' + str(m):>11}" for m in case.PLOT_ORDERS)
    print(
        f"{'E (eV)':>8} {'d/lam':>7} {'graz':>7} {'nodes':>6} {'n/osc':>6} "
        + header
        + f"{'secs':>9}"
    )
    print("-" * 92)

    rows: list[dict[str, object]] = []
    tracks: dict[int, list[tuple[float, float]]] = {m: [] for m in case.PLOT_ORDERS}
    total_seconds = 0.0

    # Ctrl-C is the expected way to end a long sweep: every point is already
    # on disk by the time it lands, so it stops the run rather than propagating.
    try:
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
                print(
                    f"{energy:>8.0f}  failed: {type(error).__name__}: {str(error)[:90]}",
                    flush=True,
                )
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
            ratio = grating.period_nm / (1239.8 / float(energy))
            node_count = IntegralOptions(
                boundary_points=args.boundary_points if args.boundary_points > 0 else "auto"
            ).resolved_boundary_points(
                period_nm=float(grating.period_nm),
                wavelength_nm=1239.8 / float(energy),
                orders=int(case.QUICK_FOURIER_ORDERS if args.quick else case.FOURIER_ORDERS),
            )
            oscillations = ratio * float(np.cos(np.deg2rad(grazing)))
            print(
                f"{energy:>8.0f} {ratio:>7.0f} {grazing:>7.3f} {node_count:>6d} "
                f"{node_count / max(oscillations, 1e-12):>6.1f} "
                + " ".join(f"{v:>11.6f}" for v in selected)
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
            write_artifacts(rows, figure, paths)
    except KeyboardInterrupt:
        print("\nInterrupted. The points already computed are saved.", flush=True)

    if not rows:
        print("No points completed.")
        return

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
