"""Probe the boundary-integral solver on the blazed 600 l/mm case, with live plot.

This is *not* the standard runner. ``run_rcwa.py`` and ``run_neviere.py`` go
through ``grax.BatchSimulationRunner``; the integral solver is not wired into
``grax.run_simulation`` yet, so this script drives it directly and compares each
point against RCWA computed on the same geometry.

Why this case. It is the cheapest real validation geometry for the integral
method: two interfaces rather than three, a three-point profile with two corners
that is still a single-valued graph, no roughness, and the lowest reachable
period-to-wavelength ratio in the suite (67 at 50 eV, against 538 for the
150 l/mm case). Cost grows with both the interface count and ``d / lambda``, so
everything else is worse.

Runtime is the thing being measured, so start small. The defaults sweep 50 to
200 eV and print seconds per point as it goes; widen with ``--max-energy`` once
you know what a point costs. The live plot shows the two solvers' order-1
efficiency and their difference, updated after every energy.

```bash
python validation/blazed/run_integral_probe.py
python validation/blazed/run_integral_probe.py --max-energy 400 --boundary-points 256
python validation/blazed/run_integral_probe.py --convergence     # two N per point
python validation/blazed/run_integral_probe.py --no-live-plot
```
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--min-energy", type=float, default=50.0, help="First photon energy in eV (default: 50)."
)
parser.add_argument(
    "--max-energy",
    type=float,
    default=200.0,
    help="Last photon energy in eV (default: 200). Cost grows with energy, because "
    "d/lambda does.",
)
parser.add_argument(
    "--stride", type=int, default=2, help="Keep every Nth energy of the 10 eV grid (default: 2)."
)
parser.add_argument(
    "--boundary-points",
    type=int,
    default=192,
    help="Collocation nodes per interface (default: 192).",
)
parser.add_argument(
    "--convergence",
    action="store_true",
    help="Also solve at half the node count, to show whether the answer has converged. "
    "Roughly doubles the runtime but is the only way to know the deviation is the "
    "solver's and not the discretization's.",
)
parser.add_argument("--no-live-plot", action="store_true", help="Disable the live plot.")
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "Agg" if args.no_live_plot else "TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grax  # noqa: E402
from grax.solvers.integral import IntegralOptions, res2_im  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grating_definition as case  # noqa: E402

RESULTS_DIR = case.CASE_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "blazed_integral_probe.csv"
OUTPUT_PLOT = RESULTS_DIR / "blazed_integral_probe.png"


def efficiency_for_order(result, order: int) -> float:
    """Return one physical diffraction order's reflected efficiency.

    graxPy indexes reflected orders negatively, so physical order ``m`` is the
    entry at ``-m``. This mirrors what ``grax.run_simulation`` does internally.

    Args:
        result: A ``DiffractionResult``.
        order: Positive physical diffraction order.

    Returns:
        Reflected efficiency.
    """

    orders = np.asarray(result.order, dtype=float)
    index = np.nonzero(np.isclose(orders, -float(order)))[0]
    if index.size != 1:
        raise ValueError(f"Order {order} is not in the computed range.")
    return float(np.real(result.efficiency[int(index[0])]))


def solve_integral(grating, *, energy_ev: float, grazing_angle_deg: float, nodes: int):
    """Run the boundary-integral solver at one sweep point.

    Args:
        grating: The blazed grating.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        nodes: Collocation nodes per interface.

    Returns:
        Pair of the order-1 efficiency and the wall-clock seconds.
    """

    wavelength_nm = 1239.8 / float(energy_ev)
    started = time.perf_counter()
    result = res2_im(
        grating=grating,
        wavelength_nm=wavelength_nm,
        period_nm=grating.period_nm,
        orders=np.arange(-3, 4, dtype=float),
        beta0=float(np.sin(np.deg2rad(90.0 - grazing_angle_deg))),
        polarization=-1 if case.POLARIZATION == "p" else 1,
        photon_energy_ev=float(energy_ev),
        options=IntegralOptions(boundary_points=nodes, discretization="nystrom"),
    )
    return efficiency_for_order(result.inc_top_reflected, 1), time.perf_counter() - started


def build_figure():
    """Return the live figure and its two axes."""

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, height_ratios=(2, 1)
    )
    upper.set_ylabel("order-1 efficiency")
    upper.set_title("Blazed 600 l/mm, cff 2.25, TM: boundary integral vs RCWA")
    upper.grid(True, alpha=0.3)
    lower.set_xlabel("photon energy (eV)")
    lower.set_ylabel("|integral - rcwa|")
    lower.set_yscale("log")
    lower.grid(True, alpha=0.3)
    return figure, upper, lower


def main() -> int:
    """Run the probe and return a process exit code."""

    grating = case.build_grating()
    energies = case.build_energies_ev(stride=args.stride)
    energies = energies[(energies >= args.min_energy) & (energies <= args.max_energy)]
    if energies.size == 0:
        print("No energies selected; widen --min-energy/--max-energy.")
        return 2

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    live = not args.no_live_plot
    figure, upper, lower = build_figure()
    if live:
        plt.ion()
        plt.show(block=False)

    print(
        f"blazed 600 l/mm, d={grating.period_nm:.0f} nm, Si + {case.LAYER_THICKNESS_NM:.0f} nm Au "
        f"(2 interfaces), TM, cff={case.CFF}"
    )
    print(f"{len(energies)} points from {energies[0]:.0f} to {energies[-1]:.0f} eV, "
          f"{args.boundary_points} nodes per interface")
    print(f"{'E (eV)':>8} {'d/lam':>7} {'graz':>7} {'rcwa':>10} {'integral':>10} "
          f"{'|diff|':>10} {'conv':>9} {'secs':>8}")
    print("-" * 82)

    rows: list[dict[str, object]] = []
    for energy in energies:
        grazing = angles[float(energy)]
        reference = grax.run_simulation(
            grating=grating,
            energy_ev=float(energy),
            grazing_angle_deg=grazing,
            diffraction_order=case.DIFFRACTION_ORDER,
            fourier_orders=case.FOURIER_ORDERS,
            polarization=case.POLARIZATION,
            solver="rcwa",
        )
        try:
            value, seconds = solve_integral(
                grating,
                energy_ev=float(energy),
                grazing_angle_deg=grazing,
                nodes=args.boundary_points,
            )
        except Exception as error:  # noqa: BLE001 - a probe reports and continues
            print(f"{energy:>8.0f}  failed: {type(error).__name__}: {str(error)[:60]}")
            continue

        coarse_gap = float("nan")
        if args.convergence:
            coarse, _ = solve_integral(
                grating,
                energy_ev=float(energy),
                grazing_angle_deg=grazing,
                nodes=max(16, args.boundary_points // 2),
            )
            coarse_gap = abs(value - coarse)

        difference = abs(value - reference.selected_efficiency)
        ratio = grating.period_nm / (1239.8 / float(energy))
        print(
            f"{energy:>8.0f} {ratio:>7.0f} {grazing:>7.3f} "
            f"{reference.selected_efficiency:>10.6f} {value:>10.6f} {difference:>10.2e} "
            f"{coarse_gap:>9.2e} {seconds:>7.1f}s",
            flush=True,
        )
        rows.append(
            {
                "energy_ev": float(energy),
                "period_over_wavelength": ratio,
                "grazing_angle_deg": grazing,
                "rcwa": reference.selected_efficiency,
                "integral": value,
                "difference": difference,
                "half_node_gap": coarse_gap,
                "seconds": seconds,
            }
        )

        upper.clear()
        lower.clear()
        upper.set_ylabel("order-1 efficiency")
        upper.set_title("Blazed 600 l/mm, cff 2.25, TM: boundary integral vs RCWA")
        upper.grid(True, alpha=0.3)
        lower.set_xlabel("photon energy (eV)")
        lower.set_ylabel("|integral - rcwa|")
        lower.set_yscale("log")
        lower.grid(True, alpha=0.3)

        x = [row["energy_ev"] for row in rows]
        upper.plot(x, [row["rcwa"] for row in rows], "b-o", ms=4, lw=1.2, label="RCWA")
        upper.plot(
            x,
            [row["integral"] for row in rows],
            "r--s",
            ms=4,
            lw=1.2,
            label=f"integral (N={args.boundary_points})",
        )
        upper.legend(loc="best")
        lower.plot(x, [row["difference"] for row in rows], "k-o", ms=4, lw=1.2, label="vs RCWA")
        if args.convergence:
            lower.plot(
                x,
                [row["half_node_gap"] for row in rows],
                "g-^",
                ms=4,
                lw=1.0,
                label="vs half the nodes",
            )
            lower.legend(loc="best")
        figure.tight_layout()
        if live:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.01)

    if rows:
        with OUTPUT_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        figure.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        total = sum(float(row["seconds"]) for row in rows)
        worst = max(float(row["difference"]) for row in rows)
        print(f"\n{len(rows)} points, {total:.1f}s total, {total / len(rows):.1f}s per point")
        print(f"worst |integral - rcwa| = {worst:.3e}")
        print(f"CSV  {OUTPUT_CSV}")
        print(f"plot {OUTPUT_PLOT}")
    if live:
        plt.ioff()
        plt.show()
    return 0


# Nothing here spawns workers, but the guard keeps the module importable.
if __name__ == "__main__":
    raise SystemExit(main())
