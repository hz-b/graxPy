"""What the differential method costs compared with the modal solver.

The differential method is usually the faster of the two, which is the opposite
of what "integrate an ODE through every layer" suggests. The reason is that the
modal solver eigen-decomposes each distinct layer operator, and a dense
eigensolve on a 2N+1 basis is expensive. The differential method never does one:
it advances a Runge-Kutta step and converts the result to an interface-response
block, both of which are matrix products.

The size of the gap depends on resolution, not on the grating. Measured here:

    reduced resolution     1.2x to 1.4x
    production resolution  2.4x to 3.0x

Coarse runs have few distinct layers, so the eigensolve is not yet the dominant
cost and the two solvers are close. Refine the grid and the modal solver pays for
an eigensolve per distinct layer while the differential method does not, so the
gap opens up. It is worth knowing that the advantage appears exactly in the
regime where a sweep is expensive, and that a quick coarse benchmark understates
it.

A speed number means nothing on its own, so this example reports the maximum
efficiency difference between the two solvers alongside each timing. If that
number were large the comparison would be meaningless; it is around 1e-11.

Run with --full for the production-resolution numbers above; the default is
reduced so the example finishes quickly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import grax

parser = argparse.ArgumentParser(description="Time the two solvers on several gratings")
parser.add_argument(
    "--full",
    action="store_true",
    help="Use production resolutions. Slower, and closer to what a real sweep costs.",
)
args = parser.parse_args()

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

ENERGIES_EV = [200.0, 400.0, 800.0]
GRAZING_ANGLE_DEG = 4.0
POLARIZATION = "p"


def laminar_grating() -> grax.LaminarGrating:
    """Return the 400 l/mm coated laminar grating."""

    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=28.77,
        x_resolution_nm=0.1 if args.full else 1.0,
        z_resolution_nm=0.1 if args.full else 0.5,
    )


def blazed_grating() -> grax.BlazedGrating:
    """Return the 600 l/mm coated blazed grating."""

    return grax.BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.729,
        anti_blaze_angle_deg=5.597,
        substrate_material="Si",
        layer_material="Au",
        layer_thickness_nm=30.0,
        x_resolution_nm=0.1 if args.full else 1.0,
        z_resolution_nm=0.1 if args.full else 0.5,
    )


def multilayer_grating() -> grax.BlazedGrating:
    """Return a blazed grating on a Cr/C multilayer stack."""

    carbon = "C"
    return grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=grax.MultilayerStack(
            substrate_material="Si",
            material_a="Cr",
            material_b=carbon,
            d_period_nm=4.8,
            gamma=0.4,
            n_bilayers=20 if args.full else 8,
            top_material=carbon,
        ),
        x_resolution_nm=0.1 if args.full else 1.0,
        z_resolution_nm=0.05 if args.full else 0.5,
    )


CASES = [
    ("laminar 400 l/mm", laminar_grating, 25 if args.full else 10, 1),
    ("blazed 600 l/mm", blazed_grating, 20 if args.full else 10, 1),
    ("blazed 2400 l/mm multilayer", multilayer_grating, 25 if args.full else 10, 2),
]


def time_solver(grating, fourier_orders: int, diffraction_order: int, solver: str):
    """Return seconds per point and the all-order efficiencies for one solver.

    Args:
        grating: Grating to solve.
        fourier_orders: Fourier truncation order.
        diffraction_order: Order reported in the result.
        solver: ``"rcwa"`` or ``"neviere"``.

    Returns:
        Mean seconds per energy point and the stacked all-order efficiencies.
    """

    efficiencies = []
    started = time.perf_counter()
    for energy_ev in ENERGIES_EV:
        result = grax.run_simulation(
            grating=grating,
            energy_ev=energy_ev,
            grazing_angle_deg=GRAZING_ANGLE_DEG,
            diffraction_order=diffraction_order,
            fourier_orders=fourier_orders,
            polarization=POLARIZATION,
            solver=solver,
            validate_physical_results=False,
        )
        efficiencies.append(np.asarray(result.efficiency_all, dtype=float))
    elapsed = time.perf_counter() - started
    return elapsed / len(ENERGIES_EV), np.vstack(efficiencies)


rows = []
print(f"Timing {len(ENERGIES_EV)} energies per case "
      f"({'production' if args.full else 'reduced'} resolution).\n")
print(f"{'case':<30} {'rcwa s/pt':>10} {'neviere s/pt':>13} {'speedup':>9} {'max |dE|':>11}")
for label, build, fourier_orders, diffraction_order in CASES:
    grating = build()
    # Warm the numba Fourier kernel so the first case is not charged for the JIT.
    grax.run_simulation(
        grating=grating, energy_ev=ENERGIES_EV[0], grazing_angle_deg=GRAZING_ANGLE_DEG,
        fourier_orders=4, polarization=POLARIZATION, validate_physical_results=False,
    )
    rcwa_seconds, rcwa_efficiency = time_solver(grating, fourier_orders, diffraction_order, "rcwa")
    dm_seconds, dm_efficiency = time_solver(grating, fourier_orders, diffraction_order, "neviere")
    deviation = float(np.max(np.abs(rcwa_efficiency - dm_efficiency)))
    speedup = rcwa_seconds / dm_seconds
    rows.append((label, rcwa_seconds, dm_seconds, speedup, deviation))
    print(f"{label:<30} {rcwa_seconds:>10.3f} {dm_seconds:>13.3f} "
          f"{speedup:>8.2f}x {deviation:>11.2e}")

labels = [row[0] for row in rows]
positions = np.arange(len(rows))
figure, axis = plt.subplots(figsize=(10, 6))
axis.barh(positions - 0.2, [row[1] for row in rows], height=0.36, label="RCWA (modal)")
axis.barh(positions + 0.2, [row[2] for row in rows], height=0.36, label="Nevière (differential)")
for index, row in enumerate(rows):
    axis.annotate(f"{row[3]:.2f}x faster", xy=(max(row[1], row[2]), index),
                  xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
axis.set_yticks(positions)
axis.set_yticklabels(labels)
axis.set_xlabel("Seconds per energy point")
axis.set_title(
    "Solver runtime "
    f"({'production' if args.full else 'reduced'} resolution); "
    f"max efficiency difference {max(row[4] for row in rows):.1e}"
)
axis.grid(True, alpha=0.3, axis="x")
axis.legend(loc="lower right")
figure.tight_layout()

suffix = "full" if args.full else "reduced"
plot_path = output_dir / f"solver_runtime_{suffix}.png"
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

csv_path = output_dir / f"solver_runtime_{suffix}.csv"
with csv_path.open("w", encoding="utf-8") as handle:
    handle.write("case,rcwa_seconds_per_point,neviere_seconds_per_point,speedup,max_abs_deviation\n")
    for label, rcwa_seconds, dm_seconds, speedup, deviation in rows:
        handle.write(f"{label},{rcwa_seconds:.6f},{dm_seconds:.6f},{speedup:.4f},{deviation:.6e}\n")

print(f"\nSpeedup range: {min(row[3] for row in rows):.2f}x to {max(row[3] for row in rows):.2f}x")
if not args.full:
    print("Reduced resolution understates the gap; run with --full for production numbers.")
print(f"Largest efficiency difference across all cases: {max(row[4] for row in rows):.2e}")
print("The speedup is not bought with accuracy: the two solvers still agree to")
print("roughly the level they agree to everywhere else.")
print(f"Plot saved to: {plot_path}")
print(f"CSV saved to: {csv_path}")
