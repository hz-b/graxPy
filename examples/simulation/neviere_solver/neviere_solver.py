"""Compare the RCWA and Nevière differential-method solvers on one grating.

Runs the same laminar grating through both solvers over a short energy sweep,
writes the per-order efficiencies for each, and plots them together with the
per-energy difference. Also demonstrates continuous permittivity sampling, which
drops the staircase approximation both solvers otherwise share.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import grax

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    x_resolution_nm=1.0,
    z_resolution_nm=0.2,
)

energies_ev = np.arange(100.0, 600.1, 50.0)
grazing_angle_deg = 4.0
diffraction_order = 1
fourier_orders = 12

# 1. One point through each solver. Only the `solver` argument changes.
single_common = dict(
    grating=grating,
    energy_ev=float(energies_ev[len(energies_ev) // 2]),
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    polarization="p",
)
rcwa_single = grax.run_simulation(**single_common, solver="rcwa")
neviere_single = grax.run_simulation(**single_common, solver="neviere")

print(f"At {single_common['energy_ev']:.0f} eV, order {diffraction_order}:")
print(f"  rcwa     : {rcwa_single.selected_efficiency:.12f}")
print(f"  neviere  : {neviere_single.selected_efficiency:.12f}")
print(
    "  difference: "
    f"{abs(rcwa_single.selected_efficiency - neviere_single.selected_efficiency):.3e}"
)
print(f"  result.solver reports: {rcwa_single.solver!r} and {neviere_single.solver!r}")

# 2. Tightening the integration drives the residual towards zero, which shows the
#    difference is Runge-Kutta truncation error rather than a modelling gap.
tight = grax.run_simulation(
    **single_common,
    solver="neviere",
    solver_options=grax.NeviereOptions(step_phase=0.005),
)
print(
    "  difference with step_phase=0.005: "
    f"{abs(rcwa_single.selected_efficiency - tight.selected_efficiency):.3e}"
)

# 3. Continuous permittivity sampling reads the true profile instead of the
#    z-sliced staircase, so its answer does not depend on z_resolution_nm. It
#    therefore differs from the two runs above, which share that staircase: a
#    much more finely sliced staircase run shows which of them is converged.
continuous = grax.run_simulation(
    **single_common,
    solver="neviere",
    solver_options=grax.NeviereOptions(z_sampling="continuous"),
)
fine_staircase = grax.run_simulation(
    **{**single_common, "grating": dataclasses.replace(grating, z_resolution_nm=0.01)},
    solver="rcwa",
)
print(f"\nStaircase at z_resolution_nm={grating.z_resolution_nm}: "
      f"{rcwa_single.selected_efficiency:.12f}")
print(f"Staircase at z_resolution_nm=0.01     : {fine_staircase.selected_efficiency:.12f}")
print(f"Continuous sampling (no staircase)    : {continuous.selected_efficiency:.12f}")
print(
    "  continuous vs converged staircase: "
    f"{abs(continuous.selected_efficiency - fine_staircase.selected_efficiency):.3e}"
)

# 4. A short sweep through each solver, via the batch runner.
efficiencies: dict[str, np.ndarray] = {}
for solver in ("rcwa", "neviere"):
    runner = grax.BatchSimulationRunner(
        diffraction_order=diffraction_order,
        fourier_orders=fourier_orders,
        solver=solver,
        on_error="fail_fast",
    )
    cases = grax.fixed_angle_cases(
        grating=grating,
        energies_ev=energies_ev,
        grazing_angle_deg=grazing_angle_deg,
        polarization="p",
    )
    results = [case for case in runner.run_cases(cases) if case.status == "ok"]
    efficiencies[solver] = np.asarray(
        [case.selected_efficiency for case in results], dtype=float
    )
    grax.write_all_orders_csv(results, output_dir / f"neviere_solver_{solver}.csv")

figure, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1.0]},
)
efficiency_axis, difference_axis = axes
efficiency_axis.plot(energies_ev, efficiencies["rcwa"], "-o", markersize=3.0, label="RCWA (modal)")
efficiency_axis.plot(
    energies_ev,
    efficiencies["neviere"],
    "--s",
    markersize=3.0,
    label="Nevière (differential)",
)
efficiency_axis.set_ylabel(f"Efficiency (order {diffraction_order})")
efficiency_axis.set_title("Laminar 400 l/mm at 4 deg grazing: RCWA vs Nevière")
efficiency_axis.grid(True, alpha=0.3)
efficiency_axis.legend(loc="best")

difference_axis.semilogy(
    energies_ev,
    np.abs(efficiencies["rcwa"] - efficiencies["neviere"]),
    "-",
    color="tab:red",
)
difference_axis.set_xlabel("Photon energy (eV)")
difference_axis.set_ylabel("|RCWA - Nevière|")
difference_axis.grid(True, alpha=0.3, which="both")

figure.tight_layout()
plot_path = output_dir / "neviere_solver_comparison.png"
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"\nMaximum sweep difference: {np.max(np.abs(efficiencies['rcwa'] - efficiencies['neviere'])):.3e}")
print(f"Comparison plot saved to: {plot_path}")
print(f"Per-solver CSVs saved to: {output_dir}")
