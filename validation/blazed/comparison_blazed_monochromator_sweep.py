#!/usr/bin/env python3
"""Compare efficiency from grax, REFLEC, and DiffMod."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Both solvers' results are plotted when present. A fresh run writes
# ``*_rcwa.csv`` / ``*_neviere.csv``; the unsuffixed CSV is the older checked-in
# artifact and is only used as a fallback, because it predates several solver
# changes and pairing it against a fresh run would show that drift as though it
# were a difference between the two methods.
SOLVER_LABELS = {"rcwa": "graxpy (RCWA)", "neviere": "graxpy (Nevière DM)"}
# The two solvers agree to ~1e-11, so without a dashed overlay the second curve
# hides the first completely and the plot looks as though one is missing.
SOLVER_STYLES = {"rcwa": {"linestyle": "-"}, "neviere": {"linestyle": (0, (6, 4))}}


def load_solver_curves(base_csv, order):
    """Return (label, energy, efficiency, style) for each solver that has been run.

    Args:
        base_csv: Historical unsuffixed all-orders CSV path for this case.
        order: Signed diffraction order to extract (reflected orders are negative).

    Returns:
        List of plottable curves, skipping solvers with no results yet.
    """

    curves = []
    for solver in ("rcwa", "neviere"):
        candidates = [base_csv.with_name(f"{base_csv.stem}_{solver}{base_csv.suffix}")]
        if solver == "rcwa":
            candidates.append(base_csv)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            print(f"  note: no {solver} results yet, skipping that curve")
            continue
        frame = pd.read_csv(path)
        selected = frame[frame["order"] == order].sort_values("energy_ev")
        if selected.empty:
            print(f"  note: {path.name} has no order {order}, skipping that curve")
            continue
        print(f"  {SOLVER_LABELS[solver]:<22} <- {path.name}  ({len(selected)} points)")
        curves.append(
            (
                SOLVER_LABELS[solver],
                selected["energy_ev"],
                selected["efficiency"],
                dict(SOLVER_STYLES[solver]),
            )
        )
    return curves

base_path = Path(__file__).resolve().parent


# Load measured data.
# The measurement file contains "--" placeholders for missing values.
meas = pd.read_csv(
    base_path / "measurements" / "GR600-BEIChem_energy-Cff2.5.dat",
    sep=r"\s+",
    header=None,
    names=["energy", "eff"],
    na_values="--",
)
meas = meas.apply(pd.to_numeric, errors="coerce").dropna()
meas_energy = meas["energy"]
meas_eff = meas["eff"]

# Load grax results, one curve per solver that has been run
print("graxpy curves:")
grax_curves = load_solver_curves(
    base_path / "results" / "blazed_comparison_monochromator_orders_1_3.csv",
    order=-1,
)

# reticolo Matlab
ret_mat = pd.read_csv(base_path / "simulations" / "reticolo_matlab.csv")
ret_mat_energy = ret_mat['PhotonEnergy_eV']
ret_mat_eff = ret_mat['DiffractionEfficiency']

# Load REFLEC results (DAT file: energy efficiency)
reflec = pd.read_csv(
    base_path / "simulations" / "REFLEC.dat",
    sep=r'\s+',
    header=None,
    names=['energy', 'eff'],
)
reflec_energy = reflec['energy']
reflec_eff = reflec['eff']

# Load DiffMod results (DAT file: energy efficiency)
diffmod = pd.read_csv(
    base_path / "simulations" / "DiffractMod.dat",
    sep=r'\s+',
    header=None,
    names=['energy', 'eff'],
)
diffmod_energy = diffmod['energy']
diffmod_eff = diffmod['eff']

# Plot
plt.figure(figsize=(10, 6))
marker_size = 1
linewidth = 1
for curve_label, curve_energy, curve_efficiency, curve_style in grax_curves:
    plt.plot(curve_energy, curve_efficiency, marker='o', label=curve_label,
             markersize=marker_size, linewidth=linewidth, **curve_style)
plt.plot(ret_mat_energy, ret_mat_eff, 'd-', label='reticolo (Matlab)', markersize=marker_size, linewidth=linewidth)
plt.plot(reflec_energy, reflec_eff, 's-', label='REFLEC', markersize=marker_size, linewidth=linewidth)
plt.plot(diffmod_energy, diffmod_eff, '^-', label='DiffMod', markersize=marker_size, linewidth=linewidth)
plt.plot(meas_energy, meas_eff, 'v-', label='Measured', markersize=marker_size, linewidth=linewidth)
plt.xlabel('Energy (eV)')
plt.ylabel('Efficiency (-1 order)')
plt.title('Comparison To Other Codes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_path = base_path / "comparison_blazed_monochromator_sweep.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Plot saved to: {output_path}')
