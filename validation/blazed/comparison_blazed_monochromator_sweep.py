#!/usr/bin/env python3
"""Compare efficiency from grax, REFLEC, and DiffMod."""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solver_comparison import load_grax_curves  # noqa: E402

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
grax_curves = load_grax_curves(
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
for curve in grax_curves:
    plt.plot(curve.energy_ev, curve.efficiency, marker='o', label=curve.label,
             markersize=marker_size, linewidth=linewidth, **curve.style)
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
