#!/usr/bin/env python3
"""Compare efficiency from grax, REFLEC, and DiffMod."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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

# Load grax results (current)
ret = pd.read_csv(base_path / "results" / "blazed_comparison_monochromator_orders_1_3.csv")
ret_m1 = ret[ret["order"] == -1].sort_values("energy_ev")
ret_energy = ret_m1["energy_ev"]
ret_eff = ret_m1["efficiency"]

# reticolo Matlab
ret_mat = pd.read_csv(base_path / "simulations" / "reticolo_matlab.csv")
ret_mat_energy = ret_mat['PhotonEnergy_eV']
ret_mat_eff = ret_mat['DiffractionEfficiency']

# Load REFLEC results (DAT file: energy efficiency)
reflec = pd.read_csv(
    base_path / "simulations" / "GT-1.dat",
    sep=r'\s+',
    header=None,
    names=['energy', 'eff'],
)
reflec_energy = reflec['energy']
reflec_eff = reflec['eff']

# Load DiffMod results (DAT file: energy efficiency)
diffmod = pd.read_csv(
    base_path / "simulations" / "RR-test-bGR600-BElChem_1order_dm_de_r_m1_single.dat",
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
plt.plot(ret_energy, ret_eff, 'o-', label='grax', markersize=marker_size, linewidth=linewidth)
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
