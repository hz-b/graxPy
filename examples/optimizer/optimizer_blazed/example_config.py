"""Shared baseline configuration for the blazed optimizer example workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
measurement_path = example_root / "GR600-BEIChem_energy-Cff2.5.dat"
results_dir = example_root / "results" / "blazed_fit"

period_lpermm = 600.0
blaze_angle_deg = 0.729
anti_blaze_angle_deg = 5.597
layer_thickness_nm = 30.0
top_cap_thickness_nm = 0.7
use_top_cap = True
top_cap_material_name = "C"
x_resolution_nm = 0.5
z_resolution_nm = 0.5

cff = 2.25
diffraction_order = 1
fourier_orders = 10
optimizer_backend = "auto"
simulation_backend = "numba"

total_trials = 50
batch_size = 15
random_seed = 7
energy1 = np.arange(51, 201, 10)      # 51 to 200 inclusive, step 10
energy2 = np.arange(200, 1851, 50)    # 200 to 1900 inclusive, step 50
evaluation_energies_ev = np.concatenate((energy1, energy2))
