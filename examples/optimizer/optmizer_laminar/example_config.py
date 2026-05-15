"""Shared baseline configuration for the laminar optimizer example workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants" / "old"
measurement_path = example_root / "measured_alpha4deg_order1.csv"
results_dir = example_root / "results" / "laminar_fit"

period_lpermm = 400.0
width_to_period_ratio = 0.67
depth_nm = 14.9
left_wall_angle_deg = 15.0
right_wall_angle_deg = 15.0
layer_thickness_nm = 28.77
top_cap_thickness_nm = 0.3
x_resolution_nm = 0.5
z_resolution_nm = 0.5

angle_mode = "fixed"
grazing_angle_deg = 4.0
cff = 2.5
diffraction_order = 1
fourier_orders = 15
optimizer_backend = "auto"
simulation_backend = "numba"

total_trials = 20
batch_size = 15
random_seed = 7
evaluation_energies_ev = np.arange(100.0, 501.0, 10.0)
