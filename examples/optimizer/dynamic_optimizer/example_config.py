"""Shared configuration for the measurement-fit laminar optimizer workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

example_root = Path(__file__).resolve().parent
repo_root = example_root.parents[2]
optical_constants_dir = repo_root / "examples" / "optical_constants"
measurement_path = example_root / "measured_alpha4deg_order1.csv"
results_dir = example_root / "results"

design_period_lpermm = 400.0
design_width_to_period_ratio = 0.67
design_depth_nm = 14.9
design_left_wall_angle_deg = 15.0
design_right_wall_angle_deg = 15.0
design_layer_thickness_nm = 28.77
design_top_cap_thickness_nm = 0.3
design_x_resolution_nm = 0.5
design_z_resolution_nm = 0.5

grazing_angle_deg = 4.0
diffraction_order = 1
fourier_orders = 15

optimizer_backend = "auto"
simulation_backend = "numba"
total_trials = 60
batch_size = 15
random_seed = 7
evaluation_energies_ev = np.arange(100.0, 501.0, 10.0)
evaluation_grazing_angles_deg = []
