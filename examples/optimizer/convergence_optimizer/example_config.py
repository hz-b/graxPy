"""Shared baseline configuration for the laminar convergence example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants" / "old"
results_dir = example_root / "results" / "convergence_study"

period_lpermm = 400.0
width_to_period_ratio = 0.67
depth_nm = 14.9
left_wall_angle_deg = 15.0
right_wall_angle_deg = 15.0
layer_thickness_nm = 28.77
top_cap_thickness_nm = 0.3
x_resolution_nm = 1.0
z_resolution_nm = 1.0

grazing_angle_deg = 4.0
diffraction_order = 1
backend = "auto"
validate_physical_results = True

fourier_orders_values = np.array([5, 7, 9, 11, 13], dtype=int)
x_resolution_values = np.array([10.0, 5.0, 2.0, 1.0, 0.5], dtype=float)
z_resolution_values = np.array([10.0, 5.0, 2.0, 1.0, 0.5], dtype=float)
relative_tolerance = 5.0e-3
energies_ev = np.array([100.0, 150.0, 200.0, 300.0, 400.0, 500.0], dtype=float)
