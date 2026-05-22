"""Run fixed-angle simulation using the measurement-fit example design parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

import grax as rp
from grax_opt import load_measurement_data
from example_config import (
    design_depth_nm,
    design_layer_thickness_nm,
    design_left_wall_angle_deg,
    design_period_lpermm,
    design_right_wall_angle_deg,
    design_top_cap_thickness_nm,
    design_width_to_period_ratio,
    design_x_resolution_nm,
    design_z_resolution_nm,
    diffraction_order,
    fourier_orders,
    grazing_angle_deg,
    evaluation_energies_ev,
    measurement_path,
    optical_constants_dir,
    results_dir,
    simulation_backend,
)

results_dir.mkdir(parents=True, exist_ok=True)

silicon = pd.read_csv(
    optical_constants_dir / "n_Si_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

platinum = pd.read_csv(
    optical_constants_dir / "n_Pt_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

measurement = load_measurement_data(measurement_path)
energies_ev = np.asarray(evaluation_energies_ev, dtype=float)
if energies_ev.size == 0:
    energies_ev = np.asarray(measurement.energy_ev, dtype=float)

design_grating = rp.LaminarGrating(
    period_lpermm=design_period_lpermm,
    width_to_period_ratio=design_width_to_period_ratio,
    depth_nm=design_depth_nm,
    left_wall_angle_deg=design_left_wall_angle_deg,
    right_wall_angle_deg=design_right_wall_angle_deg,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=design_layer_thickness_nm,
    top_cap_material=carbon,
    top_cap_thickness_nm=design_top_cap_thickness_nm,
    z_resolution_nm=design_z_resolution_nm,
    x_resolution_nm=design_x_resolution_nm,
)

cases = rp.fixed_angle_cases(
    grating=design_grating,
    energies_ev=energies_ev,
    grazing_angle_deg=grazing_angle_deg,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=diffraction_order,
    default_fourier_orders=fourier_orders,
    max_workers="auto",
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend=simulation_backend,
)
results = list(runner.run_cases(cases))

output_csv_path = results_dir / "simulated_curve_initial.csv"
rp.write_all_orders_csv(results, output_csv_path)

print(f"Initial-parameter simulation CSV: {output_csv_path}")
print(
    "Initial simulation settings: "
    f"grazing_angle_deg={grazing_angle_deg}, "
    f"fourier_orders={fourier_orders}, simulation_backend={simulation_backend}"
)
