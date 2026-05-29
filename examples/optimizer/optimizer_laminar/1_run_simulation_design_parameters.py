"""Run the fixed-angle simulation using the initial design parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import grax as rp
from example_config import (
    depth_nm,
    diffraction_order,
    fourier_orders,
    grazing_angle_deg,
    layer_thickness_nm,
    left_wall_angle_deg,
    measurement_path,
    optical_constants_dir,
    period_lpermm,
    results_dir,
    right_wall_angle_deg,
    simulation_backend,
    top_cap_thickness_nm,
    width_to_period_ratio,
    x_resolution_nm,
    z_resolution_nm,
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

measurement = pd.read_csv(
    measurement_path,
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

design_grating = rp.LaminarGrating(
    period_lpermm=period_lpermm,
    width_to_period_ratio=width_to_period_ratio,
    depth_nm=depth_nm,
    left_wall_angle_deg=left_wall_angle_deg,
    right_wall_angle_deg=right_wall_angle_deg,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=layer_thickness_nm,
    top_cap_material=carbon,
    top_cap_thickness_nm=top_cap_thickness_nm,
    z_resolution_nm=z_resolution_nm,
    x_resolution_nm=x_resolution_nm,
)

cases = rp.fixed_angle_cases(
    grating=design_grating,
    energies_ev=energies,
    grazing_angle_deg=grazing_angle_deg,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=diffraction_order,
    default_fourier_orders=fourier_orders,
    max_workers="auto",
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend="numba",
)
results = list(runner.run_cases(cases))

output_csv_path = results_dir / "simulated_curve_initial.csv"
rp.write_all_orders_csv(results, output_csv_path)

print(f"Initial-parameter simulation CSV: {output_csv_path}")
print(
    "Initial simulation settings: "
    f"grazing_angle_deg={grazing_angle_deg}, "
    f"fourier_orders={fourier_orders}, simulation_backend=numba"
)
