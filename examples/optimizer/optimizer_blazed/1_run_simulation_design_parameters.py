"""Run simulation using initial design parameters for the blazed optimizer example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import grax
from example_config import (
    anti_blaze_angle_deg,
    simulation_backend,
    blaze_angle_deg,
    cff,
    diffraction_order,
    fourier_orders,
    layer_thickness_nm,
    measurement_path,
    optical_constants_dir,
    period_lpermm,
    results_dir,
    top_cap_thickness_nm,
    use_top_cap,
    x_resolution_nm,
    z_resolution_nm,
)

results_dir.mkdir(parents=True, exist_ok=True)

silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
silicon.attrs["name"] = "Si"
gold = pd.read_csv(optical_constants_dir / "OC_Au_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
gold.attrs["name"] = "Au"
carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"
top_cap_material = carbon if use_top_cap else None

measurement = pd.read_csv(
    measurement_path,
    sep=r"\s+",
    header=None,
    names=["energy_ev", "efficiency"],
).apply(pd.to_numeric, errors="coerce").dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

design_grating = grax.BlazedGrating(
    period_lpermm=period_lpermm,
    blaze_angle_deg=blaze_angle_deg,
    anti_blaze_angle_deg=anti_blaze_angle_deg,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=layer_thickness_nm,
    top_cap_material=top_cap_material,
    top_cap_thickness_nm=top_cap_thickness_nm,
    z_resolution_nm=z_resolution_nm,
    x_resolution_nm=x_resolution_nm,
)

cases = grax.monochromator_cases(
    grating=design_grating,
    energies_ev=energies,
    period_lpermm=period_lpermm,
    diffraction_order=diffraction_order,
    cff=cff,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    max_workers="auto",
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend="numba",
)
results = list(runner.run_cases(cases))

output_csv_path = results_dir / "simulated_curve_initial.csv"
grax.write_all_orders_csv(results, output_csv_path)
print(f"Initial-parameter simulation CSV: {output_csv_path}")
print(
    "Initial simulation settings: "
    f"cff={cff}, fourier_orders={fourier_orders}, "
    "simulation_backend=numba, "
    f"x_resolution_nm={x_resolution_nm}, z_resolution_nm={z_resolution_nm}, "
    f"top_cap_material={'C' if use_top_cap else 'None'}, "
    f"top_cap_thickness_nm={top_cap_thickness_nm}"
)
