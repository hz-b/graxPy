"""Run the fixed-angle simulation using the initial design parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import grax as rp

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants" / "old"
results_dir = example_root / "results" / "laminar_fit"
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
    example_root / "measured_alpha4deg_order1.csv",
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

design_grating = rp.LaminarGrating(
    period_lpermm=400.0,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    top_cap_material=carbon,
    top_cap_thickness_nm=0.3,
    z_resolution_nm=0.1,
    x_resolution_nm=0.1,
)

cases = rp.fixed_angle_cases(
    grating=design_grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=15,
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
