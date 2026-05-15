"""Run simulation using initial design parameters for the blazed optimizer example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import grax as rp

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
results_dir = example_root / "results" / "blazed_fit"
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

measurement = pd.read_csv(
    example_root / "GR600-BEIChem_energy-Cff2.5.dat",
    sep=r"\s+",
    header=None,
    names=["energy_ev", "efficiency"],
).apply(pd.to_numeric, errors="coerce").dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

design_grating = rp.BlazedGrating(
    period_lpermm=600.0,
    blaze_angle_deg=0.729,
    anti_blaze_angle_deg=5.597,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    top_cap_material=carbon,
    top_cap_thickness_nm=0.5,
    z_resolution_nm=0.1,
    x_resolution_nm=0.1,
)

cases = rp.monochromator_cases(
    grating=design_grating,
    energies_ev=energies,
    period_lpermm=600.0,
    diffraction_order=1,
    cff=2.5,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=20,
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
