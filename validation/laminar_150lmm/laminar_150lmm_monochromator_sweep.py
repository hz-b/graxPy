"""Monochromator sweep for a 150 l/mm laminar grating (CFF=2.25)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import grax


grax.setup_logging(level="INFO", run_id="laminar_150lmm_monochromator")

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
results_dir = example_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "laminar_150lmm_monochromator_all_orders.csv"
orders_plot_path = results_dir / "laminar_150lmm_monochromator_orders_1_3.png"
profile_plot_path = results_dir / "laminar_150lmm_monochromator_profile.png"

silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

gold = pd.read_csv(
    optical_constants_dir / "OC_Au_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

# Requested geometry and coating parameters.
grating = grax.LaminarGrating(
    period_lpermm=150,
    width_to_period_ratio=0.65,
    depth_nm=60.0,
    left_wall_angle_deg=10.0,
    right_wall_angle_deg=10.0,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    x_resolution_nm=0.3,
    z_resolution_nm=0.3,
)

# Match RR-test-lGR150-gd60-nonSP-20_dm_de_r_m1_single.dat:
# 10 eV to 1000 eV in 2 eV steps.
energies_ev = np.arange(10.0, 1000.1, 2.0)

cases = grax.monochromator_cases(
    grating=grating,
    energies_ev=energies_ev,
    diffraction_order=1,
    cff=1.45,
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=10,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    live_plot_order_count=3,
    on_error="fail_fast",
    max_workers='auto',
    resume=False,
    backend="numba",
)

grating.plot_profile(profile_plot_path)
batch_result = list(
    runner.run_cases(
        cases,
        metadata={
            "description": "Laminar 150 l/mm monochromator sweep",
            "period_lpermm": 150,
            "width_to_period_ratio": 0.65,
            "depth_nm": 60.0,
            "left_wall_angle_deg": 10.0,
            "right_wall_angle_deg": 10.0,
            "layer_thickness_nm": 30.0,
            "diffraction_order": 1,
            "cff": 1.45,
            "fourier_orders": 5,
            "x_resolution_nm": 1.0,
            "z_resolution_nm": 1.0,
        },
    )
)

grax.write_all_orders_csv(batch_result, csv_path)
grax.plot_order_subset(
    batch_result,
    orders_plot_path,
    diffraction_orders=[1, 2, 3],
    title="Laminar 150 l/mm Monochromator Sweep: Orders 1-3",
)

print(f"Computed {sum(case.status == 'ok' for case in batch_result)} monochromator points.")
print(f"Monochromator all-orders CSV saved to: {csv_path}")
print(f"Monochromator orders plot saved to: {orders_plot_path}")
print(f"Grating profile plot saved to: {profile_plot_path}")
