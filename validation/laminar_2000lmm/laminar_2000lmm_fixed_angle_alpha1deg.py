"""Fixed-angle energy sweep for the 2000 l/mm laminar grating at alpha = 1 deg."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax


grax.setup_logging(level="INFO", run_id="laminar_2000lmm_fixed_angle_alpha1deg")

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
simulation_path = (
    example_root / "simulation" / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-1deg.dat"
)
results_dir = example_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_all_orders.csv"
comparison_plot_path = results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_comparison.png"
profile_plot_path = results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_profile.png"

reference_data = pd.read_csv(
    simulation_path,
    sep=r"\s+",
    engine="python",
    header=None,
    names=["energy_ev", "efficiency"],
)
energy_start_ev = float(reference_data["energy_ev"].iloc[0])
energy_stop_ev = float(reference_data["energy_ev"].iloc[-1])
num_points = 1039
energies_ev = np.linspace(energy_start_ev, energy_stop_ev, num_points, dtype=float)

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

grating = grax.LaminarGrating(
    period_lpermm=2000,
    width_to_period_ratio=0.6,
    depth_nm=5.0,
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

cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies_ev,
    grazing_angle_deg=1.0,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=10,
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    max_workers="auto",
    resume=False,
    backend="numba",
)

grating.plot_profile(profile_plot_path)
batch_result = list(
    runner.run_cases(
        cases,
        metadata={
            "description": "Laminar 2000 l/mm fixed-angle sweep",
            "period_lpermm": 2000,
            "width_to_period_ratio": 0.6,
            "depth_nm": 5.0,
            "left_wall_angle_deg": 10.0,
            "right_wall_angle_deg": 10.0,
            "layer_thickness_nm": 30.0,
            "grazing_angle_deg": 1.0,
            "diffraction_order": 1,
            "fourier_orders": 10,
            "x_resolution_nm": 0.3,
            "z_resolution_nm": 0.3,
            "polarization": "p",
            "reference_file": simulation_path.name,
        },
    )
)

grax.write_all_orders_csv(batch_result, csv_path)

successful_cases = [case for case in batch_result if case.status == "ok"]
figure, axis = plt.subplots(figsize=(10, 7))
axis.plot(
    [case.energy_ev for case in successful_cases],
    [case.selected_efficiency for case in successful_cases],
    linewidth=1.8,
    label="grax order 1",
)
axis.plot(
    reference_data["energy_ev"],
    reference_data["efficiency"],
    linestyle="--",
    linewidth=1.4,
    label=simulation_path.name,
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency")
axis.set_title("Laminar 2000 l/mm Fixed-Angle Sweep at Alpha = 1 deg")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Computed {sum(case.status == 'ok' for case in batch_result)} fixed-angle points.")
print(f"Fixed-angle all-orders CSV saved to: {csv_path}")
print(f"Fixed-angle comparison plot saved to: {comparison_plot_path}")
print(f"Grating profile plot saved to: {profile_plot_path}")
