"""Blazed multilayer energy-angle sweep matched to DiffraMod input pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax as rp

rp.setup_logging(level="INFO", run_id="blazed_multilayer_sweep")

parser = argparse.ArgumentParser(description="Blazed multilayer energy-angle sweep")
parser.add_argument(
    "--quick",
    action="store_true",
    help="Run with fewer energy-angle points for quick testing",
)
args = parser.parse_args()

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
simulation_dir = example_root / "simulation"
output_dir = example_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)

silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
chromium = pd.read_csv(
    optical_constants_dir / "OC_Cr_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
chromium.attrs["name"] = "Cr"
carbon = pd.read_csv(
    optical_constants_dir / "OC_C_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

reference_data = pd.read_csv(
    simulation_dir / "DiffractMod_CrC_d4.8_N60.dat",
    sep=r"\s+",
    engine="python",
)
reference_data = reference_data[["Energy", "Efficiency(GR)", "alpha"]].copy()
reference_data = reference_data.apply(pd.to_numeric, errors="coerce").dropna()
reference_data = reference_data.reset_index(drop=True)

if args.quick:
    sampled_reference = reference_data.iloc[::100].copy()
    x_resolution_nm = 1
    z_resolution_nm = 1
    default_fourier_orders = 10
else:
    sampled_reference = reference_data.iloc[::1].copy()
    x_resolution_nm = 0.1
    z_resolution_nm = 0.01
    default_fourier_orders = 35

energy_angle_pairs = list(
    zip(
        sampled_reference["Energy"].to_numpy(dtype=float),
        sampled_reference["alpha"].to_numpy(dtype=float),
    )
)

multilayer_stack = rp.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material=carbon,
)

grating = rp.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=x_resolution_nm,
    z_resolution_nm=z_resolution_nm,
)

cases = rp.energy_angle_cases(
    grating=grating,
    energy_angle_pairs=energy_angle_pairs,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=2,
    default_fourier_orders=default_fourier_orders,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    live_plot_order_count=2,
    live_plot_reference_data=sampled_reference[["Energy", "Efficiency(GR)"]].to_numpy(dtype=float),
    on_error="fail_fast",
    max_workers='auto',
    checkpoint_dir=output_dir / "checkpoints",
    resume=False,
    backend="numba",
)

csv_path = output_dir / "blazed_multilayer_all_orders.csv"
profile_plot_path = output_dir / "blazed_multilayer_profile.png"
stack_plot_path = output_dir / "multilayer_stack_schematic.png"
selected_order_plot_path = output_dir / "blazed_multilayer_order_2.png"

results = list(runner.run_cases(cases))
rp.write_all_orders_csv(results, csv_path)
grating.plot_profile(profile_plot_path)
multilayer_stack.plot_schematic(stack_plot_path)

successful_results = [result for result in results if result.status == "ok"]
figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(
    [result.energy_ev for result in successful_results],
    [result.selected_efficiency for result in successful_results],
    "o-",
    linewidth=1.0,
    markersize=2.0,
    label="grax",
)
axis.plot(
    sampled_reference["Energy"],
    sampled_reference["Efficiency(GR)"],
    "s-",
    linewidth=1.0,
    markersize=2.0,
    label="DiffraMod",
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency (2nd order)")
axis.set_title("Blazed Multilayer Energy-Angle Sweep: 2nd Order")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(selected_order_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Computed {len(successful_results)} energy-angle points.")
print(f"Results saved to: {csv_path}")
print(f"Selected-order plot saved to: {selected_order_plot_path}")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Stack schematic saved to: {stack_plot_path}")
