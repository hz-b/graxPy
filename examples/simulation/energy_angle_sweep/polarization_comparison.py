"""Polarization comparison for energy-angle sweep (s vs p)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import grax

example_root = Path(__file__).resolve().parent
input_path = example_root / "energy_angle_pairs.dat"
output_dir = example_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)

reference_data = pd.read_csv(input_path, sep=r"\s+", engine="python")
reference_data = reference_data.apply(pd.to_numeric, errors="coerce").dropna()
reference_data = reference_data.reset_index(drop=True)

sampled_reference = reference_data.iloc[::50].copy()
energy_angle_pairs = list(
    zip(
        sampled_reference["Energy"].to_numpy(dtype=float),
        sampled_reference["alpha"].to_numpy(dtype=float),
    )
)

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material="C",
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=1.0,
    z_resolution_nm=1.0,
)

parser = argparse.ArgumentParser(description="Polarization comparison (s vs p)")
parser.add_argument(
    "--solver",
    choices=("rcwa", "neviere"),
    default="rcwa",
    help="Electromagnetic solver to run. Both compute every diffraction order; "
    "they differ only in how each layer is crossed in z.",
)
args = parser.parse_args()

runner = grax.BatchSimulationRunner(
    solver=args.solver,
    diffraction_order=2,
    fourier_orders=5,
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend="numba",
)

results_s = list(runner.run_cases(
    grax.energy_angle_cases(grating=grating, energy_angle_pairs=energy_angle_pairs, polarization="s")
))
results_p = list(runner.run_cases(
    grax.energy_angle_cases(grating=grating, energy_angle_pairs=energy_angle_pairs, polarization="p")
))

ok_s = [r for r in results_s if r.status == "ok"]
ok_p = [r for r in results_p if r.status == "ok"]

comparison_plot_path = output_dir / f"energy_angle_pol_comparison_{args.solver}.png"
figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(
    [r.energy_ev for r in ok_s],
    [r.selected_efficiency for r in ok_s],
    "o-",
    linewidth=1.5,
    markersize=3,
    color="tab:blue",
    label="s (TE)",
)
axis.plot(
    [r.energy_ev for r in ok_p],
    [r.selected_efficiency for r in ok_p],
    "s--",
    linewidth=1.5,
    markersize=3,
    color="tab:orange",
    label="p (TM)",
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency (2nd order)")
axis.set_title("Energy-Angle Sweep: s vs p Polarization")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Sampled {len(energy_angle_pairs)} energy-angle pairs from: {input_path}")
print(f"Polarization comparison plot saved to: {comparison_plot_path}")
