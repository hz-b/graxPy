"""Fast multilayer energy-angle sweep tutorial example.

This tutorial samples predefined energy-angle pairs every 50 rows and uses
coarse RCWA settings for short runtime.
"""

from __future__ import annotations

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

cases = grax.energy_angle_cases(
    grating=grating,
    energy_angle_pairs=energy_angle_pairs,
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=2,
    default_fourier_orders=5,
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend="numba",
)

results = list(runner.run_cases(cases))

csv_path = output_dir / "energy_angle_multilayer_all_orders.csv"
plot_path = output_dir / "energy_angle_multilayer_fast.png"
profile_path = output_dir / "energy_angle_multilayer_profile.png"

grax.write_all_orders_csv(results, csv_path)
grating.plot_profile(profile_path)

successful_results = [result for result in results if result.status == "ok"]
figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(
    [result.energy_ev for result in successful_results],
    [result.selected_efficiency for result in successful_results],
    "o-",
    linewidth=1.0,
    markersize=2.0,
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency (2nd order)")
axis.set_title("Fast Multilayer Energy-Angle Sweep (grax)")
axis.grid(True, alpha=0.3)
figure.tight_layout()
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Sampled {len(energy_angle_pairs)} energy-angle pairs from: {input_path}")
print(f"Results saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")
print(f"Profile plot saved to: {profile_path}")
