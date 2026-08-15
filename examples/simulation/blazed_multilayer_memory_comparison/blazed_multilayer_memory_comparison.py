"""Run a low-memory blazed multilayer sweep with explicit p polarization."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import grax

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=6,
    gamma=0.4,
    n_bilayers=10,
    top_material="C",
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=0.1,
    z_resolution_nm=0.1,
)

energies_ev = np.linspace(3000.0, 3500.0, 50)
cases = [
    {
        **case,
        "profile_memory": True,
    }
    for case in grax.monochromator_cases(
        grating=grating,
        energies_ev=energies_ev,
        diffraction_order=1,
        cff=2.25,
        polarization="p",
    )
]

runner = grax.BatchSimulationRunner(
    show_progress=True,
    fourier_orders=20,
    max_workers="auto",
    backend="numba",
)

results = list(runner.run_cases(cases))
results_by_case_id = {result.case_id: result for result in results}

csv_path = output_dir / "blazed_multilayer_memory_comparison.csv"
plot_path = output_dir / "blazed_multilayer_memory_comparison.png"
profile_plot_path = output_dir / "blazed_multilayer_profile.png"
stack_plot_path = output_dir / "multilayer_stack_schematic.png"

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "energy_ev",
            "grazing_angle_deg",
            "selected_efficiency",
            "selected_diffraction_angle_deg",
            "peak_memory_mb",
            "wall_seconds",
        ]
    )
    for case in cases:
        result = results_by_case_id[case["case_id"]]
        writer.writerow(
            [
                float(case["energy_ev"]),
                float(case["grazing_angle_deg"]),
                float(result.selected_efficiency),
                float(result.selected_diffraction_angle_deg),
                float(result.peak_memory_bytes or 0) / (1024.0 * 1024.0),
                float(result.wall_seconds or 0.0),
            ]
        )

energy_values = np.asarray([float(case["energy_ev"]) for case in cases], dtype=float)
selected_efficiency = np.asarray(
    [float(results_by_case_id[case["case_id"]].selected_efficiency) for case in cases],
    dtype=float,
)
peak_memory_mb = np.asarray(
    [float(results_by_case_id[case["case_id"]].peak_memory_bytes or 0) / (1024.0 * 1024.0) for case in cases],
    dtype=float,
)
wall_seconds = np.asarray(
    [float(results_by_case_id[case["case_id"]].wall_seconds or 0.0) for case in cases],
    dtype=float,
)
max_peak_memory = float(np.max(peak_memory_mb))
max_wall_seconds = float(np.max(wall_seconds))

figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

efficiency_axis = axes[0]
efficiency_axis.plot(energy_values, selected_efficiency, marker="o", linewidth=1.8, color="tab:blue")
efficiency_axis.set_ylabel("Selected efficiency")
efficiency_axis.set_title("Blazed multilayer sweep with low-memory execution")
efficiency_axis.grid(True, alpha=0.3)

memory_axis = axes[1]
memory_axis.plot(energy_values, peak_memory_mb, marker="s", linewidth=1.8, color="tab:orange", label="Peak memory")
memory_axis.plot(energy_values, wall_seconds, marker="^", linewidth=1.8, color="tab:green", label="Wall seconds")
memory_axis.set_xlabel("Energy (eV)")
memory_axis.set_ylabel("Profiled cost")
memory_axis.grid(True, alpha=0.3)
memory_axis.legend(loc="best")
memory_axis.text(
    0.01,
    0.02,
    f"max peak memory = {max_peak_memory:.2f} MB\nmax wall time = {max_wall_seconds:.2f} s",
    transform=memory_axis.transAxes,
    va="bottom",
    ha="left",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
)

figure.tight_layout()
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

grating.plot_profile(profile_plot_path)
multilayer_stack.plot_schematic(stack_plot_path)

print(f"Results saved to: {csv_path}")
print(f"Plot saved to: {plot_path}")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Stack schematic saved to: {stack_plot_path}")
print(f"Max peak memory: {max_peak_memory:.3f} MB")
print(f"Max wall time: {max_wall_seconds:.3f} s")
