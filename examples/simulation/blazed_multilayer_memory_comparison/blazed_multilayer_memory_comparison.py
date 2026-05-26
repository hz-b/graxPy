"""Compare standard and low-memory solver modes for a blazed multilayer sweep."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xrt.backends.raycing import materials as xrt_materials

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import grax as rp

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
chromium = xrt_materials.Material("Cr", rho=7.19, table="Henke", name="Cr")
carbon = xrt_materials.Material("C", rho=2.20, table="Henke", name="C")

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

multilayer_stack = rp.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=6,
    gamma=0.4,
    n_bilayers=50,
    top_material=carbon,
)

grating = rp.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=0.1,
    z_resolution_nm=0.1,
)

energies_ev = np.linspace(3000.0, 3500.0, 50)
base_cases = list(
    rp.monochromator_cases(
        grating=grating,
        energies_ev=energies_ev,
        diffraction_order=1,
        cff=2.25,
    )
)

standard_cases = [
    {
        **case,
        "memory_mode": "standard",
        "profile_memory": True,
    }
    for case in base_cases
]
low_memory_cases = [
    {
        **case,
        "memory_mode": "low_memory",
        "profile_memory": True,
    }
    for case in base_cases
]

runner = rp.BatchSimulationRunner(
    show_progress=True,
    default_fourier_orders=20,
    backend="numba",
)

standard_results = list(runner.run_cases(standard_cases))
low_memory_results = list(runner.run_cases(low_memory_cases))

standard_by_case_id = {result.case_id: result for result in standard_results}
low_memory_by_case_id = {result.case_id: result for result in low_memory_results}

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
            "selected_efficiency_standard",
            "selected_efficiency_low_memory",
            "selected_diffraction_angle_deg_standard",
            "selected_diffraction_angle_deg_low_memory",
            "peak_memory_mb_standard",
            "peak_memory_mb_low_memory",
            "wall_seconds_standard",
            "wall_seconds_low_memory",
            "efficiency_abs_diff",
        ]
    )
    for case in base_cases:
        standard_result = standard_by_case_id[case["case_id"]]
        low_memory_result = low_memory_by_case_id[case["case_id"]]
        writer.writerow(
            [
                float(case["energy_ev"]),
                float(case["grazing_angle_deg"]),
                float(standard_result.selected_efficiency),
                float(low_memory_result.selected_efficiency),
                float(standard_result.selected_diffraction_angle_deg),
                float(low_memory_result.selected_diffraction_angle_deg),
                float(standard_result.peak_memory_bytes or 0) / (1024.0 * 1024.0),
                float(low_memory_result.peak_memory_bytes or 0) / (1024.0 * 1024.0),
                float(standard_result.wall_seconds or 0.0),
                float(low_memory_result.wall_seconds or 0.0),
                abs(float(standard_result.selected_efficiency) - float(low_memory_result.selected_efficiency)),
            ]
        )

energy_values = np.asarray([float(case["energy_ev"]) for case in base_cases], dtype=float)
standard_efficiency = np.asarray(
    [float(standard_by_case_id[case["case_id"]].selected_efficiency) for case in base_cases],
    dtype=float,
)
low_memory_efficiency = np.asarray(
    [float(low_memory_by_case_id[case["case_id"]].selected_efficiency) for case in base_cases],
    dtype=float,
)
standard_peak_memory = np.asarray(
    [
        float(standard_by_case_id[case["case_id"]].peak_memory_bytes or 0) / (1024.0 * 1024.0)
        for case in base_cases
    ],
    dtype=float,
)
low_memory_peak_memory = np.asarray(
    [
        float(low_memory_by_case_id[case["case_id"]].peak_memory_bytes or 0) / (1024.0 * 1024.0)
        for case in base_cases
    ],
    dtype=float,
)
max_abs_diff = float(np.max(np.abs(standard_efficiency - low_memory_efficiency)))
max_standard_peak = float(np.max(standard_peak_memory))
max_low_memory_peak = float(np.max(low_memory_peak_memory))
reduction_factor = max_standard_peak / max_low_memory_peak if max_low_memory_peak > 0.0 else float("inf")

figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

efficiency_axis = axes[0]
efficiency_axis.plot(energy_values, standard_efficiency, marker="o", linewidth=1.8, label="standard")
efficiency_axis.plot(energy_values, low_memory_efficiency, marker="s", linewidth=1.8, label="low_memory")
efficiency_axis.set_ylabel("Selected efficiency")
efficiency_axis.set_title("Blazed multilayer sweep: standard vs low_memory")
efficiency_axis.grid(True, alpha=0.3)
efficiency_axis.legend(loc="best")
efficiency_axis.text(
    0.01,
    0.02,
    f"max |eff diff| = {max_abs_diff:.3e}",
    transform=efficiency_axis.transAxes,
    va="bottom",
    ha="left",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
)

memory_axis = axes[1]
memory_axis.plot(energy_values, standard_peak_memory, marker="o", linewidth=1.8, label="standard")
memory_axis.plot(energy_values, low_memory_peak_memory, marker="s", linewidth=1.8, label="low_memory")
memory_axis.set_xlabel("Energy (eV)")
memory_axis.set_ylabel("Peak memory (MB)")
memory_axis.grid(True, alpha=0.3)
memory_axis.legend(loc="best")
memory_axis.text(
    0.01,
    0.02,
    f"max peak: std={max_standard_peak:.2f} MB, low={max_low_memory_peak:.2f} MB\n"
    f"reduction factor = {reduction_factor:.2f}x",
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
print(f"Comparison plot saved to: {plot_path}")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Stack schematic saved to: {stack_plot_path}")
print(f"Max absolute efficiency difference: {max_abs_diff:.6e}")
print(f"Max peak memory standard: {max_standard_peak:.3f} MB")
print(f"Max peak memory low_memory: {max_low_memory_peak:.3f} MB")
print(f"Peak memory reduction factor: {reduction_factor:.3f}x")
