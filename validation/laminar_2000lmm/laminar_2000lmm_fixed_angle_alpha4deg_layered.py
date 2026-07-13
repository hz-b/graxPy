"""Fixed-angle energy sweep for the 2000 l/mm laminar grating at alpha = 4 deg."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax
from fixed_angle_parameters import (
    C4O_DENSITY_G_CM3,
    C4O_THICKNESS_NM,
    DEPTH_NM,
    FOURIER_ORDERS,
    LEFT_WALL_ANGLE_DEG,
    NUM_ENERGY_POINTS,
    PERIOD_LPERMM,
    RIGHT_WALL_ANGLE_DEG,
    SI_DENSITY_G_CM3,
    SIO2_DENSITY_G_CM3,
    SIO2_THICKNESS_NM,
    WIDTH_TO_PERIOD_RATIO,
    X_RESOLUTION_NM,
    Z_RESOLUTION_NM,
    create_layered_stack,
    resolve_material,
)


grax.setup_logging(level="INFO", run_id="laminar_2000lmm_fixed_angle_alpha4deg_layered")

example_root = Path(__file__).resolve().parent
measurement_path = (
    example_root / "simulation" / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-4deg.dat"
)
results_dir = example_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_layered_all_orders.csv"
comparison_plot_path = results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_layered_comparison.png"
profile_plot_path = results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_layered_profile.png"

measurement_data = pd.read_csv(
    measurement_path,
    sep=r"\s+",
    engine="python",
    header=None,
    names=["energy_ev", "efficiency"],
)
live_plot_measurement_data = measurement_data[["energy_ev", "efficiency"]].to_numpy(dtype=float)
energy_start_ev = float(measurement_data["energy_ev"].iloc[0])
energy_stop_ev = float(measurement_data["energy_ev"].iloc[-1])
energies_ev = np.linspace(energy_start_ev, energy_stop_ev, NUM_ENERGY_POINTS, dtype=float)

silicon = resolve_material(
    material_name="Si",
    density_g_cm3=SI_DENSITY_G_CM3,
)
silicon_o2 = resolve_material(
    material_name="SiO2",
    density_g_cm3=SIO2_DENSITY_G_CM3,
)
c4o = resolve_material(
    material_name="C4O",
    density_g_cm3=C4O_DENSITY_G_CM3,
)

layered_stack = create_layered_stack(
    substrate_material=silicon,
    sio2_material=silicon_o2,
    c4o_material=c4o,
)

grating = grax.LaminarGrating(
    period_lpermm=PERIOD_LPERMM,
    width_to_period_ratio=WIDTH_TO_PERIOD_RATIO,
    depth_nm=DEPTH_NM,
    left_wall_angle_deg=LEFT_WALL_ANGLE_DEG,
    right_wall_angle_deg=RIGHT_WALL_ANGLE_DEG,
    substrate_material=silicon,
    layer_material=silicon,
    layer_thickness_nm=0.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    coating_stack=layered_stack,
    x_resolution_nm=X_RESOLUTION_NM,
    z_resolution_nm=Z_RESOLUTION_NM,
)

cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies_ev,
    grazing_angle_deg=4.0,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=FOURIER_ORDERS,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    live_plot_order_count=1,
    live_plot_reference_data=live_plot_measurement_data,
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
            "description": "Laminar 2000 l/mm fixed-angle sweep with SiO2/C4O layers",
            "period_lpermm": PERIOD_LPERMM,
            "width_to_period_ratio": WIDTH_TO_PERIOD_RATIO,
            "depth_nm": DEPTH_NM,
            "left_wall_angle_deg": LEFT_WALL_ANGLE_DEG,
            "right_wall_angle_deg": RIGHT_WALL_ANGLE_DEG,
            "substrate_material": "Si",
            "layer_stack": [
                {"material": "SiO2", "thickness_nm": SIO2_THICKNESS_NM},
                {"material": "C4O", "thickness_nm": C4O_THICKNESS_NM},
            ],
            "grazing_angle_deg": 4.0,
            "diffraction_order": 1,
            "fourier_orders": FOURIER_ORDERS,
            "x_resolution_nm": X_RESOLUTION_NM,
            "z_resolution_nm": Z_RESOLUTION_NM,
            "polarization": "p",
            "measurement_file": measurement_path.name,
        },
    )
)

grax.write_all_orders_csv(batch_result, csv_path)

successful_cases = sorted(
    [case for case in batch_result if case.status == "ok"],
    key=lambda case: float(case.energy_ev),
)
figure, axis = plt.subplots(figsize=(10, 7))
axis.plot(
    [case.energy_ev for case in successful_cases],
    [case.selected_efficiency for case in successful_cases],
    linewidth=1.8,
    label="grax order 1",
)
axis.plot(
    measurement_data["energy_ev"],
    measurement_data["efficiency"],
    linestyle="--",
    linewidth=1.4,
    label="Measurement",
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency")
axis.set_title("Laminar 2000 l/mm Fixed-Angle Sweep at Alpha = 4 deg")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Computed {sum(case.status == 'ok' for case in batch_result)} fixed-angle points.")
print(f"Fixed-angle all-orders CSV saved to: {csv_path}")
print(f"Fixed-angle comparison plot saved to: {comparison_plot_path}")
print(f"Grating profile plot saved to: {profile_plot_path}")
