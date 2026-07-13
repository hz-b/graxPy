"""Parameter study for the layered laminar 2000 l/mm fixed-angle grating."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import grax
from grax_opt import load_measurement_data

from fixed_angle_parameters import (
    C4O_DENSITY_G_CM3,
    DEPTH_NM,
    FOURIER_ORDERS,
    LEFT_WALL_ANGLE_DEG,
    PERIOD_LPERMM,
    RIGHT_WALL_ANGLE_DEG,
    SI_DENSITY_G_CM3,
    SIO2_DENSITY_G_CM3,
    WIDTH_TO_PERIOD_RATIO,
    X_RESOLUTION_NM,
    Z_RESOLUTION_NM,
    create_layered_stack,
    resolve_material,
)

POLARIZATION = "p"
NUM_STUDY_ENERGY_POINTS = 5
FOURIER_ORDERS_VALUES = np.arange(5, 31, 1, dtype=int)
X_RESOLUTION_VALUES_NM = np.geomspace(10.0, 0.1, 50, dtype=float)
Z_RESOLUTION_VALUES_NM = np.geomspace(10.0, 0.1, 50, dtype=float)
ANGLE_CONFIG = {
    1: "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-1deg.dat",
    2: "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-2deg.dat",
    4: "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-4deg.dat",
}

grax.setup_logging(level="INFO", run_id="laminar_2000lmm_layered_parameter_study")

example_root = Path(__file__).resolve().parent
measurement_dir = example_root / "simulation"
results_dir = example_root / "results" / "laminar_2000lmm_layered_parameter_study"
results_dir.mkdir(parents=True, exist_ok=True)

silicon = resolve_material(material_name="Si", density_g_cm3=SI_DENSITY_G_CM3)
silicon_o2 = resolve_material(material_name="SiO2", density_g_cm3=SIO2_DENSITY_G_CM3)
c4o = resolve_material(material_name="C4O", density_g_cm3=C4O_DENSITY_G_CM3)

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

profile_plot_path = results_dir / "laminar_2000lmm_layered_profile.png"
grating.plot_profile(profile_plot_path)

for angle_deg, measurement_filename in ANGLE_CONFIG.items():
    measurement_path = measurement_dir / measurement_filename
    measurement = load_measurement_data(measurement_path)
    energies_ev = np.linspace(
        float(np.min(measurement.energy_ev)),
        float(np.max(measurement.energy_ev)),
        NUM_STUDY_ENERGY_POINTS,
        dtype=float,
    )
    angle_results_dir = results_dir / f"alpha{angle_deg}deg"
    angle_results_dir.mkdir(parents=True, exist_ok=True)

    study = grax.run_parameter_study(
        grating=grating,
        energies_ev=energies_ev,
        grazing_angle_deg=float(angle_deg),
        diffraction_order=1,
        polarization=POLARIZATION,
        fourier_orders_values=FOURIER_ORDERS_VALUES,
        x_resolution_values=X_RESOLUTION_VALUES_NM,
        z_resolution_values=Z_RESOLUTION_VALUES_NM,
        output_dir=angle_results_dir,
        save_csv=True,
        show_progress=True,
    )

    plot_path = angle_results_dir / f"laminar_2000lmm_layered_parameter_study_alpha{angle_deg}deg.png"
    grax.plot_parameter_study(
        study,
        output_filename=plot_path,
        title=(
            "Laminar 2000 l/mm Layered Parameter Study "
            f"(alpha = {angle_deg} deg)"
        ),
    )

    print(f"Alpha {angle_deg} deg study CSVs saved to: {angle_results_dir}")
    print(f"Alpha {angle_deg} deg study plot saved to: {plot_path}")

print(f"Layered grating profile plot saved to: {profile_plot_path}")
