"""Ru/B4C configuration for the three-stage multilayer optimization example.

Edit the values here, then run ``0_ru_b4c_d_spacing_study.py``,
``1_ru_b4c_gamma_study.py`` and ``2_ru_b4c_blaze_study.py`` in order (or
``run_all.sh``). ``D_SPACING_NM`` is left as ``"auto"`` so stage 0's geometry
suggestion flows into stages 1 and 2 through ``results/optimization_state.json``;
set it (and ``GAMMA``) to a number to pin it instead. The scripts never rewrite
this file.

For a fast smoke run, shrink the grids: set ``d_spacing_points`` to ``5``,
``blaze_angle_points`` to ``2``, ``blaze_energy_points`` to ``3`` and widen the
energy steps.
"""

from __future__ import annotations

from pathlib import Path

from grax import MultilayerOptimizationConfig

CONFIG = MultilayerOptimizationConfig(
    output_dir=Path(__file__).resolve().parent / "results",
    # Selected values ("auto" consumes the previous stage's suggestion).
    d_spacing_nm="auto",
    gamma=0.5,
    blaze_angle_deg=1.1,
    # Ru/B4C on a silicon substrate; B4C is modelled with the carbon table.
    material_a=("Ru", 12.1),
    material_b=("C", 2.52),
    substrate_material=("Si", 2.33),
    n_bilayers=40,
    # Target and grating geometry.
    target_energy_ev=9000.0,
    grating_density_lpermm=2400.0,
    diffraction_order=2,
    cff=2.25,
    multilayer_bragg_order=1,
    # Per-stage energy grids.
    d_spacing_energy_min_ev=500.0,
    d_spacing_energy_max_ev=12000.0,
    d_spacing_energy_step_ev=100.0,
    gamma_energy_min_ev=500.0,
    gamma_energy_max_ev=12000.0,
    gamma_energy_step_ev=100.0,
    blaze_energy_min_ev=3000.0,
    blaze_energy_max_ev=12000.0,
    blaze_energy_points=15,
    # D-spacing scan.
    bragg_angle_min_deg=0.5,
    bragg_angle_max_deg=2.0,
    d_spacing_relative_range=0.25,
    d_spacing_min_practical_nm=2.0,
    d_spacing_max_practical_nm=8.0,
    d_spacing_points=21,
    # Gamma scan.
    gamma_min=0.3,
    gamma_max=0.8,
    gamma_step=0.1,
    # Blaze scan.
    blaze_angle_half_range_deg=0.3,
    blaze_angle_points=4,
    anti_blaze_angle_deg=0.0,
    # XRT reflectivity settings.
    xrt_window_deg=0.2,
    xrt_angle_points=2001,
    xrt_min_angle_deg=0.0,
    # graxPy theta-search settings for stage 2.
    grax_x_resolution_nm=0.5,
    grax_z_resolution_nm=0.5,
    rough_scan_half_width_deg=0.5,
    rough_scan_points=61,
    rough_fourier_orders=5,
    fine_scan_half_width_deg=0.2,
    fine_scan_points=81,
    fine_fourier_orders=15,
    final_fourier_orders=25,
    final_x_resolution_nm=0.2,
    final_z_resolution_nm=0.2,
    backend="numba",
    solver="neviere",
    polarization="p",
    # Runtime controls.
    max_workers="auto",
    on_error="fail_fast",
    resume=True,
    theta_tracking_mode="auto",
)
