"""Shared configuration for the tied-wall laminar optimizer example workflow."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_base_config_path = Path(__file__).resolve().with_name("example_config.py")
_base_spec = spec_from_file_location("_laminar_base_example_config", _base_config_path)
if _base_spec is None or _base_spec.loader is None:
    raise ImportError(f"Unable to load base laminar config from {_base_config_path}.")
_base_config = module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base_config)

angle_mode = _base_config.angle_mode
batch_size = _base_config.batch_size
cff = _base_config.cff
depth_nm = _base_config.depth_nm
diffraction_order = _base_config.diffraction_order
evaluation_energies_ev = _base_config.evaluation_energies_ev
evaluation_grazing_angles_deg = _base_config.evaluation_grazing_angles_deg
fourier_orders = _base_config.fourier_orders
grazing_angle_deg = _base_config.grazing_angle_deg
layer_thickness_nm = _base_config.layer_thickness_nm
left_wall_angle_deg = _base_config.left_wall_angle_deg
measurement_path = _base_config.measurement_path
optical_constants_dir = _base_config.optical_constants_dir
optimizer_backend = _base_config.optimizer_backend
period_lpermm = _base_config.period_lpermm
random_seed = _base_config.random_seed
results_dir = _base_config.tied_wall_results_dir
right_wall_angle_deg = _base_config.right_wall_angle_deg
simulation_backend = _base_config.simulation_backend
top_cap_thickness_nm = _base_config.top_cap_thickness_nm
total_trials = _base_config.total_trials
width_to_period_ratio = _base_config.width_to_period_ratio
x_resolution_nm = _base_config.x_resolution_nm
z_resolution_nm = _base_config.z_resolution_nm

experiment_name = "laminar_fit_tied_walls"
equality_constraints = {"right_wall_angle_deg": "left_wall_angle_deg"}

