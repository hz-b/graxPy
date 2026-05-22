"""Find coarse-to-fine simulation settings that stop changing materially."""

from __future__ import annotations

import pandas as pd
from grax import LaminarGrating

from grax_opt import SimulationConvergenceConfig, optimize_simulation_convergence
from example_config import (
    backend,
    depth_nm,
    diffraction_order,
    energies_ev,
    fourier_orders_values,
    grazing_angle_deg,
    layer_thickness_nm,
    left_wall_angle_deg,
    optical_constants_dir,
    period_lpermm,
    relative_tolerance,
    results_dir,
    right_wall_angle_deg,
    top_cap_thickness_nm,
    validate_physical_results,
    width_to_period_ratio,
    x_resolution_nm,
    x_resolution_values,
    z_resolution_nm,
    z_resolution_values,
)


def load_material(filename: str, name: str) -> pd.DataFrame:
    """Load one optical-constant table with a stable material label."""

    material = pd.read_csv(
        optical_constants_dir / filename,
        skiprows=1,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    material.attrs["name"] = name
    return material


silicon = load_material("n_Si_cxro.txt", "Si")
platinum = load_material("n_Pt_cxro.txt", "Pt")
carbon = load_material("n_C_cxro.txt", "C")
results_dir.mkdir(parents=True, exist_ok=True)


def build_grating() -> LaminarGrating:
    """Build the baseline laminar grating used for the convergence study."""

    return LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=width_to_period_ratio,
        depth_nm=depth_nm,
        left_wall_angle_deg=left_wall_angle_deg,
        right_wall_angle_deg=right_wall_angle_deg,
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=top_cap_thickness_nm,
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


config = SimulationConvergenceConfig(
    grating=build_grating(),
    energies_ev=energies_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders_values=fourier_orders_values,
    x_resolution_values=x_resolution_values,
    z_resolution_values=z_resolution_values,
    relative_tolerance=relative_tolerance,
    backend=backend,
    validate_physical_results=validate_physical_results,
)

result = optimize_simulation_convergence(config)

print(f"Backend request: {backend}")
print(f"Backend effective: {result.backend_effective}")
print(f"Energies: {result.energies_ev.tolist()}")
print(f"Relative tolerance: {result.relative_tolerance}")
print(f"Selected Fourier orders: {result.selected_fourier_orders}")
print(f"Selected x resolution (nm): {result.selected_x_resolution_nm}")
print(f"Selected z resolution (nm): {result.selected_z_resolution_nm}")
print(f"Converged across all energies: {result.converged}")
for energy_result in result.energy_results:
    print(f"Energy {energy_result.energy_ev:.1f} eV")
    print(f"  selected indices: {energy_result.selected_indices}")
    print(f"  selected values: {energy_result.selected_values}")
    print(f"  converged: {energy_result.parameter_converged}")
