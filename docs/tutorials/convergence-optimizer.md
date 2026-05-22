# Convergence Optimizer

Use `grax_opt.optimize_simulation_convergence` when you want to find the
coarsest Fourier and discretization settings that still produce stable results
for a set of photon energies.

This is the numerical counterpart to the measurement-fit workflow:

- measurement fit minimizes the difference to experimental data
- convergence optimization finds when the simulated efficiency stops changing
  enough to matter

## Basic setup

```python
from grax import BlazedGrating
from grax_opt import SimulationConvergenceConfig, optimize_simulation_convergence

grating = BlazedGrating(
    period_lpermm=600.0,
    blaze_angle_deg=0.729,
    anti_blaze_angle_deg=5.597,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    x_resolution_nm=1.0,
    z_resolution_nm=0.5,
)

config = SimulationConvergenceConfig(
    grating=grating,
    energies_ev=[100.0, 600.0, 2000.0],
    grazing_angle_deg=1.5,
    diffraction_order=1,
)

result = optimize_simulation_convergence(config)
print(result.selected_fourier_orders)
print(result.selected_x_resolution_nm)
print(result.selected_z_resolution_nm)
```

## What it does

The optimizer sweeps three candidate grids:

- `fourier_orders_values`
- `x_resolution_values`
- `z_resolution_values`

It evaluates each grid at all requested energies, then selects the coarsest
value whose relative change against all finer candidates stays below the
configured tolerance.

The candidate grids are normalized to coarse-to-fine order before evaluation:

- Fourier orders are sorted ascending
- x/z resolutions are sorted descending, because smaller resolution means a
  finer discretization

## Returned result

`optimize_simulation_convergence(...)` returns a
`SimulationConvergenceResult` with:

- the normalized candidate grids
- per-energy sweep diagnostics
- the selected Fourier order and x/z resolution values
- a `converged` flag that is `True` only when all three settings are stable for
  all energies

If you want to inspect the raw per-energy sweep curves, look at the
`energy_results` field in the returned object.
