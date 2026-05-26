# Laminar Grating

This tutorial documents the maintained optimizer workflows in
`examples/optimizer/optimizer_laminar` for fitting laminar grating models to
measured fixed-angle efficiency data.

For API details of optimizer functions/configs, see
[`grax_opt` optimization API](../api/optimization.md).



## Run the Example

Run the full workflow:

```bash
./examples/optimizer/optimizer_laminar/run_all.sh
```

The full workflow runs:

1. fit optimization (`0_fit_laminar_grating.py`)
2. simulation with design parameters (`1_run_simulation_design_parameters.py`)
3. simulation with standard fitted parameters (`2_run_simulation_fitted_parameters.py`)
4. tied-wall fit optimization (`0b_fit_laminar_grating_tied_walls.py`)
5. simulation with tied-wall fitted parameters (`2b_run_simulation_tied_wall_fitted_parameters.py`)
6. comparison plot generation (`3_plot_laminar_fit_comparison.py`)

## Design vs Optimized Parameters

The optimizer example starts from the design parameters defined in
`examples/optimizer/optimizer_laminar/0_fit_laminar_grating.py` and
`examples/optimizer/optimizer_laminar/0b_fit_laminar_grating_tied_walls.py`,
then writes optimized values in the corresponding `fitted_parameters.json`
files under `results/laminar_fit/` and `results/laminar_fit_tied_walls/`.

| Parameter | Design | Standard fit | Tied-wall fit |
| --- | --- | --- | --- |
| `period_lpermm` | `400.0` | `400.0` | `400.0` |
| `width_to_period_ratio` | `0.67` | `0.7197277882136405` | `0.7430691707159561` |
| `depth_nm` | `14.9` | `15.621464194357396` | `14.79190301894701` |
| `left_wall_angle_deg` | `15.0` | `8.110796078108251` | `5.0` |
| `right_wall_angle_deg` | `15.0` | `17.102548028342426` | `5.0` |
| `top_cap_thickness_nm` | `0.3` | `0.5566408539190888` | `0.3` |

Additional geometry/material fields in the measurement-fit `build_grating` callable are
held fixed for this run (for example substrate/layer materials,
`layer_thickness_nm`, `x_resolution_nm`, and `z_resolution_nm`).

The tied-wall fit uses the same bounds as the standard fit, but ties
`right_wall_angle_deg` to `left_wall_angle_deg`.

## Standard Fit vs Tied-Wall Fit

The two laminar optimizations use the same measurement target, same grating
model, and same parameter bounds. The key difference is how wall angles are
handled:

- Standard fit: `left_wall_angle_deg` and `right_wall_angle_deg` are free
  parameters that can vary independently.
- Tied-wall fit: `right_wall_angle_deg` is constrained to equal
  `left_wall_angle_deg`, so both walls always share one optimized angle value.

In short: the standard fit allows wall-angle asymmetry, while the tied-wall fit
enforces symmetric wall angles.

Generated figures:

```{image} images/optimizer/laminar_fit/best_fit.png
:alt: Measurement versus best-fit curve for optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/laminar_fit/optimization_loss_history.png
:alt: Optimization loss history for optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/laminar_fit/laminar_fit_measurement_comparison.png
:alt: Design versus standard fit versus tied-wall fit versus measured efficiency comparison
:align: center
:width: 80%
```
