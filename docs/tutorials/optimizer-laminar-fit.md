# Laminar Grating

This tutorial documents the maintained optimizer workflows in
`examples/optimizer/optimizer_laminar` for fitting laminar grating models to
measured fixed-angle efficiency data.

For API details of optimizer functions/configs, see
[`grax_opt` optimization API](../api/optimization.md).

## Goal

Run a full fit workflow that:

- optimizes selected laminar grating parameters against measurement data
- writes fitted parameters and trial history artifacts
- compares design-vs-fitted simulated curves against the measurement

## Prerequisites

Install the optional optimizer dependencies:

```bash
pip install .[opt]
```

Optional GPU verification:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Run the Example

Run only the fit step:

```bash
python examples/optimizer/optimizer_laminar/0_fit_laminar_grating.py
```

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

## Compute Context Banner

At optimizer startup, you should see a line like:

```text
Optimizer compute: GPU | model=NVIDIA RTX A4000 | torch=2.12.0+cu130 | cuda=13.0
```

or:

```text
Optimizer compute: CPU | model=Intel(R) Xeon(...) | torch=... | cuda=...
```

Interpretation:

- `GPU`: PyTorch/Ax sees CUDA as usable for optimizer runtime
- `CPU`: optimizer is running in CPU mode (including safe fallback when CUDA is not usable)

## Output Artifacts

Main output directory:

- `examples/optimizer/optimizer_laminar/results/laminar_fit/`
- `examples/optimizer/optimizer_laminar/results/laminar_fit_tied_walls/`

Key files:

- `laminar_fit/best_result.json`: best objective value and resolved best-fit model parameters for the standard fit
- `laminar_fit/trial_history.csv`: per-trial parameter suggestions and loss values for the standard fit
- `laminar_fit/fitted_parameters.json`: example-friendly payload used by downstream scripts for the standard fit
- `laminar_fit/simulated_curve_initial.csv`: simulated curve from the initial design parameters
- `laminar_fit/simulated_curve_fitted.csv`: simulated curve from the standard optimized fit
- `laminar_fit/best_fit.png`: measurement versus best-fit overlay from the standard fit
- `laminar_fit/optimization_loss_history.png`: trial loss and running-best history for the standard fit
- `laminar_fit/laminar_fit_measurement_comparison.png`: final design-vs-standard-fit-vs-tied-wall-fit comparison
- `laminar_fit_tied_walls/best_result.json`: best objective value and resolved best-fit model parameters for the tied-wall fit
- `laminar_fit_tied_walls/trial_history.csv`: per-trial parameter suggestions and loss values for the tied-wall fit
- `laminar_fit_tied_walls/fitted_parameters.json`: example-friendly payload used by downstream scripts for the tied-wall fit
- `laminar_fit_tied_walls/simulated_curve_fitted.csv`: simulated curve from the tied-wall optimized fit
- `laminar_fit_tied_walls/best_fit.png`: measurement versus best-fit overlay from the tied-wall fit
- `laminar_fit_tied_walls/optimization_loss_history.png`: trial loss and running-best history for the tied-wall fit
- `laminar_fit_tied_walls/laminar_fit_tied_walls_measurement_comparison.png`: tied-wall comparison artifact copied from the tied-wall fit results

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
