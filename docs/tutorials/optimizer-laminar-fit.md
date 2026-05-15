# Laminar Optimizer Example

This tutorial documents the maintained optimizer workflow in
`examples/optimizer` for fitting a laminar grating model to measured
fixed-angle efficiency data.

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
python examples/optimizer/0_fit_laminar_grating.py
```

Run the full workflow:

```bash
./examples/optimizer/run_all.sh
```

The full workflow runs:

1. fit optimization (`0_fit_laminar_grating.py`)
2. simulation with design parameters (`1_run_simulation_design_parameters.py`)
3. simulation with fitted parameters (`2_run_simulation_fitted_parameters.py`)
4. comparison plot generation (`3_plot_laminar_fit_comparison.py`)

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

- `examples/optimizer/results/laminar_fit/`

Key files:

- `best_result.json`: best objective value and resolved best-fit model parameters
- `trial_history.csv`: per-trial parameter suggestions and loss values
- `fitted_parameters.json`: example-friendly payload used by downstream scripts
- `simulated_curve_initial.csv`: simulated curve from the initial design parameters
- `simulated_curve_fitted.csv`: simulated curve from the optimized fit
- `best_fit.png`: measurement versus best-fit overlay from optimization
- `optimization_loss_history.png`: trial loss and running-best history
- `laminar_fit_measurement_comparison.png`: final design-vs-fitted-vs-measured comparison

## Design vs Optimized Parameters

The laminar optimizer example starts from the design parameters defined in
`examples/optimizer/0_fit_laminar_grating.py` and writes optimized values in
`examples/optimizer/results/laminar_fit/fitted_parameters.json`.

| Parameter | Optimized in this example | Design value | Optimized value |
| --- | --- | --- | --- |
| `period_lpermm` | No (fixed) | `400.0` | `400.0` |
| `width_to_period_ratio` | Yes | `0.67` | `0.7029613344292561` |
| `depth_nm` | Yes | `14.9` | `14.5` |
| `left_wall_angle_deg` | Yes | `15.0` | `21.378071367576183` |
| `right_wall_angle_deg` | Yes | `15.0` | `5.0` |
| `top_cap_thickness_nm` | Yes | `0.3` | `0.3811546815021951` |

Additional geometry/material fields in `InitialLaminarGrating` are held fixed
for this run (for example substrate/layer materials, `layer_thickness_nm`,
`x_resolution_nm`, and `z_resolution_nm`).

Generated figures:

```{image} images/optimizer/laminar_fit/best_fit.png
:alt: Measurement versus best-fit curve for laminar optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/laminar_fit/optimization_loss_history.png
:alt: Optimization loss history for laminar optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/laminar_fit/laminar_fit_measurement_comparison.png
:alt: Design versus fitted versus measured efficiency comparison
:align: center
:width: 80%
```