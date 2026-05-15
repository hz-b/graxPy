# Optimizer Blazed Example

This tutorial documents the blazed optimizer workflow in
`examples/optimizer_blazed` for fitting a blazed grating model to measured
monochromator efficiency data.

For API details of optimizer functions/configs, see
[`grax_opt` optimization API](../api/optimization.md).

## Goal

Run a full fit workflow that:

- optimizes selected blazed grating parameters against measurement data
- writes fitted parameters and trial history artifacts
- compares design-vs-fitted simulated curves against the measurement

## Prerequisites

Install optional optimizer dependencies:

```bash
pip install .[opt]
```

## Run the Example

Run only the fit step:

```bash
python examples/optimizer_blazed/0_fit_blazed_grating.py
```

Run the full workflow:

```bash
./examples/optimizer_blazed/run_all.sh
```

The full workflow runs:

1. fit optimization (`0_fit_blazed_grating.py`)
2. simulation with design parameters (`1_run_simulation_design_parameters.py`)
3. simulation with fitted parameters (`2_run_simulation_fitted_parameters.py`)
4. comparison plot generation (`3_plot_blazed_fit_comparison.py`)

## Compute Context Banner

At optimizer startup, you should see a line like:

```text
Optimizer compute: GPU | model=... | torch=... | cuda=...
```

or:

```text
Optimizer compute: CPU | model=... | torch=... | cuda=...
```

## Output Artifacts

Main output directory:

- `examples/optimizer_blazed/results/blazed_fit/`

Key files:

- `best_result.json`
- `trial_history.csv`
- `fitted_parameters.json`
- `simulated_curve_initial.csv`
- `simulated_curve_fitted.csv`
- `best_fit.png`
- `optimization_loss_history.png`
- `blazed_fit_measurement_comparison.png`

## Design vs Optimized Parameters

The example starts from the design parameters in
`examples/optimizer_blazed/0_fit_blazed_grating.py`.

| Parameter | Optimized in this example | Design value |
| --- | --- | --- |
| `period_lpermm` | Yes | `600.0` |
| `blaze_angle_deg` | Yes | `0.729` |
| `anti_blaze_angle_deg` | Yes | `5.597` |
| `top_cap_thickness_nm` | No (fixed) | `0.0` |

## Generated figures

```{image} images/optimizer/blazed_fit/best_fit.png
:alt: Measurement versus best-fit curve for blazed optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/blazed_fit/optimization_loss_history.png
:alt: Optimization loss history for blazed optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/blazed_fit/blazed_fit_measurement_comparison.png
:alt: Design versus fitted versus measured efficiency comparison for blazed example
:align: center
:width: 80%
```
