# Optimization API (`grax_opt`)

The optional `grax_opt` package provides Ax-based fitting helpers for grating
parameter optimization against measured efficiency data.

Install optional dependencies first:

```bash
pip install .[opt]
```

## Main entrypoints

- `grax_opt.optimize_laminar(config: LaminarAxConfig) -> OptimizationResult`
- `grax_opt.optimize_blazed(config: BlazedAxConfig) -> OptimizationResult`

Both functions run an Ax optimization loop and emit artifacts in
`config.output_dir`.

## Core configuration types

- `LaminarAxConfig`
- `BlazedAxConfig`
- `InitialLaminarGrating`
- `InitialBlazedGrating`
- `ParameterBounds`

These define:

- initial geometry/materials
- which parameters are optimized
- bounds for each optimized parameter
- solver/evaluation settings (order, Fourier terms, objective settings)

## Result type

`OptimizationResult` includes:

- `best_parameters`: optimized Ax parameter values
- `best_grating_parameters`: fully resolved grating parameters for simulation
- `best_loss`: best objective value found
- `measurement_path`: source measurement file used during fitting
- `result_json_path`: path to persisted best-result JSON
- `trial_history_csv_path`: per-trial loss and parameter history
- `best_fit_plot_path`: optional measurement-vs-fit plot
- `loss_history_plot_path`: optional optimization-loss history plot
- `trial_records`: in-memory trial summaries
- `stopped_early`, `completed_trials`, `early_stop_reason`

## Typical artifact files

Depending on workflow scripts, common files are:

- `best_result.json`
- `trial_history.csv`
- `fitted_parameters.json`
- `simulated_curve_initial.csv`
- `simulated_curve_fitted.csv`
- `best_fit.png`
- `optimization_loss_history.png`

## See tutorials

- [Laminar Grating](../tutorials/optimizer-laminar-fit.md)
- [Blazed Grating](../tutorials/optimizer-blazed-fit.md)
