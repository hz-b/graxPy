# Optimizer

Use `grax_opt.optimize_to_measurements` to fit grating parameters to measured
efficiency data.

This page explains the general setup: which spec keys are required, what each
optional key does, and what results are returned.

For complete API signatures, see
[`grax_opt` optimization API](../api/optimization.md).

For numerical convergence of Fourier and discretization settings, see
[Convergence Optimizer](convergence-optimizer.md).

For a full dynamic-spec example with tied parameters, see
[Dynamic Optimizer Specs](../how-to/dynamic-optimizer.md).

## Basic setup

```python
from pathlib import Path

from grax import LaminarGrating
from grax_opt import optimize_to_measurements


def build_grating(parameters):
    return LaminarGrating(
        period_lpermm=400.0,
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=28.77,
        top_cap_material="C",
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
    )


spec = {
    "build_grating": build_grating,
    "parameter_bounds": {
        "width_to_period_ratio": (0.60, 0.75),
        "depth_nm": (12.0, 18.0),
        "left_wall_angle_deg": (5.0, 25.0),
        "right_wall_angle_deg": (5.0, 25.0),
        "top_cap_thickness_nm": (0.0, 1.0),
    },
    "measurement_path": Path("measurement.dat"),
    "output_dir": Path("results/optimizer"),
    "evaluation_energies_ev": [100.0, 150.0, 200.0],
    "angle_mode": "fixed",
    "grazing_angle_deg": 4.0,
}

result = optimize_to_measurements(spec)
print(result.best_loss)
print(result.best_parameters)
```

## Why `build_grating` is required

The optimizer evaluates many trial parameter sets. For each trial, it must
build a grating from the current candidate values before running the
simulation.

That is why the spec takes a callable:

- `build_grating(parameters)` receives the trial parameters
- returns the grating object to evaluate for that trial

Passing one already-built grating object would keep geometry fixed and prevent
the optimizer from changing parameters across trials.

## Spec keys

`optimize_to_measurements` accepts either:

- a plain dictionary (`spec`) with these keys
- a `MeasurementFitConfig` instance (advanced/programmatic usage)

Required keys:

- `build_grating`: callable that builds a grating from a resolved parameter mapping.
- `parameter_bounds`: mapping `{name: (lower, upper)}` (or `ParameterBounds`).
- `measurement_path`: path to the measured two-column dataset.
- `output_dir`: directory where optimizer artifacts are written.
- `evaluation_energies_ev`: non-empty list of positive energies used for objective evaluation.

Important optional keys:

- `angle_mode`: `"fixed"` or `"cff"`.
- `grazing_angle_deg`: used when `angle_mode="fixed"`.
- `cff`: used when `angle_mode="cff"`.
- `evaluation_grazing_angles_deg`: optional grazing-angle list for explicit energy-angle evaluation cases.
- `equality_constraints`: map constrained targets to a source parameter (for tied parameters).
- `diffraction_order`, `fourier_orders`: solver settings for each evaluation.
- `total_trials`, `batch_size`, `random_seed`: optimization runtime controls.
- `backend`: `"auto"`, `"numba"`, or `"numpy"`.
- `enable_early_stopping` and related early-stop thresholds.
- `save_best_fit_plot`, `save_loss_plot`: artifact toggles.

## Parameter bounds and optimized variables

`parameter_bounds` does two jobs:

- defines which parameters belong to the optimization problem
- defines the allowed numeric range for each parameter

If a parameter appears in `equality_constraints` as a target, it is not a free
Ax variable. Its value is copied from the source parameter before building the
grating.

Example:

```python
"parameter_bounds": {
    "left_wall_angle_deg": (5.0, 25.0),
    "right_wall_angle_deg": (5.0, 25.0),
},
"equality_constraints": {
    "right_wall_angle_deg": "left_wall_angle_deg",
},
```

Here `left_wall_angle_deg` is optimized directly; `right_wall_angle_deg` is tied
to it.

## Evaluation inputs and geometry

Evaluation schedule:

- `evaluation_energies_ev` is mandatory and must be non-empty and positive.
- `evaluation_grazing_angles_deg` is optional. If omitted (or empty), the optimizer
  uses energy-only selection.
- If provided, angles must be positive.
- You may provide one-energy/many-angles or many-energies/one-angle.
- Many-energies/many-angles in the same spec is rejected.

Angle mode:

- `angle_mode="fixed"`: uses `grazing_angle_deg` for fixed-angle geometry.
- `angle_mode="cff"`: uses `cff` for constant-focus geometry.

## Returned result and written artifacts

`optimize_to_measurements(...)` returns an `OptimizationResult` object with the main fields:

- `best_parameters`: best free parameter values found by Ax.
- `best_grating_parameters`: resolved full parameter set after equality ties.
- `best_loss`: best objective value.
- `trial_records`: per-trial losses and parameters.
- `completed_trials`, `stopped_early`, `early_stop_reason`.
- paths: `result_json_path`, `trial_history_csv_path`, and optional plot paths.

Written files in `output_dir`:

- `best_result.json`: summary of best fit and run metadata.
- `trial_history.csv`: per-trial history.
- `best_fit.png`: optional comparison of measured vs simulated curve.
- `optimization_loss_history.png`: optional loss history plot.

```{toctree}
:maxdepth: 1

optimizer-laminar-fit
optimizer-blazed-fit
convergence-optimizer
```
