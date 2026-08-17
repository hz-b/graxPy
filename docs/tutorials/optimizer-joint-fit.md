# Joint measurement fits

`grax_opt.optimize_to_joint_measurements` fits **one** parameter set against
**several** measured curves at once. Use it when a single geometry has to explain
all of your measurements together, rather than fitting each curve separately and
comparing the results afterwards.

The curves do not have to differ only by grazing angle. Each measurement carries
its own conditions — angle, angle mode, diffraction order, polarization — so one
fit can span a whole measurement campaign.

This is a separate entrypoint from
{doc}`optimize_to_measurements <optimizer>`, which fits a single measurement.

A runnable end-to-end workflow lives in
`examples/optimizer/optimizer_joint/`:

```bash
./examples/optimizer/optimizer_joint/run_all.sh
```

It generates a four-condition measurement set from a known grating, fits it,
resumes the fit to extend it, and reports how close each parameter came to the
value the data was built from.

## Basic setup

```python
from grax_opt import optimize_to_joint_measurements

result = optimize_to_joint_measurements({
    "build_grating": build_candidate_grating,
    "parameter_bounds": {
        "width_to_period_ratio": (0.45, 0.80),
        "depth_nm": (4.5, 6.5),
        "wall_angle_deg": (1.0, 40.0),
    },
    "output_dir": "results/joint_fit",
    "grazing_angle_deg": 4.0,       # run-level default
    "diffraction_order": 1,         # run-level default
    "polarization": "s",            # run-level default
    "measurements": [
        {"measurement_path": "data/alpha1.dat", "grazing_angle_deg": 1.0},
        {"measurement_path": "data/alpha2.dat", "grazing_angle_deg": 2.0},
        {"measurement_path": "data/order2.dat", "diffraction_order": 2},
        {"measurement_path": "data/cff_p.dat",
         "angle_mode": "cff", "cff": 2.25, "polarization": "p"},
    ],
    "fourier_orders": 15,
    "total_trials": 200,
    "max_workers": "auto",
})

print(result.best_loss, result.per_measurement_best_losses)
```

Every measurement keeps its **own** energy grid. The grids do not have to match
in range or in length.

## Run-level defaults and per-measurement overrides

The conditions below are set once on the run and inherited by every measurement.
Any measurement may override any of them:

| Key | Meaning |
| --- | --- |
| `angle_mode` | `"fixed"` or `"cff"` |
| `grazing_angle_deg` | fixed angle, used when the resolved mode is `"fixed"` |
| `cff` | fixed-focus constant, used when the resolved mode is `"cff"` |
| `diffraction_order` | the order the curve was recorded in |
| `polarization` | `s`/`p`, or the equivalent `TE`/`TM` |

A measurement that sets none of them inherits all of them, so the common case —
several angles, everything else shared — stays a one-key spec per curve.

Numerical settings stay run-level and cannot be overridden per measurement:
`fourier_orders`, `solver`, `solver_options`, `backend`, `max_workers`. They
describe how the simulation is computed, not what was measured.

## Measurement keys

Each entry in `measurements` accepts the five condition keys above, plus:

- `measurement_path` (required): the measured two-column dataset.
- `evaluation_energies_ev`: energies to evaluate at. When omitted, the file's
  own energy grid is used. When given, the measured curve is interpolated onto
  it.
- `measurement_efficiency`: measured efficiencies to use **directly** instead of
  interpolating the file, as described under "Pre-prepared measurements" below.
- `weight`: relative weight for the `"weighted"` reduction. Defaults to `1.0`.
- `label`: identifier used in artifacts. Defaults to a description of whichever
  conditions the measurement sets, for example `alpha2deg` or
  `cff2p25_order1_p`, falling back to the measurement file stem when it sets
  none. Labels must be unique, so give an explicit `label` when two measurements
  would otherwise describe themselves the same way.

## Choosing the solver

`solver="rcwa"` (the default) or `solver="neviere"` selects the electromagnetic
solver for every trial, with `solver_options` carrying the differential method's
integration settings. See {doc}`choosing-a-solver`.

The solver is part of the problem fingerprint, so it cannot be swapped on a
resume — see {doc}`optimizer-resume`.

## Combining the per-measurement losses

Each measurement contributes its own mean squared error. `joint_loss_reduction`
controls how those are combined into the single value the optimizer minimizes:

| Reduction | Joint loss | Use when |
| --- | --- | --- |
| `"mean"` (default) | mean of the per-measurement MSEs | every curve should count equally |
| `"sum"` | sum of the per-measurement MSEs | same ranking as `mean`, larger magnitude |
| `"pooled"` | MSE over all points pooled together | grids have **different point counts** and every measured point should count equally |
| `"weighted"` | weighted mean using each `weight` | some curves are more trustworthy than others |

`"mean"` and `"pooled"` differ only when the measurements have unequal point
counts. If you exclude absorption edges or downsample one curve more than
another, prefer `"pooled"` — otherwise the curve with fewer points is weighted
more heavily per measured point.

A caution when the curves differ in magnitude: a weak order-2 curve contributes
a much smaller squared error than a strong order-1 curve at the same *relative*
accuracy, so an unweighted reduction quietly lets the strong curve dominate. Use
`"weighted"` when you want the weak curve to carry its share.

## Pre-prepared measurements

`grax_opt` deliberately contains no smoothing, downsampling, or edge-exclusion
logic. When you preprocess the measured curves yourself, pass the resulting
values through `measurement_efficiency` so the optimizer fits exactly the
numbers you prepared:

```python
energies, efficiencies = my_own_preprocessing(raw_path)

spec = {
    "measurement_path": raw_path,        # kept for provenance
    "grazing_angle_deg": 2.0,
    "evaluation_energies_ev": energies,
    "measurement_efficiency": efficiencies,
}
```

When `measurement_efficiency` is omitted, the measured curve is interpolated
from the file onto `evaluation_energies_ev`.

## Parallelism

Each trial evaluates **all** measurements in a single batch. With
`max_workers="auto"`, the worker pool parallelizes across measurements and
energies together, so a four-curve fit keeps the pool busy far better than four
separate single-curve fits would.

Because a batch runner may return results out of completion order, simulated
efficiencies are reassembled by result index rather than by arrival order.

## Written artifacts

- `best_result.json`: best fit, `per_measurement_best_losses`, the resolved
  conditions of every measurement, and run metadata including `solver`.
- `trial_history.csv`: per-trial history with one `loss_<label>` column per
  measurement alongside the joint `loss`.
- `best_fit.png`: one measured-versus-simulated panel per measurement, titled
  with that measurement's conditions.
- `best_fit_comparison.csv`: long-form `label, angle_mode, grazing_angle_deg,
  cff, diffraction_order, polarization, energy_ev, measured_efficiency,
  simulated_efficiency`.
- `optimization_loss_history.png`: joint-loss history.

Joint runs support checkpointing and resume on the same terms as the
single-measurement optimizer — see {doc}`optimizer-resume`.
