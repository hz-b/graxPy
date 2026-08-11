# Joint multi-angle fits

`grax_opt.optimize_to_joint_measurements` fits **one** parameter set against
**several** measured curves recorded at different grazing angles. Use it when a
single geometry has to explain all of your measurements at once, rather than
fitting each angle separately and comparing the results afterwards.

This is a separate entrypoint from
{doc}`optimize_to_measurements <optimizer>`, which fits a single measurement.

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
    "measurements": [
        {"grazing_angle_deg": 1.0, "measurement_path": "data/alpha1.dat"},
        {"grazing_angle_deg": 2.0, "measurement_path": "data/alpha2.dat"},
        {"grazing_angle_deg": 4.0, "measurement_path": "data/alpha4.dat"},
    ],
    "diffraction_order": 1,
    "fourier_orders": 15,
    "total_trials": 200,
    "max_workers": "auto",
})

print(result.best_loss, result.per_measurement_best_losses)
```

Every angle keeps its **own** energy grid. The grids do not have to match in
range or in length.

## Measurement keys

Each entry in `measurements` accepts:

- `grazing_angle_deg` (required): the fixed angle for this curve.
- `measurement_path` (required): the measured two-column dataset.
- `evaluation_energies_ev`: energies to evaluate at. When omitted, the file's
  own energy grid is used. When given, the measured curve is interpolated onto
  it.
- `measurement_efficiency`: measured efficiencies to use **directly** instead of
  interpolating the file, as described under "Pre-prepared measurements" below.
- `weight`: relative weight for the `"weighted"` reduction. Defaults to `1.0`.
- `label`: identifier used in artifacts. Defaults to `alpha<angle>deg`.

## Combining the per-angle losses

Each angle contributes its own mean squared error. `joint_loss_reduction`
controls how those are combined into the single value the optimizer minimizes:

| Reduction | Joint loss | Use when |
| --- | --- | --- |
| `"mean"` (default) | mean of the per-angle MSEs | every angle should count equally |
| `"sum"` | sum of the per-angle MSEs | same ranking as `mean`, larger magnitude |
| `"pooled"` | MSE over all points pooled together | grids have **different point counts** and every measured point should count equally |
| `"weighted"` | weighted mean using each `weight` | some angles are more trustworthy than others |

`"mean"` and `"pooled"` differ only when the angles have unequal point counts.
If you exclude absorption edges or downsample one angle more than another,
prefer `"pooled"` — otherwise the angle with fewer points is weighted more
heavily per measured point.

## Pre-prepared measurements

`grax_opt` deliberately contains no smoothing, downsampling, or edge-exclusion
logic. When you preprocess the measured curves yourself, pass the resulting
values through `measurement_efficiency` so the optimizer fits exactly the
numbers you prepared:

```python
energies, efficiencies = my_own_preprocessing(raw_path)

spec = {
    "grazing_angle_deg": 2.0,
    "measurement_path": raw_path,        # kept for provenance
    "evaluation_energies_ev": energies,
    "measurement_efficiency": efficiencies,
}
```

When `measurement_efficiency` is omitted, the measured curve is interpolated
from the file onto `evaluation_energies_ev`.

## Parallelism

Each trial evaluates **all** angles in a single batch. With
`max_workers="auto"`, the worker pool parallelizes across angles and energies
together, so a three-angle fit keeps the pool busy far better than three
separate single-angle fits would.

Because a batch runner may return results out of completion order, simulated
efficiencies are reassembled by result index rather than by arrival order.

## Written artifacts

- `best_result.json`: best fit, `per_measurement_best_losses`, per-angle
  metadata, and run metadata.
- `trial_history.csv`: per-trial history with one `loss_<label>` column per
  angle alongside the joint `loss`.
- `best_fit.png`: one measured-versus-simulated panel per angle.
- `best_fit_comparison.csv`: long-form
  `label, grazing_angle_deg, energy_ev, measured_efficiency, simulated_efficiency, diffraction_order`.
- `optimization_loss_history.png`: joint-loss history.

Joint runs support checkpointing and resume on the same terms as the
single-measurement optimizer — see {doc}`optimizer-resume`.
