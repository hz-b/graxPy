# Multilayer optimization workflow

Use {func}`grax.run_d_spacing_study`, {func}`grax.run_gamma_study` and
{func}`grax.run_blaze_study` to size a periodic multilayer coating for a blazed
grating monochromator working in a chosen diffraction order at fixed CFF. The
three stages share one {class}`grax.MultilayerOptimizationConfig` and run in
order:

1. **D-spacing.** Compute the grazing angle at the target energy and CFF with
   {func}`grax.monochromator_grazing_angles_deg`, convert it to a bilayer
   d-spacing with the first-order Bragg law `d = λ / (2 sin θ)`, build a
   practical 0.1 nm-rounded scan grid that is guaranteed to contain the rounded
   geometry value, and scan every candidate with XRT planar-multilayer
   reflectivity. The geometry value is stored as `d_suggested_nm`; the
   numerically best d at the target energy is stored separately as a diagnostic.
2. **Gamma.** At the resolved d-spacing, scan the bilayer thickness ratio and
   keep the value with the highest peak reflectivity at the target energy.
3. **Blaze.** Build the multilayer-coated blazed grating and scan the blaze
   angle, running {func}`grax.run_multilayer_theta_search` per energy (diffraction
   order 2, multilayer Bragg order 1, Nevière solver, p-polarization by default),
   and keep the blaze angle with the highest selected-order efficiency at the
   target energy.

## State file and the `"auto"` hand-off

The only channel between stages is `output_dir/optimization_state.json`. A stage
reads it only when the corresponding config value is the string `"auto"`:

- `d_spacing_nm="auto"` in stages 1 and 2 resolves `d_suggested_nm` written by
  stage 0.
- A numeric `d_spacing_nm` (or `gamma`) always wins; the state file is ignored.
- No stage ever rewrites the config. Stage 1's `gamma_suggested` is recorded for
  traceability but **not** auto-applied -- copy it into the config yourself if
  you want stage 2 to use it.

## Two inherited conventions

- The geometry d-spacing derivation uses `hc = 1239.841984` eV·nm while
  {func}`grax.monochromator_grazing_angles_deg` uses `1239.8` internally. The
  difference is far below the 0.1 nm rounding applied to the candidate grid.
- The XRT reflectivity path (stages 0-1) places `material_a` on top of a
  `material_a` substrate; the graxPy {class}`grax.MultilayerStack` used in stage
  2 places `material_b` on top of the configured substrate. Both are modelling
  choices carried over unchanged from the original workflow.

## Example

```python
from pathlib import Path

from grax import (
    MultilayerOptimizationConfig,
    run_blaze_study,
    run_d_spacing_study,
    run_gamma_study,
)

config = MultilayerOptimizationConfig(
    output_dir=Path("examples/simulation/multilayer_optimization_rub4c/results"),
    d_spacing_nm="auto",
    gamma=0.5,
    blaze_angle_deg=1.1,
    material_a=("Ru", 12.1),
    material_b=("C", 2.52),          # B4C modelled with the carbon table
    substrate_material=("Si", 2.33),
    n_bilayers=40,
    target_energy_ev=9000.0,
    grating_density_lpermm=2400.0,
    diffraction_order=2,
    cff=2.25,
    multilayer_bragg_order=1,
    solver="neviere",
    polarization="p",
)

d_result = run_d_spacing_study(config)
print(f"geometry d = {d_result.geometry_d_nm:.3f} nm -> suggested {d_result.d_suggested_nm:.1f} nm")

gamma_result = run_gamma_study(config)         # d_spacing_nm="auto" -> reads the state file
print(f"suggested gamma = {gamma_result.gamma_suggested:.3f}")

blaze_result = run_blaze_study(config)         # uses config.gamma, not the suggestion
print(f"suggested blaze = {blaze_result.blaze_suggested_deg:.4f} deg")
```

```{image} images/simulation/multilayer_optimization_rub4c_d_spacing.png
:alt: Ru/B4C d-spacing study, peak p-polarized reflectivity versus photon energy per candidate d-spacing
:align: center
:width: 90%
```

See `examples/simulation/multilayer_optimization_rub4c/` for the full runnable
workflow: `ru_b4c_parameters.py` builds the config and the three numbered
scripts run the stages. `run_all.sh` runs them in order.

## Solver selection

Stage 2 takes `solver` (`rcwa` or `neviere`) from the config; the example's
`2_ru_b4c_blaze_study.py` also exposes it as `--solver`. Stages 0 and 1 do not
use a graxPy solver -- they measure reflectivity with XRT. See
{doc}`choosing-a-solver` and {doc}`multilayer-theta-search`.
