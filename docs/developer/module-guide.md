# Module guide

This guide summarizes the source layout for contributors.

## Core package

`grax` is the simulation package documented for users.

- `gratings.py`: grating profiles, grid construction, texture/profile building,
  profile plotting, and structure debug exports
- `materials.py`: material labels, `MaterialSpec`, packaged Henke-table string
  lookup, DataFrame optical constants, xrt-like compatibility support, and
  refractive-index resolution
- `stacks.py`: coating stack models above the substrate surface
- `simulation/`: simulation workflows and orchestration (see package organization)
  - `models.py`: typed result dataclasses and small compatibility containers
  - `core.py`: one-point RCWA execution and legacy plotting/comparison helpers
  - `batch.py`: generic batch execution, checkpointing, subprocess execution, and worker calibration helpers
  - `cases.py`: lazy case-generation helpers and monochromator angle generation
  - `theta_search.py`: single-energy multilayer theta-search logic and scan helpers
  - `theta_search_sweep.py`: multi-energy adaptive theta-search sweep workflow and artifact writing
  - `serialization.py`: JSON/checkpoint serialization for single and case results
- `parameter_sweep.py`: repeated simulations for numerical convergence and
  parameter studies
- `rcwa_1d.py`: native one-dimensional TE-style RCWA numerical core
- `roughness.py`: roughness post-processing models and diagnostics (currently
  scalar Debye-Waller damping used by `rcwa_1d.res2()`)
- `slag.py`: legacy SLAG convenience helpers built on the generic public API

## Optimization package

`grax_opt` is a companion package for Ax-based fitting workflows. It is
not part of the first-class user documentation set for the core simulation
package. Keep optimization-specific user material separate unless the core docs
need to mention installation of the optional `opt` extra.

- `config.py`: `ParameterBounds`, the shared bounds value type
- `data.py`: measurement loading and interpolation onto an evaluation grid
- `evaluation.py`: normalization of evaluation energies/angles for the
  single-measurement optimizer
- `objective.py`: trial evaluation, loss functions, joint multi-angle
  evaluation, and the failure-penalty path
- `optimize.py`: Ax imports and version sniffing, candidate batching, trial
  records, plotting, and atomic artifact writes
- `loop.py`: the Ax ask-and-tell loop shared by both optimizer entrypoints,
  including best-so-far tracking and early stopping
- `dynamic.py`: `MeasurementFitConfig` and `optimize_to_measurements`, the
  single-measurement fit
- `joint.py`: `JointMeasurementFitConfig` and `optimize_to_joint_measurements`,
  the multi-angle fit
- `checkpoint.py`: checkpoint layout, problem fingerprinting, and the
  resume session used by both entrypoints

Both optimizer entrypoints share `loop.py` and `checkpoint.py`; mode-specific
work is limited to config validation, candidate evaluation, and artifact
persistence. Add new optimizer modes the same way rather than duplicating the
trial loop.

## Where to make changes

Add new grating shapes in `gratings.py` when they need to participate in the
same `BaseGrating.build_textures()` pipeline.

Add new coating models in `stacks.py` when the layer sequence above the
substrate changes but the grating profile machinery can stay unchanged.

Add new supported material input types in `materials.py` when they can resolve
to a scalar complex refractive index at a requested photon energy.

Change `simulation/` when user-facing orchestration, result packaging,
checkpointing, plotting, or validation behavior changes.

Change `rcwa_1d.py` when the Fourier conversion, stack solve, polarization
support, or diffraction-output calculation changes. Change `roughness.py` when
roughness model math or diagnostics change.

## Reading order for new maintainers

1. `examples/manual_single_simulation_example/manual_single_simulation.py`
2. `examples/simulation/fixed_angle_sweep/fixed_angle_sweep.py`
3. `src/grax/simulation/` (package overview in docs/api/simulation/package-organization.md)
4. `src/grax/gratings.py`
5. `src/grax/materials.py` and `src/grax/stacks.py`
6. `src/grax/rcwa_1d.py`
