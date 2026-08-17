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
  - `core.py`: one-point solver dispatch and legacy plotting/comparison helpers
  - `batch.py`: generic batch execution, checkpointing, subprocess execution, and worker calibration helpers
  - `cases.py`: lazy case-generation helpers and monochromator angle generation
  - `theta_search.py`: single-energy multilayer theta-search logic and scan helpers
  - `theta_search_sweep.py`: multi-energy adaptive theta-search sweep workflow and artifact writing
  - `serialization.py`: JSON/checkpoint serialization for single and case results
- `parameter_sweep.py`: repeated simulations for numerical convergence and
  parameter studies
- `solvers/`: the one-dimensional electromagnetic solvers
  - `common.py`: shared types, `res0`/`res1`, the Fourier machinery, the layer
    field operators (including the Li/fast-Fourier-factorization rules for TM),
    the interface-response cascade, and the efficiency extraction
  - `rcwa.py`: modal (eigen-decomposed) layer solve and `res2`
  - `neviere.py`: Nevière differential method, `res2_dm`, and `NeviereOptions`
- `rcwa_1d.py`: re-export shim keeping the historical `grax.rcwa_1d` import path
  working. New code should import from `grax.solvers`
- `roughness.py`: roughness post-processing models and diagnostics (currently
  scalar Debye-Waller damping applied by both solvers)
- `slag.py`: legacy SLAG convenience helpers built on the generic public API

## Optimization package

`grax_opt` is a companion package for Ax-based fitting workflows. It is
not part of the first-class user documentation set for the core simulation
package. Keep optimization-specific user material separate unless the core docs
need to mention installation of the optional `opt` extra.

## Where to make changes

Add new grating shapes in `gratings.py` when they need to participate in the
same `BaseGrating.build_textures()` pipeline.

Add new coating models in `stacks.py` when the layer sequence above the
substrate changes but the grating profile machinery can stay unchanged.

Add new supported material input types in `materials.py` when they can resolve
to a scalar complex refractive index at a requested photon energy.

Change `simulation/` when user-facing orchestration, result packaging,
checkpointing, plotting, or validation behavior changes.

Change `solvers/common.py` when the Fourier conversion, the layer field
operators, the interface cascade, or the diffraction-output calculation changes:
that code is shared, so both solvers move together. Change `solvers/rcwa.py` or
`solvers/neviere.py` only when one solver's own propagation changes. Adding a
third solver means supplying a per-layer block builder to
`solve_stack_from_layer_blocks()` and extending `SOLVER_NAMES` in
`simulation/core.py`.

Change `roughness.py` when roughness model math or diagnostics change.

## Reading order for new maintainers

1. `examples/manual_single_simulation_example/manual_single_simulation.py`
2. `examples/simulation/fixed_angle_sweep/fixed_angle_sweep.py`
3. `src/grax/simulation/` (package overview in docs/api/simulation/package-organization.md)
4. `src/grax/gratings.py`
5. `src/grax/materials.py` and `src/grax/stacks.py`
6. `src/grax/solvers/` (start with `common.py`, then read `rcwa.py` and
   `neviere.py` as the two propagation implementations)
