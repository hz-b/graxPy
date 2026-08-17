# Solver flow

This page follows the execution path from public API calls to the numerical
solver routines.

Two solvers sit at the end of that path. They share every step except the one
that turns a finite layer into an interface-response block, so the flow below is
identical for both; only step 12 branches. See
[Nevière differential method](neviere-theory.md) for what that branch does.

## Direct one-point workflow

1. The user creates a grating, usually {class}`grax.LaminarGrating` or
   {class}`grax.BlazedGrating`.
2. The user calls {func}`grax.run_simulation` with the grating,
   selected diffraction order, Fourier order count, and grazing angle.
   Roughness can be attached to the grating with `RoughnessSpec`, while the
   legacy `roughness_sigma_nm` solver argument remains available.
3. {func}`grax.run_simulation` converts photon energy to
   wavelength with `1239.8 / photon_energy_ev`.
4. `run_simulation()` computes `k_parallel` from the grazing angle.
5. `run_simulation()` calls `grating.build_textures(photon_energy_ev, n_inc=1.0 + 0.0j)`.
6. `build_textures()` resolves the coating stack and substrate material at the
   requested photon energy.
7. `build_textures()` creates the `x` grid, `z` grid, interpolated surface
   profile, optional grating-level rough interfaces, refractive-index grid,
   texture list, and RETICOLO-style profile arrays.
8. `run_simulation()` calls `res0(1)` to create the parameter bundle.
9. `run_simulation()` calls `res1(wavelength, period, textures, fourier_orders, k_parallel, parm)`.
10. `res1()` normalizes diffraction orders and converts raw textures into
    `Texture1D` objects with Fourier coefficients.
11. `run_simulation()` calls `res2(aa, profile, parm, roughness_sigma_nm=...)`,
    or `res2_dm(...)` when `solver="neviere"`.
12. The chosen solver validates the profile, compresses adjacent identical
    textures, builds one interface-response block per layer, and hands them to
    the shared `solve_stack_from_layer_blocks()`. RCWA builds each block by
    eigen-decomposing the layer operator; the differential method builds it by
    integrating the equivalent first-order system in `z`.
13. `res2()` optionally applies scalar Debye-Waller roughness damping for
    solver-level roughness.
14. `run_simulation()` finds the requested reflected diffraction order, validates
    reflected efficiencies when enabled, and returns selected-order and
    all-order arrays.

## Energy sweep workflow

Lazy helpers such as {func}`grax.fixed_angle_cases`,
{func}`grax.monochromator_cases`, and {func}`grax.energy_angle_cases`
generate stable case dictionaries. {class}`grax.BatchSimulationRunner`
streams each case to {func}`grax.run_simulation` and yields each
{class}`grax.CaseExecutionResult` immediately.

## Batch workflow

{class}`grax.BatchSimulationRunner` is an orchestration layer for many
one-point cases.

1. The user builds case dictionaries. Each case must provide `grating`,
   `energy_ev`, and `grazing_angle_deg`.
2. {meth}`grax.BatchSimulationRunner.run_cases` optionally loads an
   existing checkpoint when `resume=True`.
3. The runner saves metadata when metadata is provided.
4. For each case, the runner checks that `grating` derives from
   {class}`grax.BaseGrating`.
5. The runner clones the grating and applies optional per-case `x_resolution_nm`
   and `z_resolution_nm` overrides.
6. The runner resolves `fourier_orders`, `diffraction_order`, and
   `roughness_sigma_nm` from the case or runner defaults.
7. The runner prepares a generic per-case payload.
8. The payload is executed inline by default or in a spawned subprocess when requested.
9. The runner stores the result in a {class}`grax.CaseExecutionResult`.
10. Each case result is yielded immediately as a {class}`grax.CaseExecutionResult`.

If `on_error="continue"`, failures are stored as case results with
`status="error"`. If `on_error="fail_fast"`, the original exception is raised.

## Validation and failure points

Important runtime checks include:

- Case dictionaries must contain a grating derived from `BaseGrating`.
- `run_simulation` rejects negative roughness values and rejects simultaneous
  grating-level/API roughness plus legacy `roughness_sigma_nm`.
- Reflected efficiencies are checked for negative values, excessive single-order
  efficiency, and excessive total propagating reflected efficiency.
- Collected export or plotting helpers should validate order-grid assumptions when they need rectangular arrays.
- `res2()` rejects invalid profile arrays.
- The native Python path currently rejects unsupported dimensions and
  polarizations.
- `run_simulation` rejects an unknown `solver` name before doing any work.

## Memory-mode policy

Low-memory is the only intended user-facing simulation mode. Public entrypoints
default to the low-memory path and do not expose a user choice between solver
memory strategies.

The dense path remains only as an internal `legacy_dense` baseline for
regression tests and developer debugging. It is not expected to be maintained
as a public performance option.

## Internal solver reference

These functions are developer-facing numerical entry points:

```{autofunction} grax.solvers.common.res0
```

```{autofunction} grax.solvers.common.res1
```

```{autofunction} grax.solvers.rcwa.res2
```

```{autofunction} grax.solvers.neviere.res2_dm
```

```{autoclass} grax.solvers.common.Res1Result
:members:
```

```{autoclass} grax.solvers.common.Res2Result
:members:
```

```{autoclass} grax.solvers.common.DiffractionResult
:members:
```

The historical `grax.rcwa_1d` import path still resolves all of these names.
