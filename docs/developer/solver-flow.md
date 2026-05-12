# Solver flow

This page follows the execution path from public API calls to the numerical
solver routines.

## Direct one-point workflow

1. The user creates a grating, usually {class}`grax.LaminarGrating` or
   {class}`grax.BlazedGrating`.
2. The user calls {func}`grax.run_simulation` with the grating,
   selected diffraction order, Fourier order count, grazing angle, and
   optional roughness.
3. {func}`grax.run_simulation` converts photon energy to
   wavelength with `1239.8 / photon_energy_ev`.
4. `run_simulation()` computes `k_parallel` from the grazing angle.
5. `run_simulation()` calls `grating.build_textures(photon_energy_ev, n_inc=1.0 + 0.0j)`.
6. `build_textures()` resolves the coating stack and substrate material at the
   requested photon energy.
7. `build_textures()` creates the `x` grid, `z` grid, interpolated surface
   profile, refractive-index grid, texture list, and RETICOLO-style profile
   arrays.
8. `run_simulation()` calls `res0(1)` to create the parameter bundle.
9. `run_simulation()` calls `res1(wavelength, period, textures, fourier_orders, k_parallel, parm)`.
10. `res1()` normalizes diffraction orders and converts raw textures into
    `Texture1D` objects with Fourier coefficients.
11. `run_simulation()` calls `res2(aa, profile, parm, roughness_sigma_nm=...)`.
12. `res2()` validates the profile, compresses adjacent identical textures, and
    solves the TE stack.
13. `res2()` optionally applies scalar Debye-Waller roughness damping.
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
- `run_simulation` rejects negative roughness values.
- Reflected efficiencies are checked for negative values, excessive single-order
  efficiency, and excessive total propagating reflected efficiency.
- Collected export or plotting helpers should validate order-grid assumptions when they need rectangular arrays.
- `res2()` rejects invalid profile arrays.
- The native Python path currently rejects unsupported dimensions and
  polarizations.

## Internal solver reference

These functions are developer-facing numerical entry points:

```{autofunction} grax.rcwa_1d.res0
```

```{autofunction} grax.rcwa_1d.res1
```

```{autofunction} grax.rcwa_1d.res2
```

```{autoclass} grax.rcwa_1d.Res1Result
:members:
```

```{autoclass} grax.rcwa_1d.Res2Result
:members:
```

```{autoclass} grax.rcwa_1d.DiffractionResult
:members:
```
