# Physics to code

The table maps the physical model to the implementation. Use it as the first
orientation point when changing solver behavior.

| Physics concept | Primary class/function | Module | Where it enters the solve |
| --- | --- | --- | --- |
| Grating period | `BaseGrating.period_nm` | `grax.gratings` | Converts lines per mm to the period passed to `res1()` |
| Surface profile | `LaminarGrating.profile_points()`, `BlazedGrating.profile_points()` | `grax.gratings` | Defines substrate height over one period before grid discretization |
| Discretized structure | `BaseGrating.build_textures()` | `grax.gratings` | Builds `textures` and RETICOLO-style `profile` arrays for `res1()` and `res2()` |
| Material refractive index | Internal material-resolution helpers | `grax.materials` | Resolves each material at the current photon energy |
| DataFrame optical constants | DataFrame inputs, internal interpolation path | `grax.materials` | Supplies interpolated `n = 1 - delta + i beta` values |
| Coating sequence | `SingleLayerStack`, `MultilayerStack` | `grax.stacks` | Defines material layers above the substrate surface |
| Direct simulation | `run_simulation` | `grax.simulation` | Converts user settings into wavelength, wave vector, textures, and solver calls |
| Batch orchestration | `BatchSimulationRunner` | `grax.simulation` | Streams case dictionaries through the single-case solver and yields `CaseExecutionResult` values |
| Fourier texture conversion | `res1()` | `grax.solvers` | Converts material textures into Fourier-space texture data |
| Stack solution | `res2()` | `grax.solvers` | Solves reflected and transmitted diffraction orders |
| Diffraction outputs | `DiffractionResult`, `Res2Result` | `grax.solvers` | Carries orders, angles, amplitudes, and efficiencies back to `simulation.py` |

## Public boundary

The practical public boundary is `grax.__init__`. User-facing docs should
prefer symbols exported there. The `grax.solvers` module remains important
for contributors, but it is not the first interface users should learn.

## Geometry and materials

`gratings.py` owns the profile and discretization. A grating resolves its stack,
builds the `x` and `z` grids, interpolates the surface profile onto the grid,
and creates a refractive-index grid for the requested photon energy.

`materials.py` owns material resolution. It deliberately accepts flexible
material inputs, but all supported inputs collapse to a scalar complex
refractive index at one photon energy.

`stacks.py` owns coating order. `SingleLayerStack` is the implicit default when
a grating is configured with shortcut layer fields. `MultilayerStack` is used
when the coating itself is periodic in depth.

## Numerical core

`rcwa_1d.py` contains the native solver data structures and routines. The
high-level API calls only `res0()`, `res1()`, and `res2()` directly. Helper
functions below those entry points are implementation details unless a solver
change requires documenting a new contributor contract.
