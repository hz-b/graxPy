# Choosing a solver

`grax` ships two independent electromagnetic solvers behind one interface. Every
entry point takes a `solver` argument and defaults to `"rcwa"`, so existing code
keeps its current behaviour.

```python
import grax

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    x_resolution_nm=1.0,
    z_resolution_nm=0.1,
)

common = dict(
    grating=grating,
    energy_ev=300.0,
    grazing_angle_deg=4.0,
    fourier_orders=30,
    polarization="p",
)

rcwa = grax.run_simulation(**common, solver="rcwa")
neviere = grax.run_simulation(**common, solver="neviere")

print(rcwa.selected_efficiency, neviere.selected_efficiency)
```

Each result records which solver produced it:

```python
assert neviere.solver == "neviere"
```

## Which one to use

`"rcwa"` remains the default and is the right choice unless you have a specific
reason to switch.

Reach for `"neviere"` when you want to:

- **Cross-check a result.** The two solvers share their Fourier operators and
  their efficiency extraction but propagate through the grating by completely
  different numerics, so agreement is meaningful evidence and disagreement
  localises quickly.
- **Model a deep grating.** The modal solver evaluates `q / sinh(q d)` across a
  whole layer and overflows above roughly seven wavelengths of depth for a
  high-contrast profile. The differential method caps the optical thickness of
  anything it forms explicitly and stays stable far beyond that.
- **Remove the staircase approximation.** With `z_sampling="continuous"` the
  permittivity is read from the true profile instead of the z-sliced
  approximation of it (see below).

For the shallow X-ray gratings in `validation/`, the two agree to about `1e-10`
in absolute efficiency, and the differential method runs two to five times
faster because it never eigen-decomposes a layer.

## Batch runs

The batch runner takes a default and each case can override it:

```python
runner = grax.BatchSimulationRunner(
    default_fourier_orders=30,
    default_solver="neviere",
)

cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=[200.0, 300.0, 400.0],
    grazing_angle_deg=4.0,
    polarization="p",
)

# Or per case, to run both solvers over the same sweep:
mixed = [dict(case, solver=solver) for case in cases for solver in ("rcwa", "neviere")]
```

The solver is stored in every {class}`grax.CaseExecutionResult` and round-trips
through checkpoints, so a resumed sweep keeps that provenance.

## Tuning the differential method

{class}`grax.NeviereOptions` controls the integration. The defaults are chosen to
put the residual against RCWA well below any physically meaningful level, so you
should rarely need to change them.

```python
result = grax.run_simulation(
    **common,
    solver="neviere",
    neviere_options=grax.NeviereOptions(step_phase=0.005),
)
```

| Option | Meaning |
| --- | --- |
| `step_phase` | Optical phase per Runge–Kutta step. Accuracy improves as its fourth power; cost grows far more slowly. |
| `block_phase` | Optical phase per explicitly formed transfer matrix. Affects conditioning only, not the converged answer. |
| `max_step_nm` | Optional hard cap on the step in nanometers, on top of `step_phase`. |
| `z_sampling` | `"textures"` (default) or `"continuous"`. |
| `energy_balance_tolerance` | Upper bound on the summed propagating efficiency; raises when exceeded. |

The Fourier truncation order is the ordinary `fourier_orders` argument, shared
with the RCWA solver. There is no separate setting for this solver.

A mapping works anywhere an options object does, which is convenient in case
dictionaries:

```python
case = dict(case, solver="neviere", neviere_options={"step_phase": 0.005})
```

## Continuous permittivity sampling

Both solvers normally slice the grating into layers of `z_resolution_nm` and
treat the permittivity as constant within each. That staircase is a
discretization error you pay for by refining `z_resolution_nm`, which makes every
solve more expensive.

The differential method can skip it:

```python
result = grax.run_simulation(
    **common,
    solver="neviere",
    neviere_options=grax.NeviereOptions(z_sampling="continuous"),
)
```

In this mode the permittivity is re-expanded from the true grating profile at
every sub-block, so the result does not depend on `z_resolution_nm` at all. On a
sinusoidal test profile the continuous result matched a staircase converged at
`z_resolution_nm = 0.005`, while the staircase solvers at `z_resolution_nm = 0.1`
were still an order of magnitude further away.

Use `"textures"` (the default) when you are comparing against RCWA, since it
guarantees both solvers see identical geometry. Use `"continuous"` when you want
the profile itself resolved rather than its staircase approximation.

See `examples/simulation/neviere_solver/neviere_solver.py` for a runnable script
covering all of the above, and
[Nevière differential method](../developer/neviere-theory.md) for the underlying
formulation.
