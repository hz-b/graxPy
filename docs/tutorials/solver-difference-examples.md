# Where the two solvers differ

The RCWA and Nevière solvers agree to about 1e-11 on every validation case, so
most comparisons between them show two indistinguishable curves. Three examples
cover the cases where the choice actually matters.

Each lives under `examples/simulation/` and writes a plot and a CSV into its own
`results/` directory.

## Depth range

```bash
python examples/simulation/deep_grating_limits/deep_grating_limits.py
```

The modal solver treats each layer as z-invariant and evaluates `q / sinh(q d)`
across the whole layer. For a deep, high-contrast grating the evanescent orders
make `q d` large, `sinh` overflows, and the solve raises. The differential method
never forms that quantity: it caps the optical thickness of any transfer matrix
it builds and combines the pieces with an R-matrix cascade.

On the published RETICOLO `exemple1_1D` lamellar grating, sweeping groove depth:

| | deepest solve |
| --- | --- |
| RCWA (modal) | 8.4 wavelengths |
| Nevière (differential) | 167 wavelengths |

Where both work they agree to 3.3e-12, and the differential method's energy
balance stays within 4e-11 of one throughout.

None of the X-ray gratings in `validation/` come anywhere near this limit, so
this is a capability difference rather than a correction.

## Staircase versus continuous sampling

```bash
python examples/simulation/continuous_vs_staircase/continuous_vs_staircase.py
```

Both solvers normally see the profile as a staircase of z-slices, and that
approximation is why a converged run needs a fine `z_resolution_nm`.
`NeviereOptions(z_sampling="continuous")` re-expands the permittivity from the
true profile as it integrates, so its answer does not depend on
`z_resolution_nm` at all.

Sweeping a sinusoidal profile from 2 nm down to 0.05 nm slices:

- continuous sampling varies by **exactly zero** across the whole sweep
- the staircase carries 2.7e-3 of error at 2 nm, falling to 1.3e-5 at 0.05 nm
- the two staircase curves (RCWA and Nevière in `"textures"` mode) track each
  other to 6e-13, which is what makes the solvers comparable everywhere else

The flat line is the converged answer, so the gap to a staircase curve reads
directly as the discretization error that run is carrying.

## Runtime

```bash
python examples/simulation/solver_runtime/solver_runtime.py --full
```

The differential method is the faster of the two, which is not what "integrate an
ODE through every layer" suggests. The modal solver eigen-decomposes each
distinct layer operator; the differential method never does, using only matrix
products.

| resolution | speedup |
| --- | --- |
| reduced | 1.2x to 1.4x |
| production | 2.4x to 3.0x |

Coarse runs have few distinct layers, so the eigensolve is not yet dominant and
the two are close. The advantage appears in the regime where a sweep is
expensive — which also means a quick coarse benchmark understates it.

The example prints the maximum efficiency difference alongside every timing, so
the speed number can be read against the accuracy it was obtained at. It stays
around 1e-11.
