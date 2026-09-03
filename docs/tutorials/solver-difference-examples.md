# Where the two solvers differ

GraxPy's modal RCWA and Nevière differential-method paths solve the same
one-dimensional grating problem through different layer-propagation methods.
The general comparison example sweeps energy for a representative grating and
compares the selected diffraction-order efficiency from both paths.

```bash
python examples/simulation/neviere_solver/neviere_solver.py
```

The example writes the comparison plot and one CSV per solver to
`examples/simulation/neviere_solver/results/`.

```{figure} images/simulation/neviere_solver_comparison.png
:alt: Energy sweep comparing RCWA and Nevière diffraction efficiencies, with their absolute difference below.
:width: 100%

The upper panel compares the selected-order efficiencies from RCWA and
Nevière across the energy sweep. The lower panel shows their absolute
difference.
```

The examples below show three situations in which the solver choice changes
what can be represented, how the profile is sampled, or the cost of a solve.
For general solver-selection guidance, see {doc}`choosing-a-solver`.

## Deep-grating range

```bash
python examples/simulation/deep_grating_limits/deep_grating_limits.py
```

This RETICOLO reference geometry sweeps a high-contrast lamellar grating from
a fraction of a wavelength to deep profiles. The modal RCWA calculation can
overflow when evanescent orders make its whole-layer propagation ill
conditioned; the differential method propagates bounded sub-blocks and
cascades their interface responses.

```{figure} images/simulation/deep_grating_limits_p.png
:alt: Zeroth-order transmission and energy-balance error as a deep grating is solved by RCWA and Nevière.
:width: 100%

The upper panel marks where the modal calculation stops. The lower panel shows
the energy-balance error for the available results.
```

## Staircase and continuous sampling

```bash
python examples/simulation/continuous_vs_staircase/continuous_vs_staircase.py
```

Both solvers use the same z-sliced texture by default. With
`NeviereOptions(z_sampling="continuous")`, the differential method instead
re-expands the profile along z, removing the shared staircase approximation.

```{figure} images/simulation/continuous_vs_staircase_p.png
:alt: Staircase-convergence study comparing RCWA, Nevière texture sampling, and Nevière continuous sampling.
:width: 100%

The upper panel compares selected-order efficiencies as the z resolution is
refined. The lower panel measures each staircase result against the continuous
calculation.
```

## Runtime

```bash
python examples/simulation/solver_runtime/solver_runtime.py
```

The default command records reduced-resolution timings; add `--full` to
reproduce the checked-in production-resolution comparison. Each timing is
reported with the maximum all-order efficiency difference, so runtime can be
read alongside agreement between the two paths.

```{figure} images/simulation/solver_runtime_full.png
:alt: Production-resolution runtime comparison of RCWA and Nevière for three representative gratings.
:width: 100%

The bars show seconds per energy point for the checked-in production-resolution
benchmark cases.
```
