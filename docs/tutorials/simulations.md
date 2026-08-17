# Simulations

This section groups the core simulation workflows.

All simulation entrypoints let you set `polarization` explicitly. The accepted
values are `s` and `p`; the maintained runnable examples in this section set
`polarization="p"` on purpose rather than relying on defaults.

For fixed-angle roughness studies, use
`examples/simulation/fixed_angle_roughness/roughness_kind_comparison.py` or
`examples/simulation/fixed_angle_roughness/roughness_correlation.py`.

```{toctree}
:maxdepth: 1

single-simulation
batch-simulations
sweep-recipes
roughness
user-defined-cases
export-and-plot-results
```
