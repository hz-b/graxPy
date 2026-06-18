# Batch simulations

Use {class}`grax.BatchSimulationRunner` when you need to execute many
simulation cases with one consistent workflow.

Typical use cases:

- fixed-angle sweeps
- monochromator sweeps
- energy-angle pair sweeps
- custom user-defined case lists

Minimal pattern:

```python
runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=25,
    default_polarization="p",
    show_progress=True,
    on_error="continue",
)

results = list(runner.run_cases(cases))
```

The `cases` iterable is a list of dictionaries. Each dictionary must provide at
least:

- `grating`: the fully configured grating object to simulate
- `energy_ev`: the photon energy for that case

Common optional keys are `case_id`, `grazing_angle_deg`, `diffraction_order`,
`fourier_orders`, and `polarization`. Any extra serializable metadata such as
`label` or `depth_nm` is preserved in each result's `case_data`, which makes
downstream plotting and CSV export easier.

When you set `polarization`, the accepted values are `s` and `p`. The examples
here use explicit `p` polarization so batch runs do not depend on defaults.

Example shape:

```python
cases = [
    {
        "case_id": "depth-010",
        "label": "Laminar depth 10 nm",
        "grating": grax.LaminarGrating(depth_nm=10.0, **base_grating_kwargs),
        "energy_ev": 1000.0,
        "grazing_angle_deg": grazing_angle_deg,
        "diffraction_order": 1,
        "fourier_orders": 25,
        "polarization": "p",
        "depth_nm": 10.0,
    },
    {
        "case_id": "depth-011",
        "label": "Laminar depth 11 nm",
        "grating": grax.LaminarGrating(depth_nm=11.0, **base_grating_kwargs),
        "energy_ev": 1000.0,
        "grazing_angle_deg": grazing_angle_deg,
        "diffraction_order": 1,
        "fourier_orders": 25,
        "polarization": "p",
        "depth_nm": 11.0,
    },
]
```

For a full runnable example that builds these dictionaries programmatically, see
{doc}`user-defined-cases`.

```{toctree}
:maxdepth: 1

multiprocessing
checkpoints-and-resume
```
