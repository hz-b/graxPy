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
runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=25,
    show_progress=True,
    on_error="continue",
)

results = list(runner.run_cases(cases))
```

`total_cases` is optional. If omitted, the runner infers it automatically when
the cases object is countable.

```{toctree}
:maxdepth: 1

multiprocessing
checkpoints-and-resume
```
