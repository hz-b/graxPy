# Fixed-angle energy sweep

Use {func}`grax.fixed_angle_cases` to generate a lazy iterator of
fixed-angle energy sweep cases, then stream them through {class}`grax.BatchSimulationRunner`.

```python
import numpy as np
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
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energies = np.linspace(50.0, 650.0, 60)
cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    diffraction_order=1,
    fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
    backend="numba",
)

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, "examples/simulation/fixed_angle_sweep/results/fixed_angle_all_orders.csv")
grax.plot_order_subset(
    results,
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_orders_1_3.png",
    diffraction_orders=[1, 2, 3],
    title="Fixed-Angle Sweep: Orders 1-3 Efficiency vs Energy",
)
```

This example uses the new local Henke-backed string material path. If you
already have xrt material objects or a DataFrame with ``Energy(eV)``,
``Delta``, and ``Beta``, those inputs remain supported as well.

The example also produces an energy-vs-efficiency plot for diffraction
orders 1, 2, and 3:

This maintained example also sets `polarization="p"` explicitly for every
generated case. Accepted polarization values are `s` and `p`.

```{image} images/simulation/fixed_angle_orders_1_3.png
:alt: Fixed-angle sweep efficiencies for diffraction orders 1 to 3
:align: center
:width: 80%
```

See `examples/simulation/fixed_angle_sweep/fixed_angle_sweep.py` for the complete script.

For a roughness-specific variant, see
`examples/simulation/fixed_angle_roughness/fixed_angle_roughness.py`. That
workflow reruns the same fixed-angle laminar sweep for `sigma=0.0`, `0.5`,
`1.0`, and `2.0 nm`, enables live plotting during each run, and then writes a
combined first-order comparison figure. The current roughness implementation is
a per-case scalar Debye-Waller damping model, so the example is intended to
show how that maintained model changes the spectrum across energy.

```{image} images/simulation/fixed_angle_roughness_order1_comparison.png
:alt: First-order fixed-angle roughness comparison for four Debye-Waller roughness levels
:align: center
:width: 80%
```


For batch execution details, see {doc}`batch-simulations`, then
{doc}`multiprocessing` and {doc}`checkpoints-and-resume`.
