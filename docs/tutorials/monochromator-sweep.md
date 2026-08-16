# Monochromator sweep

Use {func}`grax.monochromator_cases` to generate cases for a fixed-cff
monochromator scan. The grazing angle is automatically calculated from the
monochromator relation for each energy.

```python
import numpy as np
import grax

grating = grax.BlazedGrating(
    period_lpermm=600,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    blaze_angle_deg=0.75,
    anti_blaze_angle_deg=None,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energies = np.linspace(50.0, 500.0, 45)
cases = grax.monochromator_cases(
    grating=grating,
    energies_ev=energies,
    diffraction_order=1,
    cff=2.25,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    fourier_orders=25,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
)

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, "examples/simulation/monochromator_sweep/results/monochromator_all_orders_rcwa.csv")
grax.plot_order_subset(
    results,
    "examples/simulation/monochromator_sweep/results/monochromator_orders_1_3_rcwa.png",
    diffraction_orders=[1, 2, 3],
    title="Monochromator Sweep: Orders 1-3 Efficiency vs Energy",
)
```

The `cff` (constant-focus factor) parameter controls the geometry:

This maintained example sets `polarization="p"` explicitly rather than relying
on the default simulation polarization. Accepted values are `s` and `p`.

- `cff = 2.25` is the standard value for many synchrotron beamlines
- The grazing angle is computed from the monochromator equation:
  `sin(alpha) = (cff * lambda * period) / (2 * d_source)` where `d_source` is implicit

The example also produces an energy-vs-efficiency plot for diffraction
orders 1, 2, and 3:

```{image} images/simulation/monochromator_orders_1_3.png
:alt: Monochromator sweep efficiencies for diffraction orders 1 to 3
:align: center
:width: 80%
```

See `examples/simulation/monochromator_sweep/monochromator_sweep.py` for the complete script.


For batch execution details, see {doc}`batch-simulations`, then
{doc}`multiprocessing` and {doc}`checkpoints-and-resume`.
