# Fixed-angle energy sweep

Use {func}`grax.fixed_angle_cases` to generate a lazy iterator of
fixed-angle energy sweep cases, then stream them through {class}`grax.BatchSimulationRunner`.

```python
import numpy as np
import grax
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energies = np.linspace(50.0, 650.0, 60)
cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=25,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
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

The example also produces an energy-vs-efficiency plot for diffraction
orders 1, 2, and 3:

```{image} images/simulation/fixed_angle_orders_1_3.png
:alt: Fixed-angle sweep efficiencies for diffraction orders 1 to 3
:align: center
:width: 80%
```

See `examples/simulation/fixed_angle_sweep/fixed_angle_sweep.py` for the complete script.


For batch execution details, see {doc}`batch-simulations`, then
{doc}`multiprocessing` and {doc}`checkpoints-and-resume`.
