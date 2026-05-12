# Monochromator sweep

Use {func}`grax.monochromator_cases` to generate cases for a fixed-cff
monochromator scan. The grazing angle is automatically calculated from the
monochromator relation for each energy.

```python
import numpy as np
import grax as rp
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
gold = xrt_materials.Material("Au", rho=19.3, table="Henke", name="Au")

grating = rp.BlazedGrating(
    period_lpermm=600,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    blaze_angle_deg=0.75,
    anti_blaze_angle_deg=None,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energies = np.linspace(50.0, 500.0, 45)
cases = rp.monochromator_cases(
    grating=grating,
    energies_ev=energies,
    diffraction_order=1,
    cff=2.25,
    case_id_prefix="mono-blazed",
)

runner = rp.BatchSimulationRunner(
    default_fourier_orders=25,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    on_error="continue",
)

results = list(runner.run_cases(cases))
rp.write_all_orders_csv(results, "examples/simulation/monochromator_sweep/results/monochromator_all_orders.csv")
rp.plot_order_subset(
    results,
    "examples/simulation/monochromator_sweep/results/monochromator_orders_1_3.png",
    diffraction_orders=[1, 2, 3],
    title="Monochromator Sweep: Orders 1-3 Efficiency vs Energy",
)
```

The `cff` (constant-focus factor) parameter controls the geometry:

- `cff = 2.25` is the standard value for many synchrotron beamlines
- The grazing angle is computed from the monochromator equation:
  `sin(alpha) = (cff * lambda * period) / (2 * d_source)` where `d_source` is implicit

Quick mode for fast testing:

```python
# --quick mode: 10 energy points, 5 Fourier orders
energies = np.linspace(100.0, 300.0, 10)
default_fourier = 5
```

The example also produces an energy-vs-efficiency plot for diffraction
orders 1, 2, and 3:

```{image} images/simulation/monochromator_orders_1_3.png
:alt: Monochromator sweep efficiencies for diffraction orders 1 to 3
:align: center
:width: 80%
```

Full mode uses the production settings:

```python
# Full mode: 45 energy points, 25 Fourier orders
energies = np.linspace(50.0, 500.0, 45)
default_fourier = 25
```

See `examples/simulation/monochromator_sweep/monochromator_sweep.py` for the complete script.


For batch execution details, see {doc}`batch-simulations`, then
{doc}`multiprocessing` and {doc}`checkpoints-and-resume`.
