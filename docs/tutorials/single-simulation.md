# Single simulation

The smallest complete workflow is:

1. Build a grating using xrt materials.
2. Call {func}`grax.run_simulation` for one energy and grazing angle.
3. Read the typed {class}`grax.SingleSimulationResult`.

```python
from xrt.backends.raycing import materials as xrt_materials
import grax

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
    x_resolution_nm=1.0,
    z_resolution_nm=0.1,
)

result = grax.run_simulation(
    grating=grating,
    energy_ev=200.0,
    grazing_angle_deg=4.0,
    diffraction_order=1,
    fourier_orders=5,
)

print(result.selected_efficiency)
```

The result contains both the selected order and all reflected orders:

```python
orders = result.orders
selected_efficiency = result.selected_efficiency
all_efficiencies = result.efficiency_all
```

To save a profile plot before or after running the simulation:

```python
grating.plot_profile("examples/simulation/single_simulation/results/single_simulation_profile.png")
```

Export helpers accept the typed single result directly:

```python
grax.write_all_orders_csv(result, "examples/simulation/single_simulation/results/single_simulation.csv")
```

See `examples/simulation/single_simulation/single_simulation.py` for the complete script.
