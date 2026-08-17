# Single simulation

The smallest complete workflow is:

1. Build a grating using built-in Henke material symbols.
2. Call {func}`grax.run_simulation` for one energy and grazing angle.
3. Read the typed {class}`grax.SingleSimulationResult`.

```python
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
    x_resolution_nm=1.0,
    z_resolution_nm=0.1,
)

result = grax.run_simulation(
    grating=grating,
    energy_ev=200.0,
    grazing_angle_deg=4.0,
    diffraction_order=1,
    fourier_orders=5,
    polarization="p",
)

print(result.selected_efficiency)
```

The result contains both the selected order and all reflected orders:

```python
orders = result.orders
selected_efficiency = result.selected_efficiency
all_efficiencies = result.efficiency_all
```

This maintained example sets `polarization="p"` explicitly so the script does
not rely on the library default.

Accepted values are `s` and `p`, plus the aliases `TE` and `TM` — `s` is TE and
`p` is TM. The alias is resolved on the way in, so a result always reports the
canonical `s` or `p` whichever spelling you passed. The two names coincide
because this solver is one-dimensional and classical; see
{func}`grax.normalize_polarization` for why that qualifier matters.

To save a profile plot before or after running the simulation:

```python
grating.plot_profile("examples/simulation/single_simulation/results/single_simulation_profile.png")
```

Export helpers accept the typed single result directly:

```python
grax.write_all_orders_csv(result, "examples/simulation/single_simulation/results/single_simulation_rcwa.csv")
```

See `examples/simulation/single_simulation/single_simulation.py` for the complete script.
