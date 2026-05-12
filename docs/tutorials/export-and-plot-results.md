# Export And Plot Results

Use the built-in helpers to export simulation outputs to CSV and create
standard efficiency plots.

## 1) Export all orders to CSV

`grax.write_all_orders_csv(...)` accepts:

- a single simulation result (`SingleSimulationResult`)
- one case result (`CaseExecutionResult`)
- an iterable of case results (for example `list(runner.run_cases(...))`)

```python
import grax as rp

# results can come from run_simulation(), runner.run_cases(), or a single case
rp.write_all_orders_csv(
    results,
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_all_orders.csv",
)
```

## 2) CSV format

The exported CSV has one row per diffraction order per successful case, with
this header:

```text
case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg
```

Column meaning:

- `case_id`: case identifier from the batch case generator
- `energy_ev`: photon energy in eV
- `grazing_angle_deg`: grazing angle in degrees used for the case
- `order`: diffraction order index from RCWA output (can be negative/zero/positive)
- `efficiency`: diffraction efficiency for that order
- `diffraction_angle_deg`: diffraction angle for that order in degrees

## 3) Plot selected orders with the predefined helper

Use `grax.plot_order_subset(...)` to generate a ready-to-use
efficiency-vs-energy figure for selected positive diffraction orders:

```python
import grax as rp

rp.plot_order_subset(
    results,
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_orders_1_3.png",
    diffraction_orders=[1, 2, 3],
    title="Fixed-Angle Sweep: Orders 1-3 Efficiency vs Energy",
)
```

## 4) Minimal analysis with pandas

You can post-process exported CSV files directly:

```python
import pandas as pd

data = pd.read_csv(
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_all_orders.csv"
)

# Keep first diffraction order
order1 = data[data["order"] == -1].copy()

print(order1[["energy_ev", "efficiency"]].head())
```

`plot_order_subset(...)` internally applies the same RCWA order convention as
the simulation utilities when selecting orders.
