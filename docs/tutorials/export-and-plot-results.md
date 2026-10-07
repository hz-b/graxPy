# Export And Plot Results

Use the built-in helpers to export simulation outputs to CSV and create
standard efficiency plots.

## 1) Export all orders to CSV

`grax.write_all_orders_csv(...)` accepts:

- a single simulation result (`SingleSimulationResult`)
- one case result (`CaseExecutionResult`)
- an iterable of case results (for example `list(runner.run_cases(...))`)

```python
import grax

# results can come from run_simulation(), runner.run_cases(), or a single case
grax.write_all_orders_csv(
    results,
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_all_orders_rcwa.csv",
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
- `order`: diffraction order index. Positive orders are the inside orders (toward the grating normal from the specular beam), negative orders are the outside orders; see {doc}`../developer/rcwa-theory`
- `efficiency`: diffraction efficiency for that order
- `diffraction_angle_deg`: diffraction angle for that order in degrees

## 3) Plot selected orders with the predefined helper

Use `grax.plot_order_subset(...)` to generate a ready-to-use
efficiency-vs-energy figure for selected positive diffraction orders:

```python
import grax

grax.plot_order_subset(
    results,
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_orders_1_3_rcwa.png",
    diffraction_orders=[1, 2, 3],
    title="Fixed-Angle Sweep: Orders 1-3 Efficiency vs Energy",
)
```

## 4) Minimal analysis with pandas

You can post-process exported CSV files directly:

```python
import pandas as pd

data = pd.read_csv(
    "examples/simulation/fixed_angle_sweep/results/fixed_angle_all_orders_rcwa.csv"
)

# Keep first diffraction order
order1 = data[data["order"] == 1].copy()

print(order1[["energy_ev", "efficiency"]].head())
```

Exported orders use the standard grating-equation sign: positive orders are the
inside orders (diffracted toward the grating normal from the specular beam),
negative orders are the outside orders. `plot_order_subset(...)` and
`efficiency_for_order(...)` select orders with the same sign.
