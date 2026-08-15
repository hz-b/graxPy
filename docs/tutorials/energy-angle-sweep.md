# Energy-angle sweep (Multilayer)

Use {func}`grax.energy_angle_cases` to run an energy-angle
sweep from a predefined set of energy/angle pairs (normally useful for multilayers).

This tutorial reads local pairs from `energy_angle_pairs.dat`, then samples one
row every 50 points for faster execution.

- Input file: `examples/simulation/energy_angle_sweep/energy_angle_pairs.dat`
- Sampling: every 50 rows
- `x_resolution_nm = 1.0`
- `z_resolution_nm = 1.0`
- `fourier_orders = 5`

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import grax

example_root = Path("examples/simulation/energy_angle_sweep")
input_path = example_root / "energy_angle_pairs.dat"
output_dir = example_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)

reference_data = pd.read_csv(input_path, sep=r"\s+", engine="python")
sampled_reference = reference_data.iloc[::50].copy()

energy_angle_pairs = list(
    zip(
        sampled_reference["Energy"].to_numpy(dtype=float),
        sampled_reference["alpha"].to_numpy(dtype=float),
    )
)

multilayer_stack = grax.MultilayerStack(
    substrate_material="Si",
    material_a="Cr",
    material_b="C",
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material="C",
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=1.0,
    z_resolution_nm=1.0,
)

cases = grax.energy_angle_cases(
    grating=grating,
    energy_angle_pairs=energy_angle_pairs,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    diffraction_order=2,
    fourier_orders=5,
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
)

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, output_dir / "energy_angle_multilayer_all_orders.csv")

successful = [result for result in results if result.status == "ok"]
figure, axis = plt.subplots(figsize=(10, 6))
axis.plot(
    [result.energy_ev for result in successful],
    [result.selected_efficiency for result in successful],
    "o-",
    linewidth=1.0,
    markersize=2.0,
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency (2nd order)")
axis.set_title("Fast Multilayer Energy-Angle Sweep (grax)")
axis.grid(True, alpha=0.3)
figure.tight_layout()
figure.savefig(output_dir / "energy_angle_multilayer_fast.png", dpi=150, bbox_inches="tight")
plt.close(figure)
```

```{image} images/simulation/energy_angle_multilayer_fast.png
:alt: Fast multilayer energy-angle sweep efficiency from sampled energy-angle pairs
:align: center
:width: 80%
```

See `examples/simulation/energy_angle_sweep/energy_angle_sweep.py` for the full runnable script.

Like the maintained runnable script, this tutorial sets `polarization="p"`
explicitly for the sampled multilayer cases. Accepted values are `s` and `p`.
