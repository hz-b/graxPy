# How To Run User-Defined Case Lists

Use this guide when you want to keep simulation conditions fixed while varying a
single grating parameter.

This recipe builds user-defined cases at the **same energy and grazing angle**,
with **different laminar grating depths**.

## 1. Define fixed simulation conditions

```python
import numpy as np
import grax

energy_ev = 1000.0
grazing_angle_deg = 3.5
diffraction_order = 1
depths_nm = np.arange(10.0, 31.0, 1.0)

base_grating_kwargs = dict(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)
```

## 2. Build user-defined cases with varying depth

```python
cases = []
for depth_nm in depths_nm:
    grating = grax.LaminarGrating(depth_nm=float(depth_nm), **base_grating_kwargs)
    cases.append(
        {
            "label": f"Depth {depth_nm:.1f} nm",
            "grating": grating,
            "energy_ev": energy_ev,
            "grazing_angle_deg": grazing_angle_deg,
            "depth_nm": float(depth_nm),
        }
    )
```

All cases share the same `energy_ev` and `grazing_angle_deg`; only `depth_nm`
changes. `case_id` is optional; the runner will generate deterministic IDs when
omitted.

## 3. Run and export

```python
runner = grax.BatchSimulationRunner(
    default_diffraction_order=diffraction_order,
    default_fourier_orders=25,
    show_progress=True,
    on_error="continue",
)

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, "results/user_defined_depth_sweep_all_orders.csv")
```

Because each case stores `depth_nm`, you can directly plot efficiency versus
Depth after execution.

```{image} ../tutorials/images/simulation/batch_user_cases_orders_1_3_vs_depth.png
:alt: User-defined depth sweep at fixed energy and angle
:align: center
:width: 80%
```
