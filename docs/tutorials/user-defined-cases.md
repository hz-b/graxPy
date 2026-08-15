# User-defined cases

Use custom case dictionaries when you want to vary geometry and keep full
control over what each case contains.

A practical example is a depth sweep for one laminar grating design.

```python
import numpy as np
import grax

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

energy_ev = 1000.0
diffraction_order = 1
cff = 2.25
depths_nm = np.arange(10.0, 31.0, 1.0)

grazing_angle_deg = float(
    grax.monochromator_grazing_angles_deg(
        [energy_ev],
        period_lpermm=base_grating_kwargs["period_lpermm"],
        diffraction_order=diffraction_order,
        cff=cff,
    )[0]
)

cases = []
for depth_nm in depths_nm:
    grating = grax.LaminarGrating(depth_nm=float(depth_nm), **base_grating_kwargs)
    cases.append(
        {
            "case_id": f"user-laminar-depth-{int(depth_nm):03d}",
            "label": f"Laminar depth {depth_nm:.1f} nm",
            "grating": grating,
            "energy_ev": energy_ev,
            "grazing_angle_deg": grazing_angle_deg,
            "polarization": "p",
            "depth_nm": float(depth_nm),
        }
    )

runner = grax.BatchSimulationRunner(
    diffraction_order=diffraction_order,
    fourier_orders=25,
    show_progress=True,
    on_error="continue",
)

results = list(runner.run_cases(cases))
grax.write_all_orders_csv(results, "results/batch_user_cases_all_orders.csv")
```

Plotting efficiency versus depth is then straightforward because each case keeps
its own `depth_nm` in `case_data`.

The maintained example also sets `polarization="p"` explicitly on each case so
the depth sweep does not depend on the default polarization. Accepted values
are `s` and `p`.

```{image} images/simulation/batch_user_cases_orders_1_3_vs_depth.png
:alt: User-defined laminar depth sweep, orders 1 to 3 versus depth
:align: center
:width: 80%
```

See `examples/simulation/batch_user_cases/batch_user_cases.py` for the complete runnable script.


For batch execution details, see {doc}`batch-simulations`, then
{doc}`multiprocessing` and {doc}`checkpoints-and-resume`.
