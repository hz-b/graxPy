# Roughness

Use `examples/simulation/fixed_angle_roughness/fixed_angle_roughness.py` to
compare the maintained Debye-Waller roughness model across several roughness
levels in a fixed-angle laminar sweep.

This example reuses the fixed-angle simulation pattern, but reruns the same
energy sweep for:

- `sigma=0.0 nm`
- `sigma=0.5 nm`
- `sigma=1.0 nm`
- `sigma=2.0 nm`

During execution, each roughness sweep uses the batch runner with:

- `live_plot=True`
- `live_plot_x_key="energy_ev"`
- `live_plot_order_count=1`
- `max_workers="auto"`

That means the first-order efficiency is plotted live while the cases are
running, and the example also uses multiprocessing through the maintained batch
execution path.

After the runs finish, the workflow writes one all-orders CSV per roughness
level and a combined comparison plot for diffraction order `1`.

The current roughness implementation is a scalar Debye-Waller damping model
applied per simulation case. It does not modify the grating geometry or RCWA
matrices; instead it scales the reflected/transmitted efficiencies after the
solve. The effect therefore varies with wavelength and grazing angle across the
sweep, while remaining uniform across orders within one single simulation case.

```{image} images/simulation/fixed_angle_roughness_order1_comparison.png
:alt: First-order fixed-angle roughness comparison for four Debye-Waller roughness levels
:align: center
:width: 80%
```

See these files for the complete workflow:

- `examples/simulation/fixed_angle_roughness/fixed_angle_roughness.py`
- `examples/simulation/fixed_angle_roughness/comparison_fixed_angle_roughness.py`

For the baseline fixed-angle workflow without the roughness sweep wrapper, see
{doc}`fixed-angle-sweep`.
