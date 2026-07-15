# Roughness

Use `examples/simulation/fixed_angle_roughness/fixed_angle_roughness.py` to
compare Debye-Waller and random-interface roughness models across several
roughness levels in a fixed-angle laminar sweep.

This example reuses the fixed-angle simulation pattern, but reruns the same
energy sweep for `debye-waller` and `random-interface` roughness at:

- `sigma=0.0 nm`
- `sigma=0.5 nm`
- `sigma=1.0 nm`
- `sigma=2.0 nm`

Roughness is selected when the grating is built:

```python
roughness=grax.RoughnessSpec(kind="debye-waller", sigma_nm=0.5)
roughness=grax.RoughnessSpec(kind="random-interface", sigma_nm=0.5, seed=0)
```

During execution, each roughness sweep uses the batch runner with:

- `live_plot=True`
- `live_plot_x_key="energy_ev"`
- `live_plot_order_count=1`
- `max_workers="auto"`

That means the first-order efficiency is plotted live while the cases are
running, and the example also uses multiprocessing through the maintained batch
execution path.

After the runs finish, the workflow writes one all-orders CSV per roughness
kind and level, plus a combined comparison plot for diffraction order `1`.

## Explanation

Grax supports two roughness implementations that enter the simulation at
different stages.

`debye-waller` roughness keeps the grating geometry unchanged. The RCWA solve
uses the original discretized structure, and the solver then applies a
Debye-Waller-style damping factor to the diffraction efficiencies. Use this mode
when you want the existing scalar roughness correction:

```python
roughness=grax.RoughnessSpec(kind="debye-waller", sigma_nm=0.5)
```

`random-interface` roughness changes the grating geometry before the RCWA
textures are built. During texture generation, each material interface receives
a deterministic random height modulation controlled by `seed`. Because the
roughness is already represented in the geometry, Debye-Waller damping is not
applied for this mode:

```python
roughness=grax.RoughnessSpec(kind="random-interface", sigma_nm=0.5, seed=0)
```

The example plot below compares the no-roughness baseline with both roughness
implementations at the configured sigma values.

```{image} images/simulation/fixed_angle_roughness_order1_comparison.png
:alt: First-order fixed-angle roughness comparison for Debye-Waller and random-interface roughness
:align: center
:width: 80%
```

See these files for the complete workflow:

- `examples/simulation/fixed_angle_roughness/fixed_angle_roughness.py`
- `examples/simulation/fixed_angle_roughness/comparison_fixed_angle_roughness.py`

For the baseline fixed-angle workflow without the roughness sweep wrapper, see
{doc}`fixed-angle-sweep`.
