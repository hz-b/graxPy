# Parameter study

Use {func}`grax.run_parameter_study` when you want a convergence study at
user-defined photon energies and a fixed user-defined grazing angle. The study
always sweeps `fourier_orders`, `x_resolution_nm`, and `z_resolution_nm`, then
plots the selected diffraction-order efficiency in a single grid figure.

```python
import grax

grating = grax.BlazedGrating(
    period_lpermm=600,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    blaze_angle_deg=0.75,
    anti_blaze_angle_deg=None,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

study = grax.run_parameter_study(
    grating=grating,
    energies_ev=[100.0, 600.0, 2000.0],
    grazing_angle_deg=1.5,
    diffraction_order=1,
    polarization="p",
    fourier_orders_values=range(5, 26, 2),
    x_resolution_values=grax.get_default_parameter_study_ranges()[1],
    z_resolution_values=grax.get_default_parameter_study_ranges()[2],
    output_dir="examples/simulation/parameter_study/results",
)

grax.plot_parameter_study(
    study,
    output_filename="examples/simulation/parameter_study/results/parameter_study_grid_rcwa.png",
    title="Blazed Parameter Study: Orders vs Fourier/x/z Resolution",
)
```

The maintained example uses the same blazed geometry as the monochromator
tutorial and produces one grid plot with rows for `100`, `600`, and `2000 eV`,
and columns for the three swept parameters.

It also sets `polarization="p"` explicitly so the convergence study matches the
intended reflected-polarization workflow without relying on defaults. Accepted
values are `s` and `p`.

The maintained example should complete without failures on current source. When
failures do happen, the CSV output records them explicitly with
`efficiency` left empty, `error=True`, and `error_message` containing the final
exception message after retries. A failed point is not a zero-efficiency point.

```{image} images/simulation/parameter_study_grid.png
:alt: Blazed parameter study grid for Fourier orders and x/z discretization
:align: center
:width: 95%
```

The default maintained example runs the full study:

```python
energies_ev = [100.0, 600.0, 2000.0]
grazing_angle_deg = 1.5
fourier_orders_values = range(5, 26, 2)
x_resolution_values = grax.get_default_parameter_study_ranges()[1]
z_resolution_values = grax.get_default_parameter_study_ranges()[2]
```

The x- and z-resolution values are logarithmically spaced from `10 nm` down to
`0.1 nm`, which keeps the smaller values more closely packed than a linear
spacing.

The generated CSV files now contain:

- `parameter`: which sweep this file represents
- `value`: the tested Fourier order or discretization value
- `efficiency`: selected-order efficiency for successful points
- `error`: whether the point still failed after retries
- `error_message`: the failure reason for unsuccessful points

The plot only draws successful points on the efficiency curve. Failed points
are shown separately with red `x` markers so they cannot be mistaken for
physical zero efficiency.

See `examples/simulation/parameter_study/parameter_study.py` for the complete
script.
