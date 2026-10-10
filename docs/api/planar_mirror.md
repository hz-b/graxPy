# Planar mirrors (`grax.planar_mirror`)

Fast specular reflectivity of flat mirrors and multilayers using the Parratt
recursion. It accepts any `grax` stack (`SingleLayerStack`, `MultilayerStack`,
`CustomStack`), honours per-layer and substrate roughness (Névot–Croce), and
supports `s`, `p` and averaged polarization.

```python
import numpy as np
import grax

stack = grax.build_multilayer_stack(
    substrate_material="Si", material_a="Ru", material_b="C",
    d_period_nm=7.0, gamma=0.4, n_bilayers=20, top_material="C",
)
energies = np.linspace(150, 400, 500)
r = grax.parratt_reflectivity(stack, energies, 20.0, polarization="s", roughness_sigma_nm=0.3)
```

## Lateral grading

If the layer thickness varies along the mirror, the beam footprint sees a
range of Bragg energies, so the measured reflectivity is broader and lower than
the value at the central position. `footprint_reflectivity` averages the
reflectivity over sample positions along the footprint:

```python
r_avg = grax.footprint_reflectivity(
    stack, energies, 20.0,
    grading=grax.LateralGrading.linear(0.0005),   # +0.05 % thickness per mm
    positions_mm=np.linspace(-20, 20, 41),
)
```

`LateralGrading` also accepts polynomial coefficients or a tabulated profile,
and any callable mapping positions (mm) to thickness scale factors works.

```{eval-rst}
.. autofunction:: grax.parratt_reflectivity

.. autofunction:: grax.footprint_reflectivity

.. autoclass:: grax.LateralGrading
   :members: linear, scale
```

The web app exposes the same calculation in the **Plane mirror** tab.
