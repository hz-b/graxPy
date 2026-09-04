# Multilayer optimization

APIs for the three-stage multilayer-grating design workflow: derive and scan the
bilayer d-spacing, scan the bilayer thickness ratio, then scan the blaze angle
with graxPy's internal theta search. Stages hand values forward only through
`optimization_state.json`, and only when a config value is `"auto"`.

```{eval-rst}
.. autoclass:: grax.MultilayerOptimizationConfig
```

```{eval-rst}
.. autofunction:: grax.run_d_spacing_study
```

```{eval-rst}
.. autofunction:: grax.run_gamma_study
```

```{eval-rst}
.. autofunction:: grax.run_blaze_study
```

```{eval-rst}
.. autoclass:: grax.DSpacingStudyResult
```

```{eval-rst}
.. autoclass:: grax.GammaStudyResult
```

```{eval-rst}
.. autoclass:: grax.BlazeStudyResult
```

## Planar-multilayer reflectivity

Stages 0 and 1 measure peak Bragg reflectivity with this XRT wrapper. `xrt` is
imported lazily, so importing `grax` does not pull it in.

```{eval-rst}
.. autoclass:: grax.MultilayerReflectivity
   :members: reflectivity_vs_energy
```
