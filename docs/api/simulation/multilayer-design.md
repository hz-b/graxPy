# Multilayer grating design

APIs for the multilayer-grating design workflow: a 2-D
d-spacing / blaze-angle survey at one optimization energy, followed by per-design
energy scans. For every `(d_spacing, blaze angle)` pair the survey builds the
multilayer-coated blazed grating and runs graxPy's single-energy multilayer theta
search, which scans the incident angle and returns the angle that maximizes the
selected-order efficiency for that pair.

```{eval-rst}
.. autoclass:: grax.MultilayerDesignConfig
```

```{eval-rst}
.. autoclass:: grax.ThetaSearchScanSettings
```

```{eval-rst}
.. autoclass:: grax.MultilayerGratingDesigner
   :members: run_survey, evaluate_survey, run_energy_scan, evaluate_energy_scan, plot_energy_scan_overlay
```

```{eval-rst}
.. autoclass:: grax.SurveyResult
```

```{eval-rst}
.. autoclass:: grax.EnergyScanResult
```

```{eval-rst}
.. autoclass:: grax.StageProgress
```

## Planar-multilayer reflectivity

The design workflow itself never touches XRT, but the planar-multilayer
reflectivity wrapper remains available for standalone Bragg-reflectivity work.
`xrt` is imported lazily, so importing `grax` does not pull it in.

```{eval-rst}
.. autoclass:: grax.MultilayerReflectivity
   :members: reflectivity_vs_energy
```
