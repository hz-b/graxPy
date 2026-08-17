# Optimization API (`grax_opt`)

Install optional dependencies first:

```bash
pip install .[opt]
```

This page exposes the primary public optimizer API for end users.

```{eval-rst}
.. autofunction:: grax_opt.optimize_to_measurements
```

## Joint measurement fitting

Fits one parameter set against several measured curves at once. Each measurement
keeps its own energy grid and its own conditions -- grazing angle, angle mode,
diffraction order and polarization -- inheriting whichever it does not set from
the run.

```{eval-rst}
.. autofunction:: grax_opt.optimize_to_joint_measurements

.. autoclass:: grax_opt.MeasurementSpec

.. autoclass:: grax_opt.JointMeasurement

.. autoclass:: grax_opt.JointMeasurementFitConfig

.. autofunction:: grax_opt.reduce_joint_losses
```

## Result types

```{eval-rst}
.. autoclass:: grax_opt.OptimizationResult

.. autoclass:: grax_opt.JointOptimizationResult
```

## See tutorials

- [Optimizer setup guide](../tutorials/optimizer.md)
- [Laminar Grating](../tutorials/optimizer-laminar-fit.md)
- [Blazed Grating](../tutorials/optimizer-blazed-fit.md)
- [Joint measurement fits](../tutorials/optimizer-joint-fit.md)
- [Resume an optimizer run](../tutorials/optimizer-resume.md)
