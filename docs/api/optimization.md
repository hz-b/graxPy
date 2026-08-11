# Optimization API (`grax_opt`)

Install optional dependencies first:

```bash
pip install .[opt]
```

This page exposes the primary public optimizer API for end users.

```{eval-rst}
.. autofunction:: grax_opt.optimize_to_measurements
```

## Joint multi-angle fitting

Fits one parameter set against several measured curves recorded at different
grazing angles, each with its own energy grid.

```{eval-rst}
.. autofunction:: grax_opt.optimize_to_joint_measurements

.. autoclass:: grax_opt.AngleMeasurementSpec

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
- [Joint multi-angle fits](../tutorials/optimizer-joint-angles.md)
- [Resume an optimizer run](../tutorials/optimizer-resume.md)
