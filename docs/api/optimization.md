# Optimization API (`grax_opt`)

Install optional dependencies first:

```bash
pip install .[opt]
```

This page exposes the public optimizer entrypoints, including the dynamic
spec-based variant for custom gratings.

```{eval-rst}
.. autofunction:: grax_opt.optimize_dynamic
```

## Configuration classes

```{eval-rst}
.. autoclass:: grax_opt.ParameterBounds
   :members:
```

```{eval-rst}
.. autoclass:: grax_opt.DynamicOptimizationConfig
   :members:
```

## Result type

```{eval-rst}
.. autoclass:: grax_opt.OptimizationResult
   :members:
```

## Utility

```{eval-rst}
.. autofunction:: grax_opt.json_safe_grating_parameters
```

```{eval-rst}
.. autofunction:: grax_opt.build_dynamic_ax_parameters
```

```{eval-rst}
.. autofunction:: grax_opt.resolve_dynamic_trial_parameters
```

## See tutorials

- [Laminar Grating](../tutorials/optimizer-laminar-fit.md)
- [Blazed Grating](../tutorials/optimizer-blazed-fit.md)
- [Dynamic optimizer how-to](../how-to/dynamic-optimizer.md)
