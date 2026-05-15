# Optimization API (`grax_opt`)

Install optional dependencies first:

```bash
pip install .[opt]
```

This page intentionally exposes only the two public optimizer entrypoints.

```{eval-rst}
.. autofunction:: grax_opt.optimize_laminar
```

```{eval-rst}
.. autofunction:: grax_opt.optimize_blazed
```

## Configuration classes

```{eval-rst}
.. autoclass:: grax_opt.LaminarAxConfig
   :members:
```

```{eval-rst}
.. autoclass:: grax_opt.BlazedAxConfig
   :members:
```

```{eval-rst}
.. autoclass:: grax_opt.ParameterBounds
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

## See tutorials

- [Laminar Grating](../tutorials/optimizer-laminar-fit.md)
- [Blazed Grating](../tutorials/optimizer-blazed-fit.md)
