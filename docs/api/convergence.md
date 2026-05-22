# Convergence API (`grax_opt`)

Use this entrypoint when you want to pick the coarsest Fourier and
discretization settings that still behave stably across a set of energies.

```{eval-rst}
.. autofunction:: grax_opt.optimize_simulation_convergence
```

## Configuration classes

```{eval-rst}
.. autoclass:: grax_opt.SimulationConvergenceConfig
   :members:
```

## Result types

```{eval-rst}
.. autoclass:: grax_opt.SimulationConvergenceResult
   :members:
```

```{eval-rst}
.. autoclass:: grax_opt.SimulationConvergenceEnergyResult
   :members:
```

## See tutorial

- [Convergence Optimizer](../tutorials/convergence-optimizer.md)
