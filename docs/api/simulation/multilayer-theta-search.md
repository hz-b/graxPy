# Multilayer theta search

APIs for multilayer theta-search sweeps and per-energy solves.

The numerical settings are split across three stages: rough scan, fine scan,
and final solve at the selected theta. Each stage has independent Fourier and
`x`/`z` resolution controls.

```{eval-rst}
.. autofunction:: grax.multilayer_theta_search_cases
```

```{eval-rst}
.. autofunction:: grax.run_multilayer_theta_search
```

```{eval-rst}
.. autofunction:: grax.run_multilayer_theta_search_sweep
```

```{eval-rst}
.. autoclass:: grax.ThetaSearchDiagnostics
```

```{eval-rst}
.. autoclass:: grax.MultilayerThetaSearchSweepResult
```
