# grax documentation

*Version {{ release }}*

`grax` is an independent Python package for one-dimensional RCWA simulations of X-ray diffraction gratings. The current solver implementation is inspired by the one-dimensional RETICOLO approach and currently supports 1D RCWA simulations, including multilayer coating stacks.

On top of the RCWA solver, `grax` provides higher-level abstractions to define and simulate different grating geometries, currently including laminar and blazed gratings, together with user-friendly simulation utilities for single simulations, batch execution, and energy-angle sweeps.

The local Web UI includes its own documentation page. Start `grax-web`, then click
`Web docs` in the GUI to read the guide for storage, grating creation,
simulation runs, plotting, exporting, and customization.

RETICOLO V9 is an important scientific reference and inspiration for this project. `grax` is not an official RETICOLO port or direct translation. If you use `grax`, please also cite RETICOLO:

> [https://doi.org/10.5281/zenodo.14631950](https://doi.org/10.5281/zenodo.14631950)

RETICOLO license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

```{toctree}
:maxdepth: 2
:caption: User guide

installation/index
validation/index
tutorials/index
how-to/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
