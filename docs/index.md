# grax documentation

*Version {{ release }}*

`graxpy` is an independent Python package for one-dimensional X-ray
diffraction-grating simulations; `grax` is its public Python import. It offers
two selectable electromagnetic solver paths for the same gratings, materials,
and workflows:

- modal rigorous coupled-wave analysis (RCWA), inspired by the one-dimensional
  RETICOLO approach;
- the Nevière differential method.

The paths share Fourier/discretization infrastructure and differ in how they
propagate fields through a layer. `solver="rcwa"` remains the default. `grax`
provides higher-level abstractions for laminar and blazed gratings, multilayer
coating stacks, single simulations, batch execution, and energy-angle sweeps.

For selection guidance and integration settings, see
[Choosing a solver](tutorials/choosing-a-solver.md).

The local Web UI includes its own documentation page. Start `grax-web`, then click
`Web docs` in the GUI to read the guide for storage, grating creation,
simulation runs, plotting, exporting, and customization.

RETICOLO V9 is an important scientific reference and inspiration for GraxPy's
modal RCWA solver. `grax` is not an official RETICOLO port or direct
translation. If you use `grax`, please also cite RETICOLO:

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
