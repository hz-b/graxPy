# GraxPy documentation

```{image} _static/GraxPy_logo.svg
:alt: GraxPy — Gratings made easy
:width: 340px
:align: center
```

*Version {{ release }}*

GraxPy (`graxpy`) is a Python package for one-dimensional X-ray grating
simulations, imported as `grax`. Build laminar, blazed, sinusoidal, or custom
grating profiles; combine them with material definitions and multilayer
coatings; then calculate diffraction efficiencies with either modal RCWA or
the Nevière differential method.

Use the same API for single simulations, energy-angle and fixed-angle sweeps,
batch runs, parameter studies, optimizer fits, and multilayer design searches.
Start with the [single simulation tutorial](tutorials/single-simulation.md),
explore [grating workflows](tutorials/gratings.md), or see
[how to choose a solver](tutorials/choosing-a-solver.md).

GraxPy's modal RCWA solver is inspired by the one-dimensional RETICOLO approach;
GraxPy is an independent implementation, not an official port. If you use it,
please cite [RETICOLO V9](https://doi.org/10.5281/zenodo.14631950) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

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
