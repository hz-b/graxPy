# RCWA theory

Rigorous coupled-wave analysis models diffraction from periodic structures by
expanding the electromagnetic fields and material permittivity in Fourier
orders. In `grax`, modal RCWA is one of two selectable electromagnetic solver
paths; it is a one-dimensional implementation for periodic grating profiles
used in X-ray optics and is inspired by RETICOLO.

This page gives the minimum physical model needed to understand the code. It is
not a replacement for RCWA references such as {cite}`moharam1995stable` and
{cite}`li1996formulation`.

## Problem geometry

The solver treats one period of a grating as a stack of horizontal slices. Each
slice can be homogeneous or patterned along `x`. The grating profile determines
which material occupies each `x, z` grid cell.

The main physical inputs are:

- Period: the spatial repeat distance along `x`, derived from lines per mm.
- Groove profile: the surface height of the substrate material over one period.
- Incident medium: currently passed as `n_inc=1.0 + 0.0j` by the high-level API.
- Substrate: the material below the grating surface.
- Coating stack: one or more material layers above the substrate surface.
- Photon energy: converted to wavelength in nanometers.
- Grazing angle: converted to the solver's in-plane wave-vector convention.

## Fourier orders

RCWA replaces a continuous periodic permittivity function with a finite Fourier
representation. The `fourier_orders` setting controls the truncation on either
side of the zeroth order. Higher values can improve convergence but increase
matrix size and runtime.

In the code, `res1()` normalizes the requested order count and converts each
texture into Fourier-space data. Patterned textures are represented by
breakpoints and refractive-index values across one period.

## Layer solve

After textures are represented in Fourier space, the stack solve propagates the
coupled modes through the grating slices and applies boundary matching at layer
interfaces. The implementation in `res2()` compresses consecutive identical
textures and calls the TE stack solver used by the native Python path.

The result contains reflected and transmitted diffraction orders, angles,
amplitudes, and efficiencies. High-level APIs usually select one reflected order
for convenience while preserving all-order arrays for export and diagnostics.

## Current limitations

The native Python solver path currently supports the one-dimensional TE-style
entry only. `res0()` rejects unsupported dimensions and polarizations, and
`res1()` raises if the parameter bundle does not match the implemented path.

Roughness is modeled as a scalar Debye-Waller damping factor applied after the
RCWA solve. It reduces reflected and transmitted efficiencies uniformly and does
not alter the geometry, Fourier matrices, diffraction amplitudes, or material
interfaces.

```{bibliography}
```
