# Laminar Grating (150 l/mm, monochromator sweep)

This comparison uses the monochromator workflow from
`validation/laminar_150lmm` with `cff = 2.25`.

## Grating design parameters

- Grating type: `LaminarGrating`
- Period: `150 l/mm`
- Duty cycle (`width_to_period_ratio`): `0.65`
- Groove depth: `60 nm`
- Left wall angle: `10 deg`
- Right wall angle: `10 deg`
- Substrate material: `Si`
- Coating material: `Au`
- Coating thickness: `30 nm`
- Diffraction order: `1`
- Fourier orders: `5`
- Spatial resolution: `x = 1 nm`, `z = 1 nm`

## Scan setup

- Mode: monochromator sweep
- Constant focus factor: `cff = 2.25`
- Energy range: `10 eV` to `1000 eV` in `2 eV` steps

## Grating profile

```{image} images/laminar_150lmm_monochromator_profile.png
:alt: Laminar 150 l/mm monochromator profile
:align: center
:width: 85%
```

## Comparison result

External reference files are read from
`validation/laminar_150lmm/simulation`.

```{image} images/comparison_laminar_150lmm_monochromator.png
:alt: Laminar 150 l/mm monochromator comparison
:align: center
:width: 85%
```
