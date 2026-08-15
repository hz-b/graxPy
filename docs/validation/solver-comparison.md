# RCWA vs Nevière differential method

The other pages in this section compare `grax` against external codes. This one
compares `grax` against itself: the same validation sweeps run through both of
its solvers.

The two solvers share their inputs, their Fourier-space operators (including the
Li/fast-Fourier-factorization rules for TM) and their efficiency extraction, and
differ only in how they propagate through a layer — RCWA eigen-decomposes it,
the differential method integrates it with fourth-order Runge–Kutta. They are
therefore two numerical treatments of one truncated system, and any difference
between them is integration error rather than a modelling difference.

## Reproducing

Run each sweep with both solvers, then the comparison script:

```bash
python validation/laminar/fixed_angle_sweep.py --solver rcwa --tag rerun
python validation/laminar/fixed_angle_sweep.py --solver neviere
python validation/compare_solvers.py --case laminar
```

`--tag rerun` writes the RCWA baseline alongside the checked-in artifacts rather
than over them. Both sides must come from the same revision; the last section on
this page explains why the checked-in RCWA artifacts are not usable as one side.

The comparison script writes `*_solver_comparison.csv` (per-order maximum, mean
and RMS deviation) and `*_solver_comparison.png` (efficiencies overlaid, with a
log-scale difference panel) into each case's `results/` directory.

## Results

Maximum absolute deviation between the two solvers over every energy of each
sweep, for reflected orders 1 to 3. `max/peak` normalizes that by the peak
efficiency of the order, so it reads as a relative figure.

| case | points | order | peak efficiency | max abs | RMS | max/peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| laminar 400 l/mm | 601 | 1 | 0.179 | 5.6e-11 | 9.0e-12 | 3.1e-10 |
|  |  | 2 | 0.021 | 1.2e-11 | 2.3e-12 | 5.6e-10 |
|  |  | 3 | 0.009 | 6.2e-12 | 1.3e-12 | 6.9e-10 |
| blazed 600 l/mm | 195 | 1 | 0.254 | 4.2e-12 | 1.6e-12 | 1.7e-11 |
|  |  | 2 | 0.138 | 2.2e-12 | 9.5e-13 | 1.6e-11 |
|  |  | 3 | 0.029 | 2.3e-12 | 1.2e-12 | 8.1e-11 |
| laminar 150 l/mm | 496 | 1 | 0.331 | 3.9e-12 | 1.6e-12 | 1.2e-11 |
|  |  | 2 | 0.116 | 2.9e-12 | 1.4e-12 | 2.5e-11 |
|  |  | 3 | 0.042 | 1.4e-12 | 6.7e-13 | 3.4e-11 |
| blazed 2400 l/mm multilayer | 173 | 1 | 0.007 | 2.3e-12 | 4.5e-13 | 3.2e-10 |
|  |  | 2 | 0.675 | 8.7e-11 | 1.8e-11 | 1.3e-10 |
|  |  | 3 | 0.002 | 1.2e-12 | 2.6e-13 | 5.8e-10 |

All four cases run p polarization at the resolution and Fourier truncation each
sweep normally uses. The `blazed_multilayer` case was run at `--stride 10`, which
samples 173 of the 1727 reference energy-angle pairs across the full 500 to
6000 eV range; the others use their full energy grids.

The largest residual, on the laminar 400 l/mm case, is simply the case that
accumulates the most Runge-Kutta error: it runs the most Fourier orders (30)
across the most layers (310 slices at `z_resolution_nm = 0.1`).

The differences are at the level of floating-point noise, not of physics. In the
difference panel of each figure they appear as a jagged band with no structure
tracking the efficiency curve, which is what accumulated Runge–Kutta truncation
error looks like; a modelling difference would instead show up correlated with
the spectral features.

Tightening `NeviereOptions.step_phase` drives the residual down as its fourth
power, confirming the same thing from the other direction. This is asserted in
`tests/unit/test_neviere.py::test_neviere_converges_to_rcwa_as_step_phase_shrinks`.

### Why the agreement is this close

The brief for this work anticipated agreement "within a few percent". It is far
better than that because the two solvers are not independent models of the
grating — they are independent *propagators* for one shared model. They build
the same Fourier operators through the same `layer_field_operators()`, see the
same z-sliced permittivity, and extract efficiencies through the same
`solve_stack_from_layer_blocks()`. The only thing that differs is how one layer
is crossed, so the only thing that can differ in the answer is integration
error.

That makes the comparison a sharp test of the propagation and a weak test of
everything upstream of it. Errors in the shared geometry, Fourier coefficients or
flux normalization would move both solvers together and stay invisible here. The
checks that do bite on the shared parts are the analytic Fresnel and
energy-conservation tests in `tests/unit/test_neviere.py`, the published RETICOLO
values in `tests/smoke/test_neviere_solver.py`, and the external-code comparisons
on the other pages of this section.

## Where the two would be expected to diverge

Nothing in these cases stresses either solver, but the places to watch are:

- **Deep gratings.** The modal solver evaluates `q / sinh(q d)` across a whole
  layer and overflows above roughly seven wavelengths of depth for a
  high-contrast profile. The differential method caps the optical thickness of
  any transfer matrix it forms and remains stable far beyond that. None of the
  X-ray validation gratings are anywhere near this limit; the effect is covered
  by `tests/smoke/test_neviere_solver.py`.
- **Metallic TM.** p-polarization at high permittivity contrast is where the
  factorization rules matter most. Both solvers use the same rules, so they
  converge together in the truncation order rather than against each other. All
  four validation cases run p-polarization.
- **High-contrast lamellar profiles.** For a genuinely lamellar grating the
  staircase is exact, so the modal solver's layer treatment is exact and the
  differential method's is a numerical approximation of it. The residual is
  bounded by `step_phase` and is visible in the difference panel as a floor
  rather than as structure.
- **Continuous versus staircase sampling.** With
  `NeviereOptions(z_sampling="continuous")` the differential method reads the
  true profile instead of the shared staircase, so it will *not* match RCWA at a
  coarse `z_resolution_nm` — it matches a much finer one. The comparisons above
  use the default `"textures"` sampling so both solvers see identical geometry.

## A note on the checked-in artifacts

The RCWA CSVs committed under each `results/` directory predate several changes
to the solver and its inputs, so they no longer reproduce from the current code.
The comparison above therefore uses a fresh RCWA run (`--tag rerun`) rather than
the committed files, and `validation/compare_solvers.py` prints the drift between
the two for reference:

| case | order 1 max drift | order 1 relative |
| --- | ---: | ---: |
| laminar 400 l/mm | 7.5e-3 | 4.1% |
| blazed 600 l/mm | 6.2e-3 | 2.4% |
| laminar 150 l/mm | 7.9e-2 | 22% |
| blazed 2400 l/mm multilayer | 3.1e-4 | 4.5% |

The laminar 150 l/mm figure is not representative of that curve: the drift is
concentrated in the 12 to 28 eV tail, where the wavelength approaches the
structure size. Its median drift over the sweep is 2.6e-4, and only 8.3% of
points exceed 1e-2.

This is independent of the differential method. The current RCWA solver and the
current differential-method solver agree with each other to the table above;
both differ from the older artifacts in the same way and by the same amount. For
scale, on the laminar 400 l/mm case the RMS difference between the current
solvers and the external references is 0.0024 (DiffractMod), 0.0036 (REFLEC) and
0.0090 (RETICOLO), so this drift sits inside the spread between established
codes — but it does mean the committed artifacts should be refreshed before
they are used as a baseline for anything.
