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
than over them. Both sides must come from the same revision — see
[a note on the checked-in artifacts](#a-note-on-the-checked-in-artifacts).

The comparison script writes `*_solver_comparison.csv` (per-order maximum, mean
and RMS deviation) and `*_solver_comparison.png` (efficiencies overlaid, with a
log-scale difference panel) into each case's `results/` directory.

## Results

<!-- RESULTS TABLE -->

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
the two for reference.

This is independent of the differential method: the current RCWA solver and the
current differential-method solver agree with each other, and both differ from
the older artifacts in the same way and by the same amount.
