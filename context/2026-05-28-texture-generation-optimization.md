# 2026-05-28 Texture Generation Optimization

## Motivation

Profiling of the blazed multilayer one-energy matrix showed that most runtime was
spent in `texture_generation`, not in the RCWA propagation or Fourier stages.
The hotspot came from the low-memory texture builder in `src/grax/gratings.py`,
which was recomputing multilayer material decisions row by row for every `z`
slice.

## What changed

- Added a prepared multilayer texture-builder path for the low-memory mode.
- Pre-resolved material refractive indices once per `build_textures()` call.
- Precomputed multilayer constants once: bilayer period, stack height, bottom
  thickness, top-cap thickness, substrate/incident indices, and bottom/top
  material indices.
- Replaced the old per-row bilayer loop with a direct vectorized row evaluation
  based on relative height within the multilayer period.
- Kept the public texture/profile output format unchanged.
- Extended the profiling tool so runs can be labeled as `baseline` or
  `candidate` and compared later with stable output filenames.
- Added a dedicated comparison utility for before/after profiling CSVs.

## Invariants preserved

- No intentional physics change.
- No resolution change.
- The low-memory builder still returns the same texture descriptor format and
  profile tuple shape as before.
- Existing `legacy_dense` remains the reference path for regression comparison.

## Manual rerun workflow

Baseline:

```bash
git switch develop
python3 tools/profiling/profile_blazed_multilayer_case.py --label baseline
```

Candidate:

```bash
git switch codex/texture-generation-optimization
python3 tools/profiling/profile_blazed_multilayer_case.py --label candidate
```

Compare:

```bash
python3 tools/profiling/compare_blazed_multilayer_profiles.py \
  --baseline tools/profiling/results/blazed_multilayer_case/profile_matrix_summary_baseline.csv \
  --candidate tools/profiling/results/blazed_multilayer_case/profile_matrix_summary_candidate.csv
```
