# Profiling Tools

Developer-only profiling scripts for investigating where RCWA runtime is spent.

These scripts are not tests and not examples. They are intended for local
performance investigation with explicit, reproducible inputs.

## Blazed Multilayer One-Energy Matrix

Run one blazed multilayer energy/angle point across a small Fourier and
discretization matrix. The tool uses the same optical constants, multilayer
stack, grating geometry, and DiffraMod energy/angle table used by:

`comparison_to_other_codes/blazed_multilayer/blazed_multilayer_sweep.py`

From the repository root:

```bash
python3 tools/profiling/profile_blazed_multilayer_case.py
```

## Branch Workflow

Baseline on `develop`:

```bash
git switch develop
python3 tools/profiling/profile_blazed_multilayer_case.py --label baseline
```

Candidate on the optimization branch:

```bash
git switch codex/texture-generation-optimization
python3 tools/profiling/profile_blazed_multilayer_case.py --label candidate
```

Compare the two matrix summaries:

```bash
python3 tools/profiling/compare_blazed_multilayer_profiles.py \
  --baseline tools/profiling/results/blazed_multilayer_case/profile_matrix_summary_baseline.csv \
  --candidate tools/profiling/results/blazed_multilayer_case/profile_matrix_summary_candidate.csv
```

Default footprint:

- `case_index = 0`
- `x_resolution_nm = 0.01, 0.1, 1.0`
- `z_resolution_nm = 0.01, 0.1, 1.0`
- `fourier_orders = 5, 10, 15`
- `backend = numpy`

Useful overrides:

```bash
python3 tools/profiling/profile_blazed_multilayer_case.py --case-index 10
```

```bash
python3 tools/profiling/profile_blazed_multilayer_case.py \
  --energy-ev 500 \
  --grazing-angle-deg 14.176
```

```bash
python3 tools/profiling/profile_blazed_multilayer_case.py \
  --x-resolution-nm 0.01 0.1 \
  --z-resolution-nm 0.01 0.1 \
  --fourier-orders 5 10
```

The tool logs live stage start/end messages while the simulation runs, for
example:

```text
stage start: texture_generation
stage end: texture_generation elapsed=1.234567s exclusive=1.234567s
stage start: res1_total
stage end: res1_total elapsed=2.345678s exclusive=0.123456s
```

Disable live stage logging with:

```bash
python3 tools/profiling/profile_blazed_multilayer_case.py --no-live-stage-log
```

Outputs are written by default to:

`tools/profiling/results/blazed_multilayer_case/`

Created files:

- `profile_report_<label>_*.txt`: human-readable stage timing report per configuration
- `profile_summary_<label>_*.json`: structured timing, counters, memory, and selected result per configuration
- `profile_matrix_summary_<label>.csv`: compact CSV summary across all configurations

The matrix summary includes:

- label
- total wall time
- texture-generation time
- Fourier-stage time
- propagation time
- peak memory
- texture count
- unique texture count
- selected efficiency

Comparison outputs are written by:

`tools/profiling/compare_blazed_multilayer_profiles.py`

Created files:

- `comparison_<baseline>_vs_<candidate>.csv`
- `comparison_<baseline>_vs_<candidate>.txt`
