# Fourier Profiling Examples

Internal benchmarking scripts for RCWA Fourier backend comparisons.

These examples are intended for profiling/investigation workflows only.

## Location

- `tools/fourier_profiling_examples/profile_single_rcwa_case.py`
- `tools/fourier_profiling_examples/profile_multi_energy_numba_vs_legacy.py`
- `tools/fourier_profiling_examples/plot_multi_energy_numba_vs_legacy.py`

## Fixed simulation settings

Both scripts use hardcoded RCWA settings:

- `grazing_angle_deg = 4.0`
- `fourier_orders = 20`
- `x_resolution_nm = 0.1`
- `z_resolution_nm = 0.1`

Single-case script also uses:

- `energy_ev = 200.0`

Multi-energy script uses a configurable energy range with defaults:

- `energy_start_ev = 120.0`
- `energy_stop_ev = 300.0`
- `num_energies = 10`

## How to run

From the repository root:

```bash
.venv/bin/python tools/fourier_profiling_examples/profile_single_rcwa_case.py
```

```bash
.venv/bin/python tools/fourier_profiling_examples/profile_multi_energy_numba_vs_legacy.py
```

```bash
.venv/bin/python tools/fourier_profiling_examples/plot_multi_energy_numba_vs_legacy.py
```

Optional arguments:

- Single-case:
  - none (fully reproducible fixed settings)
- Multi-energy:
  - none (fully reproducible fixed settings)
- Plotting:
  - `--csv-path` (default: `tools/fourier_profiling_examples/results/multi_energy_numba_vs_legacy.csv`)
  - `--output-dir` (default: `tools/fourier_profiling_examples/results`)

## Created results

Single-case script writes:

- `single_rcwa_profile_report.txt` (copy of first backend report)
- `single_rcwa_profile_report_baseline.txt`
- `single_rcwa_profile_report_numba_optional.txt`
- `single_rcwa_profile_comparison_numba_vs_legacy.txt`

When `--fourier-backend all` is used, it also writes backend-specific reports for:

- `vectorized-harmonics`
- `phase-recursive`
- `phase-table-per-texture`

Multi-energy script writes:

- `multi_energy_numba_vs_legacy.txt` (human-readable summary)
- `multi_energy_numba_vs_legacy.csv` (per-energy tabular data)

Plot script writes:

- `multi_energy_numba_vs_legacy_plots.png` (multi-panel Matplotlib summary figure)

The plot includes:

- Total runtime vs energy (baseline vs numba)
- Fourier-stage runtime vs energy (baseline vs numba)
- Speedup vs energy
- Fourier fraction of runtime
- Peak memory vs energy
- Diffracted-efficiency comparison panel (baseline reference vs numba values)

All subplot axes are rendered without scientific-notation tick labels.

All files are written into the selected `--output-dir`.
