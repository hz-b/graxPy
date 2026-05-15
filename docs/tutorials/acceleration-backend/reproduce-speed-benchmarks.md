# Reproduce Speed Benchmarks

Run the benchmark scripts in `tools/numba_speed/` from the repository root:

```bash
.venv/bin/python tools/numba_speed/profile_multi_energy_numba_vs_legacy.py
.venv/bin/python tools/numba_speed/plot_multi_energy_numba_vs_legacy.py
```

Generated outputs are written under `tools/numba_speed/results/`, including the
summary text/CSV and the plot image used in docs.

```{image} ../images/numba_speed/multi_energy_numba_vs_legacy_plots.png
:alt: Baseline vs Numba multi-energy runtime and efficiency comparison
:width: 100%
```

## Multilayer benchmark (blazed multilayer)

Run the multilayer benchmark and plot scripts from the repository root:

```bash
.venv/bin/python tools/numba_speed/profile_multi_energy_multilayer_numba_vs_legacy.py
.venv/bin/python tools/numba_speed/plot_multi_energy_multilayer_numba_vs_legacy.py
```

This run uses a fixed, reproducible energy grid of 10 points:
`500, 1000, 1500, ..., 5000` eV.
For each energy, the grazing angle is chosen by nearest lookup from the
reference blazed-multilayer table used by the comparison workflow.

Generated outputs are written under `tools/numba_speed/results/`:

- `multi_energy_multilayer_numba_vs_legacy.csv`
- `multi_energy_multilayer_numba_vs_legacy.txt`
- `multi_energy_multilayer_numba_vs_legacy_plots.png`

Interpretation:
- `speedup_baseline_over_numba > 1` means the Numba backend is faster.
- `eff_delta_order_m1` should remain near zero, indicating backend-consistent
  `-1` order efficiency.
- If Numba is unavailable, `numba-optional` falls back to baseline and this is
  noted in the report.

```{image} ../images/numba_speed/multi_energy_multilayer_numba_vs_legacy_plots.png
:alt: Baseline vs Numba multilayer multi-energy runtime, speedup, memory, and -1 order efficiency comparison
:width: 100%
```