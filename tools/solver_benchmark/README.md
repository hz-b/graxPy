# solver_benchmark

Reproducible runtime benchmarks for the RCWA and Nevière solvers. Developer
tool, not part of the shipped `grax` package and not imported by it.

```bash
# short serial grid (default: 10 energies, 3 repeats, 1 warm-up)
python tools/solver_benchmark/solver_benchmark.py --points 10

# full multiprocessing monochromator sweep
python tools/solver_benchmark/solver_benchmark.py --multiprocessing
```

Outputs (written to `--output-dir`, default `tools/solver_benchmark/`):

- `solver_runtime_benchmark_<mode>.json` / `.csv` — per-point timing records
- `solver_runtime_<mode>_<difficulty>.png` — median-seconds vs energy per case

## Execution modes measure different things

| mode | what it times |
| --- | --- |
| `serial` | one `grax.run_simulation` call per energy at a **fixed grazing angle** (no theta search) — raw per-solve cost |
| `multiprocessing` | a full **cff-locked monochromator sweep** through `grax.BatchSimulationRunner` — includes theta search and worker scheduling |

The two are not directly comparable; results are kept in separate files and
plots keyed by `execution_mode`.
