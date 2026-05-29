# Changelog

## 0.2.2 - 2026-05-29

- Added per-simulation peak RAM logging and a dedicated blazed-multilayer profiling tool under `tools/profiling/`.
- Optimized the multilayer cascade path and added profiler substage breakdowns for the cascade pair algebra.
- Made repository example and comparison scripts explicitly use the `numba` backend and removed stale solver-backend metadata from the checked-in workflows.
- Refreshed the numba-speed benchmark scripts, reports, plots, and docs so they reproduce the current backend comparison results.

## 0.2.1 - 2026-05-26

- Integrated AFM preprocessing and AFM-derived profile grating support into the public API.
- Added AFM preprocessing tests and a canonical AFM example with anonymized sample scan data.
- Expanded AFM tutorial/docs, including step-by-step plots and updated diagnostics behavior.

## 0.2.0 - 2026-05-26

- Simulation API now defaults to the low-memory solver path for user-facing workflows.
- The former dense path remains internal as `legacy_dense` for regression/debug parity only.
- Example and comparison scripts were updated to match current case-helper interfaces and avoid stale runtime kwargs.
- Flexible optimizer workflows and related examples were expanded/cleaned up across laminar and blazed use cases.
- Optimizer/simulation documentation was refreshed for consistency with the current APIs and tutorials.
- Static compile and compatibility coverage for example and comparison scripts was strengthened.
