# Changelog

## 0.4.7 - 2026-07-15
- Added a grating-level `"random-interface"` roughness kind to `grax.RoughnessSpec` as an alternative to solver-level Debye-Waller roughness, applying independent stochastic offsets per multilayer interface instead of a single analytic damping factor.
- Updated the fixed-angle roughness example, comparison script, and tutorial docs to cover both roughness kinds, including new per-sigma grating close-up previews and refreshed comparison artifacts.
- Fixed a `TypeError` in the Octave slag parity test from a stale `LaminarGrating(base_dir=...)` argument, three web-app tests left pointing at `/plots` after the plot-list/plot-form route split, and an optimizer bug that dropped the resolved worker count on the trial failure-penalty path.
- Removed an orphaned test referencing a non-existent `examples/reference_baselines/` example.
- Reorganized `tests/` and `reticolo/tests/` into `unit/` (fast, mocked) and `smoke/` (example scripts, Octave/RETICOLO parity) subdirectories, with shared test builders moved into `tests/simulation_helpers.py`.
- Added GitHub Actions CI (`.github/workflows/tests.yml`) running the unit and smoke suites on every push/PR, plus `TESTING.md` and `tests/README.md` documenting how to run tests locally.

## 0.4.6 - 2026-07-14
- Added `grax.RoughnessSpec` for construction-time roughness selection, including solver-level Debye-Waller roughness and grating-level stochastic interface roughness.
- Added trial-level optimizer multiprocessing through `BatchSimulationRunner`, so measurement-fit optimizer evaluations can use the existing batch `max_workers` worker pool.
- Added optimizer config/runtime metadata for requested and resolved worker counts, plus validation that prevents combining per-trial multiprocessing with Ax candidate batching.
- Added a maintained fixed-angle roughness simulation example with live plotting, multiprocessing, first-order comparison output, and tutorial/docs integration including synced tutorial images.
- Added flat chemical-formula material support through `grax.MaterialSpec`, so compounds such as `SiO2`, `Al2O3`, `B4C`, and similar Henke-derived formulas can be resolved even when no bundled compound table exists.
- Combined elemental Henke scattering factors and density-driven refractive-index reconstruction into the material runtime while preserving the existing elemental string workflow.
- Expanded material validation, tests, and documentation for formula parsing, required compound densities, and unknown-element handling.
- Updated the Web UI and persistence flow so formula-based `MaterialSpec` inputs round-trip cleanly and free-text compound materials can be entered alongside packaged elemental suggestions.

## 0.4.5 - 2026-06-26
- Web UI: add global CPU-pool resource manager, saved-plots list page with per-plot deletion, and dynamic version display in the docs sidebar and homepage.

## 0.4.4 - 2026-06-18
- Numba is now the dafault solver backend. Numpy will be deprecated in a future release, but is still available as an explicit `solver_backend` option for now.
- use Plotly for the web-ui plotting. minor improvements to the web-ui.
- implement s and p polarization cases

## 0.4.3 - 2026-06-17
- Split the AFM preprocessing examples into explicit blazed and laminar workflows, each with its own saved diagnostics folder and sample data file.
- Added a laminar AFM trough-detection mode that places troughs at the midpoint between consecutive vertical walls while keeping the existing blazed detection path unchanged.
- Updated the AFM tutorial, example scripts, and docs image sync to match the new profile-type switch and example output locations.
- Added density-aware material resolution with an explicit `MaterialSpec` density override, shared Henke-backed material selection in the Web UI, and deprecation warnings for xrt-backed material inputs.
- Reorganized the Web UI grating form into separate substrate, layer-stack, and top-cap sections so the material workflow matches the physical stack layout more clearly.

## 0.4.2 - 2026-06-17
- Package the Web UI templates and static assets so `grax-web` works from an installed wheel as well as from editable source checkouts.
  

## 0.4.1 - 2026-06-17

- Add web-UI documentation.
 
## 0.4.0 - 2026-06-11

- Replaced the Web UI plotting path with interactive Plotly figures for saved comparisons, live previews, and live run monitoring.
- Kept the existing server-side export workflow while switching plot export generation to Plotly-backed PNG rendering.
- Saved Web UI plot artifacts now persist as figure JSON specs instead of only static PNG files.
- Clarified the batch-simulations tutorial and runner docs with an explicit `cases` dictionary example and guidance on preserved per-case metadata.
- Improved parameter-study failure reporting so failed sweep points keep `NaN` efficiency, record `error_message` in CSV output, and are plotted separately from valid efficiency curves.
- Updated Web UI installation guidance and runtime dependency messages to distinguish PyPI installs (`graxpy[web]`) from editable local installs (`-e ".[web]"`).
- Made AFM trough detection more robust for laminar-like scans by adding prominence-based filtering and documenting how to tune it for shallow secondary minima.
- Added earlier, explicit simulation-time material validation for bare string material names and aligned AFM tutorials/examples with real optical-constants objects.

## 0.3.0 - 2026-06-11

- Introduced the first local web UI for `grax`, including abort-only run control, plot export browsing, and web startup documentation.
- Added local web run controls to use a single abort flow with explicit save-or-delete confirmation for partial runs.
- Added a dedicated web UI installation section for Linux/macOS and Windows, plus startup examples for custom ports.
- Made the `grax-web` entrypoint accept `--host` and `--port` so the web server can be started on a different port without editing code.
- Replaced the plot export dialog with a classic in-web folder browser and default-hidden dotfile handling.

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
