# Changelog

## Unreleased

### Breaking changes
- Renamed `grax.simulation.RCWASimulation` to `GratingSimulation`, with no alias. The class drives whichever solver `solver=` selects, so the old name described only one of the two things it can do. Replace `RCWASimulation(...)` with `GratingSimulation(...)`; the arguments and behaviour are unchanged.

- Added a second electromagnetic solver, the Nevière differential method, selectable with `grax.run_simulation(..., solver="neviere")` and `BatchSimulationRunner(default_solver=...)` (per-case `"solver"` also works). It expands the fields in the same truncated Fourier basis as RCWA and applies the same Li/fast-Fourier-factorization rules for TM, but integrates the coupled first-order system in `z` with fourth-order Runge-Kutta instead of eigen-decomposing each layer. Every default stays `"rcwa"`, so existing code, checkpoints and artifacts are unaffected. References: Nevière, Vincent & Petit, *Nouv. Rev. Optique* **5**, 65 (1974); Nevière, *JOSA A* **11**, 1835 (1994); Nevière & Popov, *Light Propagation in Periodic Media* (CRC, 2003).
- `grax.NeviereOptions` controls the differential method's integration. Step and sub-block sizes are given in optical phase rather than nanometers, so one setting behaves consistently across photon energies, grazing angles and truncation orders. The Fourier truncation order remains the existing `fourier_orders` argument.
- `NeviereOptions(z_sampling="continuous")` drops the staircase approximation both solvers otherwise share: the permittivity is re-expanded from the true grating profile every `sample_phase` of optical depth, so the result no longer depends on `z_resolution_nm`.
- The differential method is numerically more robust on deep gratings. The modal solver evaluates `q / sinh(q d)` across a whole layer and overflows above roughly seven wavelengths of depth for a high-contrast lamellar grating; the differential method caps the optical thickness of anything it forms explicitly and still conserves energy to `1e-9` at 167 wavelengths.
- `SingleSimulationResult` and `CaseExecutionResult` gained a `solver` field, round-tripped through checkpoints so a resumed sweep keeps that provenance.
- Split the 1D solver into a `grax.solvers` package: `solvers/common.py` holds the shared types, `res0`/`res1`, the Fourier machinery, the layer field operators, the interface cascade and the efficiency extraction; `solvers/rcwa.py` holds the modal layer solve and `res2`; `solvers/neviere.py` holds the differential method. `grax.rcwa_1d` remains as a re-export shim, and the RCWA numerics are unchanged (verified bit-identical across laminar, blazed, multilayer and sinusoidal cases in both polarizations).
- Solver selection now reaches every workflow. `run_multilayer_theta_search`, `run_multilayer_theta_search_sweep`, `run_parameter_study`, the `grax_opt` measurement fits and the web UI run form all accept a solver, alongside the existing `run_simulation` and `BatchSimulationRunner` support. Every default stays `"rcwa"`.
- Fixed the multilayer theta-search workflow silently ignoring `solver=`. `BatchSimulationRunner(default_solver="neviere")` computed those cases with RCWA, with no error or warning, because the workflow's payload carried `backend` but not `solver`. Both runner-settings mappings now come from one place so a new setting cannot reach one execution path and miss the other.
- `run_parameter_study` gained `backend`, which it previously hardcoded to `"numba"` internally.
- Results now record `solver_options` alongside `solver`, so a checkpointed differential-method run pins the integration settings that produced it and not just the solver name.
- Reorganized each validation case into a `grating_definition.py` holding the grating and sweep grid, plus a `run_rcwa.py` and a `run_neviere.py` that import it. Both solvers therefore see identical geometry, energy grid and truncation by construction. Each case's comparison script now overlays both solver curves alongside the external reference codes.
- Fixed the validation sweep scripts crashing with `BrokenProcessPool` on macOS: the batch runner spawns workers there, and without a `__main__` guard each spawned worker re-ran the whole sweep and recursively spawned more workers.
- `random-interface` roughness is now a correlated Gaussian random field (Gaussian autocorrelation) instead of per-sample white noise. `grax.RoughnessSpec` gains `correlation_length_nm`: the lateral autocorrelation length in nanometers, defaulting to one tenth of the grating period (`0.0` reproduces the previous white-noise interface). This produces physically smooth interfaces; long correlation lengths (much larger than the grating period) wash out geometrically and are better modelled by the `debye-waller` kind.
- Added per-layer roughness: `LayerSpec` accepts `roughness_sigma_nm`, and `SingleLayerStack`/`MultilayerStack`/`CustomStack` expose per-layer/per-material roughness arguments (plus a `substrate_roughness_sigma_nm` for the substrate boundary). Each value sets the roughness of that interface, falling back to the grating-level `RoughnessSpec.sigma_nm` when unset. `random-interface` perturbs each interface with its own sigma; per-layer `debye-waller` sigmas combine in quadrature into an effective damping.
- Web UI: the grating form now has a roughness sigma field for the substrate, each coating layer/material, and the top cap (persisted with the grating, schema v2). The run form has a roughness-kind dropdown (None / Debye-Waller / Random interface) that selects the model applied at run time.
- Fixed `live_plot` batch runs raising/re-focusing the plot window on every simulation point on macOS, which made the window impossible to minimize or send behind another app. The interactive figure is now only shown once (via `plt.show()`) and subsequently just redrawn in place.
- Fixed macOS batch runs crashing or exhausting memory on larger supercell counts: multiprocessing workers now use the `spawn` start method on macOS (`fork` is unsafe once native BLAS/Numba/Cocoa state has been initialized in the parent), and `auto` worker-count calibration now sizes from each solve's peak RSS high-water mark instead of only its steady-state RSS, which understated the transient memory a large-supercell solve actually needs.

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
