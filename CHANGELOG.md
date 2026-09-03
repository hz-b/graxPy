# Changelog

## 0.4.8 - 2026-09-03

- grax log records no longer leak to the terminal when the host application has not called `grax.setup_logging`. A `NullHandler` is attached to the `grax` logger at import, so Python's `logging.lastResort` handler no longer prints `WARNING`s to stderr -- most visibly from the spawned batch workers, which re-import grax but never configure logging. `setup_logging` now also sets `propagate = False` on the `grax` logger and refuses to attach a second file handler for the same path on a repeat call.
- The multilayer theta-search `Requested theta half-width ... reaches near/into 0 deg. Reducing to ...` notice is now `INFO`, not `WARNING`, and is emitted once per distinct `(center, requested half-width)` instead of on every rough/fine re-centring attempt. At small grazing angles the clamp is expected behaviour, not a fault; it belongs in the log file, not the console.
- `multilayer_theta_search_cases` gained `solver` and `solver_options`, matching `run_multilayer_theta_search` and the other case generators. The batch runner already read a per-case `"solver"` key; the generator just could not set one, so cases built directly could only take the solver from the runner default.
- The `multilayer_theta_search` example takes `--polarization {s,p,TE,TM}` alongside `--solver`, and writes each run under `results/<solver>_<polarization>/` (logs and checkpoints included) so an s run and a p run, or an rcwa and a neviere run, sit side by side.

- Added `grax_opt.optimize_to_joint_measurements` for fitting one parameter set jointly against several measured curves, each with its own measurement file and energy grid, with a configurable `joint_loss_reduction` (`mean`, `sum`, `pooled`, or `weighted`). Each trial evaluates every measurement in a single `BatchSimulationRunner` batch so trial-level `max_workers` parallelizes across measurements as well as energies.
- Joint fits are not limited to curves that differ by grazing angle. `angle_mode`, `grazing_angle_deg`, `cff`, `diffraction_order` and `polarization` are run-level defaults that any individual `MeasurementSpec` may override, so one fit can span angles, angle modes, diffraction orders and polarizations. Numerical settings (`fourier_orders`, `solver`, `backend`, `max_workers`) stay run-level, since they describe how a curve is computed rather than what was measured.
- `grax_opt` measurement fits now take `polarization`, accepting `s`/`p` or `TE`/`TM`. Both optimizer entrypoints previously had no such argument anywhere, so every fit silently ran the `run_simulation` default of `s` -- the same gap the theta-search workflow had. Default stays `s`.
- `optimize_to_joint_measurements` accepts `solver` and `solver_options`, matching `optimize_to_measurements`, and records them in `best_result.json`. Both are part of the resume fingerprint, alongside each measurement's resolved conditions, so a resumed run cannot silently switch the physics it is fitting.
- Added `examples/optimizer/optimizer_joint/`, the first runnable joint-fit workflow: it simulates a four-condition measurement set from a known grating, fits it, resumes the fit to extend the trial budget, and reports how close each parameter came to the value the data was generated from. It is also the first optimizer example with a `--solver` flag; the measurements are always generated with `rcwa`, so fitting them with `--solver neviere` recovers the same parameters to within a few parts in ten thousand.
- Fixed an interrupted optimizer run over-running its trial budget on resume. The trial-record log is appended every trial but the Ax snapshot is only rewritten every `checkpoint_interval` trials, so an interruption left the log ahead of the snapshot; the recovered history then counted trials Ax was about to generate again, duplicating a row in `trial_history.csv` and running past `total_trials`. Records at or beyond the reconciled cursor are now discarded, since Ax is authoritative for what was issued. Only reachable with `checkpoint_interval > 1`, which is why the default of `1` hid it.
- The single-measurement trial evaluation now raises on a missing batch result instead of reading uninitialized memory. It allocated its efficiency array with `np.empty` and filled by result index, so a case the runner never returned left whatever was in memory as that point's efficiency; the joint path already guarded against this.
- An unreadable measurement file now raises when the resume fingerprint is built instead of hashing to a shared `"unreadable"` sentinel, which made two different unreadable files compare equal. Both entrypoints load their measurements before fingerprinting, so this is a guard against the ordering changing rather than a bug that could be hit today.
- Added checkpoint and resume support to `optimize_to_measurements` and `optimize_to_joint_measurements` through the new `resume`, `checkpoint_dir`, and `checkpoint_interval` spec keys. The Ax client snapshot is persisted alongside optimizer run state and an append-only trial log, so a resumed run keeps its surrogate model and `total_trials` can be raised to extend a finished run. A problem fingerprint refuses to resume into a changed search space, naming the settings that differ.
- Fixed joint multi-angle evaluation assigning simulated efficiencies to the wrong angle/energy slot when `BatchSimulationRunner` returned results in completion order rather than input order; results are now reassembled by `CaseExecutionResult.index`.
- Extracted the shared Ax trial loop into `grax_opt.loop.run_ax_trial_loop`, so both optimizer entrypoints share candidate generation, best-so-far tracking, and early stopping. `early_stopping_min_relative_improvement` is now honored instead of being validated and ignored.
- Optimizer artifacts are now written atomically, the best-fit plot reuses the winning trial's cached simulated curve instead of re-simulating it after every trial, and a penalized trial logs the failing case or exception instead of failing silently.

- Every example that solves now accepts `--solver rcwa|neviere`, defaulting to `rcwa`. Solver-dependent example outputs are suffixed (`*_rcwa.*`, `*_neviere.*`) so the two runs sit side by side; geometry artifacts such as `*_profile.png` stay unsuffixed because they do not depend on the solver.
- Added three examples covering the cases where the two solvers genuinely differ, rather than duplicating existing examples whose curves would be indistinguishable: `deep_grating_limits` (the modal solver stops at 8.4 wavelengths of depth where the differential method reaches 167), `continuous_vs_staircase` (continuous z-sampling is bit-identical across every `z_resolution_nm`, while a staircase run carries 2.7e-3 of discretization error at 2 nm), and `solver_runtime` (2.4x to 3.0x faster at production resolution, with the solvers still agreeing to ~1e-11).
- Fixed `multilayer_theta_search` and `blazed_multilayer_memory_comparison` failing with `BrokenProcessPool` on macOS. Both use `max_workers`, and a spawned worker re-imports the script by path; without a `__main__` guard each worker re-ran the whole example and recursively spawned more.

- Fixed `run_multilayer_theta_search_sweep` reporting `solver="rcwa"` on every case regardless of the solver requested. The sweep builds its own `CaseExecutionResult` and copied only the theta-search diagnostics from the underlying result, so `solver` and `solver_options` fell back to their dataclass defaults. The computation was correct throughout -- an rcwa and a neviere sweep differ by ~5e-14 -- but a saved sweep recorded the wrong provenance.
- Extended the both-solver test coverage to every entry point the examples use: `monochromator_cases`, `energy_angle_cases`, `run_parameter_study`, `run_multilayer_theta_search_sweep`, `assemble_custom_stack`/`LayerSpec`, `AFMGrating`, and the `write_all_orders_csv`/`plot_order_subset`/`efficiency_for_order` output helpers. Swapping `--solver` on any example now exercises a tested path.

- Polarization arguments now accept `TE` and `TM` alongside `s` and `p`, case-insensitively, through a single shared `grax.normalize_polarization`. The library already spoke TE/TM internally (`res0`, `_solve_te_stack`) and in its theory docs while the public API took only s/p; these are the same two states. Values canonicalize to `s`/`p`, so results, CSVs and checkpoints are unchanged. Note the equivalence holds in classical mounting, which is the only mounting this solver supports.
- The multilayer theta-search workflow now takes `polarization`. It previously had no such argument anywhere, so every theta search ran the `run_simulation` default of `s` -- awkward given all four validation cases run `p`. `run_multilayer_theta_search`, `run_multilayer_theta_search_sweep`, `multilayer_theta_search_cases` and the web form all accept it, and it reaches all three stages of the search. Default stays `s`.
- `run_parameter_study` now validates `polarization` at the entry point instead of failing several frames deeper inside the simulation wrapper.

### Breaking changes

Renames only — no behaviour changed. Migration:

| old | new |
| --- | --- |
| `RCWASimulation` | `GratingSimulation` |
| `BatchSimulationRunner(default_diffraction_order=)` | `diffraction_order=` |
| `BatchSimulationRunner(default_fourier_orders=)` | `fourier_orders=` |
| `BatchSimulationRunner(default_polarization=)` | `polarization=` |
| `BatchSimulationRunner(default_solver=)` | `solver=` |
| `neviere_options=` (all entrypoints) | `solver_options=` |
| `run_simulation(min_efficiency=)` | `min_reflected_efficiency=` |

- Renamed `grax.simulation.RCWASimulation` to `GratingSimulation`. The class drives whichever solver `solver=` selects, so the old name described only one of the two things it can do.
- Dropped the `default_` prefix from the `BatchSimulationRunner` settings. The prefix was accurate — each of these is the value a case inherits when it omits its own key — but it made `default_fourier_orders: int = 25` read as the default of a default, and the runner's arguments now mirror `run_simulation`'s exactly. The per-case override relationship is documented on each argument instead.
- Renamed the `neviere_options` argument to `solver_options` everywhere, matching the `solver_options` field already present on results. One name in both directions, and the generic runner no longer names a specific solver in its API. The `NeviereOptions` class keeps its name.
- Renamed `min_efficiency` to `min_reflected_efficiency` on `run_simulation`, `GratingSimulation` and `run_multilayer_theta_search`. It was half of a min/max pair whose other half was already `max_reflected_efficiency`, and the batch runner had to translate between the two spellings.

- Added a second electromagnetic solver, the Nevière differential method, selectable with `grax.run_simulation(..., solver="neviere")` and `BatchSimulationRunner(default_solver=...)` (per-case `"solver"` also works). It expands the fields in the same truncated Fourier basis as RCWA and applies the same Li/fast-Fourier-factorization rules for TM, but integrates the coupled first-order system in `z` with fourth-order Runge-Kutta instead of eigen-decomposing each layer. Every default stays `"rcwa"`, so existing code, checkpoints and artifacts are unaffected. References: Nevière, Vincent & Petit, *Nouv. Rev. Optique* **5**, 65 (1974); Nevière, *JOSA A* **11**, 1835 (1994); Nevière & Popov, *Light Propagation in Periodic Media* (CRC, 2003).
- `grax.NeviereOptions` controls the differential method's integration. Step and sub-block sizes are given in optical phase rather than nanometers, so one setting behaves consistently across photon energies, grazing angles and truncation orders. The Fourier truncation order remains the existing `fourier_orders` argument.
- `NeviereOptions(z_sampling="continuous")` drops the staircase approximation both solvers otherwise share: the permittivity is re-expanded from the true grating profile every `sample_phase` of optical depth, so the result no longer depends on `z_resolution_nm`.
- The differential method is numerically more robust on deep gratings. The modal solver evaluates `q / sinh(q d)` across a whole layer and overflows above roughly seven wavelengths of depth for a high-contrast lamellar grating; the differential method caps the optical thickness of anything it forms explicitly and still conserves energy to `1e-9` at 167 wavelengths.
- `SingleSimulationResult` and `CaseExecutionResult` gained a `solver` field, round-tripped through checkpoints so a resumed sweep keeps that provenance.
- Split the 1D solver into a `grax.solvers` package: `solvers/common.py` holds the shared types, `res0`/`res1`, the Fourier machinery, the layer field operators, the interface cascade and the efficiency extraction; `solvers/rcwa.py` holds the modal layer solve and `res2`; `solvers/neviere.py` holds the differential method. `grax.rcwa_1d` remains as a re-export shim, and the RCWA numerics are unchanged (verified bit-identical across laminar, blazed, multilayer and sinusoidal cases in both polarizations).
- The batch progress bar now names the solver actually running (`neviere batch`) instead of always printing `RCWA batch`, and the solver-agnostic docstrings on `SingleSimulationResult`, `BatchSimulationRunner`, `gratings.py` and `run_parameter_study` no longer describe themselves as RCWA-specific.
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
