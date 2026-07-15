# tests/

## Layout

- `unit/` — fast, fully-mocked tests, no external dependencies.
  - `test_afm.py` — AFM grating preprocessing
  - `test_gratings.py` — grating geometry, roughness spec
  - `test_materials.py` — material/refractive-index resolution
  - `test_parameter_sweep.py` — parameter study sweeps
  - `test_profiling.py` — solver profiling/memory, cascade optimizations
  - `test_web_app.py` — Flask web UI (`grax.web`)
  - `test_reticolopy_opt.py` — `grax_opt` optimizer (needs the `opt` extra)
  - `test_simulation_core.py` — roughness, RCWA core, batch runner, theta-search logic
  - `test_simulation_theta_search.py` — theta-search sweep resume/retry/tracking logic
- `smoke/` — end-to-end tests: run example scripts with reduced settings, or compare against an
  external reference.
  - `test_simulation_examples.py` — example/optimizer scripts compile and run with quick configs
  - `test_simulation_reticolo_parity.py` — RCWA solver vs. RETICOLO/Octave reference values
- `conftest.py`, `optical_constants.py`, `simulation_helpers.py` — shared fixtures and test
  builders, imported as `tests.optical_constants` / `tests.simulation_helpers`.

`reticolo/tests/` (repo root, git-ignored, local-only) holds additional RETICOLO parity tests;
see the top-level `TESTING.md`.

## Running

```bash
uv sync --extra dev --extra web
uv run pytest                 # everything under tests/
uv run pytest tests/unit      # fast tests only
uv run pytest tests/smoke     # example/parity smoke tests only
```

Optimizer tests need `uv sync --extra dev --extra opt` first (not installed in CI — see
`.github/workflows/tests.yml`).
