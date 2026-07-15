# Testing

## Setup

```bash
uv sync --extra dev --extra web
```

`web` is needed because `tests/unit/test_web_app.py` exercises the Flask UI (`grax.web`).

## Running tests

```bash
uv run pytest                          # everything, including local-only reticolo/ parity tests
uv run pytest tests/unit tests/smoke   # everything CI runs (no reticolo/)
uv run pytest tests/unit               # fast, fully-mocked tests only
uv run pytest tests/smoke              # example scripts + external reference comparisons
```

- `tests/unit/` — fast tests with no external dependencies.
- `tests/smoke/` — end-to-end tests: running example scripts with reduced/"quick" settings, and comparisons against an Octave/RETICOLO reference (self-skips if `octave` isn't on `PATH`).
- `reticolo/tests/` — RETICOLO parity tests. This directory is **git-ignored** (bundles licensed RETICOLO v9 MATLAB reference material) and only exists locally, so it never runs in CI. Run it locally with `uv run pytest reticolo/tests` if you have that data checked out and (optionally) Octave installed.

## Optimizer tests

`tests/unit/test_reticolopy_opt.py` covers `grax_opt`, which depends on `torch`/`ax-platform` (the `opt` extra). It's excluded from CI to avoid pulling in that heavy, GPU-adjacent dependency stack on every push. Run it locally:

```bash
uv sync --extra dev --extra opt
uv run pytest tests/unit/test_reticolopy_opt.py
```

## Coverage

CI runs with `--cov=grax --cov-report=term-missing --cov-report=xml` and uploads `coverage.xml` as a build artifact (not gated on a threshold). Run the same locally:

```bash
uv run pytest tests/unit tests/smoke --cov=grax --cov-report=term-missing
```

## CI

See `.github/workflows/tests.yml`. It installs the `dev` and `web` extras only (no `opt`, no GPU/CUDA), sets `MPLBACKEND=Agg` (several example scripts default to an interactive backend), and runs `tests/unit` + `tests/smoke`.
