# Testing

## Setup

```bash
uv sync --extra dev --extra web
```

`web` is needed because `tests/unit/test_web_app.py` exercises the Flask UI (`grax.web`).

## Running tests

```bash
uv run pytest                          # everything, including local-only reticolo/ parity tests
uv run pytest tests/unit tests/smoke -m "not browser"  # fast CI suite (no reticolo/browser)
uv run pytest tests/unit               # fast, fully-mocked tests only
uv run pytest tests/smoke -m "not browser"  # example scripts + external references
uv run playwright install chromium     # one-time setup for the browser checks
uv run pytest tests/smoke/test_web_browser.py -m browser -v
```

- `tests/unit/` — fast tests with no external dependencies.
- `tests/smoke/` — end-to-end tests: running example scripts with reduced/"quick" settings, and comparisons against an Octave/RETICOLO reference (self-skips if `octave` isn't on `PATH`).
- `tests/smoke/test_web_browser.py` — headless Chromium checks for Flask navigation, JavaScript form behavior, and a real two-point energy sweep through result download. The test server writes only to a temporary data directory.
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
uv run pytest tests/unit tests/smoke -m "not browser" --cov=grax --cov-report=term-missing
```

## CI

See `.github/workflows/tests.yml`. The existing unit and smoke job installs the `dev` and `web` extras only (no `opt`, no GPU/CUDA), sets `MPLBACKEND=Agg` (several example scripts default to an interactive backend), and runs `tests/unit` plus non-browser `tests/smoke` on its current push and pull request triggers. A separate job runs the browser checks on pushes to `develop`, installs Chromium, and uploads a screenshot, Playwright trace, and run files if a browser check fails.
