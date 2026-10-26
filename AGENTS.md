# Repository guidance

## Changelog and release version

- For every code change beyond a truly small edit, add one concise, human-readable line to `CHANGELOG.md` explaining its purpose or user-facing effect. Write for people, not machines; avoid implementation details.
- Record these project-wide workflow rules in this file so they apply to future work in the repository.
- Do not change the package version in `pyproject.toml` during ordinary development. The release tool owns version updates.
- A changelog section may be prepared for the next release without changing `pyproject.toml`.

## Bug fixes and website testing

- Unless the user instructs otherwise, create or switch to the `bug-fixes` branch before fixing a bug, and do the bug-fix work there. Reuse that branch if it already exists.
- Run project tests with the Python environment in the repository's `.venv` (for example, `.venv/bin/python -m pytest ...` or `uv run --python .venv/bin/python pytest ...`); do not use a global Python environment. If `.venv` is missing or unusable, tell the user how to create it with `bash tools/bootstrap.sh` (requires `uv`) and wait for them to install it before running tests.
- Before starting the website for testing, check whether a development server is already running and reuse it. Keep one server on the same port through repeated checks; restart it only when a change requires a restart, and stop the old process before starting its replacement.
- Do not open a new visible browser page each time the website is tested. Reuse an existing tab when visual inspection is needed. Prefer headless browser tests and direct test-client or HTTP checks when no visible browser interaction is needed.

## GitHub Actions tests

- GitHub Actions runs unit and smoke tests with `uv sync --extra dev --extra web`; keep every package imported by those tests declared in the base dependencies or in an extra installed by that workflow. Keep browser-only requirements in the `dev` extra and install the matching browser in the workflow.
- Before pushing changes, run the same relevant commands used by `.github/workflows/tests.yml` locally. Keep tests aligned with intentional interface changes, and check that failures are not caused by stale expectations, undeclared dependencies, missing optional extras, or missing test resources.
