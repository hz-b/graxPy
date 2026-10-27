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

## Web app development

- Layout of `src/grax/web`:
  - `app.py` holds `create_app()` (app setup, request guards, shared workspace accessors), `main()` and re-exports of names that tests import.
  - Routes live in `routes/<group>.py`. Each exposes `register(app, ctx)`, which defines the view functions; `ctx` is the `WebContext` from `context.py`.
  - Logic lives in service modules: `common`, `run_state`, `run_files`, `grating_forms`, `run_inputs`, `plane_mirror_views`, `run_jobs`, `run_exports`, `multilayer_jobs`, `catalog`, `plot_builder`, `workspace`. Keep their imports acyclic, and keep new logic out of route modules.
- Keep endpoint names and URLs stable. Templates call `url_for` by endpoint name, so do not convert routes to Flask Blueprints. `tests/unit/test_web_route_inventory.py` pins every rule; update its fixture only for an intended route change.
- Monkeypatch a helper in the module that looks it up (for example `grax.web.run_jobs` or `grax.web.routes.runs`), not in `grax.web.app`. Patching a re-export has no effect.
- Frontend scripts are classic `defer` scripts under `static/js/` that share one global scope and load in the order of `FRONTEND_SCRIPTS` in `common.py`. Add new files to that list, and do not use ES modules, which bypass the `?v=` cache busting.
- `grax-web` runs without the debugger. Use `--debug` only on a local host, and note that the reloader stops running jobs. `--data-dir` selects the database folder, and `--no-browser` skips opening the system browser (the dev-server entry in `.claude/launch.json` uses it).
- Never point tests, scripts or the dev server at the real `.grax-web` database. Use a copy, such as `.grax-web-dev` (git-ignored) or a temporary folder.
- Testing layers for web changes:
  1. Unit and smoke tests, including the route inventory, the page sweep and the security tests, on every change.
  2. Browser tests (`-m browser`), including the page-script error check, after template or JavaScript changes.
  3. A parity check against the previous version (skill `web-parity-check`) after any refactoring or restructuring. Its report must show no unexplained differences.
  4. A live check in the built-in browser (skill `web-live-check`) before opening a PR that changes what users see or do.
