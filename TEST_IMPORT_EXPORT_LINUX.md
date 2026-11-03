# Test import and export on Linux (Ubuntu)

You are checking that the new import and export of gratings, plane mirrors, PGMs and multilayer gratings works on Linux. Do not change application code. Never use a real `.grax-web` folder; these tests use temporary folders only.

## 1. Set up

Requires `git` and `uv`.

```bash
git fetch origin
git switch feature/object-import-export
git pull
uv sync --extra dev --extra web
uv run playwright install --with-deps chromium
```

If `--with-deps` cannot get root rights, run `uv run playwright install chromium`. If Chromium then fails to start, report the missing system libraries.

## 2. Run the tests

```bash
export MPLBACKEND=Agg
uv run pytest tests/unit/test_web_transfer.py tests/unit/test_web_security.py tests/unit/test_web_page_sweep.py tests/unit/test_web_route_inventory.py -q
uv run pytest tests/smoke/test_web_browser.py -m browser -q
uv run pytest tests/unit tests/smoke -m "not browser" --ignore=tests/unit/test_reticolopy_opt.py -q
```

The third command is the same suite as CI, without the coverage options.

## 3. Finish

**If all three commands pass**, delete this file and record that:

```bash
git rm TEST_IMPORT_EXPORT_LINUX.md
git commit -m "Remove Linux import/export test guide after passing tests"
git pull --rebase
git push
```

**If anything fails**, keep this file and don't commit. Report:
- the Ubuntu and Python versions
- the failing test names and their full error output
- whether each failure also happens on `develop`, which shows if the new feature caused it
