# Test import and export on Windows

You are checking that the new import and export of gratings, plane mirrors, PGMs and multilayer gratings works on Windows. Do not change application code. Never use a real `.grax-web` folder; these tests use temporary folders only.

Windows is the riskiest system for this feature, so watch for:
- files that cannot be moved or deleted because they are briefly locked (`PermissionError`, `WinError 5` or `WinError 32`)
- backslashes in paths
- long paths

## 1. Set up

Requires `git` and `uv`. Use PowerShell.

```powershell
git fetch origin
git switch feature/object-import-export
git pull
uv sync --extra dev --extra web
uv run playwright install chromium
```

## 2. Run the tests

```powershell
$env:MPLBACKEND = "Agg"
uv run pytest tests/unit/test_web_transfer.py tests/unit/test_web_security.py tests/unit/test_web_page_sweep.py tests/unit/test_web_route_inventory.py -q
uv run pytest tests/smoke/test_web_browser.py -m browser -q
uv run pytest tests/unit -m "not browser" -k "web" --ignore=tests/unit/test_reticolopy_opt.py -q
```

The third command is the same suite as the Windows CI job.

## 3. Finish

**If all three commands pass**, delete this file and record that:

```powershell
git rm TEST_IMPORT_EXPORT_WINDOWS.md
git commit -m "Remove Windows import/export test guide after passing tests"
git pull --rebase
git push
```

**If anything fails**, keep this file and don't commit. Report:
- the Windows and Python versions
- the failing test names and their full error output
- whether each failure also happens on `develop`, which shows if the new feature caused it
