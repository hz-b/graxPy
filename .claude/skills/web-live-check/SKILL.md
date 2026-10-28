---
name: web-live-check
description: Exercise the running grax website in the built-in browser on a disposable copy of the database — forms, every scan type, plots and run controls — and collect console and network errors. Use to confirm a web change works for a person clicking through the site.
---

# Live website check

## Server

1. Refresh the disposable database copy, never pointing the server at `.grax-web` itself:

   ```bash
   rsync -a --delete .grax-web/ .grax-web-dev/
   ```

2. Reuse a running `grax-web` server if one is up (`preview_list`). Otherwise start the
   `grax-web` entry of `.claude/launch.json` with `preview_start`. It serves `.grax-web-dev` on
   port 5050 without the debugger or a system browser window. Restart it only after Python or template changes, and stop
   the old process first.
3. Use one browser tab for the whole check. Read pages with `get_page_text` / `read_page`, and
   take screenshots only for visual questions.

## Checklist

After each page, check `read_console_messages` (errors only) and `read_network_requests` for
4xx/5xx responses.

1. Home, guide pages, settings menu: switch decimal comma and back, reload, and the choice persists.
2. New grating: change type and stack, the live preview updates, save, edit, rename, open the delete prompt (cancel).
3. AFM grating: upload a sample from `tests/` data, preview, process, save.
4. On a grating, run each scan type with few points and one worker: energy sweep, angle sweep,
   fixed angle, monochromator (CFF), constant deviation, parameter study, and a custom grid with
   (a) only independent axes, (b) only a coupled group, (c) both. Watch progress and the RAM
   readout, then open the result, download CSV and the ZIP export.
5. Abort a longer run with "save", resume it, restart it, then abort with "delete".
6. Plots: create a plot from two runs, change series styles and loop selection, save, edit,
   download PNG, data and script, then delete.
7. Plane mirror: stack preview, compute, save a scan, run a scan with an independent range and
   a coupled row, abort/resume.
8. Multilayer design: create a small study, run the survey, watch status, abort, reset, run
   again, then run the energy scan and download the script.
9. Manage pages for gratings, runs and plane mirrors: rename, comment, bulk delete with confirmation.
10. Cross-site guard:
    `curl -s -o /dev/null -w "%{http_code}" -X POST -H "Origin: http://evil.example" http://127.0.0.1:5050/settings`
    returns `403`.

Report what you checked, anything that failed (with console or network evidence), and anything
you skipped.
