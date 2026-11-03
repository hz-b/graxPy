---
name: web-parity-check
description: Prove a grax web change keeps behavior identical by running the old and new website side by side on copies of the real database and diffing every page, export and replayed simulation. Use after refactoring or restructuring anything under src/grax/web.
---

# Web parity check

`tests/characterization/web_parity.py` starts a **baseline** server (reference code) and a
**candidate** server (your working tree), each on its own copy of a database, sends both the
same requests and writes a report. The source database is only copied, never written.

## Steps

1. Pick the reference commit (usually the branch point, e.g. `git merge-base HEAD develop`) and
   check it out into a scratch worktree, outside the repository:

   ```bash
   git worktree add <scratch>/grax-baseline <reference-commit>
   ```

2. Run the driver with the project venv. Tiers:
   - `read`: pages, JSON endpoints, exports and preview POSTs (minutes)
   - `quick`: `read` plus a replay of every saved run, mirror run and multilayer study with axes
     thinned to `--points` (default 3), a run lifecycle (start, abort, resume, restart, delete),
     and re-saving and deleting plots (about 15–30 min)
   - `full`: `quick` plus full-size replays of small saved runs (`--full-max-points`), also
     compared with the results stored in the original database (`--rtol`)

   ```bash
   .venv/bin/python tests/characterization/web_parity.py \
       --baseline-src <scratch>/grax-baseline/src --candidate-src src \
       --database .grax-web --work <scratch>/parity --tier quick
   ```

   Run long tiers in the background. Both servers use snapshots of their code, so you can keep
   editing while it runs. Ports default to 5061/5062 (`--port`).

3. Read `<work>/parity-report.md`:
   - **Differences** must be empty, or each one must be an intended change you can name.
   - **Notes** list same-content-different-order results. Parallel workers finish in any
     order; replays use one worker so these should be rare.
4. Remove the worktree afterwards: `git worktree remove <scratch>/grax-baseline`.

## Interpreting diffs

- A difference in error text or status code is a behavior change, even when it looks harmless.
  Find the cause instead of widening the normalizer.
- The normalizer masks only values that legitimately differ between two servers: timestamps,
  generated IDs, ports, data paths, `?v=` cache tokens and the list of page script tags. Extend
  it only for a new value of that kind, and say so in the PR.
