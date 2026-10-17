# Repository guidance

## Changelog and release version

- For every change, update `CHANGELOG.md` with short, human-readable sentences that explain the user-facing purpose or effect. Do not describe implementation details or write entries for machines.
- Record these project-wide workflow rules in this file so they apply to future work in the repository.
- Do not change the package version in `pyproject.toml` during ordinary development. The release tool owns version updates.
- A changelog section may be prepared for the next release without changing `pyproject.toml`.
