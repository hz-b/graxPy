## Web run abort flow and web UI startup docs

- Removed the web pause/resume workflow after repeated control-flow failures in the browser/runtime combination.
- Replaced it with a single abort flow:
  - `Abort run` opens a confirmation page
  - the user chooses to save the partial run as `aborted` or delete it entirely
- Saved partial runs keep checkpoint-backed artifacts and remain available for plotting.
- Incomplete persisted states such as `paused`, `interrupted`, and stale `running` runs now normalize to `aborted` in the web UI.
- The local `grax-web` entrypoint now accepts `--host` and `--port` so developers can start the server on a different port without editing code.
- Added explicit web UI installation and startup instructions for Linux/macOS and Windows in the docs and README.
