# Web UI

The local web UI is installed with the `web` extra and started with the
`grax-web` command.

## Linux and macOS

Create and activate a virtual environment if you do not already have one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[web]"
grax-web
```

Open <http://127.0.0.1:5050>.

To use a different port:

```bash
grax-web --port 8000
```

To change the bind address as well:

```bash
grax-web --host 0.0.0.0 --port 8000
```

## Windows

Create and activate a virtual environment in PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[web]"
grax-web
```

Open <http://127.0.0.1:5050>.

To use a different port:

```powershell
grax-web --port 8000
```

The web UI stores local data in `.grax-web/` by default:

- `saved_gratings/`
- `runs/`
- `plots/`
- `previews/`

When you change the local web UI code, restart `grax-web` so the browser sees
the updated server behavior.
