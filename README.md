# grax

`grax` is an independent Python package for diffraction-grating simulations in
X-ray optics, inspired by RETICOLO v9 and extended with higher-level workflows
for practical studies.

## Documentation

Full user and API documentation is published online.

- User guide and tutorials: see the project documentation site
- API reference: see the API section in the documentation site

For local docs builds from this repository, use:

```bash
tools/build_docs.sh --html
```

## Installation

`graxpy` supports Python `3.12` and `3.13` only.

```bash
python -m pip install graxpy
```

PyPI project page: <https://pypi.org/project/graxpy/0.1.0/>

For local editable installs:

```bash
python -m pip install -e .
```

## Local web app

Install the package and web extra, then start the local server.

From PyPI:

```bash
python -m pip install "graxpy[web]"
grax-web
```

From a local repository checkout:

```bash
python -m pip install -e ".[web]"
grax-web
```

The syntax `graxpy.[web]` is invalid. The extra must be attached directly to
the package name: `graxpy[web]`.

Then open <http://127.0.0.1:5050>. Use the home page to create and save
gratings, then open the plot page to combine saved runs and select the
diffraction orders to overlay. Plot previews, saved comparison pages, and
run-progress plots are interactive Plotly figures in the browser.

Start on a different port when needed:

```bash
grax-web --port 8000
```

You can also override the bind address:

```bash
grax-web --host 0.0.0.0 --port 8000
```

When developing the web app locally, restart `grax-web` after changing run-state
or UI logic so the browser sees the updated server behavior.

Local data is stored in `.grax-web/` by default:

- saved gratings: `.grax-web/saved_gratings/`
- run results: `.grax-web/runs/`
- combined plots: `.grax-web/plots/`
- grating previews: `.grax-web/previews/`

Each saved run lives in `.grax-web/runs/<run_id>/` and includes:

- `manifest.json`
- `summary.csv`
- `all_orders.csv`
- `selected_efficiency.json`

Use `Plots` to combine saved runs and choose which diffraction orders to
overlay for each run. Interactive plots can still be exported as PNG files
through the built-in folder browser. Use `Manage runs` to rename runs or bulk
delete them.

## Repository at a glance

- `src/grax/`: core package source code
- `examples/`: runnable examples
- `docs/`: documentation sources

## Attribution

`grax` is inspired by RETICOLO v9. This project is an independent Python
implementation and is not an official RETICOLO distribution. RETICOLO is not
bundled as part of the public `graxpy` package distribution.

- RETICOLO DOI: <https://doi.org/10.5281/zenodo.14631950>
- RETICOLO license (CC BY 4.0): <https://creativecommons.org/licenses/by/4.0/>

## License

Copyright (C) [2026] [Helmholtz-Berlin fur Materialen und Energie GmbH (HZB)]

Licensed under the European Union Public License (EUPL), Version 1.2.

You may not use this work except in compliance with the License.

A copy of the License is available at:
<https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>
