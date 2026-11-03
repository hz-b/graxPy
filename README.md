# GraxPy

`graxpy` is an independent Python package for one-dimensional X-ray
diffraction-grating simulations. Its public Python import is `grax`.

`grax` provides two selectable electromagnetic solver paths:

- modal rigorous coupled-wave analysis (RCWA), inspired by RETICOLO v9;
- the Nevière differential method.

Both solvers support the same grating, material, and workflow APIs. They share
the Fourier/discretization infrastructure and differ in how they propagate the
fields through a layer.

## Documentation

Full user and API documentation is available at
[graxpy.readthedocs.io](https://graxpy.readthedocs.io/).

- User guide and tutorials: <https://graxpy.readthedocs.io/>
- API reference: <https://graxpy.readthedocs.io/en/latest/api/index.html>

For local docs builds from this repository, use:

```bash
tools/build_docs.sh --html
```

## Installation

`graxpy` supports Python `3.12` and `3.13` only.

```bash
python -m pip install graxpy
```

`graxpy` now supports local Henke-table material lookup for elemental string
names such as `"Si"` and `"Pt"`, with optional density overrides through
`grax.MaterialSpec`. Existing DataFrame optical-constants inputs still work,
and xrt-compatible material objects remain supported for now but emit a
deprecation warning.

PyPI project page: <https://pypi.org/project/graxpy/>

For local editable installs:

```bash
python -m pip install -e .
```

## Select a solver

RCWA is the default, so existing code continues to use it. Pass `solver` to
choose the Nevière differential method instead:

```python
import grax

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    substrate_material="Si",
    layer_material="Pt",
    layer_thickness_nm=28.77,
)

common = dict(
    grating=grating,
    energy_ev=300.0,
    grazing_angle_deg=4.0,
    fourier_orders=30,
    polarization="p",
)

rcwa_result = grax.run_simulation(**common, solver="rcwa")
neviere_result = grax.run_simulation(**common, solver="neviere")
```

See the documentation's [Choosing a solver](docs/tutorials/choosing-a-solver.md)
tutorial and the runnable
`examples/simulation/neviere_solver/neviere_solver.py` for guidance and
integration options.

## SX700 PGM working range

`grax.pgm` combines SX700 geometry, a GraxPy grating, and a GraxPy plane-mirror
stack into energy-by-CFF maps of geometric and optical throughput. It currently
uses S (TE) polarization. The colleague's PGM geometry and adaptive grating
sampling are integrated directly into GraxPy; mirror reflectivity uses GraxPy's
existing plane-mirror model. See `examples/pgm_working_range.py` for an editable
Python example, or choose **PGM → New working range** in the local web app, where
which opens on the SX700 example. Edit the optics directly, fill them from saved
gratings and mirrors, and save the setup as a named PGM.

The mirror model change produces small numerical differences from the colleague's
reference: at most 0.11 percentage points for the Ir/Si single-layer case and
0.52 percentage points for the Cr/C multilayer case over the comparison grids
in `tests/unit/test_pgm.py`. Geometry and the validated grating case retain
their reference results.

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
diffraction orders to overlay.

`grax-web` now opens that local URL in your default browser automatically on
startup.

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
- `selected_efficiency.png`

Use `Plots` to combine saved runs and choose which diffraction orders to
overlay for each run. Use `Manage runs` to rename runs or bulk delete them.

## Repository at a glance

- `src/grax/`: core package source code
- `examples/`: runnable examples
- `docs/`: documentation sources

## Attribution

The modal RCWA solver is inspired by RETICOLO v9. GraxPy is an independent
implementation, not an official RETICOLO port or distribution; RETICOLO is not
bundled with the public `graxpy` package.

- RETICOLO DOI: <https://doi.org/10.5281/zenodo.14631950>
- RETICOLO license (CC BY 4.0): <https://creativecommons.org/licenses/by/4.0/>

## License

Copyright (C) [2026] [Helmholtz-Berlin fur Materialen und Energie GmbH (HZB)]

Licensed under the European Union Public License (EUPL), Version 1.2.

You may not use this work except in compliance with the License.

A copy of the License is available at:
<https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>
