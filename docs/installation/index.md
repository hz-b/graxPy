# Installation

`grax` requires Python 3.12 or 3.13. The core package dependencies are
NumPy, numba, pandas, SciPy, Matplotlib, tqdm, and `xrt` for backward compatibility
during the current material-lookup transition. Local Henke-table string
material lookup is built into `grax`, and xrt-backed material objects now emit
a deprecation warning when they are used.

## Install the package

### Install from PyPI

```bash
python -m pip install graxpy
```

To install with optimization dependencies (`grax_opt` via `opt` extra):

```bash
python -m pip install "graxpy[opt]"
```

Project page: <https://pypi.org/project/graxpy>

```{note}
The numba backend is now the default for `run_simulation` and the other public
simulation workflows. The legacy `backend="numpy"` path remains available
temporarily for compatibility, but it is deprecated and will be removed in a
future version.
```

### Install from local files (repository checkout)

From a checkout of the repository:

```bash
python -m pip install .
```

For local development, install in editable mode with the development tools:

```bash
python -m pip install -e ".[dev]"
```

To build this documentation locally, install the documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

The optional optimization package `grax_opt` uses Ax and is installed with
the separate `opt` extra:

```bash
python -m pip install -e ".[opt]"
```

```{note}
The numba backend provides **3.7x speedup** with identical numerical results.
You can still explicitly specify `backend="numpy"` for parity checks or
transition-time comparisons, but that path is deprecated.
```

The main documentation focuses on the `grax` simulation package. The
optimization package is currently treated as an advanced companion package.

## Web UI

See the clickable [`Web UI`](web-ui.md) subsection for local web app install
and startup instructions.

## Verify the install

```bash
python - <<'PY'
import grax

grating = grax.LaminarGrating()
print(grating.period_nm)
PY
```

## Build the documentation

Default (sync images + build HTML, LaTeX, and PDF):

```bash
tools/build_docs.sh
```

### Flags

- `--html`: build HTML only
- `--pdf`: build LaTeX and PDF only
- `--open`: open generated HTML index in your browser after the build
- `--skip-image-sync` or `--skip-example-sync`: skip copying images from
  `examples/` and `validation/` into `docs/`

### Common usage examples

Build only HTML:

```bash
tools/build_docs.sh --html
```

Build only PDF artifacts:

```bash
tools/build_docs.sh --pdf
```

Open HTML docs after build:

```bash
tools/build_docs.sh --html --open
```

Fast iteration when images are already up to date:

```bash
tools/build_docs.sh --html --skip-image-sync
```

```{toctree}
:maxdepth: 1
:caption: Web UI

web-ui
```
