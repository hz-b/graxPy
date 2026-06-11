# Installation

`grax` requires Python 3.12 or 3.13. The core package dependencies are
NumPy, pandas, SciPy, Matplotlib, tqdm, and xrt.

## Install the package

### Install from PyPI

```bash
python -m pip install graxpy
```

To install with optimization dependencies (`grax_opt` via `opt` extra):

```bash
python -m pip install "graxpy[opt]"
```

To install with optional Numba acceleration support (recommended for best performance):

```bash
python -m pip install "graxpy[numba]"
```

Project page: <https://pypi.org/project/graxpy>

```{note}
The numpy backend is the default for `run_simulation` and provides reliable
performance without dependencies. For maximum speed, use `backend="numba"`
which provides 3.7x speedup with identical numerical results.
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

The optional Numba backend support is installed with the `numba` extra:

```bash
python -m pip install -e ".[numba]"
```

```{note}
The Numba backend provides **3.7x speedup** with identical numerical results.
You can explicitly specify `backend="numpy"` for the pure Python
implementation or `backend="numba"` for the JIT-compiled version.
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
