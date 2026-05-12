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

To install with optional Numba acceleration support:

```bash
python -m pip install "graxpy[numba]"
```

Project page: <https://pypi.org/project/graxpy/0.1.0/>

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

The main documentation focuses on the `grax` simulation package. The
optimization package is currently treated as an advanced companion package.

## Verify the install

```bash
python - <<'PY'
import grax as rp

grating = rp.LaminarGrating()
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
  `examples/` and `comparison_to_other_codes/` into `docs/`

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
