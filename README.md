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

Install the optional web extra and start the local server:

```bash
python -m pip install -e ".[web]"
grax-web
```

Then open <http://127.0.0.1:5050>. The web app stores local saved gratings,
previews, and run artifacts in `.grax-web/`.

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
