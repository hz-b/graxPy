"""Guard that ``import grax`` stays free of heavy optional dependencies.

``xrt`` and ``matplotlib.pyplot`` are both slow to import. The package defers
them to the functions that need them, so a spawned batch worker re-importing
``grax`` does not pay for a plotting backend or the XRT dynamical-diffraction
tables it never uses.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def _modules_after_import(*import_lines: str) -> set[str]:
    """Return ``sys.modules`` keys after running the given imports in a subprocess."""

    script = textwrap.dedent(
        """
        import sys
        {imports}
        print("\\n".join(sorted(sys.modules)))
        """
    ).format(imports="\n".join(import_lines))
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return set(output.stdout.split())


def test_import_grax_does_not_import_xrt_or_pyplot() -> None:
    """A bare ``import grax`` pulls in neither ``xrt`` nor ``matplotlib.pyplot``."""

    modules = _modules_after_import("import grax")
    assert not any(name == "xrt" or name.startswith("xrt.") for name in modules)
    assert "matplotlib.pyplot" not in modules


def test_import_multilayer_modules_does_not_import_xrt() -> None:
    """Importing the multilayer workflow modules does not import ``xrt`` eagerly."""

    modules = _modules_after_import(
        "import grax.multilayer_optimization",
        "import grax.multilayer_reflectivity",
    )
    assert not any(name == "xrt" or name.startswith("xrt.") for name in modules)
