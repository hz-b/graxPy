"""Material catalog utilities for the local web app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OpticalConstantsTable:
    """Small DataFrame-like optical-constants table.

    Attributes:
        energy_ev: Photon energies in electronvolts.
        delta: Refractive-index decrement values.
        beta: Absorption index values.
        name: Material display name and catalog key.
        columns: Column names expected by ``grax.materials``.
        attrs: Metadata dictionary used by ``material_label``.
    """

    energy_ev: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    name: str
    columns: tuple[str, str, str] = ("Energy(eV)", "Delta", "Beta")
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate display metadata."""
        self.attrs["name"] = self.name

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one optical-constant column by name."""
        if key == "Energy(eV)":
            return self.energy_ev
        if key == "Delta":
            return self.delta
        if key == "Beta":
            return self.beta
        raise KeyError(key)


def default_catalog_directory() -> Path:
    """Return the preferred repository optical-constants directory."""
    project_root = Path(__file__).resolve().parents[3]
    validation_dir = project_root / "validation" / "optical_constants"
    if validation_dir.exists():
        return validation_dir
    return project_root / "examples" / "optical_constants"


def load_optical_constants_table(path: str | Path, name: str) -> OpticalConstantsTable:
    """Load a legacy optical-constants text file.

    Args:
        path: Text file with energy, delta, and beta columns.
        name: Material label.

    Returns:
        DataFrame-like optical-constants table.
    """
    data = np.loadtxt(Path(path), skiprows=2)
    return OpticalConstantsTable(
        energy_ev=np.asarray(data[:, 0], dtype=float),
        delta=np.asarray(data[:, 1], dtype=float),
        beta=np.asarray(data[:, 2], dtype=float),
        name=name,
    )


def load_material_catalog(
    catalog_dir: str | Path | None = None,
) -> dict[str, OpticalConstantsTable]:
    """Load the small built-in material catalog for the web MVP.

    Args:
        catalog_dir: Optional directory containing ``n_<material>_cxro.txt`` files.

    Returns:
        Mapping from material key to optical-constants table.
    """
    base_dir = default_catalog_directory() if catalog_dir is None else Path(catalog_dir)
    files = {
        "Au": "n_Au_cxro.txt",
        "C": "n_C_cxro.txt",
        "Cr": "n_Cr_cxro.txt",
        "Pt": "n_Pt_cxro.txt",
        "Si": "n_Si_cxro.txt",
    }
    return {
        name: load_optical_constants_table(base_dir / filename, name)
        for name, filename in files.items()
        if (base_dir / filename).exists()
    }
