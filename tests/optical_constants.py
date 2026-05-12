from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OpticalConstantsTable:
    """Minimal DataFrame-like optical-constants table for tests."""

    energy_ev: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    name: str
    columns: tuple[str, str, str] = ("Energy(eV)", "Delta", "Beta")
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate DataFrame-like attrs."""

        self.attrs["name"] = self.name

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one named optical-constant column."""

        if key == "Energy(eV)":
            return self.energy_ev
        if key == "Delta":
            return self.delta
        if key == "Beta":
            return self.beta
        raise KeyError(key)


def load_optical_constants_table(path: str | Path, name: str) -> OpticalConstantsTable:
    """Load a legacy optical-constant text file into a DataFrame-like table.

    Args:
        path: Optical-constant file path.
        name: Material label used for plotting and debug output.

    Returns:
        DataFrame-like optical-constants table with energy, delta, and beta.
    """

    data = np.loadtxt(path, skiprows=2)
    return OpticalConstantsTable(
        energy_ev=np.asarray(data[:, 0], dtype=float),
        delta=np.asarray(data[:, 1], dtype=float),
        beta=np.asarray(data[:, 2], dtype=float),
        name=name,
    )
