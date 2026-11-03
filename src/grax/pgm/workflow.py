# ruff: noqa: N815
# Energy-axis field keeps the established eV suffix.
"""High-level PGM working-range calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import calculate_working_range
from .optics import calculate_optical_efficiency
from .parameters import SX700Parameters


@dataclass(frozen=True)
class WorkingRangeResult:
    """Energy and CFF axes, geometric maps, and optical maps."""

    energies_eV: np.ndarray
    cff_values: np.ndarray
    geometry: dict[str, np.ndarray]
    optics: dict[str, np.ndarray]


def run_working_range(parameters: SX700Parameters) -> WorkingRangeResult:
    """Calculate geometry, mirror reflectivity, grating efficiency and throughput."""
    parameters.check_stopped()
    energies, cff_values, geometry = calculate_working_range(parameters)
    parameters.check_stopped()
    optics = calculate_optical_efficiency(parameters, energies, cff_values, geometry)
    parameters.check_stopped()
    return WorkingRangeResult(energies, cff_values, geometry, optics)
