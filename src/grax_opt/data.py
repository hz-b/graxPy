"""Measurement-data loading helpers for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeasurementData:
    """Measurement data used by the optimizer."""

    energy_ev: np.ndarray
    efficiency: np.ndarray
    source_path: Path

    def __post_init__(self) -> None:
        """Validate data shapes."""

        if self.energy_ev.ndim != 1 or self.efficiency.ndim != 1:
            raise ValueError("Measurement data arrays must be one-dimensional.")
        if self.energy_ev.size == 0:
            raise ValueError("Measurement data must not be empty.")
        if self.energy_ev.shape != self.efficiency.shape:
            raise ValueError("Measurement energy and efficiency arrays must have matching shape.")


def _parse_measurement_value(raw_value: str) -> float:
    """Parse one measurement token."""

    value = raw_value.strip().replace(",", ".")
    if value in {"", "--"}:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_measurement_data(path: str | Path) -> MeasurementData:
    """Load a two-column measurement file.

    Both whitespace-delimited decimal-dot files and semicolon-delimited
    decimal-comma files are accepted. Non-numeric header rows are ignored.

    Args:
        path: Input measurement path.

    Returns:
        Parsed measurement data with invalid rows removed.
    """

    source_path = Path(path)
    energy_values: list[float] = []
    efficiency_values: list[float] = []

    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split(";") if ";" in stripped else stripped.split()
            if len(columns) < 2:
                continue
            energy = _parse_measurement_value(columns[0])
            efficiency = _parse_measurement_value(columns[1])
            if np.isnan(energy) or np.isnan(efficiency):
                continue
            energy_values.append(energy)
            efficiency_values.append(efficiency)

    return MeasurementData(
        energy_ev=np.asarray(energy_values, dtype=float),
        efficiency=np.asarray(efficiency_values, dtype=float),
        source_path=source_path,
    )


def sample_measurement_data(
    measurement: MeasurementData,
    evaluation_energies_ev: list[float] | np.ndarray,
) -> MeasurementData:
    """Sample measurement data at requested energies via linear interpolation.

    Args:
        measurement: Source measurement data.
        evaluation_energies_ev: Target energies for objective evaluation.

    Returns:
        Measurement data sampled at the requested energies.
    """

    target_energies = np.asarray(evaluation_energies_ev, dtype=float)
    if target_energies.ndim != 1:
        raise ValueError("evaluation_energies_ev must be one-dimensional.")
    if target_energies.size == 0:
        raise ValueError("evaluation_energies_ev must not be empty.")
    if np.min(target_energies) < float(np.min(measurement.energy_ev)) or np.max(
        target_energies
    ) > float(np.max(measurement.energy_ev)):
        raise ValueError(
            "evaluation_energies_ev must lie within the measurement energy range for interpolation."
        )

    sampled_efficiency = np.interp(
        target_energies,
        measurement.energy_ev,
        measurement.efficiency,
    )
    return MeasurementData(
        energy_ev=target_energies.astype(float),
        efficiency=sampled_efficiency.astype(float),
        source_path=measurement.source_path,
    )
