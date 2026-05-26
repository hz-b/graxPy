"""Helpers for optimizer evaluation schedules."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationSelection:
    """Normalized evaluation inputs for one optimizer run."""

    evaluation_energies_ev: list[float]
    evaluation_grazing_angles_deg: list[float]
    evaluation_mode: str


def normalize_evaluation_selection(
    evaluation_energies_ev: Sequence[float],
    evaluation_grazing_angles_deg: Sequence[float] | None = None,
) -> EvaluationSelection:
    """Normalize and validate evaluation energies and optional angles.

    Args:
        evaluation_energies_ev: Evaluation energies supplied by the caller.
        evaluation_grazing_angles_deg: Optional grazing-angle values supplied by the caller.

    Returns:
        Normalized evaluation inputs and the derived evaluation mode.

    Raises:
        ValueError: If the inputs are empty, non-positive, or contain an ambiguous
            many-energy/many-angle grid.
    """

    normalized_energies = [float(energy_ev) for energy_ev in evaluation_energies_ev]
    if len(normalized_energies) == 0:
        raise ValueError("evaluation_energies_ev must be provided and non-empty.")
    if any(energy_ev <= 0.0 for energy_ev in normalized_energies):
        raise ValueError("evaluation_energies_ev values must be > 0.")

    normalized_angles = (
        [] if evaluation_grazing_angles_deg is None else [float(angle) for angle in evaluation_grazing_angles_deg]
    )
    if len(normalized_angles) == 0:
        return EvaluationSelection(
            evaluation_energies_ev=sorted({float(energy_ev) for energy_ev in normalized_energies}),
            evaluation_grazing_angles_deg=[],
            evaluation_mode="energy_only",
        )

    if any(angle <= 0.0 for angle in normalized_angles):
        raise ValueError("evaluation_grazing_angles_deg values must be > 0.")
    if len(normalized_energies) > 1 and len(normalized_angles) > 1:
        raise ValueError(
            "evaluation_energies_ev and evaluation_grazing_angles_deg may not both contain more than one value."
        )

    return EvaluationSelection(
        evaluation_energies_ev=normalized_energies,
        evaluation_grazing_angles_deg=normalized_angles,
        evaluation_mode="energy_angle_pairs",
    )


def build_evaluation_cases(
    evaluation_energies_ev: Sequence[float],
    evaluation_grazing_angles_deg: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Build evaluation cases for optimizer objective evaluation.

    Args:
        evaluation_energies_ev: Evaluation energies supplied by the caller.
        evaluation_grazing_angles_deg: Optional grazing-angle values supplied by the caller.

    Returns:
        A tuple of ``(energies_ev, grazing_angles_deg, evaluation_mode)``.
        ``grazing_angles_deg`` is ``None`` for the current geometry-driven
        single-angle behavior.
    """

    selection = normalize_evaluation_selection(
        evaluation_energies_ev,
        evaluation_grazing_angles_deg,
    )
    energies = np.asarray(selection.evaluation_energies_ev, dtype=float)
    if selection.evaluation_mode == "energy_only":
        return energies, None, selection.evaluation_mode

    angles = np.asarray(selection.evaluation_grazing_angles_deg, dtype=float)
    if energies.size == 1 and angles.size > 1:
        energies = np.repeat(energies, angles.size)
    elif angles.size == 1 and energies.size > 1:
        angles = np.repeat(angles, energies.size)
    return energies, angles, selection.evaluation_mode
