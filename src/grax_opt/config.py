"""Shared configuration primitives for optimizer modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterBounds:
    """Bounds for one optimization parameter."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate bounds ordering."""

        if self.upper <= self.lower:
            raise ValueError("Parameter bounds must satisfy upper > lower.")

    def as_list(self) -> list[float]:
        """Return bounds in Ax-compatible list form."""

        return [float(self.lower), float(self.upper)]
