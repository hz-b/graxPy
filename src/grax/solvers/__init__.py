"""Electromagnetic solvers for one-dimensional gratings.

Solvers share the same inputs, the same Fourier-space operators (including the
Li/Fast-Fourier-Factorization rules for TM), and the same efficiency
extraction:

- :mod:`grax.solvers.rcwa` - rigorous coupled-wave analysis. Each layer is
  z-invariant and its field operator is eigen-decomposed once.

Shared machinery lives in :mod:`grax.solvers.common`. Callers normally select a
solver through ``grax.run_simulation(..., solver=...)`` rather than importing
these modules directly.
"""

from __future__ import annotations

from .common import (
    DiffractionResult,
    FourierBackend,
    FourierBackendName,
    Parameters,
    Res1Parameters,
    Res1Result,
    Res2Parameters,
    Res2Result,
    Texture1D,
    propagating_energy_balance,
    propagating_order_mask,
    res0,
    res1,
    safe_linalg_solve,
)
from .rcwa import res2

__all__ = [
    "DiffractionResult",
    "FourierBackend",
    "FourierBackendName",
    "Parameters",
    "Res1Parameters",
    "Res1Result",
    "Res2Parameters",
    "Res2Result",
    "Texture1D",
    "propagating_energy_balance",
    "propagating_order_mask",
    "res0",
    "res1",
    "res2",
    "safe_linalg_solve",
]
