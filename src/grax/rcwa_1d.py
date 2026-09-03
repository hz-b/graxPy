"""Backward-compatible aliases for the 1D solver internals.

The solver implementation moved to the :mod:`grax.solvers` package when a second
solver (the Nevière differential method) was added:

- shared types, ``res0``/``res1``, the Fourier machinery, the layer field
  operators, the interface cascade and the efficiency extraction now live in
  :mod:`grax.solvers.common`
- the modal RCWA layer solve and ``res2`` live in :mod:`grax.solvers.rcwa`

This module re-exports the historical names so existing imports of
``grax.rcwa_1d`` keep working. New code should import from :mod:`grax.solvers`.

Note for tests and tooling: patching a name here rebinds only this alias module.
To intercept a function the solvers actually call, patch it on the module that
defines it (:mod:`grax.solvers.common` or :mod:`grax.solvers.rcwa`).
"""

from __future__ import annotations

from .roughness import (  # noqa: F401
    apply_debye_waller_roughness,
    debye_waller_roughness_diagnostics,
    incidence_sine_from_beta0,
)
from .solvers.common import (  # noqa: F401
    ArrayLike,
    BoundaryBlockCache,
    DiffractionResult,
    EigenCache,
    FourierBackend,
    FourierBackendName,
    LayerFieldOperators,
    Parameters,
    Res1Parameters,
    Res1Result,
    Res2Parameters,
    Res2Result,
    Texture1D,
    _angles_from_kx,
    _apply_debye_waller_roughness,
    _bottom_field_from_boundary_block,
    _cascade_boundary_pair,
    _cascade_layer_boundary_blocks,
    _convert_texture,
    _convolution_matrix,
    _debye_waller_roughness_factor,
    _incidence_sine_from_beta0,
    _kz_branch,
    _kz_branch_array,
    _normalize_orders,
    _piecewise_fourier_coefficients,
    _piecewise_fourier_coefficients_baseline,
    _piecewise_fourier_coefficients_numba,
    _resolve_fourier_backend,
    _texture_signature_for_metadata,
    _top_admittance_from_boundary_block,
    layer_field_operators,
    res0,
    res1,
    safe_linalg_solve,
)
from .solvers.neviere import NeviereOptions, res2_dm  # noqa: F401
from .solvers.rcwa import (  # noqa: F401
    _layer_boundary_block,
    _modal_function_matrices,
    _modal_function_matrix,
    _modal_q_coth,
    _modal_q_csch,
    _solve_te_stack,
    _solve_tm_stack,
    res2,
)

__all__ = [
    "ArrayLike",
    "BoundaryBlockCache",
    "DiffractionResult",
    "EigenCache",
    "FourierBackend",
    "FourierBackendName",
    "LayerFieldOperators",
    "NeviereOptions",
    "Parameters",
    "Res1Parameters",
    "Res1Result",
    "Res2Parameters",
    "Res2Result",
    "Texture1D",
    "apply_debye_waller_roughness",
    "debye_waller_roughness_diagnostics",
    "incidence_sine_from_beta0",
    "layer_field_operators",
    "res0",
    "res1",
    "res2",
    "res2_dm",
    "safe_linalg_solve",
]
