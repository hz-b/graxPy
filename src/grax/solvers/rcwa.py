"""Rigorous coupled-wave (modal) solver for 1D gratings.

Each finite layer is treated as z-invariant, its Fourier-space field operator is
eigen-decomposed once, and the resulting modal admittances are assembled into a
Dirichlet-to-Neumann interface-response block. Blocks are cascaded by the shared
routine in :mod:`grax.solvers.common`, which also extracts the diffraction
efficiencies.

The differential-method solver in :mod:`grax.solvers.neviere` replaces only the
per-layer block construction; everything else is shared, so the two solvers can
be compared directly.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import nullcontext as _nullcontext
from time import perf_counter

import numpy as np

from ..simulation._profiling import SolverProfiler
from .common import (
    ArrayLike,
    BoundaryBlockCache,
    EigenCache,
    Parameters,
    Res1Result,
    Res2Result,
    Texture1D,
    _MAX_BASIS_SIZE,
    apply_optional_roughness,
    layer_field_operators,
    prepare_layer_stack,
    solve_stack_from_layer_blocks,
)

logger = logging.getLogger(__name__)


def res2(
    aa: Res1Result,
    profile: tuple[ArrayLike, ArrayLike],
    parm: Parameters | None = None,
    *,
    roughness_sigma_nm: float | None = None,
    _profiler: SolverProfiler | None = None,
) -> Res2Result:
    """Solve the RCWA layer stack.

    Args:
        aa: Fourier-space texture data returned by :func:`grax.res1`.
        profile: Thickness and texture-index profile for one RCWA layer stack.
        parm: Optional solver parameters.
        roughness_sigma_nm: Optional rms interface roughness in nanometers. When
            provided, reflected and transmitted diffraction efficiencies are
            uniformly damped by a scalar Debye-Waller factor,
            ``exp(-(4*pi*sigma*sin(theta)/wavelength)^2)``. Here ``theta`` is
            the grazing incidence angle from the surface. This simple model
            reduces total intensity to mimic scattering losses; it does not
            alter geometry, RCWA matrices, diffraction amplitudes, or introduce
            stochastic rough surfaces.

    Returns:
        Reflected and transmitted diffraction results for incidence from top.
    """

    if roughness_sigma_nm is not None and roughness_sigma_nm < 0.0:
        raise ValueError("roughness_sigma_nm must be >= 0 when provided.")

    # Polarization comes from aa (set by res1); parm is ignored for polarization selection.
    polarization = aa.polarization
    n_top, n_bottom, layers = prepare_layer_stack(aa, profile)

    with _profiler.record("res2_total") if _profiler is not None else _nullcontext():
        reflected, transmitted = _solve_modal_stack(
            wavelength=aa.wavelength,
            period=aa.period,
            orders=aa.orders,
            beta0=aa.beta0,
            n_top=n_top,
            n_bottom=n_bottom,
            polarization=polarization,
            layers=layers,
            _profiler=_profiler,
        )

    reflected, transmitted = apply_optional_roughness(
        reflected,
        transmitted,
        aa=aa,
        roughness_sigma_nm=roughness_sigma_nm,
    )

    return Res2Result(
        inc_top_reflected=reflected,
        inc_top_transmitted=transmitted,
    )


def _solve_te_stack(
    wavelength: float,
    period: float,
    orders: np.ndarray,
    beta0: float,
    n_top: complex,
    n_bottom: complex,
    layers: list[tuple[float, Texture1D]],
    *,
    _profiler: SolverProfiler | None = None,
):
    """Solve the 1D TE stack with modal layer blocks."""

    return _solve_modal_stack(
        wavelength=wavelength,
        period=period,
        orders=orders,
        beta0=beta0,
        n_top=n_top,
        n_bottom=n_bottom,
        polarization=1,
        layers=layers,
        _profiler=_profiler,
    )


def _solve_tm_stack(
    wavelength: float,
    period: float,
    orders: np.ndarray,
    beta0: float,
    n_top: complex,
    n_bottom: complex,
    layers: list[tuple[float, Texture1D]],
    *,
    _profiler: SolverProfiler | None = None,
):
    """Solve the 1D TM stack with modal layer blocks.

    Uses the inverse-permittivity convolution operator and TM boundary
    admittances. Only the layer operator and the semi-infinite derivatives
    differ from the TE path.
    """

    return _solve_modal_stack(
        wavelength=wavelength,
        period=period,
        orders=orders,
        beta0=beta0,
        n_top=n_top,
        n_bottom=n_bottom,
        polarization=-1,
        layers=layers,
        _profiler=_profiler,
    )


def _solve_modal_stack(
    *,
    wavelength: float,
    period: float,
    orders: np.ndarray,
    beta0: float,
    n_top: complex,
    n_bottom: complex,
    polarization: int,
    layers: list[tuple[float, Texture1D]],
    _profiler: SolverProfiler | None = None,
):
    """Solve one 1D stack using eigen-decomposed modal layer blocks."""

    eigen_cache: EigenCache = {}
    boundary_block_cache: BoundaryBlockCache = {}

    def layer_block_fn(**kwargs: object) -> np.ndarray:
        return _layer_boundary_block(
            eigen_cache=eigen_cache,
            boundary_block_cache=boundary_block_cache,
            **kwargs,  # type: ignore[arg-type]
        )

    return solve_stack_from_layer_blocks(
        wavelength=wavelength,
        period=period,
        orders=orders,
        beta0=beta0,
        n_top=n_top,
        n_bottom=n_bottom,
        polarization=polarization,
        layers=layers,
        layer_block_fn=layer_block_fn,
        _profiler=_profiler,
    )


def _layer_boundary_block(
    *,
    thickness: float,
    texture: Texture1D,
    orders: np.ndarray,
    k0: float,
    kx: np.ndarray,
    kx_matrix_sq: np.ndarray,
    polarization: int = 1,
    eigen_cache: EigenCache,
    boundary_block_cache: BoundaryBlockCache,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray:
    """Return the interface-response block for one finite RCWA layer."""

    basis_size = len(orders)
    logger.debug(f"  _layer_boundary_block: thickness={thickness:.4f} nm, basis_size={basis_size}")

    if basis_size > _MAX_BASIS_SIZE:
        raise ValueError(f"Fourier orders too large: {basis_size} modes. "
                        "Try reducing fourier_orders to 50 or less.")

    boundary_cache_key = (
        texture.signature,
        float(thickness),
        tuple(int(order) for order in orders),
        float(k0),
        int(polarization),
    )
    cached_block = boundary_block_cache.get(boundary_cache_key)
    if cached_block is not None:
        if _profiler is not None:
            _profiler.add_detail_count("layer_boundary_block_cache_hits", 1)
        return cached_block
    if _profiler is not None:
        _profiler.add_detail_count("layer_boundary_block_cache_misses", 1)

    with _profiler.record("layer_operator_build") if _profiler is not None else _nullcontext():
        operators = layer_field_operators(
            texture=texture,
            orders=orders,
            k0=k0,
            kx=kx,
            kx_matrix_sq=kx_matrix_sq,
            polarization=polarization,
        )
        operator = operators.operator
        inv_epsilon_conv_tm = operators.inv_epsilon_conv

    cache_key = (
        texture.signature,
        tuple(int(order) for order in orders),
        float(k0),
        int(polarization),
    )
    if _profiler is not None:
        operator_key = hashlib.sha1(operator.tobytes()).hexdigest()
        _profiler.add_unique_value("layer_operator_unique", operator_key)
        _profiler.add_detail_count("layer_operator_rows_total", int(operator.shape[0]))
        _profiler.add_detail_count("layer_operator_cols_total", int(operator.shape[1]))
        _profiler.add_detail_count("layer_operator_calls", 1)
        _profiler.update_detail_peak("layer_operator_rows_peak", float(operator.shape[0]))
        _profiler.update_detail_peak("layer_operator_cols_peak", float(operator.shape[1]))
    if cache_key in eigen_cache:
        eigenvalues, eigenvectors = eigen_cache[cache_key]
        if _profiler is not None:
            _profiler.add_detail_count("layer_eigensolve_cache_hits", 1)
    else:
        if _profiler is not None:
            _profiler.add_detail_count("layer_eigensolve_cache_misses", 1)
        with _profiler.record("layer_eigensolve") if _profiler is not None else _nullcontext():
            t0 = perf_counter() if _profiler is not None else None
            eigenvalues, eigenvectors = np.linalg.eig(operator)
            if _profiler is not None and t0 is not None:
                _profiler.add_detail_timing("layer_eigensolve_call", perf_counter() - t0)
        eigen_cache[cache_key] = (eigenvalues, eigenvectors)
    if _profiler is not None:
        _profiler.increment("layer_eigensolve_calls")

    with _profiler.record("layer_modal_values") if _profiler is not None else _nullcontext():
        q_values = np.sqrt(eigenvalues + 0j)
        q_coth = _modal_q_coth(q_values, thickness)
        q_csch = _modal_q_csch(q_values, thickness)
    with _profiler.record("layer_modal_matrices") if _profiler is not None else _nullcontext():
        t0 = perf_counter() if _profiler is not None else None
        admittance, coupling = _modal_function_matrices(eigenvectors, q_coth, q_csch)
        if _profiler is not None and t0 is not None:
            _profiler.add_detail_timing("layer_modal_matrices_call", perf_counter() - t0)
            _profiler.add_detail_count("layer_modal_matrices_calls", 1)
    with _profiler.record("layer_block_assembly") if _profiler is not None else _nullcontext():
        if polarization == -1 and inv_epsilon_conv_tm is not None:
            # TM: block tracks [H; E_x] where E_x = [1/eps] @ dH/dz
            admittance = inv_epsilon_conv_tm @ admittance
            coupling = inv_epsilon_conv_tm @ coupling
        block = np.empty((2 * basis_size, 2 * basis_size), dtype=complex)
        block[:basis_size, :basis_size] = -admittance
        block[:basis_size, basis_size:] = coupling
        block[basis_size:, :basis_size] = -coupling
        block[basis_size:, basis_size:] = admittance
    boundary_block_cache[boundary_cache_key] = block
    return block


def _modal_q_coth(q_values: np.ndarray, thickness: float) -> np.ndarray:
    """Return modal ``q / tanh(q * thickness)`` with a small-argument limit."""

    argument = q_values * thickness
    values = np.empty_like(q_values, dtype=complex)
    small = np.abs(argument) < 1e-8
    values[small] = (1.0 / thickness) + (q_values[small] ** 2 * thickness / 3.0)
    values[~small] = q_values[~small] / np.tanh(argument[~small])
    return values


def _modal_q_csch(q_values: np.ndarray, thickness: float) -> np.ndarray:
    """Return modal ``q / sinh(q * thickness)`` with a small-argument limit."""

    argument = q_values * thickness
    values = np.empty_like(q_values, dtype=complex)
    small = np.abs(argument) < 1e-8
    values[small] = (1.0 / thickness) - (q_values[small] ** 2 * thickness / 6.0)
    values[~small] = q_values[~small] / np.sinh(argument[~small])
    return values


def _modal_function_matrix(
    eigenvectors: np.ndarray,
    modal_values: np.ndarray,
) -> np.ndarray:
    """Return ``V @ diag(modal_values) @ V^-1`` without forming ``V^-1``."""

    left_factor = eigenvectors * modal_values[np.newaxis, :]
    result = np.linalg.solve(eigenvectors.T, left_factor.T).T
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        raise ValueError("Modal layer function produced NaN/Inf values")
    return result


def _modal_function_matrices(
    eigenvectors: np.ndarray,
    first_modal_values: np.ndarray,
    second_modal_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two ``V @ diag(m) @ V^-1`` matrices from one shared solve."""

    first_left_factor = eigenvectors * first_modal_values[np.newaxis, :]
    second_left_factor = eigenvectors * second_modal_values[np.newaxis, :]
    stacked_rhs = np.hstack((first_left_factor.T, second_left_factor.T))
    solved = np.linalg.solve(eigenvectors.T, stacked_rhs)
    split_index = eigenvectors.shape[0]
    first_result = solved[:, :split_index].T
    second_result = solved[:, split_index:].T
    if (
        np.any(np.isnan(first_result))
        or np.any(np.isinf(first_result))
        or np.any(np.isnan(second_result))
        or np.any(np.isinf(second_result))
    ):
        raise ValueError("Modal layer function produced NaN/Inf values")
    return first_result, second_result
