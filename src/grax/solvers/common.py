"""Shared data structures and numerics for the 1D grating solvers.

This module holds everything the modal (RCWA) solver in :mod:`grax.solvers.rcwa`
and the differential-method solver in :mod:`grax.solvers.neviere` have in common:
the pseudo-periodic Fourier machinery (:func:`res1`), the layer field operators
including the Li/Fast-Fourier-Factorization rules, the interface-response
cascade, and the diffraction-efficiency extraction.

The two solvers differ only in how one finite layer is turned into an
interface-response block: RCWA eigen-decomposes the layer operator, while the
differential method integrates the equivalent first-order system in ``z``.
Everything before and after that step is shared, which is what makes the two
solvers directly comparable.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ..roughness import (
    _apply_debye_waller_roughness as _apply_debye_waller_roughness_impl,
    _debye_waller_roughness_factor as _debye_waller_roughness_factor_impl,
    apply_debye_waller_roughness,
    debye_waller_roughness_diagnostics,
    incidence_sine_from_beta0,
)
from ..simulation._profiling import SolverProfiler

logger = logging.getLogger(__name__)


ArrayLike = Any
EigenCache = dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]
BoundaryBlockCache = dict[tuple[Any, ...], np.ndarray]

# Callable that turns one finite layer into its interface-response block. The
# modal and differential-method solvers each provide their own implementation.
LayerBlockFunction = Callable[..., np.ndarray]

# Upper bound on the modal basis size. Beyond this the dense linear algebra
# becomes impractical and the caller almost certainly wants fewer orders.
_MAX_BASIS_SIZE = 201


class FourierBackend(str, Enum):
    """Internal Fourier coefficient implementations.

    Two backends are available:
    - NUMPY: Reference implementation kept for transition-time compatibility
    - NUMBA: JIT-compiled kernel and the default production path
    """

    NUMPY = "numpy"
    NUMBA = "numba"


FourierBackendName = Literal["numpy", "numba"]

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_AVAILABLE = False


@dataclass
class Res1Parameters:
    trace: int = 0


@dataclass
class Res2Parameters:
    result: int = 1


@dataclass
class Parameters:
    dim: int = 1
    polarization: int = 1
    not_io: int = 0
    res1: Res1Parameters = field(default_factory=Res1Parameters)
    res2: Res2Parameters = field(default_factory=Res2Parameters)


@dataclass
class Texture1D:
    period: float
    breaks: np.ndarray
    refractive_index: np.ndarray
    epsilon_fourier: np.ndarray
    homogeneous_index: complex | None = None
    inv_epsilon_fourier: np.ndarray | None = None

    @property
    def signature(self) -> tuple[Any, ...]:
        if self.homogeneous_index is not None:
            n = self.homogeneous_index
            return ("homogeneous", round(float(np.real(n)), 12), round(float(np.imag(n)), 12))
        return (
            "patterned",
            tuple(np.round(self.breaks, 12)),
            tuple((round(float(np.real(v)), 12), round(float(np.imag(v)), 12)) for v in self.refractive_index),
        )


@dataclass
class Res1Result:
    wavelength: float
    period: float
    orders: np.ndarray
    beta0: float
    polarization: int
    textures: list[Texture1D]


@dataclass
class DiffractionResult:
    order: np.ndarray
    theta: np.ndarray
    efficiency: np.ndarray
    amplitude: np.ndarray


@dataclass
class Res2Result:
    inc_top_reflected: DiffractionResult
    inc_top_transmitted: DiffractionResult


if _NUMBA_AVAILABLE:  # pragma: no branch
    @njit(cache=True)
    def _piecewise_fourier_coefficients_numba_kernel(
        breaks: np.ndarray,
        refractive_index: np.ndarray,
        period: float,
        max_order: int,
    ) -> np.ndarray:
        """Return Fourier coefficients using a Numba-compiled harmonic loop."""

        harmonic_count = (2 * max_order) + 1
        coeffs = np.zeros(harmonic_count, dtype=np.complex128)
        epsilon = refractive_index**2
        for harmonic_index in range(harmonic_count):
            order = harmonic_index - max_order
            if order == 0:
                total = 0.0 + 0.0j
                for segment_index in range(epsilon.size):
                    total += epsilon[segment_index] * (breaks[segment_index + 1] - breaks[segment_index])
                coeffs[harmonic_index] = total / period
                continue

            total = 0.0 + 0.0j
            denominator = -1j * 2.0 * np.pi * order
            for segment_index in range(epsilon.size):
                phase_right = np.exp(-1j * 2.0 * np.pi * order * breaks[segment_index + 1] / period)
                phase_left = np.exp(-1j * 2.0 * np.pi * order * breaks[segment_index] / period)
                total += epsilon[segment_index] * (phase_right - phase_left) / denominator
            coeffs[harmonic_index] = total
        return coeffs


def _resolve_fourier_backend(
    backend: FourierBackendName | str,
    profiler: SolverProfiler | None = None,
) -> FourierBackend:
    """Return the effective Fourier backend after validation."""

    if backend == "numba" and not _NUMBA_AVAILABLE:
        raise RuntimeError(
            "numba backend requested but numba is not installed. "
            "Reinstall the standard graxpy package so the required numba dependency is available."
        )
    try:
        return FourierBackend(backend)
    except ValueError as error:
        raise ValueError(f"Unsupported Fourier backend: {backend}") from error


def res0(dim: int) -> Parameters:
    if dim == 0 or abs(dim) != 1:
        raise NotImplementedError("The Python port currently supports 1D TE/TM-style entry only.")
    return Parameters(dim=1, polarization=1 if dim > 0 else -1)


def res1(
    wavelength: float,
    period: float,
    textures: list[ArrayLike],
    nn: int | tuple[int, int] | list[int],
    beta0: float,
    parm: Parameters | None = None,
    *,
    _profiler: SolverProfiler | None = None,
    _fourier_backend: FourierBackendName | str = "numpy",
) -> Res1Result:
    parm = parm or res0(1)

    with _profiler.record("res1_total") if _profiler is not None else _nullcontext():
        orders = _normalize_orders(nn)
        harmonic_count = (2 * int(np.max(np.abs(orders)))) + 1
        resolved_backend = _resolve_fourier_backend(_fourier_backend, _profiler)
        if _profiler is not None:
            _profiler.set_metadata("fourier_backend_requested", str(_fourier_backend))
            _profiler.set_metadata("fourier_backend_actual", resolved_backend.value)
            _profiler.set_metadata("numba_available", _NUMBA_AVAILABLE)
            _profiler.set_metadata("texture_count", len(textures))
            _profiler.set_metadata("input_texture_count", len(textures))
            _profiler.set_metadata("harmonic_count", harmonic_count)
        unique_signatures = {_texture_signature_for_metadata(texture, period) for texture in textures}
        if _profiler is not None:
            _profiler.set_metadata("unique_texture_signatures", len(unique_signatures))
        texture_cache: dict[tuple[Any, ...], Texture1D] = {}
        converted: list[Texture1D] = []
        cache_hits = 0
        cache_misses = 0
        max_order = 2 * int(np.max(np.abs(orders)))
        for texture in textures:
            cache_key = (
                _texture_signature_for_metadata(texture, period),
                float(period),
                int(max_order),
                resolved_backend.value,
            )
            if cache_key in texture_cache:
                converted.append(texture_cache[cache_key])
                cache_hits += 1
                if _profiler is not None:
                    _profiler.add_detail_count("texture_conversion_cache_hits", 1)
                continue
            converted_texture = _convert_texture(
                texture,
                period,
                orders,
                _profiler=_profiler,
                _fourier_backend=resolved_backend,
            )
            texture_cache[cache_key] = converted_texture
            converted.append(converted_texture)
            cache_misses += 1
            if _profiler is not None:
                _profiler.add_detail_count("texture_conversion_cache_misses", 1)
        if _profiler is not None:
            _profiler.set_metadata("texture_conversion_cache_hits", cache_hits)
            _profiler.set_metadata("texture_conversion_cache_misses", cache_misses)
    return Res1Result(
        wavelength=float(wavelength),
        period=float(period),
        orders=orders,
        beta0=float(beta0),
        polarization=parm.polarization,
        textures=converted,
    )


def prepare_layer_stack(
    aa: Res1Result,
    profile: tuple[ArrayLike, ArrayLike],
) -> tuple[complex, complex, list[tuple[float, Texture1D]]]:
    """Validate one profile and return the semi-infinite media and finite layers.

    Args:
        aa: Fourier-space texture data returned by :func:`res1`.
        profile: Thickness and texture-index profile for one layer stack.

    Returns:
        Superstrate index, substrate index, and the compressed finite-layer list
        where consecutive identical textures have been merged.
    """

    thicknesses = np.asarray(profile[0], dtype=float)
    texture_indices = np.asarray(profile[1], dtype=int)
    if thicknesses.shape != texture_indices.shape:
        raise ValueError("Profile thicknesses and texture indices must have the same length.")
    if len(texture_indices) < 2:
        raise ValueError("The profile must contain at least superstrate and substrate textures.")

    top_texture = aa.textures[int(texture_indices[0])]
    bottom_texture = aa.textures[int(texture_indices[-1])]
    if top_texture.homogeneous_index is None or bottom_texture.homogeneous_index is None:
        raise ValueError("The Python port expects homogeneous superstrate and substrate textures.")

    compressed: list[tuple[float, Texture1D]] = []
    for thickness, texture_index in zip(thicknesses[1:-1], texture_indices[1:-1]):
        if thickness <= 0:
            continue
        texture = aa.textures[int(texture_index)]
        if compressed and compressed[-1][1].signature == texture.signature:
            compressed[-1] = (compressed[-1][0] + float(thickness), texture)
        else:
            compressed.append((float(thickness), texture))

    return top_texture.homogeneous_index, bottom_texture.homogeneous_index, compressed


def apply_optional_roughness(
    reflected: DiffractionResult,
    transmitted: DiffractionResult,
    *,
    aa: Res1Result,
    roughness_sigma_nm: float | None,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Apply optional scalar Debye-Waller damping to one solver's output."""

    if roughness_sigma_nm is None:
        return reflected, transmitted

    diagnostics = debye_waller_roughness_diagnostics(
        sigma_nm=float(roughness_sigma_nm),
        wavelength_nm=aa.wavelength,
        beta0=float(aa.beta0),
        theta_surface_rad=float(np.arccos(np.clip(float(aa.beta0), -1.0, 1.0))),
    )
    logger.debug(
        "Debye-Waller roughness diagnostics: "
        "sigma_nm=%(sigma_nm).6g, wavelength_nm=%(wavelength_nm).6g, "
        "theta_surface_rad=%(theta_surface_rad).6g, "
        "theta_normal_rad=%(theta_normal_rad).6g, "
        "beta0=%(beta0).6g, sin_theta_used=%(sin_theta_used).6g, A=%(A).6g, "
        "A_squared=%(A_squared).6g, damping_factor=%(damping_factor).6g",
        diagnostics,
    )
    return apply_debye_waller_roughness(
        reflected=reflected,
        transmitted=transmitted,
        wavelength_nm=aa.wavelength,
        beta0=float(aa.beta0),
        roughness_sigma_nm=float(roughness_sigma_nm),
    )


def _apply_debye_waller_roughness(
    result: DiffractionResult,
    *,
    wavelength_nm: float,
    incidence_sine: float,
    roughness_sigma_nm: float,
) -> DiffractionResult:
    """Compatibility wrapper for the extracted roughness module."""

    return _apply_debye_waller_roughness_impl(
        result,
        wavelength_nm=wavelength_nm,
        incidence_sine=incidence_sine,
        roughness_sigma_nm=roughness_sigma_nm,
    )


def _debye_waller_roughness_factor(
    *,
    wavelength_nm: float,
    incidence_sine: float,
    roughness_sigma_nm: float | None,
) -> float:
    """Compatibility wrapper for the extracted roughness module."""

    return _debye_waller_roughness_factor_impl(
        wavelength_nm=wavelength_nm,
        incidence_sine=incidence_sine,
        roughness_sigma_nm=roughness_sigma_nm,
    )


def _incidence_sine_from_beta0(beta0: float) -> float:
    """Compatibility wrapper for the extracted roughness module."""

    return incidence_sine_from_beta0(beta0)


def _normalize_orders(nn: int | tuple[int, int] | list[int]) -> np.ndarray:
    if np.isscalar(nn):
        n = int(nn)
        return np.arange(-n, n + 1, dtype=int)
    values = np.asarray(nn, dtype=int).ravel()
    if values.size != 2:
        raise ValueError("Only scalar or two-value 1D order definitions are supported.")
    return np.arange(values[0], values[1] + 1, dtype=int)


def _convert_texture(
    texture: ArrayLike,
    period: float,
    orders: np.ndarray,
    *,
    _profiler: SolverProfiler | None = None,
    _fourier_backend: FourierBackend = FourierBackend.NUMPY,
) -> Texture1D:
    if not isinstance(texture, (list, tuple)):
        refractive_index = complex(texture)
        _breaks = np.array([0.0, period], dtype=float)
        _n = np.array([refractive_index], dtype=complex)
        _max_order = 2 * int(np.max(np.abs(orders)))
        return Texture1D(
            period=period,
            breaks=_breaks,
            refractive_index=_n,
            epsilon_fourier=_piecewise_fourier_coefficients(
                _breaks, _n, period=period, max_order=_max_order,
                _profiler=_profiler, _fourier_backend=_fourier_backend,
            ),
            homogeneous_index=refractive_index,
            inv_epsilon_fourier=_piecewise_fourier_coefficients(
                _breaks, 1.0 / _n, period=period, max_order=_max_order,
                _profiler=None, _fourier_backend=_fourier_backend,
            ),
        )

    if len(texture) == 1:
        refractive_index = complex(texture[0])
        return _convert_texture(
            refractive_index,
            period,
            orders,
            _profiler=_profiler,
            _fourier_backend=_fourier_backend,
        )

    if len(texture) != 2:
        raise ValueError("1D Python textures must be homogeneous or [x_positions, n_left].")

    x_positions = np.asarray(texture[0], dtype=float).ravel()
    n_left = np.asarray(texture[1], dtype=complex).ravel()
    if x_positions.size != n_left.size:
        raise ValueError("Texture boundary and refractive-index vectors must have the same length.")
    if x_positions.size == 0:
        raise ValueError("Patterned textures require at least one discontinuity.")

    breaks = np.concatenate(([0.0], x_positions, [period]))
    refractive_index = np.concatenate((n_left, [n_left[0]]))
    _max_order = 2 * int(np.max(np.abs(orders)))
    return Texture1D(
        period=period,
        breaks=breaks,
        refractive_index=refractive_index,
        epsilon_fourier=_piecewise_fourier_coefficients(
            breaks, refractive_index, period=period, max_order=_max_order,
            _profiler=_profiler, _fourier_backend=_fourier_backend,
        ),
        inv_epsilon_fourier=_piecewise_fourier_coefficients(
            breaks, 1.0 / refractive_index, period=period, max_order=_max_order,
            _profiler=None, _fourier_backend=_fourier_backend,
        ),
    )


def _texture_signature_for_metadata(texture: ArrayLike, period: float) -> tuple[Any, ...]:
    """Return a hashable texture signature for one ``res1`` call."""

    if not isinstance(texture, (list, tuple)):
        value = complex(texture)
        return ("homogeneous", round(float(np.real(value)), 12), round(float(np.imag(value)), 12))

    if len(texture) == 1:
        value = complex(texture[0])
        return ("homogeneous", round(float(np.real(value)), 12), round(float(np.imag(value)), 12))

    x_positions = np.asarray(texture[0], dtype=float).ravel()
    n_left = np.asarray(texture[1], dtype=complex).ravel()
    breaks = np.concatenate(([0.0], x_positions, [period]))
    refractive_index = np.concatenate((n_left, [n_left[0]]))
    return (
        "patterned",
        tuple(np.round(breaks, 12)),
        tuple(
            (round(float(np.real(value)), 12), round(float(np.imag(value)), 12))
            for value in refractive_index
        ),
    )


def _record_fourier_array_allocations(
    profiler: SolverProfiler | None,
    *arrays: np.ndarray,
) -> None:
    """Record temporary allocation estimates for Fourier helpers."""

    if profiler is None:
        return
    bytes_total = sum(int(array.nbytes) for array in arrays)
    profiler.add_detail_count("fourier_allocation_bytes", bytes_total)
    profiler.update_detail_peak("fourier_temp_buffer_bytes_peak", float(bytes_total))


def _record_fourier_common_stats(
    profiler: SolverProfiler | None,
    refractive_index: np.ndarray,
    breaks: np.ndarray,
    max_order: int,
    backend: FourierBackend,
) -> None:
    """Record common Fourier call metadata and counters."""

    if profiler is None:
        return
    profiler.add_detail_count("fourier_calls", 1)
    profiler.add_detail_count("fourier_harmonics_total", (2 * max_order) + 1)
    profiler.add_detail_count("fourier_segments_total", int(refractive_index.size))
    profiler.add_detail_count("fourier_backend_calls", 1)
    profiler.add_unique_value("fourier_backend_actual", backend.value)
    profiler.update_detail_peak("fourier_segment_count_peak", float(refractive_index.size))
    profiler.update_detail_peak("fourier_harmonic_count_peak", float((2 * max_order) + 1))
    _record_fourier_array_allocations(profiler, refractive_index, breaks)


def _record_fourier_call_totals(
    profiler: SolverProfiler | None,
    total_seconds: float,
    exp_seconds: float,
    sum_seconds: float,
) -> None:
    """Record comparable detailed Fourier timings across backends."""

    if profiler is None:
        return
    overhead_seconds = max(0.0, total_seconds - exp_seconds - sum_seconds)
    profiler.add_detail_timing("fourier_call_total", total_seconds)
    profiler.add_detail_timing("fourier_exp", exp_seconds)
    profiler.add_detail_timing("fourier_sum", sum_seconds)
    profiler.add_detail_timing("fourier_loop_overhead", overhead_seconds)


def _piecewise_fourier_coefficients_baseline(
    breaks: np.ndarray,
    refractive_index: np.ndarray,
    period: float,
    max_order: int,
    *,
    profiler: SolverProfiler | None,
) -> np.ndarray:
    """Return Fourier coefficients using the legacy harmonic loop."""

    call_start = perf_counter() if profiler is not None else None
    g = np.arange(-max_order, max_order + 1, dtype=int)
    epsilon = refractive_index**2
    coeffs = np.zeros_like(g, dtype=complex)
    _record_fourier_array_allocations(profiler, g, epsilon, coeffs)

    exp_seconds = 0.0
    sum_seconds = 0.0
    for idx, order in enumerate(g):
        if profiler is not None:
            profiler.add_detail_count("fourier_loop_iterations", 1)
        if order == 0:
            t0 = perf_counter() if profiler is not None else None
            coeffs[idx] = np.sum(epsilon * (breaks[1:] - breaks[:-1])) / period
            if profiler is not None and t0 is not None:
                sum_seconds += perf_counter() - t0
                profiler.add_detail_count("fourier_sum_calls", 1)
            continue

        t0 = perf_counter() if profiler is not None else None
        phase_right = np.exp(-1j * 2.0 * np.pi * order * breaks[1:] / period)
        phase_left = np.exp(-1j * 2.0 * np.pi * order * breaks[:-1] / period)
        if profiler is not None and t0 is not None:
            exp_seconds += perf_counter() - t0
            profiler.add_detail_count("fourier_exp_calls", int(phase_right.size + phase_left.size))
            _record_fourier_array_allocations(profiler, phase_right, phase_left)

        t1 = perf_counter() if profiler is not None else None
        coeffs[idx] = np.sum(
            epsilon * (phase_right - phase_left) / (-1j * 2.0 * np.pi * order)
        )
        if profiler is not None and t1 is not None:
            sum_seconds += perf_counter() - t1
            profiler.add_detail_count("fourier_sum_calls", 1)

    if profiler is not None and call_start is not None:
        _record_fourier_call_totals(profiler, perf_counter() - call_start, exp_seconds, sum_seconds)
    return coeffs


def _piecewise_fourier_coefficients_numba(
    breaks: np.ndarray,
    refractive_index: np.ndarray,
    period: float,
    max_order: int,
    *,
    profiler: SolverProfiler | None,
) -> np.ndarray:
    """Return Fourier coefficients using the Numba kernel."""

    call_start = perf_counter() if profiler is not None else None
    coeffs = _piecewise_fourier_coefficients_numba_kernel(breaks, refractive_index, period, max_order)
    if profiler is not None:
        profiler.add_detail_count("fourier_loop_iterations", int((2 * max_order) + 1))
        profiler.add_detail_count("fourier_numba_calls", 1)
        profiler.add_detail_count("fourier_sum_calls", int((2 * max_order) + 1))
        profiler.add_detail_count("fourier_exp_calls", int((2 * max_order) * 2 * refractive_index.size))
        _record_fourier_array_allocations(profiler, coeffs)
    if profiler is not None and call_start is not None:
        total_seconds = perf_counter() - call_start
        _record_fourier_call_totals(profiler, total_seconds, total_seconds, 0.0)
    return coeffs


def _piecewise_fourier_coefficients(
    breaks: np.ndarray,
    refractive_index: np.ndarray,
    period: float,
    max_order: int,
    *,
    _profiler: SolverProfiler | None = None,
    _fourier_backend: FourierBackend = FourierBackend.NUMPY,
) -> np.ndarray:
    with _profiler.record("fourier_coefficients") if _profiler is not None else _nullcontext():
        _record_fourier_common_stats(_profiler, refractive_index, breaks, max_order, _fourier_backend)
        if _fourier_backend is FourierBackend.NUMPY:
            return _piecewise_fourier_coefficients_baseline(
                breaks,
                refractive_index,
                period,
                max_order,
                profiler=_profiler,
            )
        if _fourier_backend is FourierBackend.NUMBA:
            return _piecewise_fourier_coefficients_numba(
                breaks,
                refractive_index,
                period,
                max_order,
                profiler=_profiler,
            )
        raise ValueError(f"Unsupported Fourier backend: {_fourier_backend}")


def _convolution_matrix(coefficients: np.ndarray, orders: np.ndarray) -> np.ndarray:
    max_order = (len(coefficients) - 1) // 2
    coefficient_indices = orders[:, np.newaxis] - orders[np.newaxis, :] + max_order
    return coefficients[coefficient_indices]


@dataclass(frozen=True)
class LayerFieldOperators:
    """Fourier-space operators describing one finite layer.

    The tangential field pair ``[F; G]`` (``[E_y; dE_y/dz]`` in TE,
    ``[H_y; E_x]`` in TM) obeys the first-order system::

        d/dz [F; G] = [[0, A], [B, 0]] [F; G]

    which is equivalent to the second-order form ``d2F/dz2 = (A @ B) F``. The
    modal solver eigen-decomposes ``operator = A @ B``; the differential method
    integrates the first-order system directly.

    Attributes:
        operator: Second-order layer operator ``A @ B``.
        first_order_a: ``A``. ``None`` means the identity (TE).
        first_order_b: ``B``.
        inv_epsilon_conv: TM inverse-permittivity convolution matrix ``[[1/eps]]``
            used to convert ``dF/dz`` into the tangential electric field.
            ``None`` in TE.
    """

    operator: np.ndarray
    first_order_a: np.ndarray | None
    first_order_b: np.ndarray
    inv_epsilon_conv: np.ndarray | None


def layer_field_operators(
    *,
    texture: Texture1D,
    orders: np.ndarray,
    k0: float,
    kx: np.ndarray,
    kx_matrix_sq: np.ndarray,
    polarization: int,
) -> LayerFieldOperators:
    """Return the Fourier-space field operators for one finite layer.

    TE uses the direct (Laurent) rule. TM uses the Li/Fast-Fourier-Factorization
    inverse rule so that the products of the discontinuous permittivity and the
    discontinuous normal field component converge properly; this is what keeps
    p-polarized convergence usable for high-contrast profiles.

    Args:
        texture: Fourier-space texture for the layer.
        orders: Retained diffraction orders.
        k0: Vacuum wavenumber.
        kx: In-plane wavevector for each retained order.
        kx_matrix_sq: ``diag(kx**2)``, precomputed once per stack.
        polarization: ``1`` for TE, ``-1`` for TM.

    Returns:
        Layer operators for both the modal and differential formulations.
    """

    basis_size = len(orders)
    epsilon_conv = _convolution_matrix(texture.epsilon_fourier, orders)
    if polarization == 1:
        operator = kx_matrix_sq - (k0**2) * epsilon_conv
        if np.any(np.isnan(operator)) or np.any(np.isinf(operator)):
            raise ValueError("Layer operator contains NaN/Inf values")
        return LayerFieldOperators(
            operator=operator,
            first_order_a=None,
            first_order_b=operator,
            inv_epsilon_conv=None,
        )

    # TM: Li factorization - A = a1 @ a2
    # a2 = kx @ inv([eps]) @ kx - k0^2 I
    # a1 = inv([1/eps])
    # The correct eigenproblem is a1*a2 (not a2*a1), consistent with
    # dH/dz = ik0*a1*E_x and dE_x/dz = ik0*a2*H -> d2H/dz2 = -k0^2*a1*a2*H.
    kx_diag = np.diag(kx)
    inv_epsilon_conv_tm = _convolution_matrix(texture.inv_epsilon_fourier, orders)
    inv_eps_matrix = np.linalg.inv(epsilon_conv)
    inv_inveps_matrix = np.linalg.inv(inv_epsilon_conv_tm)
    a2 = kx_diag @ inv_eps_matrix @ kx_diag - (k0**2) * np.eye(basis_size)
    operator = inv_inveps_matrix @ a2
    if np.any(np.isnan(operator)) or np.any(np.isinf(operator)):
        raise ValueError("Layer operator contains NaN/Inf values")
    return LayerFieldOperators(
        operator=operator,
        first_order_a=inv_inveps_matrix,
        first_order_b=a2,
        inv_epsilon_conv=inv_epsilon_conv_tm,
    )


def solve_stack_from_layer_blocks(
    *,
    wavelength: float,
    period: float,
    orders: np.ndarray,
    beta0: float,
    n_top: complex,
    n_bottom: complex,
    polarization: int,
    layers: list[tuple[float, Texture1D]],
    layer_block_fn: LayerBlockFunction,
    _profiler: SolverProfiler | None = None,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Solve one 1D stack from per-layer interface-response blocks.

    Each finite layer is represented by its Dirichlet-to-Neumann boundary block.
    Blocks are cascaded by eliminating shared interface fields, which keeps the
    propagation stable for deep gratings: no growing exponential is ever formed
    across the whole stack. ``layer_block_fn`` decides how one layer's block is
    built, and is the only thing that differs between the modal (RCWA) and
    differential-method solvers.

    Args:
        wavelength: Vacuum wavelength in nanometers.
        period: Grating period in nanometers.
        orders: Retained diffraction orders.
        beta0: In-plane incidence direction cosine.
        n_top: Superstrate refractive index.
        n_bottom: Substrate refractive index.
        polarization: ``1`` for TE, ``-1`` for TM.
        layers: Finite layers as ``(thickness, texture)`` pairs, top to bottom.
        layer_block_fn: Callable returning one layer's interface-response block.
        _profiler: Optional solver profiler.

    Returns:
        Reflected and transmitted diffraction results for incidence from top.
    """

    k0 = 2 * np.pi / wavelength
    kx = k0 * beta0 + (2 * np.pi * orders / period)
    kx_matrix_sq = np.diag(kx**2)
    basis_size = len(orders)

    kz_top = _kz_branch_array((k0 * n_top) ** 2 - kx**2)
    kz_bottom = _kz_branch_array((k0 * n_bottom) ** 2 - kx**2)
    if polarization == 1:
        derivative_top = 1j * np.diag(kz_top)
        derivative_bottom = 1j * np.diag(kz_bottom)
    else:
        derivative_top = 1j * np.diag(kz_top / n_top**2)
        derivative_bottom = 1j * np.diag(kz_bottom / n_bottom**2)

    stack_boundary: np.ndarray | None = None
    with _profiler.record("layer_propagation_cascade") if _profiler is not None else _nullcontext():
        for thickness, texture in layers:
            block = layer_block_fn(
                thickness=thickness,
                texture=texture,
                orders=orders,
                k0=k0,
                kx=kx,
                kx_matrix_sq=kx_matrix_sq,
                polarization=polarization,
                _profiler=_profiler,
            )
            if _profiler is not None:
                _profiler.add_detail_count("layer_boundary_blocks_constructed", 1)
                _profiler.update_detail_peak("layer_boundary_block_temp_peak", 1.0)
                _profiler.update_detail_peak("layer_boundary_block_bytes_peak", float(block.nbytes))
            if stack_boundary is None:
                stack_boundary = block
                continue
            stack_boundary = _cascade_boundary_pair(
                stack_boundary,
                block,
                basis_size,
                _profiler=_profiler,
            )
    top_stack_admittance = _top_admittance_from_boundary_block(
        stack_boundary=stack_boundary,
        derivative_bottom=derivative_bottom,
    )

    incident = np.zeros(basis_size, dtype=complex)
    incident[np.where(orders == 0)[0][0]] = 1.0

    with _profiler.record("matrix_solves") if _profiler is not None else _nullcontext():
        reflected_amplitude = np.linalg.solve(
            derivative_top + top_stack_admittance,
            (derivative_top - top_stack_admittance) @ incident,
        )
    top_field = incident + reflected_amplitude
    transmitted_amplitude = _bottom_field_from_boundary_block(
        top_field=top_field,
        stack_boundary=stack_boundary,
        derivative_bottom=derivative_bottom,
        _profiler=_profiler,
    )

    incident_kz = kz_top[np.where(orders == 0)[0][0]]
    if polarization == 1:
        reflected_efficiency = np.real(kz_top / incident_kz) * np.abs(reflected_amplitude) ** 2
        transmitted_efficiency = np.real(kz_bottom / incident_kz) * np.abs(transmitted_amplitude) ** 2
    else:
        # TM Poynting flux ~ Re(kz/n^2); normalise by incident admittance Re(kz_inc/n_top^2)
        incident_admittance = np.real(incident_kz / n_top**2)
        reflected_efficiency = (
            np.real(kz_top / n_top**2) / incident_admittance * np.abs(reflected_amplitude) ** 2
        )
        transmitted_efficiency = (
            np.real(kz_bottom / n_bottom**2) / incident_admittance * np.abs(transmitted_amplitude) ** 2
        )

    reflected_theta = _angles_from_kx(kx, k0, n_top)
    transmitted_theta = _angles_from_kx(kx, k0, n_bottom)

    return (
        DiffractionResult(
            order=orders.copy(),
            theta=reflected_theta,
            efficiency=reflected_efficiency,
            amplitude=reflected_amplitude,
        ),
        DiffractionResult(
            order=orders.copy(),
            theta=transmitted_theta,
            efficiency=transmitted_efficiency,
            amplitude=transmitted_amplitude,
        ),
    )


def propagating_order_mask(
    *,
    wavelength: float,
    period: float,
    orders: np.ndarray,
    beta0: float,
    refractive_index: complex,
) -> np.ndarray:
    """Return a mask selecting orders that propagate in one semi-infinite medium.

    Args:
        wavelength: Vacuum wavelength in nanometers.
        period: Grating period in nanometers.
        orders: Retained diffraction orders.
        beta0: In-plane incidence direction cosine.
        refractive_index: Refractive index of the medium.

    Returns:
        Boolean mask that is ``True`` for propagating orders.
    """

    k0 = 2 * np.pi / wavelength
    kx = k0 * beta0 + (2 * np.pi * orders / period)
    kz_squared = (k0 * refractive_index) ** 2 - kx**2
    return np.real(kz_squared) > 0.0


def propagating_energy_balance(
    reflected: DiffractionResult,
    transmitted: DiffractionResult,
    *,
    wavelength: float,
    period: float,
    beta0: float,
    n_top: complex,
    n_bottom: complex,
) -> dict[str, float]:
    """Return the propagating energy balance for one solve.

    For a lossless structure the total should equal one. For the absorbing
    materials used in the X-ray regime it is below one, and the shortfall is the
    absorbed fraction rather than a solver error.

    Args:
        reflected: Reflected diffraction result.
        transmitted: Transmitted diffraction result.
        wavelength: Vacuum wavelength in nanometers.
        period: Grating period in nanometers.
        beta0: In-plane incidence direction cosine.
        n_top: Superstrate refractive index.
        n_bottom: Substrate refractive index.

    Returns:
        Reflected, transmitted, and total propagating efficiency sums.
    """

    reflected_mask = propagating_order_mask(
        wavelength=wavelength,
        period=period,
        orders=np.asarray(reflected.order),
        beta0=beta0,
        refractive_index=n_top,
    )
    transmitted_mask = propagating_order_mask(
        wavelength=wavelength,
        period=period,
        orders=np.asarray(transmitted.order),
        beta0=beta0,
        refractive_index=n_bottom,
    )
    reflected_sum = float(np.sum(np.real(np.asarray(reflected.efficiency))[reflected_mask]))
    transmitted_sum = float(np.sum(np.real(np.asarray(transmitted.efficiency))[transmitted_mask]))
    return {
        "reflected": reflected_sum,
        "transmitted": transmitted_sum,
        "total": reflected_sum + transmitted_sum,
    }


def _cascade_boundary_pair(
    left: np.ndarray,
    right: np.ndarray,
    basis_size: int,
    *,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray:
    """Cascade two adjacent interface-response blocks into one block."""

    if _profiler is not None:
        _profiler.add_detail_count("layer_cascade_pair_calls", 1)
    with _profiler.record("layer_block_cascade_pair") if _profiler is not None else _nullcontext():
        with _profiler.record("layer_cascade_pair_partition") if _profiler is not None else _nullcontext():
            l11 = left[:basis_size, :basis_size]
            l12 = left[:basis_size, basis_size:]
            l21 = left[basis_size:, :basis_size]
            l22 = left[basis_size:, basis_size:]
            r11 = right[:basis_size, :basis_size]
            r12 = right[:basis_size, basis_size:]
            r21 = right[basis_size:, :basis_size]
            r22 = right[basis_size:, basis_size:]

        with _profiler.record("layer_cascade_pair_matrix_setup") if _profiler is not None else _nullcontext():
            matrix_to_solve = l22 - r11
            rhs = np.empty((basis_size, 2 * basis_size), dtype=left.dtype)
            rhs[:, :basis_size] = l21
            rhs[:, basis_size:] = r12
        logger.debug("  _cascade_boundary_pair: solving interface system...")
        if logger.isEnabledFor(logging.DEBUG):
            try:
                cond = np.linalg.cond(matrix_to_solve)
                if cond > 1e12:
                    logger.warning(f"  interface matrix is nearly singular (cond={cond:.2e})")
            except Exception:
                pass

        try:
            with _profiler.record("layer_cascade_pair_solve") if _profiler is not None else _nullcontext():
                solved_blocks = np.linalg.solve(matrix_to_solve, rhs)
        except np.linalg.LinAlgError as e:
            logger.error(f"  np.linalg.solve failed in cascade: {e}")
            raise

        solved_l21 = solved_blocks[:, :basis_size]
        solved_r12 = solved_blocks[:, basis_size:]

        with _profiler.record("layer_cascade_pair_multiply") if _profiler is not None else _nullcontext():
            top_left = l11 - (l12 @ solved_l21)
            top_right = l12 @ solved_r12
            bottom_left = -(r21 @ solved_l21)
            bottom_right = r22 + (r21 @ solved_r12)

        with _profiler.record("layer_cascade_pair_assemble") if _profiler is not None else _nullcontext():
            result = np.empty_like(left)
            result[:basis_size, :basis_size] = top_left
            result[:basis_size, basis_size:] = top_right
            result[basis_size:, :basis_size] = bottom_left
            result[basis_size:, basis_size:] = bottom_right

            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("  cascade produced NaN/Inf values")

    return result


def _cascade_layer_boundary_blocks(
    layer_blocks: list[np.ndarray],
    basis_size: int,
    *,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray | None:
    """Return the global interface-response block for all finite layers."""

    if not layer_blocks:
        return None
    stack_boundary = layer_blocks[0]
    with _profiler.record("layer_propagation_cascade") if _profiler is not None else _nullcontext():
        for block in layer_blocks[1:]:
            stack_boundary = _cascade_boundary_pair(
                stack_boundary,
                block,
                basis_size,
                _profiler=_profiler,
            )
    return stack_boundary


def _top_admittance_from_boundary_block(
    *,
    stack_boundary: np.ndarray | None,
    derivative_bottom: np.ndarray,
) -> np.ndarray:
    """Return the top admittance implied by the cascaded layer-response block."""

    if stack_boundary is None:
        return derivative_bottom

    basis_size = derivative_bottom.shape[0]
    s11 = stack_boundary[:basis_size, :basis_size]
    s12 = stack_boundary[:basis_size, basis_size:]
    s21 = stack_boundary[basis_size:, :basis_size]
    s22 = stack_boundary[basis_size:, basis_size:]
    bottom_response = np.linalg.solve(derivative_bottom - s22, s21)
    return s11 + s12 @ bottom_response


def _bottom_field_from_boundary_block(
    *,
    top_field: np.ndarray,
    stack_boundary: np.ndarray | None,
    derivative_bottom: np.ndarray,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray:
    """Return the substrate field for a boundary-block-cascaded stack."""

    if stack_boundary is None:
        return top_field

    basis_size = top_field.size
    s21 = stack_boundary[basis_size:, :basis_size]
    s22 = stack_boundary[basis_size:, basis_size:]
    with _profiler.record("matrix_solves") if _profiler is not None else _nullcontext():
        return np.linalg.solve(derivative_bottom - s22, s21 @ top_field)


def _angles_from_kx(kx: np.ndarray, k0: float, refractive_index: complex) -> np.ndarray:
    if abs(np.imag(refractive_index)) > 1e-12:
        n_for_angles = np.real(refractive_index)
    else:
        n_for_angles = float(np.real(refractive_index))
    ratio = np.real(kx / (k0 * n_for_angles))
    ratio = np.clip(ratio, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))


def _kz_branch(value: complex) -> complex:
    """Select correct branch for sqrt when computing kz.

    Uses Petit cutoff for branch selection, matching MATLAB's retsqrt(x,0) behavior.

    Args:
        value: Complex value to compute sqrt of.

    Returns:
        kz with correct branch selected: imag(kz) > 0, or (imag(kz) == 0 and real(kz) >= 0).
    """
    kz = np.sqrt(value + 0j)

    # Petit cutoff: keep if imag > 0, or if (imag == 0 and real >= 0)
    # Otherwise flip sign
    if kz.imag < 0 or (np.abs(kz.imag) < 1e-15 and kz.real < 0):
        kz = -kz

    return kz


def _kz_branch_array(values: np.ndarray) -> np.ndarray:
    """Select the outgoing/decaying sqrt branch for an array of kz values."""

    kz = np.sqrt(np.asarray(values, dtype=complex) + 0j)
    flip_mask = (np.imag(kz) < 0) | ((np.abs(np.imag(kz)) < 1e-15) & (np.real(kz) < 0))
    kz[flip_mask] = -kz[flip_mask]
    return kz


def safe_linalg_solve(A: np.ndarray, b: np.ndarray, context: str = "") -> np.ndarray:
    """Solve linear system with error handling and logging.

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector or matrix.
        context: Context string for logging.

    Returns:
        Solution vector/matrix.

    Raises:
        np.linalg.LinAlgError: If the system cannot be solved.
    """
    try:
        cond = np.linalg.cond(A)
        if cond > 1e10:
            logger.warning(f"Matrix condition number is high ({cond:.2e}) in {context}")
        if cond > 1e15:
            logger.error(f"Matrix is nearly singular ({cond:.2e}) in {context}")
    except Exception:
        pass

    try:
        result = np.linalg.solve(A, b)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            logger.error(f"Solution contains NaN/Inf in {context}")
        return result
    except np.linalg.LinAlgError as e:
        logger.error(f"Linear solve failed in {context}: {e}")
        raise
