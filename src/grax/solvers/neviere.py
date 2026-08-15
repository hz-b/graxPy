"""Nevière differential-method solver for 1D gratings.

The differential method expands the fields and the permittivity in the same
truncated Fourier basis as RCWA, but instead of eigen-decomposing a z-invariant
layer operator it integrates the resulting coupled first-order system in ``z``
with a fourth-order Runge-Kutta scheme.

References:
    Nevière, Vincent & Petit, *Nouv. Rev. Optique* **5**, 65 (1974) - original
    formulation. Nevière, *J. Opt. Soc. Am. A* **11**, 1835 (1994) - multilayer
    (Bragg-Fresnel) stacks. Nevière & Popov, *Light Propagation in Periodic
    Media* (CRC, 2003) - reference treatment including the fast-Fourier-
    factorization rules for TM.

Formulation
-----------
With ``F`` the tangential field (``E_y`` in TE, ``H_y`` in TM) and ``G`` its
conjugate tangential partner (``dE_y/dz`` in TE, ``E_x = [[1/eps]] dH_y/dz`` in
TM), Maxwell's equations reduce to::

    d/dz [F; G] = [[0, A], [B, 0]] [F; G]

TE uses ``A = I`` and ``B = Kx^2 - k0^2 [[eps]]``. TM uses the Li /
fast-Fourier-factorization inverse rule, ``A = [[1/eps]]^-1`` and
``B = Kx [[eps]]^-1 Kx - k0^2 I``, which is what keeps p-polarized convergence
usable across a discontinuous permittivity. Both operators are built by
:func:`grax.solvers.common.layer_field_operators`, shared with the RCWA solver,
so the two solvers integrate the same truncated system and differ only in how
they propagate it.

Numerical stability
-------------------
A raw transfer matrix is only ever formed across a short sub-block whose optical
thickness is bounded by ``NeviereOptions.block_phase``, so no growing
exponential is built up. Each sub-block is converted to a Dirichlet-to-Neumann
interface-response block and cascaded by
:func:`grax.solvers.common._cascade_boundary_pair`, which eliminates the shared
interface fields through a linear solve. That cascade is an R-matrix (impedance)
propagation, not a transfer-matrix product, so deep gratings and strongly
evanescent orders stay well conditioned.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal

import numpy as np

from ..simulation._profiling import SolverProfiler
from .common import (
    ArrayLike,
    BoundaryBlockCache,
    FourierBackend,
    FourierBackendName,
    Parameters,
    Res1Result,
    Res2Result,
    Texture1D,
    _MAX_BASIS_SIZE,
    _cascade_boundary_pair,
    _convert_texture,
    _resolve_fourier_backend,
    apply_optional_roughness,
    layer_field_operators,
    prepare_layer_stack,
    propagating_energy_balance,
    solve_stack_from_layer_blocks,
)

logger = logging.getLogger(__name__)

ZSampling = Literal["textures", "continuous"]


@dataclass(frozen=True)
class EpsilonSampler:
    """Continuous permittivity along ``z`` for one grating at one photon energy.

    Calling the sampler with a depth below the top of the modelled stack returns
    the Fourier-space permittivity of the grating cut at that depth. Unlike the
    z-sliced textures, a sampler can be evaluated between solver rows, which is
    what lets the differential method integrate the true profile.

    Attributes:
        total_depth_nm: Depth of the modelled stack in nanometers. Taken from the
            grating geometry rather than from a z-sliced profile, so it does not
            depend on ``z_resolution_nm``.
        sample: Maps a depth in nanometers to the permittivity there.
    """

    total_depth_nm: float
    sample: Callable[[float], Texture1D]

    def __call__(self, depth_nm: float) -> Texture1D:
        """Return the Fourier-space permittivity at one depth."""

        return self.sample(depth_nm)


@dataclass(frozen=True)
class NeviereOptions:
    """Integration settings for the Nevière differential-method solver.

    The step and block sizes are expressed in units of optical phase rather than
    nanometers so that one setting behaves consistently across photon energies,
    grazing angles, and Fourier truncation orders. The relevant scale is
    ``q``, the layer's modal decay/propagation constant; it is bounded per layer
    from the row norm of the layer operator.

    Attributes:
        z_sampling: How the permittivity is sampled along ``z``. ``"textures"``
            integrates through the same z-sliced permittivity the RCWA solver
            uses, which makes the two solvers directly comparable.
            ``"continuous"`` re-expands the permittivity from the true grating
            profile every ``sample_phase`` of optical depth, which is the
            textbook differential method and does not inherit the staircase
            approximation. ``"continuous"`` requires a grating and is selected
            through ``run_simulation``.
        step_phase: Target ``|q| * h`` for one Runge-Kutta step. Accuracy
            improves as the fourth power of this value while cost grows far more
            slowly, because most of the per-layer work is the interface-response
            conversion and cascade rather than the Runge-Kutta stages: on a
            lossless dielectric grating, tightening ``0.05 -> 0.02`` improved the
            energy balance from ``1e-7`` to ``1e-9`` for 30% more runtime.
        block_phase: Maximum ``|q| * d`` accumulated before a sub-block is
            converted to an interface-response block and cascaded. Bounds the
            dynamic range of any transfer matrix that is explicitly formed, and
            so affects conditioning rather than the converged answer.
        sample_phase: Optical phase between permittivity samples, used only by
            ``z_sampling="continuous"``. Unlike ``block_phase`` this one does set
            accuracy: it is the depth quadrature step at which the true profile
            is read, so it plays the role ``z_resolution_nm`` plays for the
            staircase modes. Ignored when ``z_sampling="textures"``.
        max_step_nm: Optional hard upper bound on the Runge-Kutta step in
            nanometers, applied on top of ``step_phase``.
        max_steps_per_layer: Runaway guard on the number of Runge-Kutta steps
            spent on one layer.
        energy_balance_tolerance: When set, the summed propagating reflected and
            transmitted efficiency must not exceed this value or the solve
            raises. Leave ``None`` to rely on the checks in
            :func:`grax.run_simulation`. Note that absorbing materials make the
            balance legitimately smaller than one, so only an upper bound is
            enforced.
    """

    z_sampling: ZSampling = "textures"
    step_phase: float = 0.02
    block_phase: float = 2.0
    sample_phase: float = 0.02
    max_step_nm: float | None = None
    max_steps_per_layer: int = 4096
    energy_balance_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate the integration settings."""

        if self.z_sampling not in ("textures", "continuous"):
            raise ValueError("z_sampling must be 'textures' or 'continuous'.")
        if not self.step_phase > 0.0:
            raise ValueError("step_phase must be > 0.")
        if not self.block_phase > 0.0:
            raise ValueError("block_phase must be > 0.")
        if not self.sample_phase > 0.0:
            raise ValueError("sample_phase must be > 0.")
        if self.max_step_nm is not None and not self.max_step_nm > 0.0:
            raise ValueError("max_step_nm must be > 0 when provided.")
        if self.max_steps_per_layer < 1:
            raise ValueError("max_steps_per_layer must be >= 1.")
        if self.energy_balance_tolerance is not None and not self.energy_balance_tolerance > 0.0:
            raise ValueError("energy_balance_tolerance must be > 0 when provided.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the options."""

        return {
            "z_sampling": self.z_sampling,
            "step_phase": float(self.step_phase),
            "block_phase": float(self.block_phase),
            "sample_phase": float(self.sample_phase),
            "max_step_nm": None if self.max_step_nm is None else float(self.max_step_nm),
            "max_steps_per_layer": int(self.max_steps_per_layer),
            "energy_balance_tolerance": (
                None
                if self.energy_balance_tolerance is None
                else float(self.energy_balance_tolerance)
            ),
        }


def coerce_neviere_options(options: NeviereOptions | dict[str, object] | None) -> NeviereOptions:
    """Return a :class:`NeviereOptions` from an options object, mapping, or ``None``.

    Args:
        options: Options instance, keyword mapping, or ``None`` for defaults.

    Returns:
        Validated options instance.
    """

    if options is None:
        return NeviereOptions()
    if isinstance(options, NeviereOptions):
        return options
    if isinstance(options, dict):
        return NeviereOptions(**options)  # type: ignore[arg-type]
    raise TypeError("neviere_options must be a NeviereOptions, a mapping, or None.")


def res2_dm(
    aa: Res1Result,
    profile: tuple[ArrayLike, ArrayLike],
    parm: Parameters | None = None,
    *,
    roughness_sigma_nm: float | None = None,
    options: NeviereOptions | dict[str, object] | None = None,
    epsilon_sampler: EpsilonSampler | None = None,
    _profiler: SolverProfiler | None = None,
) -> Res2Result:
    """Solve one layer stack with the Nevière differential method.

    Signature-compatible with :func:`grax.solvers.rcwa.res2` so the two solvers
    are interchangeable behind ``grax.run_simulation(..., solver=...)``.

    Args:
        aa: Fourier-space texture data returned by :func:`grax.res1`.
        profile: Thickness and texture-index profile for one layer stack.
        parm: Optional solver parameters. Polarization comes from ``aa``.
        roughness_sigma_nm: Optional rms interface roughness in nanometers,
            applied as the same scalar Debye-Waller damping the RCWA path uses.
        options: Integration settings, see :class:`NeviereOptions`.
        epsilon_sampler: Required when ``options.z_sampling == "continuous"``.
            Maps a depth below the top of the finite stack, in nanometers, to
            the Fourier-space permittivity there. Build one with
            :func:`build_grating_epsilon_sampler`.
        _profiler: Optional solver profiler.

    Returns:
        Reflected and transmitted diffraction results for incidence from top.
    """

    if roughness_sigma_nm is not None and roughness_sigma_nm < 0.0:
        raise ValueError("roughness_sigma_nm must be >= 0 when provided.")

    resolved_options = coerce_neviere_options(options)
    if resolved_options.z_sampling == "continuous" and epsilon_sampler is None:
        raise ValueError(
            "z_sampling='continuous' requires an epsilon_sampler. Use "
            "run_simulation(solver='neviere', ...) so the sampler is built from the grating."
        )

    # Polarization comes from aa (set by res1); parm is ignored for polarization selection.
    polarization = aa.polarization
    n_top, n_bottom, layers = prepare_layer_stack(aa, profile)

    if resolved_options.z_sampling == "continuous":
        layers = _resample_layers_continuously(
            layers,
            epsilon_sampler=epsilon_sampler,  # type: ignore[arg-type]
            options=resolved_options,
        )

    with _profiler.record("res2_total") if _profiler is not None else _nullcontext():
        boundary_block_cache: BoundaryBlockCache = {}

        def layer_block_fn(**kwargs: object) -> np.ndarray:
            return _differential_layer_block(
                options=resolved_options,
                boundary_block_cache=boundary_block_cache,
                **kwargs,  # type: ignore[arg-type]
            )

        reflected, transmitted = solve_stack_from_layer_blocks(
            wavelength=aa.wavelength,
            period=aa.period,
            orders=aa.orders,
            beta0=aa.beta0,
            n_top=n_top,
            n_bottom=n_bottom,
            polarization=polarization,
            layers=layers,
            layer_block_fn=layer_block_fn,
            _profiler=_profiler,
        )

    balance = propagating_energy_balance(
        reflected,
        transmitted,
        wavelength=aa.wavelength,
        period=aa.period,
        beta0=aa.beta0,
        n_top=n_top,
        n_bottom=n_bottom,
    )
    logger.debug(
        "Neviere energy balance: reflected=%.9g, transmitted=%.9g, total=%.9g",
        balance["reflected"],
        balance["transmitted"],
        balance["total"],
    )
    if _profiler is not None:
        _profiler.set_metadata("neviere_energy_balance_total", balance["total"])
    tolerance = resolved_options.energy_balance_tolerance
    if tolerance is not None and balance["total"] > tolerance:
        raise ValueError(
            "Nevière differential-method solve violated the energy balance: "
            f"total propagating efficiency {balance['total']:.9g} exceeds "
            f"tolerance {tolerance:.9g}. Reduce step_phase or block_phase."
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


def _resample_layers_continuously(
    layers: list[tuple[float, Texture1D]],
    *,
    epsilon_sampler: EpsilonSampler,
    options: NeviereOptions,
) -> list[tuple[float, Texture1D]]:
    """Return slabs sampled from the continuous profile instead of the staircase.

    The staircase layers are discarded entirely: both the integration depth and
    the permittivity come from the grating geometry, so the result does not
    depend on ``z_resolution_nm``. Each slab carries the permittivity at its
    midpoint, which makes the depth quadrature second-order accurate. Sub-block
    splitting and Runge-Kutta stepping then happen inside each slab as usual.

    Args:
        layers: Compressed staircase layers, used only to bound the slab count
            and to estimate the permittivity scale.
        epsilon_sampler: Continuous permittivity for the grating.
        options: Integration settings.

    Returns:
        Slabs spanning the stack, carrying continuously sampled permittivities.
    """

    total_depth_nm = float(epsilon_sampler.total_depth_nm)
    if total_depth_nm <= 0.0:
        return layers

    # sample_phase, not block_phase: here the slab thickness sets how finely the
    # true profile is read along z, which is an accuracy knob, not a
    # conditioning one.
    scale = max(
        (_spectral_scale_from_texture(texture) for _, texture in layers),
        default=1.0,
    )
    target_nm = options.sample_phase / scale if scale > 0.0 else total_depth_nm
    if options.max_step_nm is not None:
        target_nm = min(target_nm, options.max_step_nm)
    slab_count = max(1, int(math.ceil(total_depth_nm / max(target_nm, 1e-12))))
    slab_count = min(slab_count, options.max_steps_per_layer * max(len(layers), 1))
    slab_nm = total_depth_nm / slab_count

    return [
        (slab_nm, epsilon_sampler((index + 0.5) * slab_nm))
        for index in range(slab_count)
    ]


def build_grating_epsilon_sampler(
    grating: object,
    *,
    photon_energy_ev: float,
    period_nm: float,
    orders: np.ndarray,
    n_inc: complex = 1.0 + 0.0j,
    fourier_backend: FourierBackendName | str = "numba",
) -> EpsilonSampler:
    """Return a continuous permittivity sampler for one grating.

    The sampler evaluates the grating's material stack at an arbitrary depth
    rather than on the discrete solver z-grid, and Fourier-expands that cut. It
    is what makes ``z_sampling="continuous"`` integrate the true profile instead
    of a staircase approximation of it.

    Args:
        grating: Grating derived from :class:`grax.BaseGrating`.
        photon_energy_ev: Photon energy used to resolve optical constants.
        period_nm: Grating period spanned by the sampler in nanometers.
        orders: Retained diffraction orders.
        n_inc: Incident-medium refractive index.
        fourier_backend: Fourier coefficient backend selector.

    Returns:
        Sampler mapping a depth below the top of the modelled stack, in
        nanometers, to a Fourier-space texture.
    """

    from ..materials import resolve_refractive_index

    resolved_backend: FourierBackend = _resolve_fourier_backend(fourier_backend)
    coating_stack = grating.resolved_stack()  # type: ignore[attr-defined]
    n_sub = complex(resolve_refractive_index(coating_stack.substrate_material, photon_energy_ev))
    num_periods = grating._roughness_num_supercells()  # type: ignore[attr-defined]
    x_grid = grating._build_x_grid(num_periods=num_periods)  # type: ignore[attr-defined]
    z_grid = grating._build_solver_z_grid(coating_stack)  # type: ignore[attr-defined]
    surface = grating._surface_profile_on_grid(x_grid, num_periods=num_periods)  # type: ignore[attr-defined]
    z_top_nm = float(z_grid[0])

    cache: dict[float, Texture1D] = {}

    def sampler(depth_nm: float) -> Texture1D:
        key = round(float(depth_nm), 9)
        cached = cache.get(key)
        if cached is not None:
            return cached
        descriptor = grating._texture_descriptor_for_row(  # type: ignore[attr-defined]
            z_value=z_top_nm - float(depth_nm),
            x_grid=x_grid,
            surface=surface,
            coating_stack=coating_stack,
            photon_energy_ev=photon_energy_ev,
            n_inc=complex(n_inc),
            n_sub=n_sub,
        )
        texture = _convert_texture(
            descriptor,
            period_nm,
            orders,
            _fourier_backend=resolved_backend,
        )
        cache[key] = texture
        return texture

    # The solver z grid runs from the top of the modelled stack down to zero, so
    # its first entry is the stack depth. Reading it here rather than summing a
    # z-sliced profile keeps the continuous mode independent of z_resolution_nm.
    return EpsilonSampler(total_depth_nm=z_top_nm, sample=sampler)


def _spectral_scale_from_texture(texture: Texture1D) -> float:
    """Return a crude propagation-constant scale from a texture alone.

    Used only to pick a slab thickness before any layer operator exists.
    """

    magnitude = float(np.max(np.abs(texture.epsilon_fourier)))
    return max(math.sqrt(magnitude), 1e-12)


def _spectral_scale(operator: np.ndarray) -> float:
    """Return an upper bound on ``|q|`` for one layer operator.

    ``operator`` has eigenvalues ``q**2``. The induced infinity norm (the largest
    absolute row sum) bounds the spectral radius, and is tight for the nearly
    diagonal operators produced by the low-contrast permittivities typical of
    X-ray optics.

    Args:
        operator: Second-order layer operator.

    Returns:
        Bound on the magnitude of the layer's propagation constants.
    """

    row_sums = np.sum(np.abs(operator), axis=1)
    largest = float(np.max(row_sums)) if row_sums.size else 0.0
    return math.sqrt(max(largest, 0.0))


def _rk4_constant_propagator(system: np.ndarray, step_nm: float) -> np.ndarray:
    """Return the RK4 propagator for a z-invariant first-order system.

    For a constant coefficient matrix ``M`` the classical fourth-order
    Runge-Kutta update reduces exactly to the fourth-order truncation of
    ``exp(M h)``.

    Args:
        system: Coefficient matrix ``M``.
        step_nm: Step size ``h`` in nanometers.

    Returns:
        Propagator advancing the state by one step.
    """

    scaled = system * step_nm
    size = scaled.shape[0]
    propagator = np.eye(size, dtype=complex)
    term = np.eye(size, dtype=complex)
    for order in range(1, 5):
        term = (term @ scaled) / order
        propagator = propagator + term
    return propagator


def _rk4_fundamental_matrix(
    system: np.ndarray,
    thickness_nm: float,
    step_count: int,
) -> np.ndarray:
    """Return the fundamental solution across one z-invariant slab.

    Args:
        system: Coefficient matrix of the first-order system.
        thickness_nm: Slab thickness in nanometers.
        step_count: Number of Runge-Kutta steps across the slab.

    Returns:
        Transfer matrix mapping the state at the top of the slab to the bottom.
    """

    step_nm = thickness_nm / step_count
    propagator = _rk4_constant_propagator(system, step_nm)
    result = propagator
    for _ in range(step_count - 1):
        result = result @ propagator
    return result


def _first_order_system(
    first_order_a: np.ndarray | None,
    first_order_b: np.ndarray,
) -> np.ndarray:
    """Assemble ``[[0, A], [B, 0]]`` for the coupled first-order system."""

    basis_size = first_order_b.shape[0]
    system = np.zeros((2 * basis_size, 2 * basis_size), dtype=complex)
    if first_order_a is None:
        system[:basis_size, basis_size:] = np.eye(basis_size, dtype=complex)
    else:
        system[:basis_size, basis_size:] = first_order_a
    system[basis_size:, :basis_size] = first_order_b
    return system


def _boundary_block_from_transfer(transfer: np.ndarray, basis_size: int) -> np.ndarray:
    """Convert a slab transfer matrix into an interface-response block.

    ``transfer`` maps ``[F; G]`` at the top of the slab to ``[F; G]`` at the
    bottom. The cascade routine instead needs the Dirichlet-to-Neumann block
    mapping ``[F_top; F_bottom]`` to ``[G_top; G_bottom]``, which for a
    homogeneous slab reproduces the modal ``[[-q coth(qd), q csch(qd)],
    [-q csch(qd), q coth(qd)]]`` exactly.

    Args:
        transfer: Slab transfer matrix.
        basis_size: Number of retained orders.

    Returns:
        Interface-response block for the slab.
    """

    t11 = transfer[:basis_size, :basis_size]
    t12 = transfer[:basis_size, basis_size:]
    t21 = transfer[basis_size:, :basis_size]
    t22 = transfer[basis_size:, basis_size:]

    rhs = np.empty((basis_size, 2 * basis_size), dtype=complex)
    rhs[:, :basis_size] = t11
    rhs[:, basis_size:] = np.eye(basis_size, dtype=complex)
    try:
        solved = np.linalg.solve(t12, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Nevière slab transfer matrix is singular. Reduce block_phase so "
            "shorter sub-blocks are cascaded."
        ) from error
    t12_inv_t11 = solved[:, :basis_size]
    t12_inv = solved[:, basis_size:]

    block = np.empty((2 * basis_size, 2 * basis_size), dtype=complex)
    block[:basis_size, :basis_size] = -t12_inv_t11
    block[:basis_size, basis_size:] = t12_inv
    block[basis_size:, :basis_size] = t21 - (t22 @ t12_inv_t11)
    block[basis_size:, basis_size:] = t22 @ t12_inv
    if np.any(np.isnan(block)) or np.any(np.isinf(block)):
        raise ValueError("Nevière layer block produced NaN/Inf values")
    return block


def _differential_layer_block(
    *,
    thickness: float,
    texture: Texture1D,
    orders: np.ndarray,
    k0: float,
    kx: np.ndarray,
    kx_matrix_sq: np.ndarray,
    polarization: int,
    options: NeviereOptions,
    boundary_block_cache: BoundaryBlockCache,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray:
    """Return the interface-response block for one layer by RK4 integration."""

    basis_size = len(orders)
    if basis_size > _MAX_BASIS_SIZE:
        raise ValueError(
            f"Fourier orders too large: {basis_size} modes. "
            "Try reducing fourier_orders to 50 or less."
        )

    cache_key = (
        texture.signature,
        float(thickness),
        tuple(int(order) for order in orders),
        float(k0),
        int(polarization),
    )
    cached_block = boundary_block_cache.get(cache_key)
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
        system = _first_order_system(operators.first_order_a, operators.first_order_b)

    scale = _spectral_scale(operators.operator)
    block_count, step_count = _resolve_integration_schedule(
        thickness_nm=float(thickness),
        scale=scale,
        options=options,
    )
    if _profiler is not None:
        _profiler.add_detail_count("neviere_sub_blocks", block_count)
        _profiler.add_detail_count("neviere_rk4_steps", block_count * step_count)
        total_steps = float(block_count * step_count)
        _profiler.update_detail_peak("neviere_steps_per_layer_peak", total_steps)

    sub_thickness_nm = float(thickness) / block_count
    integration_stage = (
        _profiler.record("neviere_layer_integration") if _profiler is not None else _nullcontext()
    )
    with integration_stage:
        transfer = _rk4_fundamental_matrix(system, sub_thickness_nm, step_count)
        sub_block = _boundary_block_from_transfer(transfer, basis_size)

    block = sub_block
    if block_count > 1:
        cascade_stage = (
            _profiler.record("layer_propagation_cascade")
            if _profiler is not None
            else _nullcontext()
        )
        with cascade_stage:
            for _ in range(block_count - 1):
                block = _cascade_boundary_pair(
                    block,
                    sub_block,
                    basis_size,
                    _profiler=_profiler,
                )

    boundary_block_cache[cache_key] = block
    return block


def _resolve_integration_schedule(
    *,
    thickness_nm: float,
    scale: float,
    options: NeviereOptions,
) -> tuple[int, int]:
    """Return the sub-block count and per-block Runge-Kutta step count.

    Both are derived from the layer's optical thickness ``scale * thickness_nm``:
    ``block_phase`` bounds how much phase one explicitly formed transfer matrix
    may span, and ``step_phase`` bounds the phase per Runge-Kutta step.

    Args:
        thickness_nm: Layer thickness in nanometers.
        scale: Bound on the layer's propagation constants.
        options: Integration settings.

    Returns:
        Number of cascaded sub-blocks and Runge-Kutta steps per sub-block.
    """

    optical_thickness = scale * max(thickness_nm, 0.0)
    block_count = max(1, int(math.ceil(optical_thickness / options.block_phase)))
    sub_thickness_nm = thickness_nm / block_count
    step_count = max(1, int(math.ceil((optical_thickness / block_count) / options.step_phase)))
    if options.max_step_nm is not None:
        step_count = max(step_count, int(math.ceil(sub_thickness_nm / options.max_step_nm)))
    step_count = min(step_count, options.max_steps_per_layer)
    return block_count, step_count
