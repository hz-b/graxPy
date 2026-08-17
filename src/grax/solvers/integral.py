"""Boundary-integral (Maystre) solver for 1D gratings.

Where the modal solver in :mod:`grax.solvers.rcwa` and the differential-method
solver in :mod:`grax.solvers.neviere` both discretize the *volume* -- expanding a
z-sliced permittivity in a truncated Fourier basis -- the integral method never
leaves the surface. It solves for the field and its normal derivative on the
grating boundary itself and recovers the diffracted orders by projecting the
resulting surface densities onto the Rayleigh basis.

That makes it an independent cross-check of the two Fourier solvers: its error
sources (boundary quadrature, corner resolution) have nothing in common with
theirs (Fourier truncation, z-slicing). Two consequences follow directly:

- ``z_resolution_nm`` and ``x_resolution_nm`` do not enter at all, because the
  exact polyline from ``profile_points()`` is used;
- ``fourier_orders`` no longer controls accuracy. It selects only *how many
  orders are reported*, so the returned order grid matches the other solvers and
  the results stay directly comparable. Accuracy is set by
  ``IntegralOptions.boundary_points``.

Formulation
-----------
With ``phi`` the field on the boundary (``E_y`` in TE, ``H_y`` in TM) and ``psi``
its normal derivative taken from the medium *above* the interface, Green's second
identity applied in each homogeneous region gives, per interface, one equation
from the medium below and one from the medium above. Writing ``S`` and ``D`` for
the single- and double-layer operators of the relevant medium, the two equations
bounding a single interface are

    (1/2) phi - D_above phi + S_above psi         = u_inc
    (1/2) phi + D_below phi - tau S_below psi     = 0

where ``tau`` is ``1`` in TE and ``eps_below / eps_above`` in TM, which is the
only place the two polarizations differ. Multiple conformal interfaces extend
this to one coupled block system; see :func:`_assemble_system`.

Status: not yet wired into ``run_simulation``
--------------------------------------------
This module is not reachable through ``solver=``. Nothing in
``grax.simulation.core``, ``grax.solvers.__init__`` or ``grax.__init__`` refers
to it, so it is inert for users and costs nothing at import time. What remains
before wiring it in is runtime, not correctness.

Discretization
--------------
Two schemes are available through ``IntegralOptions.discretization``, and they
produce the same two operators with the same meaning, so they stay directly
comparable on identical geometry.

``"panel"``
    Flat panels with piecewise-constant densities. Converges as ``O(h^2)``.
    Parametrized by arc length, so it is the only one of the two that handles a
    profile which is not a graph -- a laminar grating with exactly vertical
    sidewalls, for instance.

``"nystrom"``
    Trigonometric Nystrom with Martensen-Kussmaul product quadrature, graded at
    corners. Converges at fourth order or better. Parametrized by ``x``, so it
    requires a single-valued profile.

The difference is large and was measured rather than assumed. Against RCWA on a
shallow sinusoid, the unknown count needed for 1e-4 agreement::

    d/lambda   panel   nystrom   reduction
    25         3672    128       29x
    50         6266    256       24x
    100        ~12000  384       31x

On a flat interface, where the answer is analytic, 256 nodes give 5.9e-8
relative for Nystrom against 8.7e-2 for panels, and Nystrom is also faster at
equal N because it needs neither a quadrature multiplier nor near-panel
refinement.

Why the gap is that large is worth recording, because it is the whole reason the
second scheme exists. The boundary densities are strongly band-limited: at
``d/lambda = 50`` the field density carries 8 significant harmonics and its
normal derivative 31, while the panel scheme needed 6266 unknowns. Panels were
being spent resolving the oscillatory *kernel* rather than the unknowns, because
a collocation scheme uses one grid for both. As ``d/lambda`` grows at fixed
depth the density envelope gets *smoother* while the kernel oscillates *faster*;
a single-grid scheme is dragged by the faster of the two, and a Nystrom scheme is
not.

What is left
------------
Runtime. At ``d/lambda = 100`` a converged solve is still seconds to minutes per
energy point, against a target of about one second. The identified work is
engineering rather than method development: the Ewald splitting default is a
measured factor of two off its optimum, the kernel evaluation is numpy-bound at
roughly 0.3 microseconds per term where a compiled kernel should reach tens of
nanoseconds, and the conformal-interface structure means only one distinct block
per vertical offset needs building rather than one per interface pair.

Scope
-----
Classical mounting only, matching the rest of the package. Debye-Waller
roughness is supported through the shared damping factor; stochastic
``random-interface`` roughness is not. The boundary count is capped by
``IntegralOptions.max_boundaries``: the system is dense and grows as the square
of the total panel count, so a 60-bilayer multilayer is out of reach here and
raises rather than grinding.

References:
    Maystre, in Petit (ed.), *Electromagnetic Theory of Gratings* (Springer,
    1980), pp. 63-100. Maystre & Popov, *Gratings: Theory and Numeric
    Applications*, 2nd ed., ch. 4 (2014). Popov, Bozhkov, Maystre & Hoose,
    *Appl. Opt.* **38**, 47 (1999) -- layered echelles. *J. Opt. Soc. Am. A*
    **4**, 834 (1987) -- differential/integral comparison in TM.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..materials import resolve_refractive_index
from ..simulation._profiling import SolverProfiler
from ._boundary import BoundaryPanels, build_panels
from ._green import PeriodicGreen, default_ewald_splitting
from ._nystrom import (
    TrigBoundary,
    build_graded_boundary,
    build_trig_boundary,
    has_corners,
    nystrom_operators,
)
from ._operators import layer_operators
from .common import (
    DiffractionResult,
    Res2Result,
    apply_optional_roughness,
    safe_linalg_solve,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IntegralOptions",
    "coerce_integral_options",
    "res2_im",
]

GreenMethod = Literal["ewald", "spectral"]
Discretization = Literal["panel", "nystrom"]


@dataclass(frozen=True)
class IntegralOptions:
    """Discretization settings for the boundary-integral solver.

    Attributes:
        boundary_points: Collocation panels per interface, or ``"auto"`` to size
            them from the period-to-wavelength ratio and the reported order
            count. This is the accuracy knob: unlike the Fourier solvers, whose
            convergence is governed by ``fourier_orders``, the integral method
            converges as the boundary discretization is refined.
        corner_grading: Power-law clustering of panels toward the corners of the
            profile polyline, where the surface densities are singular. ``1.0``
            spaces panels uniformly along each facet; the default clusters
            moderately. Laminar profiles with near-vertical walls benefit most.
        quadrature_order: Gauss-Legendre nodes per panel. Panels close to a
            collocation point are additionally bisected until resolved, so this
            controls the smooth part of the integrand only.
        green_function: ``"ewald"`` (default) evaluates the quasi-periodic Green
            function by Ewald summation, which converges exponentially even for
            two points on the same interface. ``"spectral"`` uses the plain
            plane-wave series; it is exact for well-separated interfaces but
            stalls as the vertical separation goes to zero, so it exists for
            cross-checking rather than production.
        ewald_splitting: Splitting parameter of the Ewald sum, in inverse
            nanometers. The exact value cancels analytically, so this changes
            cost and conditioning but not the converged answer. ``None`` picks a
            default that keeps both halves convergent.
        max_boundaries: Refuse geometries with more interfaces than this. The
            coupled system is dense in the *total* panel count, so cost grows as
            the cube of the boundary count.
        energy_balance_tolerance: When set, the summed propagating reflected and
            transmitted efficiency must not exceed this value or the solve
            raises. Leave ``None`` to rely on the checks in
            :func:`grax.run_simulation`. Absorbing materials make the balance
            legitimately smaller than one, so only an upper bound is enforced.
    """

    boundary_points: int | Literal["auto"] = "auto"
    discretization: Discretization = "panel"
    corner_grading: float = 2.0
    quadrature_order: int = 8
    green_function: GreenMethod = "ewald"
    ewald_splitting: float | None = None
    max_boundaries: int = 8
    energy_balance_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If any setting is outside its supported range.
        """

        if self.boundary_points != "auto":
            if not isinstance(self.boundary_points, (int, np.integer)):
                raise ValueError("boundary_points must be an integer or 'auto'.")
            if int(self.boundary_points) < 8:
                raise ValueError("boundary_points must be at least 8 when given.")
        if self.discretization not in ("panel", "nystrom"):
            raise ValueError("discretization must be 'panel' or 'nystrom'.")
        if self.corner_grading < 1.0:
            raise ValueError("corner_grading must be >= 1.")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2.")
        if self.green_function not in ("ewald", "spectral"):
            raise ValueError("green_function must be 'ewald' or 'spectral'.")
        if self.ewald_splitting is not None and self.ewald_splitting <= 0.0:
            raise ValueError("ewald_splitting must be > 0 when provided.")
        if self.max_boundaries < 1:
            raise ValueError("max_boundaries must be >= 1.")
        if self.energy_balance_tolerance is not None and self.energy_balance_tolerance <= 0.0:
            raise ValueError("energy_balance_tolerance must be > 0 when provided.")

    def to_dict(self) -> dict[str, object]:
        """Return the settings as a plain mapping for checkpoints."""

        return {
            "boundary_points": self.boundary_points,
            "discretization": self.discretization,
            "corner_grading": self.corner_grading,
            "quadrature_order": self.quadrature_order,
            "green_function": self.green_function,
            "ewald_splitting": self.ewald_splitting,
            "max_boundaries": self.max_boundaries,
            "energy_balance_tolerance": self.energy_balance_tolerance,
        }

    def resolved_boundary_points(
        self, *, period_nm: float, wavelength_nm: float, orders: int
    ) -> int:
        """Return the panel count to use for one solve.

        The ``"auto"`` heuristic asks for enough panels to resolve both the
        reported orders and the surface field itself. The surface field carries
        the incident phase ``exp(i alpha_0 x)``, which oscillates roughly
        ``period / wavelength`` times across one period, so that ratio sets the
        floor.

        Args:
            period_nm: Grating period in nanometers.
            wavelength_nm: Vacuum wavelength in nanometers.
            orders: Half-width of the reported order range.

        Returns:
            Panel count per interface.
        """

        if self.boundary_points != "auto":
            return int(self.boundary_points)
        per_wavelength = 6.0 * period_nm / max(wavelength_nm, 1e-12)
        per_order = 8.0 * (2 * orders + 1)
        return int(max(64, min(per_wavelength, per_order)))


def coerce_integral_options(
    options: IntegralOptions | dict[str, object] | None,
) -> IntegralOptions:
    """Return an :class:`IntegralOptions` from an options object, mapping, or ``None``.

    Args:
        options: Options instance, keyword mapping, or ``None`` for defaults.

    Returns:
        Validated options instance.

    Raises:
        TypeError: If the argument is none of the accepted forms.
    """

    if options is None:
        return IntegralOptions()
    if isinstance(options, IntegralOptions):
        return options
    if isinstance(options, dict):
        return IntegralOptions(**options)  # type: ignore[arg-type]
    raise TypeError("solver_options must be an IntegralOptions, a mapping, or None.")


@dataclass(frozen=True)
class _Stack:
    """Resolved geometry and materials for one integral-method solve.

    Attributes:
        interfaces: Panel discretizations, bottom-up. Interface ``j`` separates
            medium ``j`` (below) from medium ``j + 1`` (above).
        indices: Refractive indices of the media, bottom-up, one longer than
            ``interfaces``.
        offsets: Vertical offset of each interface above the profile datum.
    """

    interfaces: tuple[BoundaryPanels, ...]
    indices: tuple[complex, ...]
    offsets: tuple[float, ...]

    @property
    def interface_count(self) -> int:
        """Return the number of interfaces."""

        return len(self.interfaces)


def build_stack(
    grating: Any,
    *,
    photon_energy_ev: float,
    wavelength_nm: float,
    n_inc: complex,
    orders: int,
    options: IntegralOptions,
) -> _Stack:
    """Return the interface stack for one grating at one photon energy.

    Coatings are conformal in :mod:`grax.gratings`: every interface is the
    substrate profile displaced by the cumulative thickness beneath it. The
    panels are therefore built once and shifted, which also makes every interface
    share the same tangents, normals and lengths.

    Args:
        grating: Grating providing the profile and the material stack.
        photon_energy_ev: Photon energy in electronvolts.
        wavelength_nm: Vacuum wavelength in nanometers.
        n_inc: Refractive index of the incident medium.
        orders: Half-width of the reported order range.
        options: Discretization settings.

    Returns:
        The resolved stack.

    Raises:
        ValueError: If the geometry has more interfaces than ``max_boundaries``.
    """

    coating_stack = grating.resolved_stack()
    # A zero-thickness layer would put two interfaces at the same height, making
    # the coupled system singular. The Fourier solvers absorb such a layer into
    # its neighbour through their z-slicing; here it simply has no boundary.
    layers = [
        (material, float(thickness))
        for material, thickness in coating_stack.layer_sequence_bottom_up()
        if float(thickness) > 0.0
    ]
    interface_count = len(layers) + 1
    if interface_count > options.max_boundaries:
        raise ValueError(
            f"solver='integral' supports at most {options.max_boundaries} interfaces, but "
            f"{type(grating).__name__} with {type(coating_stack).__name__} has "
            f"{interface_count} ({len(layers)} coating layers plus the substrate). The "
            "coupled boundary system is dense in the total panel count, so this geometry "
            "is out of reach for the integral solver. Use solver='rcwa' or "
            "solver='neviere' for it, or raise IntegralOptions(max_boundaries=...) if you "
            "are prepared for the cost."
        )

    positions, heights = grating.profile_points()
    panel_count = options.resolved_boundary_points(
        period_nm=float(grating.period_nm),
        wavelength_nm=float(wavelength_nm),
        orders=int(orders),
    )
    if options.discretization == "nystrom":
        # The Kress weights need an even node count.
        even_count = panel_count + (panel_count % 2)
        if has_corners(positions, heights):
            # Corners break the analyticity the trigonometric grid relies on, so
            # the parametrization is graded to vanish at each of them and the
            # geometry is read straight off the polyline. Spectral
            # differentiation would ring on a piecewise-linear profile.
            base = build_graded_boundary(
                positions,
                heights,
                period=float(grating.period_nm),
                count=even_count,
                grading=float(options.corner_grading),
            )
        else:
            base = build_trig_boundary(
                positions,
                heights,
                period=float(grating.period_nm),
                count=even_count,
            )
    else:
        base = build_panels(
            positions,
            heights,
            period=float(grating.period_nm),
            panel_count=panel_count,
            corner_grading=float(options.corner_grading),
        )

    indices: list[complex] = [
        complex(resolve_refractive_index(coating_stack.substrate_material, photon_energy_ev))
    ]
    offsets: list[float] = [0.0]
    cumulative = 0.0
    for material, thickness in layers:
        indices.append(complex(resolve_refractive_index(material, photon_energy_ev)))
        cumulative += float(thickness)
        offsets.append(cumulative)
    indices.append(complex(n_inc))

    interfaces = tuple(base.shifted(offset) for offset in offsets)
    return _Stack(interfaces=interfaces, indices=tuple(indices), offsets=tuple(offsets))


def res2_im(
    *,
    grating: Any,
    wavelength_nm: float,
    period_nm: float,
    orders: np.ndarray,
    beta0: float,
    polarization: int,
    photon_energy_ev: float,
    n_inc: complex = 1.0 + 0.0j,
    options: IntegralOptions | dict[str, object] | None = None,
    roughness_sigma_nm: float | None = None,
    _profiler: SolverProfiler | None = None,
) -> Res2Result:
    """Solve one grating with the boundary-integral method.

    Args:
        grating: Grating providing the profile polyline and the material stack.
        wavelength_nm: Vacuum wavelength in nanometers.
        period_nm: Grating period in nanometers.
        orders: Diffraction orders to report, as produced by ``res1``.
        beta0: In-plane incidence direction cosine, matching the other solvers.
        polarization: ``1`` for TE, ``-1`` for TM.
        photon_energy_ev: Photon energy in electronvolts, used for the optical
            constants.
        n_inc: Refractive index of the incident medium.
        options: Discretization settings.
        roughness_sigma_nm: Optional rms roughness for the shared Debye-Waller
            damping.
        _profiler: Optional solver profiler.

    Returns:
        Reflected and transmitted diffraction results for incidence from top.

    Raises:
        ValueError: If the geometry or the settings are unsupported, or if an
            energy-balance tolerance is set and violated.
    """

    resolved = coerce_integral_options(options)
    if roughness_sigma_nm is not None and roughness_sigma_nm < 0.0:
        raise ValueError("roughness_sigma_nm must be >= 0 when provided.")

    orders = np.asarray(orders)
    order_halfwidth = int(np.max(np.abs(orders)))
    k0 = 2.0 * np.pi / float(wavelength_nm)
    alpha0 = k0 * float(beta0)

    with _profiler.record("integral_geometry") if _profiler is not None else _nullcontext():
        stack = build_stack(
            grating,
            photon_energy_ev=float(photon_energy_ev),
            wavelength_nm=float(wavelength_nm),
            n_inc=n_inc,
            orders=order_halfwidth,
            options=resolved,
        )

    greens = tuple(
        PeriodicGreen(
            period=float(period_nm),
            wavenumber=k0 * index,
            alpha0=alpha0,
            method=resolved.green_function,
            splitting=(
                resolved.ewald_splitting
                if resolved.ewald_splitting is not None
                else default_ewald_splitting(float(period_nm), k0 * index)
            ),
        )
        for index in stack.indices
    )

    with _profiler.record("integral_assembly") if _profiler is not None else _nullcontext():
        matrix, rhs = _assemble_system(
            stack=stack,
            greens=greens,
            polarization=int(polarization),
            alpha0=alpha0,
            k0=k0,
            options=resolved,
        )

    with _profiler.record("matrix_solves") if _profiler is not None else _nullcontext():
        solution = safe_linalg_solve(matrix, rhs, context="integral method boundary system")

    panels_per_interface = stack.interfaces[0].count
    densities = solution.reshape(stack.interface_count, 2, panels_per_interface)

    reflected, transmitted = _project_rayleigh(
        stack=stack,
        densities=densities,
        orders=orders,
        polarization=int(polarization),
        alpha0=alpha0,
        k0=k0,
        period_nm=float(period_nm),
    )

    reflected, transmitted = apply_optional_roughness(
        reflected,
        transmitted,
        aa=_RoughnessContext(wavelength=float(wavelength_nm), beta0=float(beta0)),
        roughness_sigma_nm=roughness_sigma_nm,
    )

    if resolved.energy_balance_tolerance is not None:
        total = float(np.sum(np.real(reflected.efficiency))) + float(
            np.sum(np.real(transmitted.efficiency))
        )
        if total > resolved.energy_balance_tolerance:
            raise ValueError(
                "The integral-method solve violated its energy-balance tolerance: "
                f"total propagating efficiency {total:.6g} exceeds "
                f"{resolved.energy_balance_tolerance:.6g}. Raise boundary_points or "
                "corner_grading, or relax the tolerance."
            )

    return Res2Result(inc_top_reflected=reflected, inc_top_transmitted=transmitted)


@dataclass(frozen=True)
class _RoughnessContext:
    """Minimal stand-in carrying what the shared roughness helper reads.

    ``apply_optional_roughness`` only needs the wavelength and the incidence
    direction cosine, both of which the integral solver has without ever building
    a :class:`~grax.solvers.common.Res1Result`.

    Attributes:
        wavelength: Vacuum wavelength in nanometers.
        beta0: In-plane incidence direction cosine.
    """

    wavelength: float
    beta0: float


def _tau(polarization: int, index_below: complex, index_above: complex) -> complex:
    """Return the normal-derivative continuity factor across one interface.

    In TE the normal derivative itself is continuous. In TM it is
    ``(1/eps) du/dn`` that is continuous, so the derivative taken from below is
    the one from above scaled by ``eps_below / eps_above``.

    Args:
        polarization: ``1`` for TE, ``-1`` for TM.
        index_below: Refractive index below the interface.
        index_above: Refractive index above the interface.

    Returns:
        The scaling factor.
    """

    if polarization == 1:
        return 1.0 + 0.0j
    return (index_below**2) / (index_above**2)


def _assemble_system(
    *,
    stack: _Stack,
    greens: tuple[PeriodicGreen, ...],
    polarization: int,
    alpha0: float,
    k0: float,
    options: IntegralOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the coupled boundary system and its right-hand side.

    Unknowns are ordered interface by interface as ``[phi_0, psi_0, phi_1, ...]``
    with ``psi`` taken from the medium above each interface. Every medium
    contributes one block row per interface it touches: the medium above an
    interface sees it from the ``+n`` side, the medium below from the ``-n``
    side, which is where the two ``1/2`` jump terms come from with opposite
    signs.

    Args:
        stack: Resolved interfaces and media.
        greens: Green function per medium, bottom-up.
        polarization: ``1`` for TE, ``-1`` for TM.
        alpha0: In-plane wavenumber of the incident field.
        k0: Vacuum wavenumber.
        options: Discretization settings.

    Returns:
        The dense system matrix and right-hand side.
    """

    interface_count = stack.interface_count
    panels = stack.interfaces[0].count
    size = 2 * interface_count * panels
    matrix = np.zeros((size, size), dtype=complex)
    rhs = np.zeros(size, dtype=complex)
    identity = np.eye(panels, dtype=complex)

    def block(row_interface: int, row_half: int) -> slice:
        start = (2 * row_interface + row_half) * panels
        return slice(start, start + panels)

    def unknown(interface: int, which: int) -> slice:
        start = (2 * interface + which) * panels
        return slice(start, start + panels)

    # Row bookkeeping: interface j gets row half 0 from the medium above and row
    # half 1 from the medium below.
    for medium in range(interface_count + 1):
        lower = medium - 1 if medium >= 1 else None
        upper = medium if medium <= interface_count - 1 else None
        green = greens[medium]

        pieces = [index for index in (lower, upper) if index is not None]
        operators: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for observed in pieces:
            for sourced in pieces:
                operators[(observed, sourced)] = _interface_operators(
                    green,
                    target=stack.interfaces[observed],
                    source=stack.interfaces[sourced],
                    same_interface=observed == sourced,
                    options=options,
                )

        if lower is not None:
            # Observed on the lower interface, approached from above.
            row = block(lower, 0)
            single, double = operators[(lower, lower)]
            matrix[row, unknown(lower, 0)] += 0.5 * identity - double
            matrix[row, unknown(lower, 1)] += single
            if upper is not None:
                cross_single, cross_double = operators[(lower, upper)]
                scale = _tau(polarization, stack.indices[upper], stack.indices[upper + 1])
                matrix[row, unknown(upper, 0)] += cross_double
                matrix[row, unknown(upper, 1)] += -scale * cross_single
            else:
                rhs[row] = _incident_field(
                    stack.interfaces[lower], alpha0=alpha0, k0=k0, green=green
                )

        if upper is not None:
            # Observed on the upper interface, approached from below.
            row = block(upper, 1)
            single, double = operators[(upper, upper)]
            scale = _tau(polarization, stack.indices[upper], stack.indices[upper + 1])
            matrix[row, unknown(upper, 0)] += 0.5 * identity + double
            matrix[row, unknown(upper, 1)] += -scale * single
            if lower is not None:
                cross_single, cross_double = operators[(upper, lower)]
                matrix[row, unknown(lower, 0)] += -cross_double
                matrix[row, unknown(lower, 1)] += cross_single

    return matrix, rhs


def _interface_operators(
    green: PeriodicGreen,
    *,
    target: BoundaryPanels | TrigBoundary,
    source: BoundaryPanels | TrigBoundary,
    same_interface: bool,
    options: IntegralOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single- and double-layer blocks for one pair of interfaces.

    Dispatches on the requested discretization. Both schemes produce the same two
    matrices with the same meaning, which is what lets the coupled system and the
    Rayleigh projection above stay unchanged between them -- and what makes the
    two directly comparable on identical geometry.

    Args:
        green: Green function of the medium bounded by these interfaces.
        target: Interface carrying the collocation points.
        source: Interface carrying the densities.
        same_interface: Whether the two are the same interface.
        options: Discretization settings.

    Returns:
        ``(S, D)``.
    """

    if options.discretization == "nystrom":
        return nystrom_operators(
            green,
            target=target,
            source=source,
            same_boundary=same_interface,
        )
    return layer_operators(
        green,
        target=target,
        source=source,
        same_interface=same_interface,
        quadrature_order=options.quadrature_order,
    )


def _incident_field(
    panels: BoundaryPanels,
    *,
    alpha0: float,
    k0: float,
    green: PeriodicGreen,
) -> np.ndarray:
    """Return the incident plane wave sampled at the collocation points.

    The wave travels downward, ``exp(i alpha_0 x - i beta_0 y)``, with unit
    amplitude at ``y = 0`` so that the reported amplitudes share the phase
    reference of the profile datum.

    Args:
        panels: Interface carrying the collocation points.
        alpha0: In-plane wavenumber.
        k0: Vacuum wavenumber.
        green: Green function of the incident medium, used for ``beta_0``.

    Returns:
        Complex array of length ``panels.count``.
    """

    beta0 = complex(green.beta(np.asarray([0.0]))[0])
    x = panels.collocation[:, 0]
    y = panels.collocation[:, 1]
    return np.exp(1j * alpha0 * x - 1j * beta0 * y)


def _project_rayleigh(
    *,
    stack: _Stack,
    densities: np.ndarray,
    orders: np.ndarray,
    polarization: int,
    alpha0: float,
    k0: float,
    period_nm: float,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Return the reflected and transmitted diffraction results.

    Above the top interface the scattered field is a Rayleigh series, and its
    coefficients follow from the same representation used to set up the system,
    with the Green function replaced by its plane-wave form. The same holds
    below the bottom interface for the transmitted field.

    Args:
        stack: Resolved interfaces and media.
        densities: Surface densities shaped ``(interfaces, 2, panels)``.
        orders: Reported diffraction orders.
        polarization: ``1`` for TE, ``-1`` for TM.
        alpha0: In-plane wavenumber of the incident field.
        k0: Vacuum wavenumber.
        period_nm: Grating period in nanometers.

    Returns:
        Reflected and transmitted diffraction results.
    """

    orders_float = np.asarray(orders, dtype=float)
    alpha = alpha0 + 2.0 * np.pi * orders_float / period_nm

    top_interface = stack.interfaces[-1]
    bottom_interface = stack.interfaces[0]
    n_top = stack.indices[-1]
    n_bottom = stack.indices[0]

    beta_top = _beta(k0 * n_top, alpha)
    beta_bottom = _beta(k0 * n_bottom, alpha)

    phi_top = densities[-1, 0]
    psi_top = densities[-1, 1]
    phi_bottom = densities[0, 0]
    psi_bottom = densities[0, 1]

    reflected_amplitude = _rayleigh_amplitudes(
        panels=top_interface,
        phi=phi_top,
        psi=psi_top,
        alpha=alpha,
        beta=beta_top,
        period_nm=period_nm,
        upward=True,
    )
    tau_bottom = _tau(polarization, stack.indices[0], stack.indices[1])
    transmitted_amplitude = _rayleigh_amplitudes(
        panels=bottom_interface,
        phi=phi_bottom,
        psi=tau_bottom * psi_bottom,
        alpha=alpha,
        beta=beta_bottom,
        period_nm=period_nm,
        upward=False,
    )

    zero_index = int(np.where(orders_float == 0.0)[0][0])
    incident_beta = beta_top[zero_index]
    if polarization == 1:
        reflected_efficiency = np.real(beta_top / incident_beta) * np.abs(reflected_amplitude) ** 2
        transmitted_efficiency = (
            np.real(beta_bottom / incident_beta) * np.abs(transmitted_amplitude) ** 2
        )
    else:
        incident_admittance = np.real(incident_beta / n_top**2)
        reflected_efficiency = (
            np.real(beta_top / n_top**2) / incident_admittance * np.abs(reflected_amplitude) ** 2
        )
        transmitted_efficiency = (
            np.real(beta_bottom / n_bottom**2)
            / incident_admittance
            * np.abs(transmitted_amplitude) ** 2
        )

    return (
        DiffractionResult(
            order=np.asarray(orders).copy(),
            theta=_angles_from_alpha(alpha, k0, n_top),
            efficiency=reflected_efficiency,
            amplitude=reflected_amplitude,
        ),
        DiffractionResult(
            order=np.asarray(orders).copy(),
            theta=_angles_from_alpha(alpha, k0, n_bottom),
            efficiency=transmitted_efficiency,
            amplitude=transmitted_amplitude,
        ),
    )


def _rayleigh_amplitudes(
    *,
    panels: BoundaryPanels,
    phi: np.ndarray,
    psi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    period_nm: float,
    upward: bool,
) -> np.ndarray:
    """Return the Rayleigh coefficients of the field radiated by one interface.

    Args:
        panels: The radiating interface.
        phi: Field density on the interface.
        psi: Normal-derivative density on the interface, taken in the medium the
            field is being evaluated in.
        alpha: In-plane wavenumbers of the reported orders.
        beta: Out-of-plane wavenumbers in the receiving medium.
        period_nm: Grating period in nanometers.
        upward: Whether the field radiates upward (reflected) or downward
            (transmitted), which flips the sign of ``beta`` in the kernel.

    Returns:
        Complex amplitudes, one per reported order.
    """

    sign = -1.0 if upward else 1.0
    x = panels.collocation[:, 0]
    y = panels.collocation[:, 1]
    normal_x = panels.normal[:, 0]
    normal_y = panels.normal[:, 1]
    weight = panels.weight

    phase = np.exp(
        -1j * alpha[:, None] * x[None, :] + 1j * sign * beta[:, None] * y[None, :]
    )
    normal_derivative = (
        -1j * alpha[:, None] * normal_x[None, :]
        + 1j * sign * beta[:, None] * normal_y[None, :]
    )

    if upward:
        integrand = phi[None, :] * normal_derivative - psi[None, :]
    else:
        integrand = psi[None, :] - phi[None, :] * normal_derivative
    contribution = np.sum(integrand * phase * weight[None, :], axis=1)
    return 0.5j / period_nm * contribution / beta


def _beta(wavenumber: complex, alpha: np.ndarray) -> np.ndarray:
    """Return out-of-plane wavenumbers on the ``Im beta >= 0`` branch."""

    beta = np.sqrt(complex(wavenumber) ** 2 - np.asarray(alpha, dtype=float) ** 2 + 0j)
    flip = (np.imag(beta) < 0) | ((np.abs(np.imag(beta)) < 1e-15) & (np.real(beta) < 0))
    return np.where(flip, -beta, beta)


def _angles_from_alpha(alpha: np.ndarray, k0: float, refractive_index: complex) -> np.ndarray:
    """Return diffraction angles in degrees, matching the other solvers."""

    n_for_angles = float(np.real(refractive_index))
    ratio = np.clip(np.real(alpha / (k0 * n_for_angles)), -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))
