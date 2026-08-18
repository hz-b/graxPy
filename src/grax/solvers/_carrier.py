"""Carrier-extracted (high-frequency) discretization of the boundary integrals.

The classical Nystrom scheme in :mod:`grax.solvers._nystrom` solves for the
physical boundary densities, which carry the incident Floquet phase
``exp(i alpha_0 x)``. At grazing incidence that phase oscillates
``(d / lambda) cos(theta_g)`` times across one period, so the node count has to
track ``d / lambda`` -- 4030 oscillations for a 2.5 um period at 2 keV. This
module solves for the *envelope* instead.

Why the obvious substitution is not enough
------------------------------------------
Writing ``u(x) = exp(i alpha_0 x) u~(x)`` and carrying the phase through the
integral equation gives the reduced kernel

    G~(t, s) = exp(-i alpha_0 (x_t - x_s)) G(r_t, r_s)

and a right-hand side ``exp(-i beta_0 y_t)``, which is genuinely smooth. But on
the *same node set* this is column ``j`` scaled by ``exp(i alpha_0 x_j)`` and row
``i`` by ``exp(-i alpha_0 x_i)``: a diagonal similarity transform of the
assembled system. It leaves the spectrum, the conditioning and the quadrature
error exactly where they were. ``tools/highfreq_study/similarity_check.py``
checks this numerically -- condition numbers and eigenvalues agree to 1e-13 --
and it is the reason this module does something else.

What actually changes
---------------------
The *representation* of the unknown changes. The envelope is expanded in a
truncated Fourier basis carried on the Floquet phase,

    phi(x) = sum_m phi~_m exp(i alpha_m x),   alpha_m = alpha_0 + 2 pi m / d

with ``m`` running over a small band, and the integral equation is imposed in
the Galerkin sense against the same basis. The unknown count becomes ``2 M + 1``
per density per interface and is set by how much the *grating* modulates the
field -- a geometric quantity -- rather than by the wavelength.

The basis is the diffraction-order basis, which is why the flat interface is
exact at ``M = 0``: a flat surface couples nothing, its envelope is constant, and
one unknown per density carries the whole answer at any energy.

What does not change
--------------------
The quadrature. The projection is formed from the classical nodal blocks, so the
node count still has to resolve the *kernel*, whose own Fourier content spans the
propagating orders. This module therefore separates the two bottlenecks rather
than removing both: the unknown count stops tracking ``d / lambda``, the
quadrature does not. See the study in ``tools/highfreq_study`` for the measured
split and what it implies for the next stage.

Scope
-----
Everything physical is inherited from :mod:`grax.solvers.integral`: the same
stack, the same coupled block system, the same tau continuity factors, the same
incident field, the same Rayleigh reconstruction. This module only changes the
basis the system is expressed in, which is what keeps the classical solver
available unchanged as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CarrierBasis",
    "build_carrier_basis",
    "project_system",
    "res2_hf",
]


@dataclass(frozen=True)
class CarrierBasis:
    """Truncated Floquet-Fourier basis for one boundary's envelopes.

    Attributes:
        orders: Envelope harmonics ``m``, shape ``(2 M + 1,)``.
        synthesis: ``B[j, m] = exp(i alpha_m x_j)``, mapping envelope
            coefficients to nodal density values, shape ``(nodes, modes)``.
        analysis: ``P[m, j]``, the discrete ``(1/d) int (.) exp(-i alpha_m x) dx``
            projector, shape ``(modes, nodes)``.
        alpha0: Floquet wavenumber of the incident field.
        period: Grating period in nanometers.
    """

    orders: np.ndarray
    synthesis: np.ndarray
    analysis: np.ndarray
    alpha0: float
    period: float

    @property
    def mode_count(self) -> int:
        """Return the number of envelope harmonics."""

        return int(self.orders.size)

    def orthogonality_defect(self) -> float:
        """Return ``max |P B - I|``, the quadrature's own error on the basis.

        The basis is orthonormal under the exact inner product, because
        ``alpha_m - alpha_m' = 2 pi (m - m') / d`` is a plain periodic harmonic:
        the carrier cancels between analysis and synthesis. Any departure is the
        node grid failing to integrate those harmonics, which is a direct check
        that the quadrature resolves the basis even when it cannot resolve the
        carrier.

        Returns:
            The largest deviation from the identity.
        """

        gram = self.analysis @ self.synthesis
        return float(np.max(np.abs(gram - np.eye(self.mode_count))))


def build_carrier_basis(boundary, *, alpha0: float, modes: int) -> CarrierBasis:
    """Return the envelope basis carried on the Floquet phase for one boundary.

    The measure is ``dx``, not arc length: the envelope is periodic in ``x`` with
    the grating period, and the harmonics that diagonalize a flat interface are
    harmonics in ``x``. On a boundary parametrized by ``t`` that makes the
    quadrature weight ``(dx/dt) (2 pi / N) / d``.

    Args:
        boundary: Nodal boundary carrying ``position`` and ``derivative``.
        alpha0: Floquet wavenumber of the incident field, inverse nanometers.
        modes: Half-width ``M`` of the retained envelope harmonics.

    Returns:
        The basis.

    Raises:
        ValueError: If the basis is not coarser than the node grid, which would
            make the projection a change of coordinates rather than a reduction.
    """

    count = int(boundary.count)
    harmonics = np.arange(-int(modes), int(modes) + 1, dtype=float)
    if harmonics.size >= count:
        raise ValueError(
            f"The envelope basis needs {harmonics.size} modes but the quadrature has "
            f"{count} nodes. The basis has to stay well inside the grid it is "
            "integrated on; raise the node count or lower the mode count."
        )

    period = float(boundary.period)
    x = boundary.position[:, 0]
    alpha = float(alpha0) + 2.0 * np.pi * harmonics / period

    # exp(i alpha_m x_j): envelope coefficients -> nodal physical density.
    synthesis = np.exp(1j * alpha[None, :] * x[:, None])

    # (1/d) int (.) exp(-i alpha_m x) dx, as a quadrature over the t grid.
    weight = boundary.derivative[:, 0] * (2.0 * np.pi / count) / period
    analysis = np.exp(-1j * alpha[:, None] * x[None, :]) * weight[None, :]

    return CarrierBasis(
        orders=harmonics,
        synthesis=synthesis,
        analysis=analysis,
        alpha0=float(alpha0),
        period=period,
    )


def project_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    bases: tuple[CarrierBasis, ...],
    interface_count: int,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nodal boundary system expressed in the envelope basis.

    The nodal system is laid out as ``[phi_0, psi_0, phi_1, psi_1, ...]`` with
    ``nodes`` entries each. Each ``(nodes, nodes)`` sub-block is replaced by
    ``P A B`` and each right-hand side segment by ``P b``, which is the Galerkin
    projection of the same integral equation onto the truncated envelope basis.

    Doing it this way rather than re-deriving the assembly is deliberate: every
    physical ingredient -- the jump terms, the tau continuity factors, the
    cross-interface coupling, the incident field -- is inherited unchanged from
    :func:`grax.solvers.integral._assemble_system`, so the two formulations
    cannot silently drift apart on the physics. Only the basis differs.

    Args:
        matrix: Assembled nodal system.
        rhs: Assembled nodal right-hand side.
        bases: Envelope basis per interface, bottom-up.
        interface_count: Number of interfaces.
        nodes: Collocation nodes per interface.

    Returns:
        The projected matrix and right-hand side.
    """

    blocks = 2 * interface_count
    mode_counts = [bases[index // 2].mode_count for index in range(blocks)]
    offsets = np.cumsum([0, *mode_counts])
    size = int(offsets[-1])

    projected = np.zeros((size, size), dtype=complex)
    projected_rhs = np.zeros(size, dtype=complex)

    for row in range(blocks):
        analysis = bases[row // 2].analysis
        row_slice = slice(int(offsets[row]), int(offsets[row + 1]))
        nodal_rows = slice(row * nodes, (row + 1) * nodes)
        projected_rhs[row_slice] = analysis @ rhs[nodal_rows]
        for column in range(blocks):
            synthesis = bases[column // 2].synthesis
            column_slice = slice(int(offsets[column]), int(offsets[column + 1]))
            nodal_columns = slice(column * nodes, (column + 1) * nodes)
            projected[row_slice, column_slice] = (
                analysis @ matrix[nodal_rows, nodal_columns] @ synthesis
            )

    return projected, projected_rhs


def res2_hf(
    *,
    grating,
    wavelength_nm: float,
    period_nm: float,
    orders: np.ndarray,
    beta0: float,
    polarization: int,
    photon_energy_ev: float,
    n_inc: complex = 1.0 + 0.0j,
    envelope_modes: int = 8,
    quadrature_nodes: int | str = "auto",
    corner_grading: float = 1.0,
    return_envelopes: bool = False,
):
    """Solve one grating with the carrier-extracted formulation.

    Mirrors :func:`grax.solvers.integral.res2_im` argument for argument, so the
    two can be swapped in a comparison without any other change, and reuses its
    stack, assembly and Rayleigh reconstruction verbatim. What differs is that
    the boundary system is projected onto ``2 * envelope_modes + 1`` Floquet
    harmonics per density before it is solved.

    Args:
        grating: Grating providing the profile polyline and the material stack.
        wavelength_nm: Vacuum wavelength in nanometers.
        period_nm: Grating period in nanometers.
        orders: Diffraction orders to report.
        beta0: In-plane incidence direction cosine.
        polarization: ``1`` for TE, ``-1`` for TM.
        photon_energy_ev: Photon energy in electronvolts.
        n_inc: Refractive index of the incident medium.
        envelope_modes: Half-width ``M`` of the retained envelope harmonics.
            This is the unknown count knob, and the quantity that should stop
            tracking ``d / lambda``.
        quadrature_nodes: Nodes carrying the kernel quadrature, or ``"auto"``.
        corner_grading: Node clustering toward profile corners.
        return_envelopes: Also return the solved envelope coefficients, for the
            smoothness diagnostics.

    Returns:
        A :class:`~grax.solvers.common.Res2Result`, or that paired with the
        envelope coefficients when ``return_envelopes`` is set.
    """

    from ._green import PeriodicGreen, default_ewald_splitting
    from .integral import IntegralOptions, _assemble_system, _project_rayleigh, build_stack

    orders = np.asarray(orders)
    order_halfwidth = int(np.max(np.abs(orders)))
    k0 = 2.0 * np.pi / float(wavelength_nm)
    alpha0 = k0 * float(beta0)

    options = IntegralOptions(
        boundary_points=quadrature_nodes,
        discretization="nystrom",
        corner_grading=float(corner_grading),
    )
    stack = build_stack(
        grating,
        photon_energy_ev=float(photon_energy_ev),
        wavelength_nm=float(wavelength_nm),
        n_inc=n_inc,
        orders=order_halfwidth,
        options=options,
    )
    greens = tuple(
        PeriodicGreen(
            period=float(period_nm),
            wavenumber=k0 * index,
            alpha0=alpha0,
            method="ewald",
            splitting=default_ewald_splitting(float(period_nm), k0 * index),
        )
        for index in stack.indices
    )
    matrix, rhs = _assemble_system(
        stack=stack,
        greens=greens,
        polarization=int(polarization),
        alpha0=alpha0,
        k0=k0,
        options=options,
    )

    nodes = stack.interfaces[0].count
    bases = tuple(
        build_carrier_basis(interface, alpha0=alpha0, modes=int(envelope_modes))
        for interface in stack.interfaces
    )
    projected, projected_rhs = project_system(
        matrix,
        rhs,
        bases=bases,
        interface_count=stack.interface_count,
        nodes=nodes,
    )
    solution = np.linalg.solve(projected, projected_rhs)
    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError(
            "The carrier-extracted system produced a non-finite solution; its condition "
            f"number is {np.linalg.cond(projected):.3g}. Lower envelope_modes, or raise "
            "quadrature_nodes so the projection is better resolved."
        )

    # Back to nodal densities, so the existing Rayleigh reconstruction applies
    # unchanged and the two formulations are compared on identical downstream code.
    mode_count = bases[0].mode_count
    envelopes = solution.reshape(stack.interface_count, 2, mode_count)
    densities = np.empty((stack.interface_count, 2, nodes), dtype=complex)
    for interface in range(stack.interface_count):
        for which in range(2):
            densities[interface, which] = (
                bases[interface].synthesis @ envelopes[interface, which]
            )

    reflected, transmitted = _project_rayleigh(
        stack=stack,
        densities=densities,
        orders=orders,
        polarization=int(polarization),
        alpha0=alpha0,
        k0=k0,
        period_nm=float(period_nm),
    )
    from .common import Res2Result

    result = Res2Result(inc_top_reflected=reflected, inc_top_transmitted=transmitted)
    if return_envelopes:
        return result, envelopes
    return result
