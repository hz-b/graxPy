"""Specular reflectivity of planar mirrors and multilayers (Parratt recursion).

This module is a fast, closed-form alternative to the RCWA solvers for
surfaces without a grating profile. It supports arbitrary layer stacks built
from :mod:`grax.stacks`, per-interface Névot–Croce roughness, ``s`` and ``p``
polarization, and lateral grading of the layer thicknesses along the beam
footprint.

References:
    L. G. Parratt, Phys. Rev. 95, 359 (1954).
    L. Névot and P. Croce, Rev. Phys. Appl. 15, 761 (1980).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from .materials import resolve_refractive_index
from .stacks import BaseStack

HC_EV_NM = 1239.8419843320026
"""Planck constant times speed of light in eV*nm."""

PlanarPolarization = Literal["s", "p", "average"]

_POLARIZATION_ALIASES = {
    "s": "s",
    "te": "s",
    "p": "p",
    "tm": "p",
    "average": "average",
    "avg": "average",
    "unpolarized": "average",
}


def _normalize_planar_polarization(value: str) -> str:
    """Return ``"s"``, ``"p"`` or ``"average"`` for one polarization name."""

    normalized = _POLARIZATION_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported polarization {value!r}; use 's', 'p', 'TE', 'TM' or 'average'."
        )
    return normalized


def _physical_sqrt(values: np.ndarray) -> np.ndarray:
    """Square root on the branch with non-negative imaginary part."""

    root = np.sqrt(np.asarray(values, dtype=complex))
    flip = (root.imag < 0.0) | ((root.imag == 0.0) & (root.real < 0.0))
    return np.where(flip, -root, root)


def _refractive_indices(
    materials: list[Any], energies_ev: np.ndarray
) -> list[np.ndarray]:
    """Return one complex-index array over energy per material, caching repeats."""

    cache: dict[int, np.ndarray] = {}
    result: list[np.ndarray] = []
    for material in materials:
        key = id(material)
        if key not in cache:
            cache[key] = np.array(
                [resolve_refractive_index(material, float(energy)) for energy in energies_ev],
                dtype=complex,
            )
        result.append(cache[key])
    return result


def _parratt_single(
    polarization: str,
    cos_theta_sq: np.ndarray,
    k0_nm: np.ndarray,
    indices_top_down: list[np.ndarray],
    thicknesses_top_down: list[Any],
    sigmas_top_down: list[float],
) -> np.ndarray:
    """Run the Parratt recursion for one polarization.

    Media are ordered top-down: vacuum, layers, substrate. ``indices`` entries
    have shape ``(E, 1, 1)``, ``cos_theta_sq`` has shape ``(1, T, 1)``, and
    layer thicknesses broadcast against ``(E, T, P)``. ``sigmas_top_down``
    holds the roughness of the interface below each medium except the
    substrate.
    """

    n_media = len(indices_top_down)
    kz = [_physical_sqrt(n**2 - cos_theta_sq) for n in indices_top_down]
    eps = [n**2 for n in indices_top_down]

    def interface(i: int) -> np.ndarray:
        kz_a, kz_b = kz[i], kz[i + 1]
        if polarization == "s":
            r = (kz_a - kz_b) / (kz_a + kz_b)
        else:
            r = (eps[i + 1] * kz_a - eps[i] * kz_b) / (eps[i + 1] * kz_a + eps[i] * kz_b)
        sigma = sigmas_top_down[i]
        if sigma > 0.0:
            r = r * np.exp(-2.0 * (k0_nm**2) * kz_a * kz_b * sigma**2)
        return r

    cumulative = interface(n_media - 2)
    for i in range(n_media - 3, -1, -1):
        phase = np.exp(2j * k0_nm * kz[i + 1] * thicknesses_top_down[i + 1])
        r_i = interface(i)
        cumulative = (r_i + cumulative * phase) / (1.0 + r_i * cumulative * phase)
    return np.abs(cumulative) ** 2


def parratt_reflectivity(
    stack: BaseStack,
    energies_ev: Any,
    grazing_angles_deg: Any,
    *,
    polarization: str = "s",
    roughness_sigma_nm: float = 0.0,
    thickness_scale: Any = None,
) -> np.ndarray:
    """Return the specular reflectivity of a layer stack.

    Args:
        stack: Coating stack above the substrate. Any :class:`~grax.stacks.BaseStack`
            works (single layer, multilayer, custom). Per-layer and substrate
            roughness overrides on the stack are honoured.
        energies_ev: Photon energies in eV, scalar or 1-D.
        grazing_angles_deg: Grazing angles of incidence in degrees (measured
            from the surface), scalar or 1-D.
        polarization: ``"s"``/``"TE"``, ``"p"``/``"TM"`` or ``"average"``
            (incoherent mean of ``s`` and ``p``).
        roughness_sigma_nm: Default rms roughness in nm applied to every
            interface without its own override (Névot–Croce).
        thickness_scale: Optional 1-D array of dimensionless factors applied
            to every layer thickness, for example the local value of a laterally
            graded ``d``. Adds a trailing axis to the result.

    Returns:
        Reflectivity (intensity) with shape ``(n_energies, n_angles)``, or
        ``(n_energies, n_angles, n_scale)`` when ``thickness_scale`` is given.

    Raises:
        ValueError: On non-positive energies, angles outside ``(0, 90]``,
            negative roughness, non-positive scale factors, or an unknown
            polarization.
    """

    mode = _normalize_planar_polarization(polarization)
    energies = np.atleast_1d(np.asarray(energies_ev, dtype=float))
    angles = np.atleast_1d(np.asarray(grazing_angles_deg, dtype=float))
    if energies.ndim != 1 or angles.ndim != 1:
        raise ValueError("energies_ev and grazing_angles_deg must be scalar or 1-D.")
    if energies.size == 0 or angles.size == 0:
        raise ValueError("energies_ev and grazing_angles_deg must not be empty.")
    if np.any(energies <= 0.0):
        raise ValueError("energies_ev must be > 0.")
    if np.any(angles <= 0.0) or np.any(angles > 90.0):
        raise ValueError("grazing_angles_deg must be in (0, 90].")
    if float(roughness_sigma_nm) < 0.0:
        raise ValueError("roughness_sigma_nm must be >= 0.")

    scale: np.ndarray | None = None
    if thickness_scale is not None:
        scale = np.atleast_1d(np.asarray(thickness_scale, dtype=float))
        if scale.ndim != 1 or scale.size == 0:
            raise ValueError("thickness_scale must be a non-empty 1-D array.")
        if np.any(scale <= 0.0):
            raise ValueError("thickness_scale must be > 0.")

    layers_bottom_up = stack.layer_sequence_bottom_up()
    sigmas_bottom_up = stack.interface_roughness_sigmas_bottom_up(float(roughness_sigma_nm))
    if any(sigma < 0.0 for sigma in sigmas_bottom_up):
        raise ValueError("Roughness sigmas must be >= 0.")

    materials = [material for material, _ in layers_bottom_up] + [stack.substrate_material]
    indices = _refractive_indices(materials, energies)
    layer_indices_top_down = indices[: len(layers_bottom_up)][::-1]
    substrate_index = indices[-1]

    vacuum = np.ones((energies.size, 1, 1), dtype=complex)
    indices_top_down = [vacuum] + [
        index.reshape(-1, 1, 1) for index in layer_indices_top_down
    ] + [substrate_index.reshape(-1, 1, 1)]

    scale_axis = np.ones(1) if scale is None else scale
    scale_axis = scale_axis.reshape(1, 1, -1)
    thicknesses_top_down: list[Any] = [0.0]
    for _, thickness_nm in reversed(layers_bottom_up):
        thicknesses_top_down.append(float(thickness_nm) * scale_axis)
    thicknesses_top_down.append(0.0)

    # sigmas_bottom_up[0] is the substrate interface; sigmas_bottom_up[j + 1] is
    # the top of layer j. Top-down, interface i lies below medium i.
    sigmas_top_down = list(reversed(sigmas_bottom_up))

    cos_theta_sq = (np.cos(np.deg2rad(angles)) ** 2).reshape(1, -1, 1)
    k0_nm = (2.0 * np.pi * energies / HC_EV_NM).reshape(-1, 1, 1)

    if mode == "average":
        reflectivity = 0.5 * (
            _parratt_single("s", cos_theta_sq, k0_nm, indices_top_down, thicknesses_top_down, sigmas_top_down)
            + _parratt_single("p", cos_theta_sq, k0_nm, indices_top_down, thicknesses_top_down, sigmas_top_down)
        )
    else:
        reflectivity = _parratt_single(
            mode, cos_theta_sq, k0_nm, indices_top_down, thicknesses_top_down, sigmas_top_down
        )
    reflectivity = np.broadcast_to(reflectivity, (energies.size, angles.size, scale_axis.shape[2]))
    reflectivity = np.clip(np.array(reflectivity), 0.0, 1.0)
    return reflectivity[:, :, 0] if scale is None else reflectivity


@dataclass(frozen=True)
class LateralGrading:
    """Lateral variation of layer thickness along the mirror.

    The local thickness of every layer is its nominal value times
    ``scale(s)``, where ``s`` is the position along the mirror in mm measured
    from the mirror centre (``s = 0``).

    Attributes:
        coefficients: Polynomial coefficients in ascending order,
            ``scale(s) = c0 + c1*s + c2*s**2 + ...``. The default ``(1.0,)``
            is an ungraded mirror. For a linear grading of ``g`` per mm use
            ``(1.0, g)``.
        table_positions_mm: Optional sample positions for a tabulated profile.
            When given together with ``table_scale``, the table is linearly
            interpolated (and clamped at its ends) and ``coefficients`` is
            ignored.
        table_scale: Scale factors at ``table_positions_mm``.
    """

    coefficients: tuple[float, ...] = (1.0,)
    table_positions_mm: tuple[float, ...] | None = None
    table_scale: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the grading definition."""

        if (self.table_positions_mm is None) != (self.table_scale is None):
            raise ValueError("table_positions_mm and table_scale must be given together.")
        if self.table_positions_mm is not None and self.table_scale is not None:
            if len(self.table_positions_mm) != len(self.table_scale) or len(self.table_scale) < 2:
                raise ValueError("Grading table needs >= 2 matching positions and scale values.")
            if np.any(np.diff(np.asarray(self.table_positions_mm, dtype=float)) <= 0.0):
                raise ValueError("table_positions_mm must be strictly increasing.")
        elif len(self.coefficients) == 0:
            raise ValueError("coefficients must not be empty.")

    @classmethod
    def linear(cls, relative_change_per_mm: float) -> "LateralGrading":
        """Return a linear grading with the given relative thickness change per mm."""

        return cls(coefficients=(1.0, float(relative_change_per_mm)))

    def scale(self, positions_mm: Any) -> np.ndarray:
        """Return the thickness scale factor at the given positions in mm."""

        positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
        if self.table_positions_mm is not None and self.table_scale is not None:
            return np.interp(
                positions,
                np.asarray(self.table_positions_mm, dtype=float),
                np.asarray(self.table_scale, dtype=float),
            )
        return np.polynomial.polynomial.polyval(positions, np.asarray(self.coefficients, dtype=float))


def footprint_reflectivity(
    stack: BaseStack,
    energies_ev: Any,
    grazing_angles_deg: Any,
    *,
    grading: LateralGrading | Callable[[np.ndarray], Any],
    positions_mm: Any,
    weights: Any = None,
    polarization: str = "s",
    roughness_sigma_nm: float = 0.0,
) -> np.ndarray:
    """Return the reflectivity averaged over a laterally graded footprint.

    The beam illuminates positions ``positions_mm`` along the mirror, each
    with its own local thickness. At fixed energy and angle the Bragg
    condition differs along the footprint, so the average is broader and
    lower than the reflectivity at the central position. The positions are
    averaged incoherently with ``weights`` (uniform by default).

    Args:
        stack: Coating stack, see :func:`parratt_reflectivity`.
        energies_ev: Photon energies in eV.
        grazing_angles_deg: Grazing angles in degrees.
        grading: A :class:`LateralGrading` or a callable mapping positions in
            mm to thickness scale factors.
        positions_mm: Sample positions along the footprint in mm (use tens of
            points for a smooth average).
        weights: Optional non-negative intensity weights, one per position.
        polarization: See :func:`parratt_reflectivity`.
        roughness_sigma_nm: See :func:`parratt_reflectivity`.

    Returns:
        Footprint-averaged reflectivity with shape ``(n_energies, n_angles)``.
    """

    positions = np.atleast_1d(np.asarray(positions_mm, dtype=float))
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("positions_mm must be a non-empty 1-D array.")
    scale_fn = grading.scale if isinstance(grading, LateralGrading) else grading
    scale = np.atleast_1d(np.asarray(scale_fn(positions), dtype=float))
    if scale.shape != positions.shape:
        raise ValueError("The grading must return one scale factor per position.")

    if weights is None:
        weight_array = np.full(positions.shape, 1.0 / positions.size)
    else:
        weight_array = np.atleast_1d(np.asarray(weights, dtype=float))
        if weight_array.shape != positions.shape:
            raise ValueError("weights must have one value per position.")
        if np.any(weight_array < 0.0) or weight_array.sum() <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum.")
        weight_array = weight_array / weight_array.sum()

    reflectivity = parratt_reflectivity(
        stack,
        energies_ev,
        grazing_angles_deg,
        polarization=polarization,
        roughness_sigma_nm=roughness_sigma_nm,
        thickness_scale=scale,
    )
    return reflectivity @ weight_array


__all__ = [
    "LateralGrading",
    "footprint_reflectivity",
    "parratt_reflectivity",
]
