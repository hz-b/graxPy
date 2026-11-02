"""Coupled-wave model of a Bragg reflection modulated by a comb-electrode grating.

A piezoelectric crystal carries comb-shaped (interdigital) electrodes. A voltage
across the combs deforms the lattice below the fingers periodically along the
surface, so the Bragg reflection splits into satellites (Vadilonga et al., Opt.
Express 29, 34962, 2021). This module computes the rocking curve of that device
for a symmetric Bragg reflection, for a given amplitude of the deformation.

The model is the two-beam (transmitted and diffracted wave) Takagi-Taupin
picture, expanded in lateral Fourier orders ``m`` along the surface so that the
orders are coupled by the periodic lattice displacement. Depth is treated by
integrating the reflection matrix of the crystal from a perfect substrate up to
the surface, which is numerically stable. The Al fingers enter as a thin
amplitude and phase mask on the entry and exit paths.

Lengths are in micrometres unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ._threads import single_threaded_blas
from .materials import resolve_refractive_index

HC_EV_ANGSTROM = 12398.419843320026
_ANGSTROM_IN_UM = 1e-4
_NM_IN_UM = 1e-3


@dataclass(frozen=True)
class BraggCrystal:
    """Bragg reflection constants of a crystal at one photon energy."""

    name: str
    hkl: tuple[int, int, int]
    energy_ev: float
    wavelength_um: float
    d_spacing_um: float
    theta_b_rad: float
    chi0: complex
    chih: complex
    chihbar: complex

    @property
    def wavenumber_per_um(self) -> float:
        """Vacuum wavenumber ``2*pi/lambda`` in 1/um."""

        return 2.0 * np.pi / self.wavelength_um

    @property
    def h_per_um(self) -> float:
        """Length of the reciprocal-lattice vector ``2*pi/d`` in 1/um."""

        return 2.0 * np.pi / self.d_spacing_um


@dataclass(frozen=True)
class IdtGrating:
    """Comb electrodes on the surface and the geometry of the deformation they drive.

    The comb has fingers of alternating polarity. One period is ``+`` finger,
    gap, ``-`` finger, gap, so the deformation period is
    ``2 * (finger_um + gap_um)`` and the Al period is half of it.

    The surface-normal lattice displacement follows the electrode pattern at the
    surface (``+A`` below one finger polarity, ``-A`` below the other, smooth
    across the gaps, ``A`` being the amplitude given to :func:`rocking_curve`).
    Lateral harmonic ``n`` of the pattern decays with depth as
    ``exp(-|n| * z / penetration_um)``, so the fundamental decays over
    ``penetration_um``. ``electrode_length_um`` (along the grating vector) and
    ``aperture_um`` (across it) bound the active area for :func:`active_fraction`.
    """

    finger_um: float = 1.0
    gap_um: float = 1.0
    penetration_um: float = 1.59
    al_thickness_um: float = 0.1
    al_material: str = "Al"
    electrode_length_um: float = 1200.0
    aperture_um: float = 200.0

    @property
    def period_um(self) -> float:
        """Period of the lattice deformation (and of the grating vector)."""

        return 2.0 * (self.finger_um + self.gap_um)

    @property
    def decay_per_um(self) -> float:
        """Depth decay rate of the fundamental harmonic in 1/um."""

        return 1.0 / self.penetration_um


@dataclass(frozen=True)
class RockingCurve:
    """Efficiency versus angle, per exit order and summed over orders."""

    delta_theta_rad: np.ndarray
    orders: np.ndarray
    order_efficiency: np.ndarray
    total: np.ndarray
    amplitude_nm: float


CRYSTALS = ("LiNbO3", "Si", "Ge", "diamond")


def bragg_crystal(
    material: str, hkl: tuple[int, int, int], energy_ev: float
) -> BraggCrystal:
    """Bragg constants of a crystal for a symmetric reflection (planes parallel to the surface).

    Args:
        material: One of :data:`CRYSTALS`.
        hkl: Miller indices of the reflection.
        energy_ev: Photon energy in electronvolts.

    Raises:
        ValueError: For an unknown material, a reflection that is forbidden, or one
            that cannot be reached at this photon energy.
    """

    from xrt.backends.raycing import materials as xrt_materials

    hkl = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if hkl == (0, 0, 0):
        raise ValueError("The Miller indices must not all be zero.")
    if material == "LiNbO3":
        atoms, positions = _lithium_niobate_cell()
        cell = xrt_materials.CrystalFromCell(
            "LiNbO3",
            list(hkl),
            a=5.1483,
            c=13.8631,
            alpha=90,
            beta=90,
            gamma=120,
            atoms=atoms,
            atomsXYZ=positions,
        )
    elif material == "Si":
        cell = xrt_materials.CrystalSi(hkl=list(hkl))
    elif material == "Ge":
        cell = xrt_materials.CrystalFromCell("Ge", list(hkl), a=5.6579, atoms=[32] * 8)
    elif material == "diamond":
        cell = xrt_materials.CrystalDiamond(hkl, 3.5668)
    else:
        raise ValueError(f"Unknown crystal {material!r}; choose from {', '.join(CRYSTALS)}.")
    d_angstrom = float(cell.d)
    wavelength_angstrom = HC_EV_ANGSTROM / energy_ev
    if wavelength_angstrom >= 2.0 * d_angstrom:
        raise ValueError(
            f"The {material} {hkl} reflection (d = {d_angstrom:.3f} Å) cannot be reached at "
            f"{energy_ev:.0f} eV; raise the energy or choose a lower-order reflection."
        )
    theta_b = float(cell.get_Bragg_angle(energy_ev))
    _, f_h, _, chi0, chih, chihbar = cell.get_F_chi(energy_ev, 1.0 / (2.0 * d_angstrom))
    if abs(f_h) < 1e-6:
        raise ValueError(f"The {material} {hkl} reflection is forbidden by the crystal symmetry.")
    return BraggCrystal(
        name=material,
        hkl=hkl,
        energy_ev=float(energy_ev),
        wavelength_um=HC_EV_ANGSTROM / energy_ev * _ANGSTROM_IN_UM,
        d_spacing_um=d_angstrom * _ANGSTROM_IN_UM,
        theta_b_rad=theta_b,
        chi0=complex(chi0),
        chih=complex(chih),
        chihbar=complex(chihbar),
    )


def _lithium_niobate_cell() -> tuple[list[int], list[list[float]]]:
    """Atoms (30 per hexagonal cell) of congruent LiNbO3 from its R3c symmetry."""

    operations = (
        lambda x, y, z: (x, y, z),
        lambda x, y, z: (-y, x - y, z),
        lambda x, y, z: (-x + y, -x, z),
        lambda x, y, z: (-y, -x, z + 0.5),
        lambda x, y, z: (-x + y, y, z + 0.5),
        lambda x, y, z: (x, x - y, z + 0.5),
    )
    centrings = ((0.0, 0.0, 0.0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3))
    sites = ((41, (0.0, 0.0, 0.0)), (3, (0.0, 0.0, 0.2829)), (8, (0.0492, 0.3446, 0.0647)))
    atoms: list[int] = []
    positions: list[list[float]] = []
    for atomic_number, site in sites:
        found: list[np.ndarray] = []
        for operation in operations:
            for centring in centrings:
                point = (np.array(operation(*site)) + centring) % 1.0
                if not any(_same_fractional_point(point, other) for other in found):
                    found.append(point)
        atoms.extend([atomic_number] * len(found))
        positions.extend(point.tolist() for point in found)
    return atoms, positions


def _same_fractional_point(first: np.ndarray, second: np.ndarray) -> bool:
    difference = np.abs(first - second)
    return bool(np.all(np.minimum(difference, 1.0 - difference) < 2e-3))


def surface_potential(grating: IdtGrating, n_x: int) -> np.ndarray:
    """Surface pattern over one period with unit amplitude.

    ``+1`` below one finger polarity, ``-1`` below the other, and linear ramps
    across the gaps.
    """

    period = grating.period_um
    x = (np.arange(n_x) + 0.5) / n_x * period
    f, g = grating.finger_um, grating.gap_um
    knots = np.array([0.0, f, f + g, 2 * f + g, 2 * (f + g)])
    values = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return np.interp(x, knots, values)


def al_mask_profile(grating: IdtGrating, n_x: int) -> np.ndarray:
    """1 below an Al finger and 0 elsewhere, over one deformation period."""

    period = grating.period_um
    x = (np.arange(n_x) + 0.5) / n_x * period
    f, g = grating.finger_um, grating.gap_um
    below_positive = x < f
    below_negative = (x >= f + g) & (x < 2 * f + g)
    return (below_positive | below_negative).astype(float)


def surface_displacement(
    grating: IdtGrating,
    amplitude_um: float,
    depths_um: np.ndarray,
    n_x: int = 256,
) -> np.ndarray:
    """Surface-normal lattice displacement in um, shape ``(len(depths), n_x)`` over one period.

    The surface pattern has zero-to-peak amplitude ``amplitude_um``; harmonic ``n``
    decays with depth as ``exp(-|n| z / penetration)``.
    """

    spectrum = np.fft.fft(surface_potential(grating, n_x))
    harmonics = np.fft.fftfreq(n_x, d=1.0 / n_x)
    decay = np.exp(-np.abs(harmonics)[None, :] * grating.decay_per_um * depths_um[:, None])
    return amplitude_um * np.fft.ifft(spectrum[None, :] * decay, axis=1).real


def lattice_phase(
    grating: IdtGrating,
    crystal: BraggCrystal,
    amplitude_um: float,
    depths_um: np.ndarray,
    n_x: int = 256,
) -> np.ndarray:
    """Phase ``h * u(x, z)`` of the Bragg Fourier component, shape ``(len(depths), n_x)``."""

    return crystal.h_per_um * surface_displacement(grating, amplitude_um, depths_um, n_x)


def fourier_coefficients(values: np.ndarray, max_index: int) -> np.ndarray:
    """Coefficients ``c_k`` for ``k = -max_index..max_index`` of samples along the last axis."""

    n_x = values.shape[-1]
    spectrum = np.fft.fft(values, axis=-1) / n_x
    indices = np.arange(-max_index, max_index + 1)
    return spectrum[..., indices % n_x]


def toeplitz_stack(coefficients: np.ndarray, n_orders: int) -> np.ndarray:
    """Matrices ``X[m, n] = c[m - n]`` for each leading index of ``coefficients``."""

    half = (n_orders - 1) // 2
    rows = np.arange(n_orders)[:, None]
    columns = np.arange(n_orders)[None, :]
    return coefficients[..., (rows - columns) + 2 * half]


@njit(cache=True)
def _matmul(a, b, out):
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            total = 0.0 + 0.0j
            for k in range(n):
                total += a[i, k] * b[k, j]
            out[i, j] = total


@njit(cache=True)
def _riccati_rhs(r, p_diag, s_diag, q_mat, t_mat, scratch_a, scratch_b, out):
    """``R' = T + S R - R P - R Q R`` with ``S`` and ``P`` diagonal."""

    n = r.shape[0]
    _matmul(r, q_mat, scratch_a)
    _matmul(scratch_a, r, scratch_b)
    for i in range(n):
        for j in range(n):
            out[i, j] = t_mat[i, j] + s_diag[i] * r[i, j] - r[i, j] * p_diag[j] - scratch_b[i, j]


@njit(cache=True)
def _integrate_reflection(r0, p_diag, s_diag, q_nodes, t_nodes, step):
    """RK4 from the substrate (last node) to the surface (node 0); returns ``R`` at the surface."""

    n = r0.shape[0]
    n_steps = (q_nodes.shape[0] - 1) // 2
    r = r0.copy()
    k1 = np.empty((n, n), dtype=np.complex128)
    k2 = np.empty((n, n), dtype=np.complex128)
    k3 = np.empty((n, n), dtype=np.complex128)
    k4 = np.empty((n, n), dtype=np.complex128)
    trial = np.empty((n, n), dtype=np.complex128)
    scratch_a = np.empty((n, n), dtype=np.complex128)
    scratch_b = np.empty((n, n), dtype=np.complex128)
    h = -step
    for index in range(n_steps):
        node = 2 * (n_steps - index)
        _riccati_rhs(r, p_diag, s_diag, q_nodes[node], t_nodes[node], scratch_a, scratch_b, k1)
        for i in range(n):
            for j in range(n):
                trial[i, j] = r[i, j] + 0.5 * h * k1[i, j]
        _riccati_rhs(trial, p_diag, s_diag, q_nodes[node - 1], t_nodes[node - 1], scratch_a, scratch_b, k2)
        for i in range(n):
            for j in range(n):
                trial[i, j] = r[i, j] + 0.5 * h * k2[i, j]
        _riccati_rhs(trial, p_diag, s_diag, q_nodes[node - 1], t_nodes[node - 1], scratch_a, scratch_b, k3)
        for i in range(n):
            for j in range(n):
                trial[i, j] = r[i, j] + h * k3[i, j]
        _riccati_rhs(trial, p_diag, s_diag, q_nodes[node - 2], t_nodes[node - 2], scratch_a, scratch_b, k4)
        for i in range(n):
            for j in range(n):
                r[i, j] += h / 6.0 * (k1[i, j] + 2.0 * k2[i, j] + 2.0 * k3[i, j] + k4[i, j])
    return r


@njit(cache=True, parallel=True)
def _surface_reflection_matrices(betas, base_p, base_s, r_substrate, q_nodes, t_nodes, step):
    n_angles = betas.shape[0]
    n = base_p.shape[0]
    out = np.empty((n_angles, n, n), dtype=np.complex128)
    for index in prange(n_angles):
        s_diag = base_s - 1j * betas[index]
        out[index] = _integrate_reflection(
            r_substrate[index], base_p, s_diag, q_nodes, t_nodes, step
        )
    return out


def _semi_infinite_reflection(p_diag, s_diag, q_scalar, t_scalar):
    """Diagonal ``R`` of the perfect substrate: the bounded root of ``Q R^2 + (P-S) R - T = 0``."""

    difference = p_diag - s_diag
    root = np.sqrt(difference**2 + 4.0 * q_scalar * t_scalar)
    first = (-difference + root) / (2.0 * q_scalar)
    second = (-difference - root) / (2.0 * q_scalar)
    return np.where(np.abs(first) <= np.abs(second), first, second)


def rocking_curve(
    crystal: BraggCrystal,
    grating: IdtGrating,
    amplitude_nm: float,
    delta_theta_rad: np.ndarray,
    *,
    n_orders: int = 17,
    polarization: str = "sigma",
    include_al_mask: bool = True,
    coating: tuple[tuple[object, float], ...] = (),
    max_step_um: float = 0.02,
    n_x: int = 256,
) -> RockingCurve:
    """Diffraction efficiency into each exit order for a scan of incidence angles.

    Args:
        crystal: Bragg constants from :func:`bragg_crystal`.
        grating: Electrode geometry and displacement model.
        amplitude_nm: Zero-to-peak amplitude of the surface-normal displacement in nm.
        delta_theta_rad: Offsets from the kinematical Bragg angle in radians.
        n_orders: Odd number of lateral Fourier orders kept.
        polarization: ``"sigma"`` or ``"pi"``.
        include_al_mask: Apply the Al fingers as an entry and exit phase mask.
        coating: ``(material, thickness_um)`` layers covering the surface. A
            conformal coating has the same thickness everywhere, so it only
            attenuates and phase-shifts the entry and exit beams uniformly.
        max_step_um: Upper bound on the depth step of the integration.
        n_x: Samples per period for the Fourier expansion of the deformation.
    """

    if n_orders % 2 == 0 or n_orders < 1:
        raise ValueError("n_orders must be a positive odd integer.")
    delta_theta = np.atleast_1d(np.asarray(delta_theta_rad, dtype=float))
    half = (n_orders - 1) // 2
    orders = np.arange(-half, half + 1)

    k = crystal.wavenumber_per_um
    theta = crystal.theta_b_rad
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    q = 2.0 * np.pi / grating.period_um
    polarization_factor = 1.0 if polarization == "sigma" else abs(np.cos(2.0 * theta))
    coupling = 1j * k * polarization_factor / (2.0 * sin_t)

    p_diag = 1j * (k * crystal.chi0 / 2.0 - orders * q * cos_t) / sin_t
    s_diag_base = 1j * (orders * q * cos_t - k * crystal.chi0 / 2.0) / sin_t
    betas = k * np.sin(2.0 * theta) * delta_theta
    s_per_angle = s_diag_base[None, :] - 1j * betas[:, None] / sin_t

    omega = np.abs(s_per_angle[:, :, None] - p_diag[None, None, :]).max()
    step = min(max_step_um, 1.5 / max(omega, 1e-9))
    decay_length = grating.penetration_um
    depth_max = 8.0 * decay_length
    n_steps = int(np.ceil(depth_max / step))
    step = depth_max / n_steps
    depths = np.arange(2 * n_steps + 1) * step / 2.0

    phase = lattice_phase(grating, crystal, amplitude_nm * _NM_IN_UM, depths, n_x)
    chih_coefficients = crystal.chih * fourier_coefficients(np.exp(-1j * phase), 2 * half)
    chihbar_coefficients = crystal.chihbar * fourier_coefficients(np.exp(1j * phase), 2 * half)
    q_nodes = np.ascontiguousarray(coupling * toeplitz_stack(chihbar_coefficients, n_orders))
    t_nodes = np.ascontiguousarray(-coupling * toeplitz_stack(chih_coefficients, n_orders))

    q_scalar = coupling * crystal.chihbar
    t_scalar = -coupling * crystal.chih
    r_substrate = np.zeros((delta_theta.size, n_orders, n_orders), dtype=np.complex128)
    for index in range(delta_theta.size):
        diagonal = _semi_infinite_reflection(p_diag, s_per_angle[index], q_scalar, t_scalar)
        r_substrate[index][np.diag_indices(n_orders)] = diagonal

    with single_threaded_blas():
        r_surface = _surface_reflection_matrices(
            betas / sin_t, p_diag, s_diag_base, r_substrate, q_nodes, t_nodes, step
        )

    entry = np.zeros(n_orders, dtype=complex)
    entry[half] = 1.0
    exit_matrix = np.eye(n_orders, dtype=complex)
    if include_al_mask and grating.al_thickness_um > 0.0:
        entry, exit_matrix = _al_mask_operators(grating, crystal, n_orders, n_x)

    amplitudes = np.einsum("aij,j->ai", r_surface, entry)
    amplitudes = np.einsum("ij,aj->ai", exit_matrix, amplitudes)
    amplitudes = amplitudes * coating_transmission(coating, crystal) ** 2
    direction = (cos_t + orders * q / k) ** 2
    exit_sin = np.sqrt(np.clip(1.0 - direction, 0.0, None))
    flux = np.abs(amplitudes) ** 2 * (exit_sin / sin_t)[None, :]
    return RockingCurve(
        delta_theta_rad=delta_theta,
        orders=orders,
        order_efficiency=flux,
        total=flux.sum(axis=1),
        amplitude_nm=float(amplitude_nm),
    )


def _al_mask_operators(
    grating: IdtGrating, crystal: BraggCrystal, n_orders: int, n_x: int
) -> tuple[np.ndarray, np.ndarray]:
    """Entry-mask harmonics and exit-mask convolution matrix of the Al fingers."""

    half = (n_orders - 1) // 2
    index = resolve_refractive_index(grating.al_material, crystal.energy_ev)
    delta, beta = 1.0 - index.real, index.imag
    path = grating.al_thickness_um / np.sin(crystal.theta_b_rad)
    k = crystal.wavenumber_per_um
    transmission = np.exp(-k * (beta + 1j * delta) * path)
    mask = 1.0 + (transmission - 1.0) * al_mask_profile(grating, n_x)
    coefficients = fourier_coefficients(mask, 2 * half)
    entry = coefficients[(np.arange(n_orders) - half) + 2 * half]
    exit_matrix = toeplitz_stack(coefficients, n_orders)
    return entry, exit_matrix


def coating_transmission(coating: tuple[tuple[object, float], ...], crystal: BraggCrystal) -> complex:
    """Amplitude transmission of uniform layers for one pass at the Bragg angle."""

    k = crystal.wavenumber_per_um
    transmission = 1.0 + 0.0j
    for material, thickness_um in coating:
        index = resolve_refractive_index(material, crystal.energy_ev)
        path = float(thickness_um) / np.sin(crystal.theta_b_rad)
        transmission *= np.exp(-k * (index.imag + 1j * (1.0 - index.real)) * path)
    return complex(transmission)


def gaussian_resolution(delta_theta_rad: np.ndarray, curve: np.ndarray, fwhm_rad: float) -> np.ndarray:
    """Convolve a curve on a uniform angle grid with a Gaussian of the given FWHM."""

    if fwhm_rad <= 0.0:
        return curve
    step = float(delta_theta_rad[1] - delta_theta_rad[0])
    sigma = fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0))) / step
    half_width = int(np.ceil(4.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half_width, half_width + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(curve, half_width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def active_fraction(
    grating: IdtGrating, crystal: BraggCrystal, beam_x_um: float, beam_y_um: float
) -> float:
    """Fraction of a Gaussian beam that lands on the active electrode area.

    The beam is centred on the electrodes. ``beam_x_um`` is the beam size (FWHM)
    in the scattering plane, whose footprint on the surface is longer by
    ``1/sin(theta_B)`` along the grating vector; ``beam_y_um`` is the size across it.
    """

    def inside(length_um: float, fwhm_um: float) -> float:
        if fwhm_um <= 0.0:
            return 1.0
        sigma = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return float(math.erf(length_um / 2.0 / (sigma * np.sqrt(2.0))))

    footprint = beam_x_um / np.sin(crystal.theta_b_rad)
    return inside(grating.electrode_length_um, footprint) * inside(grating.aperture_um, beam_y_um)


def simulate_rocking_curves(
    crystal: BraggCrystal,
    grating: IdtGrating,
    amplitudes_nm: list[float],
    delta_theta_rad: np.ndarray,
    *,
    beam_x_um: float = 0.0,
    beam_y_um: float = 0.0,
    resolution_rad: float = 0.0,
    n_orders: int = 17,
    polarization: str = "sigma",
    coating: tuple[tuple[object, float], ...] = (),
    progress=None,
    stop=None,
) -> dict:
    """Measured-like rocking curves for several deformation amplitudes.

    Each curve is the active part (with the deformation) plus the part of the
    beam outside the electrodes (undeformed crystal, no Al), weighted by
    :func:`active_fraction`, smoothed by the angular resolution, and normalized to
    the peak of the undeformed crystal curve. ``progress(amplitude, curve,
    peak_angle_rad)`` is called after each amplitude and a set ``stop`` event ends the loop early.
    """

    fraction = active_fraction(grating, crystal, beam_x_um, beam_y_um)
    options = {"n_orders": n_orders, "polarization": polarization, "coating": tuple(coating)}
    bare = rocking_curve(crystal, grating, 0.0, delta_theta_rad, include_al_mask=False, **options)
    reference = rocking_curve(crystal, grating, 0.0, delta_theta_rad, **options)
    undeformed = gaussian_resolution(
        delta_theta_rad, fraction * reference.total + (1.0 - fraction) * bare.total, resolution_rad
    )
    peak_index = int(np.argmax(undeformed))
    norm = float(undeformed[peak_index])
    curves: dict[float, np.ndarray] = {}
    for amplitude in amplitudes_nm:
        if stop is not None and stop.is_set():
            break
        active = reference if amplitude == 0.0 else rocking_curve(
            crystal, grating, amplitude, delta_theta_rad, **options
        )
        mixed = fraction * active.total + (1.0 - fraction) * bare.total
        curves[float(amplitude)] = gaussian_resolution(delta_theta_rad, mixed, resolution_rad) / norm
        if progress is not None:
            progress(float(amplitude), curves[float(amplitude)], float(delta_theta_rad[peak_index]))
    return {
        "curves": curves,
        "active_fraction": fraction,
        "peak_angle_rad": float(delta_theta_rad[peak_index]),
        "reference_peak_efficiency": norm,
    }
