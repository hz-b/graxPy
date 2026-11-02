"""Comb-electrode (IDT) grating under a coating, for grazing-incidence solvers.

This is the second way to use the piezoelectric grating of
:mod:`grax.bragg_grating`: the crystal with its Al fingers is covered by a
coating or a multilayer and used at grazing incidence, in total external
reflection or near the multilayer Bragg peak. There the crystal lattice plays
no role; the device is a surface grating whose profile is

* the surface-normal displacement ``h(x)`` that the voltage drives (the same
  pattern and amplitude as in :func:`grax.bragg_grating.surface_displacement`),
* plus the Al fingers, which fill ``h <= z < h + t_Al`` below each finger,
* plus the coating layers, which follow the outer envelope.

By default every layer copies the interface below it shifted straight up. A
layer with a planarization length ``nu`` instead smooths it while it grows: the
Fourier component ``q`` of the profile is damped by ``exp(-nu q^2 t)`` across a
layer of thickness ``t`` (linear growth model). The fingers alone are a strong
grating with half the deformation period, so even orders are present at 0 V and
odd orders appear only with the voltage-driven displacement.

:class:`IdtSurfaceGrating` is a :class:`~grax.gratings.BaseGrating`, so the
RCWA and Nevière solvers, :func:`grax.run_simulation` and the batch runner take
it as is. Its solver slices are placed at the interfaces, with
``ripple_slices`` sub-slices across each interface's height range, because the
sub-nanometre displacement is far below any uniform z grid.

Lengths are in nanometres unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .bragg_grating import IdtGrating, gaussian_resolution
from .gratings import BaseGrating
from .materials import MaterialSpec, material_label, resolve_refractive_index, validate_material_input
from .stacks import BareStack, BaseStack, CustomStack, LayerSpec

CRYSTAL_MATERIALS: dict[str, MaterialSpec] = {
    "LiNbO3": MaterialSpec("LiNbO3", 4.65),
    "Si": MaterialSpec("Si", 2.329),
    "Ge": MaterialSpec("Ge", 5.323),
    "diamond": MaterialSpec("C", 3.515),
}

_INCIDENT, _CRYSTAL, _FINGER = 0, 1, 2


@dataclass(frozen=True)
class CoatingLayer:
    """One layer above the fingers; ``planarization_nm`` is the smoothing length ``nu``."""

    material: Any
    thickness_nm: float
    planarization_nm: float = 0.0

    def __post_init__(self) -> None:
        if not float(self.thickness_nm) > 0.0:
            raise ValueError("Coating layer thickness must be above 0 nm.")
        if not float(self.planarization_nm) >= 0.0:
            raise ValueError("Coating planarization length must be at least 0 nm.")


@dataclass
class IdtSurfaceGrating(BaseGrating):
    """Crystal with comb electrodes and a coating stack, as a 1D surface grating.

    Attributes:
        idt: Electrode geometry; its period sets the grating period.
        amplitude_nm: Zero-to-peak surface displacement driven by the voltage.
        crystal_material: Substrate material (the crystal).
        layers: Coating layers from the bottom (on the fingers) up.
        ripple_slices: Solver slices across the height range of each interface.
    """

    idt: IdtGrating = field(default_factory=IdtGrating)
    amplitude_nm: float = 0.0
    crystal_material: Any = field(default_factory=lambda: CRYSTAL_MATERIALS["LiNbO3"])
    layers: tuple[CoatingLayer, ...] = ()
    ripple_slices: int = 8
    x_resolution_nm: float = 2.0
    z_resolution_nm: float = 0.5

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        if self.amplitude_nm < 0.0:
            raise ValueError("The displacement amplitude must be at least 0 nm.")
        if self.ripple_slices < 1:
            raise ValueError("ripple_slices must be at least 1.")
        self.period_lpermm = 1e3 / self.idt.period_um
        self.substrate_material = self.crystal_material
        self.coating_stack = self.stack()

    def stack(self) -> BaseStack:
        """The coating as a flat stack on the crystal (no fingers, no displacement)."""

        if not self.layers:
            return BareStack(substrate_material=self.crystal_material)
        return CustomStack(
            substrate_material=self.crystal_material,
            layers_bottom_up=[LayerSpec(layer.material, layer.thickness_nm) for layer in self.layers],
        )

    # Geometry -----------------------------------------------------------

    def _nodes(self, num_periods: int = 1) -> np.ndarray:
        n_x = max(int(round(self.period_nm / self.x_resolution_nm)), 16)
        return np.linspace(0.0, num_periods * self.period_nm, num_periods * n_x + 1)

    def _geometry(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Crystal surface, finger mask and the interfaces ``s_0`` (envelope) .. ``s_L``.

        ``x`` must be a uniform grid over whole periods that includes both ends.
        Heights are shifted so the lowest point of the crystal surface is 0.
        """

        period = self.period_nm
        f, g = self.idt.finger_um * 1e3, self.idt.gap_um * 1e3
        phase = np.mod(x, period)
        knots = np.array([0.0, f, f + g, 2 * f + g, 2 * (f + g)])
        pattern = np.interp(phase, knots, [1.0, 1.0, -1.0, -1.0, 1.0])
        crystal = self.amplitude_nm * (pattern + 1.0)
        fingers = (phase < f) | ((phase >= f + g) & (phase < 2 * f + g))
        envelope = crystal + self.idt.al_thickness_um * 1e3 * fingers
        interfaces = [envelope]
        q = 2.0 * np.pi * np.fft.rfftfreq(x.size - 1, d=float(x[1] - x[0]))
        for layer in self.layers:
            lower = interfaces[-1]
            upper = lower + layer.thickness_nm
            if layer.planarization_nm > 0.0:
                periodic = lower[:-1]
                mean = periodic.mean()
                damping = np.exp(-layer.planarization_nm * q**2 * layer.thickness_nm)
                smooth = mean + np.fft.irfft(np.fft.rfft(periodic - mean) * damping, n=periodic.size)
                upper = np.maximum(np.append(smooth, smooth[0]) + layer.thickness_nm, lower)
            interfaces.append(upper)
        return crystal, fingers, interfaces

    def interface_profiles(self, num_periods: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
        """x grid and the heights of every material boundary, crystal surface first."""

        x = self._nodes(num_periods)
        crystal, _, interfaces = self._geometry(x)
        return x, [crystal, *interfaces]

    def _codes_at(self, z: float, crystal: np.ndarray, fingers: np.ndarray, interfaces: list[np.ndarray]) -> np.ndarray:
        """Material code per x node at height ``z``: 0 incident, 1 crystal, 2 fingers, 3 + j layer j."""

        codes = np.full(crystal.shape, _INCIDENT, dtype=np.int32)
        for index in range(len(interfaces) - 1):
            codes[(z >= interfaces[index]) & (z < interfaces[index + 1])] = 3 + index
        codes[fingers & (z < interfaces[0])] = _FINGER
        codes[z < crystal] = _CRYSTAL
        return codes

    def _slice_boundaries(self, crystal: np.ndarray, interfaces: list[np.ndarray]) -> np.ndarray:
        """Heights of the slice boundaries, from top to bottom."""

        boundaries: list[np.ndarray] = []
        for profile in (crystal, *interfaces):
            values = np.unique(profile)
            splits = np.nonzero(np.diff(values) > self.z_resolution_nm)[0]
            for band in np.split(values, splits + 1):
                low, high = float(band[0]), float(band[-1])
                if high - low <= 1e-9:
                    boundaries.append(np.array([low]))
                    continue
                count = max(self.ripple_slices, int(math.ceil((high - low) / self.z_resolution_nm)))
                boundaries.append(np.linspace(low, high, count + 1))
        merged = np.unique(np.round(np.concatenate(boundaries), 9))
        return merged[::-1]

    def solver_slice_count(self) -> int:
        """Number of solver slices (before identical neighbours are merged)."""

        crystal, _, interfaces = self._geometry(self._nodes())
        return max(len(self._slice_boundaries(crystal, interfaces)) - 1, 0)

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        x = self._nodes()
        _, _, interfaces = self._geometry(x)
        return x, interfaces[0]

    def profile_depth_nm(self) -> float:
        _, envelope = self.profile_points()
        return float(np.max(envelope) - np.min(envelope))

    def material_names(self) -> list[str]:
        """Labels for codes 1 (crystal), 2 (fingers) and 3 + j (layer j)."""

        return [
            material_label(self.crystal_material),
            material_label(self.idt.al_material),
            *(material_label(layer.material) for layer in self.layers),
        ]

    # Solver interface ---------------------------------------------------

    def _validate_simulation_materials(self) -> None:
        validate_material_input(self.crystal_material, field_name="crystal_material")
        validate_material_input(self.idt.al_material, field_name="finger_material")
        for index, layer in enumerate(self.layers):
            validate_material_input(layer.material, field_name=f"layers[{index}]")

    def build_textures(
        self,
        photon_energy_ev: float,
        *,
        n_inc: complex = 1.0 + 0.0j,
        _memory_mode: str = "low_memory",
    ) -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
        """Textures and profile (thicknesses, texture indices) from the top down."""

        del _memory_mode
        self._validate_simulation_materials()
        x = self._nodes()
        crystal, fingers, interfaces = self._geometry(x)
        n_crystal = complex(resolve_refractive_index(self.crystal_material, photon_energy_ev))
        lookup = np.array(
            [
                complex(n_inc),
                n_crystal,
                complex(resolve_refractive_index(self.idt.al_material, photon_energy_ev)),
                *(complex(resolve_refractive_index(layer.material, photon_energy_ev)) for layer in self.layers),
            ],
            dtype=complex,
        )
        textures: list[object] = []
        registry: dict[tuple[object, ...], int] = {}

        def register(texture: object) -> int:
            signature = self._texture_descriptor_signature(texture)
            if signature not in registry:
                registry[signature] = len(textures)
                textures.append(texture)
            return registry[signature]

        thicknesses = [0.0]
        indices = [register(complex(n_inc))]
        boundaries = self._slice_boundaries(crystal, interfaces)
        for top, bottom in zip(boundaries[:-1], boundaries[1:]):
            values = lookup[self._codes_at(0.5 * (top + bottom), crystal, fingers, interfaces)]
            changes = np.nonzero(np.diff(values) != 0)[0]
            texture = complex(values[0]) if changes.size == 0 else [x[changes + 1], values[changes]]
            row = register(texture)
            if len(indices) > 1 and indices[-1] == row:
                thicknesses[-1] += float(top - bottom)
            else:
                thicknesses.append(float(top - bottom))
                indices.append(row)
        thicknesses.append(0.0)
        indices.append(register(n_crystal))
        return textures, (np.asarray(thicknesses, dtype=float), np.asarray(indices, dtype=int))

    def _material_plot_data(self, num_periods: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        x = self._nodes(num_periods)
        crystal, fingers, interfaces = self._geometry(x)
        top = float(np.max(interfaces[-1]))
        bottom = -max(0.1 * top, 1.0)
        z = np.linspace(bottom, top + max(0.05 * top, 0.5), 400)
        labels = list(dict.fromkeys(self.material_names()))
        label_codes = np.array([-1] + [labels.index(name) for name in self.material_names()])
        rows = [label_codes[self._codes_at(float(height), crystal, fingers, interfaces)] for height in z]
        return x, z, np.asarray(rows), labels

    def _build_material_code_grid(self, **_: Any) -> np.ndarray:
        raise NotImplementedError("IdtSurfaceGrating builds its material map in _material_plot_data.")


def grazing_active_fraction(idt: IdtGrating, theta_rad: np.ndarray, beam_x_um: float, beam_y_um: float) -> np.ndarray:
    """Fraction of a Gaussian beam on the electrodes at grazing angle ``theta``.

    ``beam_x_um`` (FWHM, in the scattering plane) is stretched to ``beam_x/sin(theta)``
    along the grating vector; ``beam_y_um`` is the size across it.
    """

    def inside(length_um: float, fwhm_um: np.ndarray) -> np.ndarray:
        sigma = np.asarray(fwhm_um, dtype=float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        with np.errstate(divide="ignore"):
            argument = np.where(sigma > 0.0, length_um / 2.0 / (sigma * np.sqrt(2.0)), np.inf)
        return np.vectorize(math.erf)(argument)

    theta = np.asarray(theta_rad, dtype=float)
    footprint = beam_x_um / np.sin(theta) if beam_x_um > 0.0 else np.zeros_like(theta)
    across = inside(idt.aperture_um, np.array(beam_y_um)) if beam_y_um > 0.0 else 1.0
    return inside(idt.electrode_length_um, footprint) * across


def multilayer_period_nm(layers: tuple[CoatingLayer, ...]) -> float | None:
    """Period of the longest repeated pair of layers, or ``None`` without a multilayer."""

    best, best_repeats = None, 1
    for start in range(len(layers) - 1):
        pair = layers[start:start + 2]
        repeats = 1
        while layers[start + 2 * repeats:start + 2 * repeats + 2] == pair:
            repeats += 1
        if repeats > best_repeats:
            best, best_repeats = pair[0].thickness_nm + pair[1].thickness_nm, repeats
    return best


def theta_window(grating: IdtSurfaceGrating, energy_ev: float) -> dict:
    """Suggested grazing-angle window at one energy.

    Returns ``center_deg`` and ``half_range_deg``, the critical angle of the top
    material and ``multilayer_peak_deg``: the first-order Bragg peak of the flat
    multilayer (refraction-corrected, then refined on its reflectivity), or
    ``None`` without a multilayer. The window covers that peak and its
    satellites, else the total external reflection up to twice the critical angle.
    """

    from .planar_mirror import parratt_reflectivity

    top = grating.layers[-1].material if grating.layers else grating.crystal_material
    delta = 1.0 - float(np.real(resolve_refractive_index(top, energy_ev)))
    critical = math.degrees(math.sqrt(max(2.0 * delta, 1e-12)))
    low, high = 0.05, max(2.0 * critical, 0.1)
    reflection = {
        "center_deg": (low + high) / 2.0,
        "half_range_deg": (high - low) / 2.0,
        "critical_deg": critical,
        "multilayer_peak_deg": None,
    }
    period = multilayer_period_nm(grating.layers)
    if period is None:
        return reflection
    wavelength_nm = 1239.84198 / energy_ev
    pair = [layer for layer in grating.layers if layer.thickness_nm > 0][:2]
    mean_delta = sum(
        (1.0 - float(np.real(resolve_refractive_index(layer.material, energy_ev)))) * layer.thickness_nm
        for layer in pair
    ) / max(sum(layer.thickness_nm for layer in pair), 1e-12)
    sine = math.sqrt((wavelength_nm / (2.0 * period)) ** 2 + 2.0 * mean_delta)
    if sine >= 1.0:
        # The wavelength is too long for the multilayer period: no Bragg peak.
        return reflection
    guess = math.degrees(math.asin(sine))
    angles = np.linspace(0.8 * guess, min(1.2 * guess, 89.99), 2001)
    reflectivity = parratt_reflectivity(grating.stack(), energy_ev, angles)[0]
    peak = float(angles[np.argmax(reflectivity)])
    shift = math.degrees(wavelength_nm / (grating.period_nm * math.sin(math.radians(peak))))
    half = min(max(1.5 * shift, 0.02), peak - 0.01)
    return {"center_deg": peak, "half_range_deg": half, "critical_deg": critical, "multilayer_peak_deg": peak}


def default_theta_range_deg(grating: IdtSurfaceGrating, energy_ev: float) -> tuple[float, float]:
    """Grazing-angle window: around the strongest multilayer peak, else up to twice the critical angle."""

    window = theta_window(grating, energy_ev)
    return window["center_deg"] - window["half_range_deg"], window["center_deg"] + window["half_range_deg"]


def surface_rocking_scan(
    grating: IdtSurfaceGrating,
    amplitudes_nm: list[float],
    theta_deg: np.ndarray,
    *,
    energy_ev: float,
    polarization: str = "s",
    solver: str = "rcwa",
    fourier_orders: int = 25,
    beam_x_um: float = 0.0,
    beam_y_um: float = 0.0,
    resolution_deg: float = 0.0,
    max_workers: int | str | None = None,
    min_efficiency: float = 1e-12,
    progress=None,
    stop=None,
    progress_interval_s: float = 0.5,
) -> dict:
    """Rocking curves per diffraction order, as an ideal line detector records them.

    Every order leaves at its own exit angle and is recorded on its own. Each
    curve mixes the beam on the electrodes (the grating) with the beam outside
    them (the flat coated crystal, order 0 only) by :func:`grazing_active_fraction`,
    then is smoothed by the angular resolution. Orders below ``min_efficiency``
    for every amplitude are dropped.

    ``progress(amplitude, curves, total, exit_angles, computed)`` follows the scan
    point by point, at most every ``progress_interval_s`` and always when an
    amplitude is complete (``computed`` equals the number of angles). Angles not
    computed yet are NaN, and the angular resolution is applied only to complete
    curves. A set ``stop`` event ends the scan; the amplitude in progress is then
    returned with NaN at its missing angles.
    """

    import time

    from .planar_mirror import parratt_reflectivity
    from .simulation import BatchSimulationRunner

    theta = np.asarray(theta_deg, dtype=float)
    amplitudes = [float(value) for value in amplitudes_nm]
    flat = parratt_reflectivity(grating.stack(), energy_ev, theta, polarization=polarization)[0]
    fraction = grazing_active_fraction(grating.idt, np.radians(theta), beam_x_um, beam_y_um)
    cases = [
        {
            "case_id": f"a{a_index}-t{t_index}",
            "grating": replace(grating, amplitude_nm=amplitude),
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": float(angle),
            "amplitude_index": a_index,
            "theta_index": t_index,
        }
        for a_index, amplitude in enumerate(amplitudes)
        for t_index, angle in enumerate(theta)
    ]
    runner = BatchSimulationRunner(
        diffraction_order=1,
        fourier_orders=int(fourier_orders),
        polarization=polarization,
        solver=solver,
        max_workers=max_workers,
        stop_event=stop,
        validate_physical_results=False,
        on_error="fail_fast",
    )
    raw: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {index: {} for index in range(len(amplitudes))}
    exit_angles: dict[int, np.ndarray] = {}
    curves: dict[float, dict[int, np.ndarray]] = {}
    totals: dict[float, np.ndarray] = {}

    def assemble(a_index: int) -> tuple[float, dict[int, np.ndarray], np.ndarray]:
        rows = raw[a_index]
        complete = len(rows) == theta.size
        missing = np.ones(theta.size, dtype=bool)
        missing[list(rows)] = False
        orders = sorted({int(round(order)) for values in rows.values() for order in values[0]})
        per_order = {order: np.zeros(theta.size) for order in orders}
        for t_index, (row_orders, efficiency, angles) in rows.items():
            for order, value, angle in zip(row_orders, efficiency, angles):
                key = int(round(order))
                per_order[key][t_index] = max(float(value), 0.0)
                exit_angles.setdefault(key, np.full(theta.size, np.nan))[t_index] = float(angle)
        mixed = {}
        for order, values in per_order.items():
            curve = fraction * values + (1.0 - fraction) * (flat if order == 0 else 0.0)
            if complete and theta.size > 1:
                curve = gaussian_resolution(np.radians(theta), curve, math.radians(resolution_deg))
            curve[missing] = np.nan
            mixed[order] = curve
        total = np.sum(list(mixed.values()), axis=0) if mixed else np.full(theta.size, np.nan)
        return amplitudes[a_index], mixed, total

    def report(a_index: int) -> None:
        amplitude, mixed, total = assemble(a_index)
        if len(raw[a_index]) == theta.size:
            curves[amplitude] = mixed
            totals[amplitude] = total
        if progress is not None:
            progress(amplitude, mixed, total, dict(exit_angles), len(raw[a_index]))

    last_report = time.monotonic()
    for result in runner.run_cases(cases):
        if stop is not None and stop.is_set():
            break
        data = result.case_data
        a_index = int(data["amplitude_index"])
        raw[a_index][int(data["theta_index"])] = (
            np.asarray(result.orders, dtype=float),
            np.asarray(result.efficiency_all, dtype=float),
            np.asarray(result.diffraction_angle_all, dtype=float),
        )
        now = time.monotonic()
        if len(raw[a_index]) == theta.size or now - last_report >= progress_interval_s:
            report(a_index)
            last_report = now
    if stop is not None and stop.is_set():
        for a_index, rows in raw.items():
            if rows and len(rows) < theta.size:
                amplitude, mixed, total = assemble(a_index)
                curves[amplitude], totals[amplitude] = mixed, total

    kept = sorted(
        order
        for order in {order for per_order in curves.values() for order in per_order}
        if max(float(np.nanmax(per_order.get(order, 0.0), initial=0.0)) for per_order in curves.values()) > min_efficiency
    )
    return {
        "theta_deg": theta,
        "orders": kept,
        "curves": {amplitude: {order: per_order[order] for order in kept if order in per_order} for amplitude, per_order in curves.items()},
        "total": totals,
        "exit_angles_deg": {order: exit_angles[order] for order in kept if order in exit_angles},
        "flat_reference": flat,
        "active_fraction": fraction,
    }
