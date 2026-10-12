"""Piezoelectric X-ray pulse-picker geometry and Bragg rocking model.

The RCWA geometry treats the substrate as a continuous optical medium.  The
rocking model uses a crystalline Bragg amplitude and a thin periodic screen;
the two calculations deliberately expose their different approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .gratings import BaseGrating
from .materials import MaterialSpec, material_label, resolve_refractive_index
from .stacks import BareStack


@dataclass
class PulsePickerGrating(BaseGrating):
    """One piezoelectric period with two interdigitated aluminium fingers.

    ``displacement_pv_nm`` is the peak-to-valley vertical displacement.  It
    describes the static state supplied by the caller, not a presumed material
    voltage response.  ``component`` selects diagnostic sub-geometries.
    """

    period_lpermm: float = 250.0
    substrate_material: object = MaterialSpec("LiNbO3", density_g_cm3=4.64)
    layer_material: object = "Al"
    layer_thickness_nm: float = 0.0
    x_resolution_nm: float = 20.0
    z_resolution_nm: float = 0.025
    electrode_pitch_nm: float = 2000.0
    finger_width_nm: float = 1000.0
    aluminium_thickness_nm: float = 100.0
    electrode_offset_nm: float = 0.0
    displacement_pv_nm: float = 0.0
    voltage_v: float = 0.0
    # Effective Fig. 4 fit for this reduced model; not a measured strain law.
    displacement_per_volt_nm: float = 0.0104
    displacement_harmonics: tuple[float, ...] = ()
    component: Literal["combined", "deformation", "aluminium"] = "combined"
    sample: Literal["LNB", "LGS"] = "LNB"

    def __post_init__(self) -> None:
        """Check that both electrode fingers fit exactly into the cell."""
        positive = (
            "period_lpermm",
            "x_resolution_nm",
            "z_resolution_nm",
            "electrode_pitch_nm",
            "finger_width_nm",
            "aluminium_thickness_nm",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("displacement_pv_nm", "voltage_v", "displacement_per_volt_nm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.electrode_offset_nm):
            raise ValueError("electrode_offset_nm must be finite.")
        if not all(np.isfinite(value) for value in self.displacement_harmonics):
            raise ValueError("displacement_harmonics must be finite.")
        if self.component not in {"combined", "deformation", "aluminium"}:
            raise ValueError("Unsupported pulse-picker component.")
        if self.sample not in {"LNB", "LGS"}:
            raise ValueError("sample must be LNB or LGS.")
        if self.finger_width_nm > self.electrode_pitch_nm:
            raise ValueError("finger_width_nm cannot exceed electrode_pitch_nm.")
        ratio = self.period_nm / self.electrode_pitch_nm
        if not np.isclose(ratio, 2.0, rtol=0.0, atol=1e-8):
            raise ValueError("The grating cell must contain exactly two electrode pitches.")
        if self.roughness is not None:
            raise ValueError("PulsePickerGrating does not support interface roughness yet.")
        if self.coating_stack is not None and not isinstance(self.coating_stack, BareStack):
            raise ValueError("PulsePickerGrating requires a bare substrate stack.")

    @classmethod
    def paper_preset(cls, sample: Literal["LNB", "LGS"] = "LNB") -> PulsePickerGrating:
        """Return the electrode geometry reported in Table 1 of the paper."""
        if sample == "LNB":
            return cls()
        if sample == "LGS":
            return cls(
                sample="LGS",
                period_lpermm=1e6 / 2800.0,
                electrode_pitch_nm=1400.0,
                finger_width_nm=700.0,
                displacement_per_volt_nm=0.0,
                substrate_material=MaterialSpec("La3Ga5SiO14", density_g_cm3=5.75),
            )
        raise ValueError("sample must be LNB or LGS.")

    def resolved_stack(self) -> BareStack:
        """Return the crystalline substrate without a conformal film."""
        return BareStack(substrate_material=self.substrate_material)

    def with_component(self, component: Literal["combined", "deformation", "aluminium"]):
        """Return a diagnostic geometry sharing all physical parameters."""
        if component == "deformation":
            return replace(self, component=component)
        if component == "aluminium":
            return replace(self, component=component, displacement_pv_nm=0.0)
        return replace(self, component=component)

    def at_voltage(self, voltage_v: float) -> PulsePickerGrating:
        """Apply the caller's linear displacement-voltage calibration."""
        if not np.isfinite(voltage_v) or voltage_v < 0:
            raise ValueError("voltage_v must be finite and nonnegative.")
        return replace(
            self,
            voltage_v=float(voltage_v),
            displacement_pv_nm=float(voltage_v) * self.displacement_per_volt_nm,
        )

    def displacement_on(self) -> bool:
        """Return whether the crystal deformation is included."""
        return self.component != "aluminium" and self.displacement_pv_nm > 0.0

    def aluminium_on(self) -> bool:
        """Return whether the electrode material is included."""
        return self.component != "deformation"

    def displacement_nm(self, x_nm: np.ndarray) -> np.ndarray:
        """Return the periodic vertical displacement, shifted to minimum zero."""
        x = np.asarray(x_nm, dtype=float)
        if not self.displacement_on():
            return np.zeros_like(x)
        phase = 2.0 * np.pi * x / self.period_nm
        raw = np.cos(phase)
        for harmonic, coefficient in enumerate(self.displacement_harmonics, start=2):
            raw = raw + coefficient * np.cos(harmonic * phase)
        # Harmonic normalization keeps displacement_pv_nm literal.
        samples = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        shape = np.cos(samples)
        for harmonic, coefficient in enumerate(self.displacement_harmonics, start=2):
            shape = shape + coefficient * np.cos(harmonic * samples)
        return self.displacement_pv_nm * (raw - np.min(shape)) / np.ptp(shape)

    def electrode_mask(self, x_nm: np.ndarray) -> np.ndarray:
        """Return the two Al fingers as a periodic Boolean array."""
        x = np.asarray(x_nm, dtype=float)
        if not self.aluminium_on():
            return np.zeros_like(x, dtype=bool)
        phase = np.mod(x - self.electrode_offset_nm, self.electrode_pitch_nm)
        return phase < self.finger_width_nm

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample one period of the deformed substrate surface."""
        count = max(128, int(np.ceil(self.period_nm / self.x_resolution_nm)))
        x = np.linspace(0.0, self.period_nm, count + 1)
        return x, self.displacement_nm(x)

    def profile_depth_nm(self) -> float:
        """Return the crystalline surface peak-to-valley displacement."""
        return float(self.displacement_pv_nm if self.displacement_on() else 0.0)

    def plot_profile(self, output_filename: str | Path) -> None:
        """Preview electrodes and strain on separate vertical scales."""
        import matplotlib.pyplot as plt

        x = np.linspace(0.0, 3.0 * self.period_nm, 1201)
        fig, (electrodes, deformation) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        if self.aluminium_on():
            for left in np.arange(0.0, 3.0 * self.period_nm, self.electrode_pitch_nm):
                electrodes.fill_between(
                    [left / 1000.0, (left + self.finger_width_nm) / 1000.0],
                    [0.0, 0.0],
                    [self.aluminium_thickness_nm] * 2,
                    color="#a7a7a7",
                )
        electrodes.set(
            title=f"{self.sample} pulse picker — electrodes and deformation (three periods)",
            ylabel="Al thickness (nm)",
            ylim=(0.0, 1.2 * self.aluminium_thickness_nm),
        )
        deformation.plot(x / 1000.0, self.displacement_nm(x), color="#255da4")
        deformation.set(
            xlabel="Lateral position (µm)",
            ylabel="Crystal displacement (nm)",
            ylim=(
                -0.05 * max(self.displacement_pv_nm, 0.1),
                1.2 * max(self.displacement_pv_nm, 0.1),
            ),
        )
        for axis in (electrodes, deformation):
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_filename, dpi=150)
        plt.close(fig)

    def _material_labels(self, *, include_incident_medium: bool) -> list[str]:
        labels = ["Incident Medium"] if include_incident_medium else []
        if self.aluminium_on():
            labels.append("Al")
        labels.append(material_label(self.substrate_material))
        return labels

    def _build_solver_z_grid(self, coating_stack: BareStack) -> np.ndarray:
        del coating_stack
        top = self.profile_depth_nm() + (
            self.aluminium_thickness_nm if self.aluminium_on() else 0.0
        )
        top += 2.0 * self.z_resolution_nm
        count = int(np.ceil(top / self.z_resolution_nm))
        return np.arange(count, -1, -1, dtype=float) * self.z_resolution_nm

    def _refractive_index_row(
        self,
        *,
        z_value: float,
        x_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BareStack,
        photon_energy_ev: float,
        n_inc: complex,
        n_sub: complex,
    ) -> np.ndarray:
        del coating_stack
        row = np.full(x_grid.shape, complex(n_sub), dtype=complex)
        row[z_value >= surface] = complex(n_inc)
        if self.aluminium_on():
            mask = self.electrode_mask(x_grid) & (z_value >= surface)
            mask &= z_value < surface + self.aluminium_thickness_nm
            row[mask] = resolve_refractive_index("Al", photon_energy_ev)
        return row

    def _build_refractive_index_grid(
        self,
        *,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BareStack,
        photon_energy_ev: float,
        n_inc: complex,
    ) -> np.ndarray:
        n_sub = resolve_refractive_index(self.substrate_material, photon_energy_ev)
        return np.asarray(
            [
                self._refractive_index_row(
                    z_value=float(z),
                    x_grid=x_grid,
                    surface=surface,
                    coating_stack=coating_stack,
                    photon_energy_ev=photon_energy_ev,
                    n_inc=n_inc,
                    n_sub=n_sub,
                )
                for z in z_grid
            ]
        )

    def _build_material_code_grid(
        self,
        *,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        surface: np.ndarray,
        coating_stack: BareStack,
        material_codes: dict[str, int],
        include_incident_medium: bool,
    ) -> np.ndarray:
        del coating_stack
        substrate_code = material_codes[material_label(self.substrate_material)]
        incident_code = material_codes.get("Incident Medium", -1) if include_incident_medium else -1
        grid = np.full((z_grid.size, x_grid.size), substrate_code, dtype=int)
        for index, z in enumerate(z_grid):
            grid[index, z >= surface] = incident_code
            if self.aluminium_on():
                mask = self.electrode_mask(x_grid) & (z >= surface)
                mask &= z < surface + self.aluminium_thickness_nm
                grid[index, mask] = material_codes["Al"]
        return grid


@dataclass(frozen=True)
class PulsePickerRockingResult:
    """Rocking intensities relative to incident flux for each grating order."""

    angle_offset_deg: np.ndarray
    orders: np.ndarray
    intensity: np.ndarray
    bragg_center_deg: float
    satellite_spacing_deg: float
    displacement_pv_nm: float
    component: str


@dataclass(frozen=True)
class PulsePickerTimingResult:
    """Effective selection of one bunch and leakage from neighbouring bunches."""

    bunch_times_ns: np.ndarray
    voltage_v: np.ndarray
    selected_order_intensity: np.ndarray
    selected_intensity: float
    neighbour_leakage: float
    rejection_ratio: float


def pulse_picker_rcwa_amplitudes(
    grating: PulsePickerGrating,
    *,
    energy_ev: float = 8000.0,
    grazing_angle_deg: float = 16.45,
    fourier_orders: int = 8,
    polarization: Literal["s", "p"] = "s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complex RCWA reflected/transmitted amplitudes and efficiencies.

    This is a continuous-index optical calculation, not crystalline Bragg
    diffraction.  Returned orders use the solver's Fourier sign convention.
    """
    if energy_ev <= 0 or not np.isfinite(energy_ev):
        raise ValueError("energy_ev must be finite and positive.")
    if not 0 < grazing_angle_deg < 90:
        raise ValueError("grazing_angle_deg must be between 0 and 90 degrees.")
    if fourier_orders < 1:
        raise ValueError("fourier_orders must be positive.")
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be s or p.")
    from .solvers import res0, res1, res2

    textures, profile = grating.build_textures(energy_ev)
    parm = res0(1 if polarization == "s" else -1)
    aa = res1(
        1239.841984 / energy_ev,
        grating.period_nm,
        textures,
        fourier_orders,
        np.cos(np.deg2rad(grazing_angle_deg)),
        parm,
        _fourier_backend="numba",
    )
    result = res2(aa, profile, parm)
    return (
        result.inc_top_reflected.order,
        result.inc_top_reflected.amplitude,
        result.inc_top_transmitted.amplitude,
        result.inc_top_reflected.efficiency,
        result.inc_top_transmitted.efficiency,
    )


def _aluminium_transfer_mask(
    grating: PulsePickerGrating,
    *,
    energy_ev: float,
    grazing_angle_deg: float,
    fourier_orders: int,
    x_nm: np.ndarray,
    polarization: Literal["s", "p"],
) -> np.ndarray:
    """Reconstruct the electrode transmission at the crystal boundary.

    The return pass uses the same local mask by reciprocity. This thin-screen
    approximation is distinct from a full two-pass RCWA crystal calculation.
    """
    if not grating.aluminium_on():
        return np.ones_like(x_nm, dtype=complex)
    al = grating.with_component("aluminium")
    orders, _, transmitted, _, _ = pulse_picker_rcwa_amplitudes(
        al,
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=fourier_orders,
        polarization=polarization,
    )
    bare = replace(grating.with_component("deformation"), displacement_pv_nm=0.0)
    _, _, bare_transmitted, _, _ = pulse_picker_rcwa_amplitudes(
        bare,
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=fourier_orders,
        polarization=polarization,
    )
    reference = bare_transmitted[np.where(orders == 0)[0][0]]
    if abs(reference) < 1e-12:
        raise ValueError("The flat-substrate RCWA transmission vanished.")
    phase = np.exp(2j * np.pi * orders[:, None] * x_nm[None, :] / grating.period_nm)
    return np.sum((transmitted / reference)[:, None] * phase, axis=0)


def _lnb_bragg_amplitude(energy_ev: float, angles_deg: np.ndarray, polarization: str) -> np.ndarray:
    """Evaluate the dynamical 104 reflection with xrt's crystal structure."""
    from xrt.backends.raycing.materials_crystals import LiNbO3

    crystal = LiNbO3(hkl=(1, 0, 4), geom="Bragg reflected")
    s, p = crystal.get_amplitude(
        np.full_like(angles_deg, energy_ev, dtype=float),
        -np.sin(np.deg2rad(angles_deg)),
    )[:2]
    return np.ravel(s if polarization == "s" else p)


def simulate_pulse_picker_rocking_curve(
    grating: PulsePickerGrating,
    *,
    energy_ev: float = 8000.0,
    angle_offset_deg: np.ndarray | None = None,
    max_order: int = 4,
    fourier_orders: int = 8,
    polarization: Literal["s", "p"] = "s",
    energy_bandwidth_fwhm_ev: float | None = None,
    divergence_fwhm_deg: float = 0.0,
    aperture_width_deg: float = 0.0,
    bragg_reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> PulsePickerRockingResult:
    """Calculate a pulse-picker rocking curve using a Bragg kernel and RCWA mask.

    For LGS supply ``bragg_reference=(angle_deg, complex_amplitude)``. The
    default LNB kernel is xrt's dynamical 104 reflectivity. Instrumental
    broadening acts on intensity, after coherent grating diffraction.
    """
    offsets = (
        np.linspace(-0.025, 0.025, 501)
        if angle_offset_deg is None
        else np.asarray(angle_offset_deg, dtype=float)
    )
    if offsets.ndim != 1 or offsets.size < 3 or not np.all(np.diff(offsets) > 0):
        raise ValueError("angle_offset_deg must be a strictly increasing 1D array.")
    if not np.all(np.isfinite(offsets)) or max_order < 0:
        raise ValueError("Invalid rocking angle grid or max_order.")
    if energy_bandwidth_fwhm_ev is None:
        energy_bandwidth_fwhm_ev = 0.85 * energy_ev / 8000.0
    if any(
        v < 0 or not np.isfinite(v)
        for v in (
            energy_bandwidth_fwhm_ev,
            divergence_fwhm_deg,
            aperture_width_deg,
        )
    ):
        raise ValueError("Broadening values must be finite and nonnegative.")
    if grating.sample == "LGS" and bragg_reference is None:
        raise ValueError("LGS rocking curves require a measured complex Bragg reference.")

    wavelength_nm = 1239.841984 / energy_ev
    if grating.sample == "LNB":
        from xrt.backends.raycing.materials_crystals import LiNbO3

        crystal = LiNbO3(hkl=(1, 0, 4), geom="Bragg reflected")
        d_nm = crystal.d * 0.1
        nominal_center = float(np.rad2deg(crystal.get_Bragg_angle(energy_ev)))
        fine = nominal_center + np.linspace(-0.008, 0.008, 1601)
        fine_intensity = np.abs(_lnb_bragg_amplitude(energy_ev, fine, polarization)) ** 2
        center = float(fine[np.argmax(fine_intensity)])
    else:
        reference_angles = np.asarray(bragg_reference[0], dtype=float)
        reference_amplitudes = np.asarray(bragg_reference[1], dtype=complex)
        if reference_angles.ndim != 1 or reference_angles.size < 3:
            raise ValueError("bragg_reference needs at least three angle samples.")
        if reference_angles.shape != reference_amplitudes.shape or not np.all(
            np.diff(reference_angles) > 0
        ):
            raise ValueError("Invalid bragg_reference angle/amplitude arrays.")
        center = float(reference_angles[np.argmax(abs(reference_amplitudes) ** 2)])
        d_nm = wavelength_nm / (2.0 * np.sin(np.deg2rad(center)))

    spacing_deg = float(np.rad2deg(d_nm / grating.period_nm))
    samples = max(1024, 128 * (2 * max_order + 1))
    x_nm = np.arange(samples, dtype=float) * grating.period_nm / samples
    mask = _aluminium_transfer_mask(
        grating,
        energy_ev=energy_ev,
        grazing_angle_deg=center,
        fourier_orders=fourier_orders,
        x_nm=x_nm,
        polarization=polarization,
    )
    q_bragg = 4.0 * np.pi * np.sin(np.deg2rad(center)) / wavelength_nm
    periodic_field = mask**2 * np.exp(1j * q_bragg * grating.displacement_nm(x_nm))
    fourier = np.fft.fft(periodic_field) / samples
    orders = np.arange(-max_order, max_order + 1, dtype=int)
    coefficients = fourier[orders % samples]
    intensities = np.zeros((orders.size, offsets.size), dtype=float)

    if energy_bandwidth_fwhm_ev > 0:
        sigma_ev = energy_bandwidth_fwhm_ev / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        energy_offsets = sigma_ev * np.linspace(-3.0, 3.0, 9)
        weights = np.exp(-0.5 * (energy_offsets / sigma_ev) ** 2)
        weights /= np.sum(weights)
    else:
        energy_offsets = np.array([0.0])
        weights = np.array([1.0])

    for delta_energy, weight in zip(energy_offsets, weights, strict=True):
        sample_energy = energy_ev + float(delta_energy)
        if sample_energy <= 0:
            continue
        for index, order in enumerate(orders):
            rocking_angles = center + offsets - order * spacing_deg
            if grating.sample == "LNB":
                amplitude = _lnb_bragg_amplitude(sample_energy, rocking_angles, polarization)
            else:
                re = np.interp(
                    rocking_angles, reference_angles, reference_amplitudes.real, left=0, right=0
                )
                im = np.interp(
                    rocking_angles, reference_angles, reference_amplitudes.imag, left=0, right=0
                )
                amplitude = re + 1j * im
            intensities[index] += (
                float(weight) * abs(coefficients[index]) ** 2 * abs(amplitude) ** 2
            )

    dx = float(np.median(np.diff(offsets)))
    if divergence_fwhm_deg > 0:
        sigma_points = divergence_fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)) * dx)
        intensities = gaussian_filter1d(intensities, sigma_points, axis=1, mode="constant")
    if aperture_width_deg > 0:
        intensities = uniform_filter1d(
            intensities,
            size=max(1, round(aperture_width_deg / dx)),
            axis=1,
            mode="constant",
        )
    return PulsePickerRockingResult(
        angle_offset_deg=offsets,
        orders=orders,
        intensity=intensities,
        bragg_center_deg=center,
        satellite_spacing_deg=spacing_deg,
        displacement_pv_nm=grating.displacement_pv_nm,
        component=grating.component,
    )


def simulate_pulse_picker_timing(
    voltages_v: np.ndarray,
    order_efficiencies: np.ndarray,
    *,
    peak_voltage_v: float = 9.0,
    gate_fwhm_ns: float = 1.0,
    bunch_spacing_ns: float = 2.0,
    neighbour_count: int = 2,
    pulse_times_ns: np.ndarray | None = None,
    pulse_voltages_v: np.ndarray | None = None,
) -> PulsePickerTimingResult:
    """Interpolate a static order-efficiency curve onto an effective voltage gate."""
    voltage_grid = np.asarray(voltages_v, dtype=float)
    efficiency_grid = np.asarray(order_efficiencies, dtype=float)
    if (
        voltage_grid.ndim != 1
        or voltage_grid.size < 2
        or not np.all(np.isfinite(voltage_grid))
        or np.any(np.diff(voltage_grid) <= 0)
    ):
        raise ValueError("voltages_v must be a strictly increasing array.")
    if (
        efficiency_grid.shape != voltage_grid.shape
        or not np.all(np.isfinite(efficiency_grid))
        or np.any((efficiency_grid < 0) | (efficiency_grid > 1))
    ):
        raise ValueError("order_efficiencies must match voltages and lie in [0, 1].")
    if (
        not all(np.isfinite(value) for value in (peak_voltage_v, gate_fwhm_ns, bunch_spacing_ns))
        or peak_voltage_v < voltage_grid[0]
        or peak_voltage_v > voltage_grid[-1]
        or gate_fwhm_ns <= 0
        or bunch_spacing_ns <= 0
        or neighbour_count < 1
    ):
        raise ValueError("Invalid timing parameters.")
    bunch_times = np.arange(-neighbour_count, neighbour_count + 1) * bunch_spacing_ns
    if pulse_times_ns is None:
        voltage = peak_voltage_v * np.exp(-4.0 * np.log(2.0) * (bunch_times / gate_fwhm_ns) ** 2)
    else:
        times = np.asarray(pulse_times_ns, dtype=float)
        samples = np.asarray(pulse_voltages_v, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or samples.shape != times.shape
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(samples))
            or np.any(np.diff(times) <= 0)
            or np.any((samples < voltage_grid[0]) | (samples > voltage_grid[-1]))
        ):
            raise ValueError("Custom waveform needs matching time and voltage arrays.")
        voltage = np.interp(bunch_times, times, samples, left=samples[0], right=samples[-1])
    selected_order_intensity = np.interp(voltage, voltage_grid, efficiency_grid)
    middle = neighbour_count
    selected = float(selected_order_intensity[middle])
    leakage = float(np.sum(selected_order_intensity) - selected)
    return PulsePickerTimingResult(
        bunch_times_ns=bunch_times,
        voltage_v=voltage,
        selected_order_intensity=selected_order_intensity,
        selected_intensity=selected,
        neighbour_leakage=leakage,
        rejection_ratio=np.inf if leakage == 0 else selected / leakage,
    )
