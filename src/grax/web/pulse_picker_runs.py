"""Saved web calculations for the piezoelectric pulse picker."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from grax.pulse_picker import (
    PulsePickerGrating,
    pulse_picker_rcwa_amplitudes,
    simulate_pulse_picker_rocking_curve,
)


def run_pulse_picker_web(
    grating: PulsePickerGrating,
    workflow: str,
    form: Any,
    output_dir: Path,
) -> list[str]:
    """Run one pulse-picker workflow and return generated artifact names."""
    if workflow not in {"pulse_picker_rocking", "pulse_picker_rcwa"}:
        raise ValueError("Unsupported pulse-picker workflow.")
    energy = float(form.get("energy_ev", 8000.0))
    voltage = float(form.get("voltage_v", 9.0))
    response = float(form.get("displacement_per_volt_nm", grating.displacement_per_volt_nm))
    orders = int(form.get("fourier_orders", 8))
    max_order = int(form.get("max_order", 4))
    polarization = str(form.get("polarization", "s"))
    if not np.isfinite(energy) or energy <= 0 or not np.isfinite(voltage) or voltage < 0:
        raise ValueError("Energy must be positive and voltage nonnegative.")
    if not np.isfinite(response) or response < 0 or not 2 <= orders <= 30:
        raise ValueError("Invalid displacement calibration or Fourier order count.")
    if not 1 <= max_order <= 8 or polarization not in {"s", "p"}:
        raise ValueError("Invalid satellite order or polarization.")
    if workflow == "pulse_picker_rocking" and grating.sample == "LGS":
        raise ValueError("LGS rocking curves require an external complex Bragg reference.")
    on = replace(
        grating,
        voltage_v=voltage,
        displacement_per_volt_nm=response,
        displacement_pv_nm=voltage * response,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if workflow == "pulse_picker_rcwa":
        return _run_rcwa(on, energy, orders, polarization, output_dir)
    half_width = float(form.get("angle_half_width_deg", 0.02))
    points = int(form.get("angle_points", 401))
    bandwidth = float(form.get("energy_bandwidth_fwhm_ev", 0.85 * energy / 8000.0))
    divergence = float(form.get("divergence_fwhm_deg", 0.0))
    aperture = float(form.get("aperture_width_deg", 0.0))
    if not np.isfinite(half_width) or not 0.001 <= half_width <= 0.2:
        raise ValueError("Angle half-width must be between 0.001 and 0.2 degrees.")
    if not 51 <= points <= 2001:
        raise ValueError("Angle sample count must be between 51 and 2001.")
    if any(not np.isfinite(v) or v < 0 for v in (bandwidth, divergence, aperture)):
        raise ValueError("Instrument widths must be finite and nonnegative.")
    return _run_rocking(
        on,
        energy,
        orders,
        max_order,
        polarization,
        half_width,
        points,
        bandwidth,
        divergence,
        aperture,
        output_dir,
    )


def _run_rcwa(
    grating: PulsePickerGrating,
    energy: float,
    fourier_orders: int,
    polarization: str,
    output_dir: Path,
) -> list[str]:
    results = {}
    for component in ("deformation", "aluminium", "combined"):
        diagnostic = grating.with_component(component)
        order, reflection, transmission, efficiency_r, efficiency_t = pulse_picker_rcwa_amplitudes(
            diagnostic,
            energy_ev=energy,
            grazing_angle_deg=16.45,
            fourier_orders=fourier_orders,
            polarization=polarization,
        )
        results[component] = (order, reflection, transmission, efficiency_r, efficiency_t)
    with (output_dir / "rcwa_orders.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "component",
                "order",
                "reflected_efficiency",
                "transmitted_efficiency",
                "reflected_amplitude_real",
                "reflected_amplitude_imag",
                "transmitted_amplitude_real",
                "transmitted_amplitude_imag",
            )
        )
        for component, (
            order,
            reflection,
            transmission,
            efficiency_r,
            efficiency_t,
        ) in results.items():
            for row in zip(
                order, reflection, transmission, efficiency_r, efficiency_t, strict=True
            ):
                m, r, t, er, et = row
                writer.writerow((component, int(m), er, et, r.real, r.imag, t.real, t.imag))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for component, (order, _, _, efficiency_r, efficiency_t) in results.items():
        axes[0].semilogy(order, np.maximum(efficiency_r, 1e-15), marker="o", label=component)
        axes[1].semilogy(order, np.maximum(efficiency_t, 1e-15), marker="o", label=component)
    for ax, title in zip(axes, ("Reflected", "Transmitted"), strict=True):
        ax.set(xlabel="Order in 4 µm cell", ylabel="RCWA efficiency", title=title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Continuous-index RCWA at {energy:g} eV — not atomic Bragg diffraction")
    fig.tight_layout()
    fig.savefig(output_dir / "rcwa_orders.png", dpi=160)
    plt.close(fig)
    totals = {
        key: {"reflected": float(np.sum(value[3])), "transmitted": float(np.sum(value[4]))}
        for key, value in results.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"model": "continuous-index RCWA", "totals": totals}, indent=2) + "\n",
        encoding="utf-8",
    )
    return ["rcwa_orders.png", "rcwa_orders.csv", "summary.json"]


def _run_rocking(
    grating: PulsePickerGrating,
    energy: float,
    fourier_orders: int,
    max_order: int,
    polarization: str,
    half_width: float,
    points: int,
    bandwidth: float,
    divergence: float,
    aperture: float,
    output_dir: Path,
) -> list[str]:
    grid = np.linspace(-half_width, half_width, points)
    kwargs = {
        "energy_ev": energy,
        "angle_offset_deg": grid,
        "max_order": max_order,
        "fourier_orders": fourier_orders,
        "polarization": polarization,
        "energy_bandwidth_fwhm_ev": bandwidth,
        "divergence_fwhm_deg": divergence,
        "aperture_width_deg": aperture,
    }
    off = replace(grating, displacement_pv_nm=0.0, voltage_v=0.0)
    off_curve = simulate_pulse_picker_rocking_curve(off, **kwargs)
    on_curve = simulate_pulse_picker_rocking_curve(grating, **kwargs)
    off_total = np.sum(off_curve.intensity, axis=0)
    on_total = np.sum(on_curve.intensity, axis=0)
    normalizer = max(float(np.max(off_total)), 1e-20)
    with (output_dir / "rocking_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "angle_offset_deg",
                "off_incident_fraction",
                "on_incident_fraction",
                "off_normalized",
                "on_normalized",
            )
        )
        for angle, off_value, on_value in zip(grid, off_total, on_total, strict=True):
            writer.writerow(
                (angle, off_value, on_value, off_value / normalizer, on_value / normalizer)
            )
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    # The paper's angular origin and xrt's Bragg centre are not the same. This
    # nuisance shift was fitted on the rasterized OFF curve in the example.
    paper_origin_shift_deg = -0.00188 if energy == 8000.0 else 0.0
    ax.plot(grid + paper_origin_shift_deg, off_total / normalizer, label="0 V — Bragg + Al", lw=2)
    ax.plot(
        grid + paper_origin_shift_deg,
        on_total / normalizer,
        label=f"{grating.voltage_v:g} V — Bragg + phase + Al",
        lw=2,
    )
    digitized = files("grax").joinpath("pulse_picker_data/fig4_digitized.csv")
    if digitized.is_file() and energy == 8000.0 and grating.sample == "LNB":
        with digitized.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for voltage, label in ((0, "Fig. 4 digitized, 0 V"), (9, "Fig. 4 digitized, 9 V")):
            if voltage == 9 and grating.voltage_v != 9:
                continue
            selected = [row for row in rows if int(row["voltage_v"]) == voltage]
            measured_x = np.asarray([float(row["angle_offset_deg"]) for row in selected])
            measured_y = np.asarray([float(row["normalized_intensity"]) for row in selected])
            prediction = np.interp(
                measured_x - paper_origin_shift_deg,
                grid,
                off_total / normalizer if voltage == 0 else on_total / normalizer,
            )
            ax.scatter(
                measured_x,
                measured_y,
                s=7,
                alpha=0.38,
                label=label,
            )
            residual_ax.plot(measured_x, prediction - measured_y, lw=1, label=f"{voltage} V")
    ax.set_xlabel("Δθ from Bragg centre (deg)")
    ax.set_ylabel("Intensity / calculated OFF peak")
    ax.set_title("Pulse picker rocking curve — crystalline Bragg model")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    residual_ax.axhline(0, color="black", lw=0.7)
    residual_ax.set(xlabel="Δθ from paper origin (deg)", ylabel="Model − data")
    residual_ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "rocking_curve.png", dpi=160)
    plt.close(fig)
    summary = {
        "model": "dynamical Bragg amplitude + periodic thin screen",
        "energy_ev": energy,
        "voltage_v": grating.voltage_v,
        "displacement_pv_nm": grating.displacement_pv_nm,
        "bragg_center_deg": off_curve.bragg_center_deg,
        "satellite_spacing_deg": off_curve.satellite_spacing_deg,
        "off_peak_incident_fraction": normalizer,
        "on_peak_incident_fraction": float(np.max(on_total)),
        "note": "Fig. 4 points are raster-digitized, not raw measurement data.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return ["rocking_curve.png", "rocking_curve.csv", "summary.json"]
