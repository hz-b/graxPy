"""Reproduce the LNB pulse-picker geometry and compare with rasterized Fig. 4.

Run from the repository root with ``python examples/pulse_picker/paper_lnb.py``.
The fitted deformation is a parameter of the thin-screen approximation, not
a measured piezoelectric displacement or an electrical calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from grax.pulse_picker import (
    PulsePickerGrating,
    pulse_picker_rcwa_amplitudes,
    simulate_pulse_picker_rocking_curve,
    simulate_pulse_picker_timing,
)

ROOT = Path(__file__).resolve().parents[2]


def _digitized(voltage: int) -> tuple[np.ndarray, np.ndarray]:
    with (ROOT / "data/pulse_picker/fig4_digitized.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["voltage_v"]) == voltage]
    return (
        np.array([float(row["angle_offset_deg"]) for row in rows]),
        np.array([float(row["normalized_intensity"]) for row in rows]),
    )


def _geometry_figure(grating: PulsePickerGrating, output: Path) -> None:
    x = np.linspace(0, grating.period_nm, 1200)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for left in (0, grating.electrode_pitch_nm):
        top.fill_between(
            [left / 1000, (left + grating.finger_width_nm) / 1000],
            [0, 0],
            [grating.aluminium_thickness_nm, grating.aluminium_thickness_nm],
            color="#8a8a8a",
            alpha=0.7,
        )
    top.set(
        ylabel="Al thickness (nm)",
        ylim=(0, 125),
        title="LNB cell: 4 µm deformation period, two 1 µm Al fingers",
    )
    bottom.plot(x / 1000, grating.displacement_nm(x), color="#2455a4")
    bottom.set(
        xlabel="Position in cell (µm)",
        ylabel="Displacement (nm)",
        ylim=(-0.015, max(0.15, grating.displacement_pv_nm * 1.15)),
    )
    for ax in (top, bottom):
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    """Fit Fig. 4, run the diagnostic RCWA cases, and write all figures/data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--harmonics",
        default="",
        help="Comma-separated cosine coefficients for the 2nd, 3rd, ... displacement harmonics",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    harmonics = tuple(float(value) for value in args.harmonics.split(",") if value.strip())
    grating = replace(PulsePickerGrating.paper_preset("LNB"), displacement_harmonics=harmonics)
    grid = np.linspace(-0.02, 0.02, 401)
    kwargs = {"angle_offset_deg": grid, "max_order": 5, "fourier_orders": 8}
    x0, y0 = _digitized(0)
    x9, y9 = _digitized(9)

    # Fit the effective instrumental bandwidth and raster angle origin on OFF.
    # This is not a claimed monochromator bandwidth measurement.
    candidates = []
    for bandwidth in np.linspace(0.5, 1.1, 13):
        off_candidate = simulate_pulse_picker_rocking_curve(
            grating,
            energy_bandwidth_fwhm_ev=float(bandwidth),
            **kwargs,
        )
        normalized = off_candidate.intensity.sum(axis=0)
        normalized /= normalized.max()
        shift_fit = minimize_scalar(
            lambda shift, curve=normalized: np.mean((np.interp(x0 - shift, grid, curve) - y0) ** 2),
            bounds=(-0.003, 0.003),
            method="bounded",
        )
        candidates.append((float(shift_fit.fun), float(bandwidth), float(shift_fit.x)))
    _, bandwidth, shift = min(candidates)
    kwargs["energy_bandwidth_fwhm_ev"] = bandwidth
    off = simulate_pulse_picker_rocking_curve(grating, **kwargs)
    off_total = off.intensity.sum(axis=0)
    scale = off_total.max()

    def on_prediction(displacement_nm: float) -> np.ndarray:
        candidate = replace(grating, displacement_pv_nm=displacement_nm)
        result = simulate_pulse_picker_rocking_curve(candidate, **kwargs)
        return result.intensity.sum(axis=0) / scale

    fit = minimize_scalar(
        lambda displacement: np.mean(
            (np.interp(x9 - shift, grid, on_prediction(displacement)) - y9) ** 2
        ),
        bounds=(0.0, 0.35),
        method="bounded",
        options={"xatol": 0.001},
    )
    h9 = float(fit.x)
    calibrated = replace(
        grating, displacement_per_volt_nm=h9 / 9, displacement_pv_nm=h9, voltage_v=9
    )
    on = simulate_pulse_picker_rocking_curve(calibrated, **kwargs)
    on_total = on.intensity.sum(axis=0) / scale
    off_normalized = off_total / scale
    off_residual = np.interp(x0 - shift, grid, off_normalized) - y0
    on_residual = np.interp(x9 - shift, grid, on_total) - y9

    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(grid + shift, off_normalized, color="#1557bd", label="0 V model")
    ax.plot(grid + shift, on_total, color="#5f9f39", label="9 V model")
    ax.scatter(x0, y0, s=9, color="#1557bd", alpha=0.38, label="Fig. 4 digitized: 0 V")
    ax.scatter(x9, y9, s=9, color="#5f9f39", alpha=0.38, label="Fig. 4 digitized: 9 V")
    ax.set(ylabel="Normalized intensity", title="Fig. 4 comparison — Bragg + periodic-screen model")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    residual_ax.plot(x0, off_residual, color="#1557bd", lw=1)
    residual_ax.plot(x9, on_residual, color="#5f9f39", lw=1)
    residual_ax.axhline(0, color="black", lw=0.7)
    residual_ax.set(xlabel="Δθ (deg)", ylabel="Model − data")
    residual_ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "fig4_comparison.png", dpi=170)
    plt.close(fig)

    with (output / "rocking_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "angle_offset_deg",
                "off_normalized",
                "on_normalized",
                "off_incident_fraction",
                "on_incident_fraction",
            )
        )
        for values in zip(
            grid + shift, off_normalized, on_total, off_total, on_total * scale, strict=True
        ):
            writer.writerow(values)
    with (output / "digitized_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "voltage_v",
                "angle_offset_deg",
                "digitized_normalized",
                "model_normalized",
                "residual",
            )
        )
        for voltage, angles, data, residual in (
            (0, x0, y0, off_residual),
            (9, x9, y9, on_residual),
        ):
            for angle, datum, error in zip(angles, data, residual, strict=True):
                writer.writerow((voltage, angle, datum, datum + error, error))

    _geometry_figure(calibrated, output / "geometry.png")
    rcwa = {}
    for component in ("deformation", "aluminium", "combined"):
        order, _, _, reflected, transmitted = pulse_picker_rcwa_amplitudes(
            calibrated.with_component(component),
            fourier_orders=8,
        )
        rcwa[component] = (order, reflected, transmitted)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for component, (order, reflected, transmitted) in rcwa.items():
        axes[0].semilogy(order, np.maximum(reflected, 1e-15), marker="o", label=component)
        axes[1].semilogy(order, np.maximum(transmitted, 1e-15), marker="o", label=component)
    for ax, title in zip(axes, ("Reflected", "Transmitted"), strict=True):
        ax.set(xlabel="Order in 4 µm cell", ylabel="RCWA efficiency", title=title)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Continuous-index RCWA diagnostic (not atomic Bragg diffraction)")
    fig.tight_layout()
    fig.savefig(output / "rcwa_diagnostic.png", dpi=170)
    plt.close(fig)
    with (output / "rcwa_orders.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("component", "order", "reflected_efficiency", "transmitted_efficiency"))
        for component, (order, reflected, transmitted) in rcwa.items():
            writer.writerows(
                zip([component] * len(order), order, reflected, transmitted, strict=True)
            )

    voltage_grid = np.linspace(0, 9, 19)
    efficiencies = []
    for voltage in voltage_grid:
        curve = simulate_pulse_picker_rocking_curve(calibrated.at_voltage(float(voltage)), **kwargs)
        efficiencies.append(float(np.max(curve.intensity[curve.orders == 1])))
    timing = simulate_pulse_picker_timing(voltage_grid, np.asarray(efficiencies))
    with (output / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("bunch_time_ns", "effective_voltage_v", "selected_order_intensity"))
        writer.writerows(
            zip(
                timing.bunch_times_ns,
                timing.voltage_v,
                timing.selected_order_intensity,
                strict=True,
            )
        )

    peak_windows = {
        "first_negative": (-0.0055, -0.0030),
        "central": (-0.0018, 0.0005),
        "first_positive": (0.0025, 0.0055),
    }
    peak_metrics = {}
    for name, (left, right) in peak_windows.items():
        measured = (x9 >= left) & (x9 <= right)
        simulated = (grid + shift >= left) & (grid + shift <= right)
        measured_peak = float(np.max(y9[measured]))
        model_peak = float(np.max(on_total[simulated]))
        peak_metrics[name] = {
            "digitized_9v": measured_peak,
            "model_9v": model_peak,
            "model_minus_digitized": model_peak - measured_peak,
        }

    summary = {
        "branch_model": "xrt LiNbO3 104 Bragg kernel + RCWA-derived Al thin screen",
        "displacement_harmonics": harmonics,
        "experimental_data": "raster-digitized Fig. 4, not raw measurements",
        "fitted_9v_displacement_peak_to_valley_nm": h9,
        "fitted_displacement_per_volt_nm": h9 / 9,
        "fitted_effective_bandwidth_fwhm_ev": bandwidth,
        "fitted_angle_origin_shift_deg": shift,
        "off_rms_normalized": float(np.sqrt(np.mean(off_residual**2))),
        "on_rms_normalized": float(np.sqrt(np.mean(on_residual**2))),
        "on_peak_windows": peak_metrics,
        "full_fig4_reproduction": False,
        "limitation": "Pure sinusoidal thin-screen model underestimates ON satellites.",
        "satellite_spacing_deg": off.satellite_spacing_deg,
        "rcwa_energy_totals": {
            component: {"R": float(sum(values[1])), "T": float(sum(values[2]))}
            for component, values in rcwa.items()
        },
        "timing": {
            "gate_fwhm_ns": 1.0,
            "bunch_spacing_ns": 2.0,
            "selected_intensity": timing.selected_intensity,
            "neighbour_leakage": timing.neighbour_leakage,
            "rejection_ratio": timing.rejection_ratio,
            "note": "Effective Gaussian gate; not a measured electrode voltage trace.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
