"""Compare baseline vs optional Numba Fourier backend for multilayer benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import grax as rp
from grax.rcwa_1d import _numba_fourier_available
from grax.simulation._profiling import SolverProfiler

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "comparison_to_other_codes" / "blazed_multilayer"
OPTICAL_CONSTANTS_DIR = EXAMPLE_ROOT / "optical_constants"
REFERENCE_PATH = EXAMPLE_ROOT / "simulation" / "DiffractMod_CrC_d4.8_N60.dat"

ENERGIES_EV = np.arange(500.0, 5000.0 + 1.0, 500.0, dtype=float)
FOURIER_ORDERS = 20
X_RESOLUTION_NM = 0.1
Z_RESOLUTION_NM = 0.1
DIFFRACTION_ORDER = -1
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _load_reference_energy_alpha_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load reference energy-angle pairs from DiffraMod input table."""

    reference_data = pd.read_csv(path, sep=r"\s+", engine="python")
    reference_data = reference_data[["Energy", "alpha"]].copy()
    reference_data = reference_data.apply(pd.to_numeric, errors="coerce").dropna()
    energies = reference_data["Energy"].to_numpy(dtype=float)
    angles = reference_data["alpha"].to_numpy(dtype=float)
    return energies, angles


def _nearest_angle_deg(energy_ev: float, reference_energies: np.ndarray, reference_angles: np.ndarray) -> float:
    """Return the nearest reference grazing angle for one energy."""

    nearest_index = int(np.argmin(np.abs(reference_energies - float(energy_ev))))
    return float(reference_angles[nearest_index])


def _build_multilayer_blazed_grating() -> rp.BlazedGrating:
    """Build the blazed multilayer grating used for comparison workflow."""

    silicon = pd.read_csv(
        OPTICAL_CONSTANTS_DIR / "OC_Si_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    silicon.attrs["name"] = "Si"

    chromium = pd.read_csv(
        OPTICAL_CONSTANTS_DIR / "OC_Cr_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    chromium.attrs["name"] = "Cr"

    carbon = pd.read_csv(
        OPTICAL_CONSTANTS_DIR / "OC_C_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    carbon.attrs["name"] = "C"

    multilayer_stack = rp.MultilayerStack(
        substrate_material=silicon,
        material_a=chromium,
        material_b=carbon,
        d_period_nm=4.8,
        gamma=0.4,
        n_bilayers=60,
        top_material=carbon,
    )

    return rp.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=multilayer_stack,
        x_resolution_nm=X_RESOLUTION_NM,
        z_resolution_nm=Z_RESOLUTION_NM,
    )


def _fourier_stage_seconds(summary: dict[str, object]) -> float:
    """Return exclusive Fourier stage seconds from one profiling summary."""

    for stage in summary["stages"]:
        if stage["stage"] == "fourier_coefficients":
            return float(stage["seconds_exclusive"])
    return 0.0


def _efficiency_for_exact_order(result: rp.SingleSimulationResult, order: int) -> float:
    """Return reflected efficiency for one exact diffraction order value."""

    order_indices = np.where(result.orders == int(order))[0]
    if order_indices.size != 1:
        raise ValueError(f"Diffraction order {order} not found exactly once in result orders.")
    return float(result.efficiency_all[int(order_indices[0])])


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
reference_energies, reference_angles = _load_reference_energy_alpha_pairs(REFERENCE_PATH)
grating = _build_multilayer_blazed_grating()

rows: list[dict[str, float | str]] = []

for energy_ev in tqdm(ENERGIES_EV, desc="Multilayer RCWA comparison", unit="energy"):
    angle_deg = _nearest_angle_deg(float(energy_ev), reference_energies, reference_angles)

    profiler_baseline = SolverProfiler()
    profiler_baseline.enable_memory_tracking()
    baseline_result = rp.run_simulation(
        grating=grating,
        energy_ev=float(energy_ev),
        grazing_angle_deg=angle_deg,
        diffraction_order=abs(DIFFRACTION_ORDER),
        fourier_orders=FOURIER_ORDERS,
        _profiler=profiler_baseline,
        _fourier_backend="baseline",
    )
    summary_baseline = profiler_baseline.summary_dict()

    profiler_numba = SolverProfiler()
    profiler_numba.enable_memory_tracking()
    numba_result = rp.run_simulation(
        grating=grating,
        energy_ev=float(energy_ev),
        grazing_angle_deg=angle_deg,
        diffraction_order=abs(DIFFRACTION_ORDER),
        fourier_orders=FOURIER_ORDERS,
        _profiler=profiler_numba,
        _fourier_backend="numba-optional",
    )
    summary_numba = profiler_numba.summary_dict()

    baseline_total = float(summary_baseline["total_wall_seconds"])
    numba_total = float(summary_numba["total_wall_seconds"])
    baseline_fourier = _fourier_stage_seconds(summary_baseline)
    numba_fourier = _fourier_stage_seconds(summary_numba)
    baseline_peak_mb = float(summary_baseline["peak_memory_bytes"]) / (1024.0 * 1024.0)
    numba_peak_mb = float(summary_numba["peak_memory_bytes"]) / (1024.0 * 1024.0)
    speedup = baseline_total / numba_total if numba_total > 0.0 else 0.0
    baseline_eff_m1 = _efficiency_for_exact_order(baseline_result, order=DIFFRACTION_ORDER)
    numba_eff_m1 = _efficiency_for_exact_order(numba_result, order=DIFFRACTION_ORDER)

    rows.append(
        {
            "energy_ev": float(energy_ev),
            "grazing_angle_deg": float(angle_deg),
            "baseline_actual_backend": str(summary_baseline["metadata"].get("fourier_backend_actual", "baseline")),
            "numba_actual_backend": str(summary_numba["metadata"].get("fourier_backend_actual", "numba-optional")),
            "baseline_total_s": baseline_total,
            "numba_total_s": numba_total,
            "baseline_fourier_s": baseline_fourier,
            "numba_fourier_s": numba_fourier,
            "baseline_peak_mb": baseline_peak_mb,
            "numba_peak_mb": numba_peak_mb,
            "speedup_baseline_over_numba": speedup,
            "baseline_eff_order_m1": baseline_eff_m1,
            "numba_eff_order_m1": numba_eff_m1,
            "eff_delta_order_m1": float(numba_eff_m1 - baseline_eff_m1),
        }
    )

csv_path = OUTPUT_DIR / "multi_energy_multilayer_numba_vs_legacy.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

mean_speedup = float(np.mean([float(row["speedup_baseline_over_numba"]) for row in rows]))
mean_baseline_total = float(np.mean([float(row["baseline_total_s"]) for row in rows]))
mean_numba_total = float(np.mean([float(row["numba_total_s"]) for row in rows]))

lines = [
    "RCWA Multilayer Multi-Energy Fourier Backend Comparison (baseline vs numba-optional)",
    "",
    f"numba_installed={_numba_fourier_available()}",
    f"num_energies={len(ENERGIES_EV)}",
    "energies_ev=500,1000,1500,2000,2500,3000,3500,4000,4500,5000",
    f"fourier_orders={FOURIER_ORDERS}",
    f"x_resolution_nm={X_RESOLUTION_NM}",
    f"z_resolution_nm={Z_RESOLUTION_NM}",
    "angle_strategy=nearest_reference_alpha_from_DiffractMod_CrC_d4.8_N60.dat",
    f"efficiency_order={DIFFRACTION_ORDER}",
    "",
    "aggregates",
    f"- mean_baseline_total_s: {mean_baseline_total:.6f}",
    f"- mean_numba_total_s: {mean_numba_total:.6f}",
    f"- mean_speedup_baseline_over_numba: {mean_speedup:.6f}",
    "",
    "per_energy",
    "energy_ev  grazing_angle_deg  baseline_total_s  numba_total_s  baseline_fourier_s  numba_fourier_s  speedup  eff_delta_order_m1",
]

for row in rows:
    lines.append(
        f"{row['energy_ev']:>8.1f}  "
        f"{row['grazing_angle_deg']:>17.6f}  "
        f"{row['baseline_total_s']:>16.6f}  "
        f"{row['numba_total_s']:>13.6f}  "
        f"{row['baseline_fourier_s']:>18.6f}  "
        f"{row['numba_fourier_s']:>15.6f}  "
        f"{row['speedup_baseline_over_numba']:>7.4f}  "
        f"{row['eff_delta_order_m1']:+.3e}"
    )

if not _numba_fourier_available():
    lines.extend(
        [
            "",
            "note",
            "- numba-optional fell back to baseline because Numba is not installed in this environment.",
        ]
    )

report_path = OUTPUT_DIR / "multi_energy_multilayer_numba_vs_legacy.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved summary: {report_path}")
print(f"Saved CSV: {csv_path}")
