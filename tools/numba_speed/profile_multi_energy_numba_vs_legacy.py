"""Compare RCWA numpy vs numba Fourier backend across many energies."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

import grax as rp
from grax.simulation._profiling import SolverProfiler
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")


def _build_grating(*, x_resolution_nm: float, z_resolution_nm: float) -> rp.LaminarGrating:
    """Return the laminar benchmark grating."""

    return rp.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=28.77,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )


def _fourier_stage_seconds(summary: dict[str, object]) -> float:
    """Return exclusive Fourier stage seconds from one profiling summary."""

    stages = summary["stages"]
    for stage in stages:
        if stage["stage"] == "fourier_coefficients":
            return float(stage["seconds_exclusive"])
    return 0.0


def _efficiency_for_exact_order(result: rp.SingleSimulationResult, order: int) -> float:
    """Return reflected efficiency for one exact diffraction order value."""

    order_indices = np.where(result.orders == int(order))[0]
    if order_indices.size != 1:
        raise ValueError(f"Diffraction order {order} not found exactly once in result orders.")
    return float(result.efficiency_all[int(order_indices[0])])


energy_start_ev = 100.0
energy_stop_ev = 1200.0
num_energies = 10
grazing_angle_deg = 4.0
fourier_orders = 20
x_resolution_nm = 0.1
z_resolution_nm = 0.1
output_dir = Path(__file__).resolve().parent / "results"

output_dir.mkdir(parents=True, exist_ok=True)
energies = np.linspace(energy_start_ev, energy_stop_ev, num_energies, dtype=float)
grating = _build_grating(x_resolution_nm=x_resolution_nm, z_resolution_nm=z_resolution_nm)

rows: list[dict[str, float | str]] = []

for energy_ev in tqdm(energies, desc="Multi-energy RCWA comparison", unit="energy"):
    profiler_numpy = SolverProfiler()
    profiler_numpy.enable_memory_tracking()
    numpy_result = rp.run_simulation(
        grating=grating,
        energy_ev=float(energy_ev),
        grazing_angle_deg=grazing_angle_deg,
        diffraction_order=1,
        fourier_orders=fourier_orders,
        _profiler=profiler_numpy,
        fourier_backend="numpy",
    )
    summary_numpy = profiler_numpy.summary_dict()

    profiler_numba = SolverProfiler()
    profiler_numba.enable_memory_tracking()
    numba_result = rp.run_simulation(
        grating=grating,
        energy_ev=float(energy_ev),
        grazing_angle_deg=grazing_angle_deg,
        diffraction_order=1,
        fourier_orders=fourier_orders,
        _profiler=profiler_numba,
        fourier_backend="numba",
    )
    summary_numba = profiler_numba.summary_dict()

    numpy_total = float(summary_numpy["total_wall_seconds"])
    numba_total = float(summary_numba["total_wall_seconds"])
    baseline_fourier = _fourier_stage_seconds(summary_baseline)
    numba_fourier = _fourier_stage_seconds(summary_numba)
    baseline_peak_mb = float(summary_baseline["peak_memory_bytes"]) / (1024.0 * 1024.0)
    numba_peak_mb = float(summary_numba["peak_memory_bytes"]) / (1024.0 * 1024.0)
    speedup = baseline_total / numba_total if numba_total > 0.0 else 0.0
    baseline_eff_m1 = _efficiency_for_exact_order(baseline_result, order=-1)
    numba_eff_m1 = _efficiency_for_exact_order(numba_result, order=-1)

    rows.append(
        {
            "energy_ev": float(energy_ev),
            "numpy_actual_backend": str(summary_numpy["metadata"].get("fourier_backend_actual", "numpy")),
            "numba_actual_backend": str(summary_numba["metadata"].get("fourier_backend_actual", "numba")),
            "numpy_total_s": numpy_total,
            "numba_total_s": numba_total,
            "numpy_fourier_s": _fourier_stage_seconds(summary_numpy),
            "numba_fourier_s": numba_fourier,
            "numpy_peak_mb": float(summary_numpy["peak_memory_bytes"]) / (1024.0 * 1024.0),
            "numba_peak_mb": numba_peak_mb,
            "speedup_numpy_over_numba": speedup,
            "numpy_eff_order_m1": _efficiency_for_exact_order(numpy_result, order=-1),
            "numba_eff_order_m1": numba_eff_m1,
            "eff_delta_order_m1": float(numba_eff_m1 - _efficiency_for_exact_order(numpy_result, order=-1)),
        }
    )

csv_path = output_dir / "multi_energy_numba_vs_numpy.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

mean_speedup = float(np.mean([float(row["speedup_baseline_over_numba"]) for row in rows]))
mean_baseline_total = float(np.mean([float(row["baseline_total_s"]) for row in rows]))
mean_numba_total = float(np.mean([float(row["numba_total_s"]) for row in rows]))

lines = [
    "RCWA Multi-Energy Fourier Backend Comparison (numpy vs numba)",
    "",
    f"num_energies={num_energies}",
    f"energy_start_ev={energy_start_ev}",
    f"energy_stop_ev={energy_stop_ev}",
    f"grazing_angle_deg={grazing_angle_deg}",
    f"fourier_orders={fourier_orders}",
    f"x_resolution_nm={x_resolution_nm}",
    f"z_resolution_nm={z_resolution_nm}",
    "",
    "aggregates",
    f"- mean_numpy_total_s: {mean_numpy_total_s:.6f}",
    f"- mean_numba_total_s: {mean_numba_total_s:.6f}",
    f"- mean_speedup_numpy_over_numba: {mean_speedup:.6f}",
    "",
    "per_energy",
    "energy_ev  numpy_total_s  numba_total_s  numpy_fourier_s  numba_fourier_s  speedup  eff_delta_order_m1",
]

for row in rows:
    lines.append(
        f"{row['energy_ev']:>8.3f}  "
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

report_path = output_dir / "multi_energy_numba_vs_legacy.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved summary: {report_path}")
print(f"Saved CSV: {csv_path}")
