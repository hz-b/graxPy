"""Profile one laminar RCWA simulation and compare internal Fourier backends."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import grax as rp
from grax.simulation._profiling import SolverProfiler
from xrt.backends.raycing import materials as xrt_materials

BACKEND_CHOICES = (
    "compare-numpy-numba",
    "numpy",
    "numba",
)

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")


def _build_grating(*, x_resolution_nm: float, z_resolution_nm: float) -> rp.LaminarGrating:
    """Return the lightweight laminar benchmark grating."""

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


def _backend_run_order(selection: str) -> list[str]:
    """Return the backend run order for one CLI selection."""

    if selection == "compare-numpy-numba":
        return ["numpy", "numba"]
    return [selection]


def _sanitize_backend_name(name: str) -> str:
    """Return a filesystem-safe backend label."""

    return name.replace("-", "_")


def _configure_profiler(
    profiler: SolverProfiler,
    *,
    backend: str,
    energy_ev: float,
    grazing_angle_deg: float,
    fourier_orders: int,
    x_resolution_nm: float,
    z_resolution_nm: float,
) -> None:
    """Attach benchmark metadata to one profiling run."""

    profiler.enable_memory_tracking()
    profiler.set_metadata("benchmark_case", "laminar_single_case")
    profiler.set_metadata("energy_ev", energy_ev)
    profiler.set_metadata("grazing_angle_deg", grazing_angle_deg)
    profiler.set_metadata("fourier_orders", fourier_orders)
    profiler.set_metadata("x_resolution_nm", x_resolution_nm)
    profiler.set_metadata("z_resolution_nm", z_resolution_nm)
    profiler.set_metadata("python_version", sys.version.split()[0])
    profiler.set_metadata("numpy_version", np.__version__)
    profiler.set_metadata("tool_backend_selection", backend)


def _run_backend_case(
    *,
    backend: str,
    energy_ev: float,
    grazing_angle_deg: float,
    fourier_orders: int,
    x_resolution_nm: float,
    z_resolution_nm: float,
) -> tuple[rp.SingleSimulationResult, SolverProfiler]:
    """Run one profiled simulation for one Fourier backend."""

    profiler = SolverProfiler()
    _configure_profiler(
        profiler,
        backend=backend,
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=fourier_orders,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )
    result = rp.run_simulation(
        grating=_build_grating(x_resolution_nm=x_resolution_nm, z_resolution_nm=z_resolution_nm),
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        diffraction_order=1,
        fourier_orders=fourier_orders,
        _profiler=profiler,
        fourier_backend=backend,
    )
    return result, profiler


def _write_report(path: Path, report: str) -> None:
    """Write one benchmark report."""

    path.write_text(report, encoding="utf-8")


def _efficiency_for_exact_order(result: rp.SingleSimulationResult, order: int) -> float:
    """Return reflected efficiency for one exact diffraction order value."""

    order_indices = np.where(result.orders == int(order))[0]
    if order_indices.size != 1:
        raise ValueError(f"Diffraction order {order} not found exactly once in result orders.")
    return float(result.efficiency_all[int(order_indices[0])])


def _comparison_report(
    results: dict[str, rp.SingleSimulationResult],
    profilers: dict[str, SolverProfiler],
) -> str:
    """Return a concise cross-backend comparison summary."""

    baseline_summary = profilers["numpy"].summary_dict()
    baseline_actual = str(baseline_summary["metadata"].get("fourier_backend_actual", "numpy"))
    lines = [
        "RCWA Fourier Backend Comparison",
        "",
        f"baseline_actual_backend={baseline_actual}",
        "",
        "backend                    actual_backend              total_s    fourier_s   peak_mb   speedup_vs_baseline",
        "-----------------------------------------------------------------------------------------------------------",
    ]

    baseline_total = float(baseline_summary["total_wall_seconds"])
    for backend_name, profiler in profilers.items():
        summary = profiler.summary_dict()
        metadata = summary["metadata"]
        actual_backend = str(metadata.get("fourier_backend_actual", backend_name))
        total_seconds = float(summary["total_wall_seconds"])
        peak_mb = float(summary["peak_memory_bytes"]) / (1024.0 * 1024.0)
        fourier_stage = next(
            (
                float(stage["seconds_exclusive"])
                for stage in summary["stages"]
                if stage["stage"] == "fourier_coefficients"
            ),
            0.0,
        )
        speedup = baseline_total / total_seconds if total_seconds > 0.0 else 0.0
        lines.append(
            f"{backend_name:<25} {actual_backend:<25} {total_seconds:>8.6f}  "
            f"{fourier_stage:>8.6f}  {peak_mb:>7.3f}  {speedup:>18.6f}"
        )

    lines.extend(["", "order_minus_1_efficiency_deltas_vs_numpy"])
    numpy_efficiency = _efficiency_for_exact_order(results["numpy"], order=-1)
    for backend_name, result in results.items():
        backend_efficiency = _efficiency_for_exact_order(result, order=-1)
        lines.append(
            f"- {backend_name}: {backend_efficiency - numpy_efficiency:.12e}"
        )
    return "\n".join(lines)


energy_ev = 200.0
grazing_angle_deg = 4.0
fourier_orders = 20
x_resolution_nm = 0.1
z_resolution_nm = 0.1
fourier_backend_selection = "compare-numba-legacy"
output_dir = Path(__file__).resolve().parent / "results"

output_dir.mkdir(parents=True, exist_ok=True)
backend_names = _backend_run_order(fourier_backend_selection)
results_by_backend: dict[str, rp.SingleSimulationResult] = {}
profilers_by_backend: dict[str, SolverProfiler] = {}

for backend_name in backend_names:
    result, profiler = _run_backend_case(
        backend=backend_name,
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=fourier_orders,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )
    results_by_backend[backend_name] = result
    profilers_by_backend[backend_name] = profiler
    report = profiler.format_report()
    report_path = output_dir / f"single_rcwa_profile_report_{_sanitize_backend_name(backend_name)}.txt"
    _write_report(report_path, report)
    if backend_name == backend_names[0]:
        _write_report(output_dir / "single_rcwa_profile_report.txt", report)
    print(f"\n=== backend={backend_name} ===")
    print(report)
    print(f"\nSaved report: {report_path}")
    print(f"Selected efficiency: {result.selected_efficiency:.6g}")

if "numpy" in profilers_by_backend and "numba" in profilers_by_backend:
    comparison = _comparison_report(results_by_backend, profilers_by_backend)
    comparison_path = output_dir / "single_rcwa_profile_comparison_numba_vs_numpy.txt"
    _write_report(comparison_path, comparison)
    print("\n=== comparison=numba-vs-numpy ===")
    print(comparison)
    print(f"\nSaved comparison: {comparison_path}")
