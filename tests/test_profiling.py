from __future__ import annotations

import numpy as np
import pytest

from grax.gratings import LaminarGrating
from grax.simulation import run_simulation
from grax.simulation._profiling import SolverProfiler
from tests.optical_constants import load_optical_constants_table
from pathlib import Path

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[1] / "comparison_to_other_codes" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")


def _grating() -> LaminarGrating:
    return LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=2.0,
        z_resolution_nm=0.2,
    )


def test_profiler_disabled_does_not_change_result() -> None:
    baseline = run_simulation(grating=_grating(), energy_ev=200.0, grazing_angle_deg=4.0, fourier_orders=5)
    profiled = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _profiler=SolverProfiler(),
    )

    assert baseline.selected_efficiency == profiled.selected_efficiency
    assert np.allclose(baseline.efficiency_all, profiled.efficiency_all)
    assert np.allclose(baseline.diffraction_angle_all, profiled.diffraction_angle_all)


def test_profiler_summary_contains_expected_stages() -> None:
    profiler = SolverProfiler()
    run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _profiler=profiler,
    )
    summary = profiler.summary_dict()
    stage_names = {row["stage"] for row in summary["stages"]}

    assert summary["total_wall_seconds"] >= 0.0
    assert summary["profiled_exclusive_seconds"] >= 0.0
    assert summary["unprofiled_seconds"] >= 0.0
    assert "metadata" in summary
    assert "derived_kpis" in summary
    assert "texture_generation" in stage_names
    assert "res1_total" in stage_names
    assert "fourier_coefficients" in stage_names
    assert "res2_total" in stage_names
    assert "layer_propagation_cascade" in stage_names
    assert "matrix_solves" in stage_names
    assert "postprocessing" in stage_names
    assert "details" in summary


def test_profiler_report_handles_empty_stage_set() -> None:
    profiler = SolverProfiler()
    report = profiler.format_report()

    assert "RCWA Profiling Summary" in report
    assert "(no recorded stages)" in report


def test_profiler_exclusive_percentages_are_normalized() -> None:
    profiler = SolverProfiler()
    run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _profiler=profiler,
    )
    summary = profiler.summary_dict()
    percent_sum = sum(float(row["percent_exclusive"]) for row in summary["stages"])

    assert 99.0 <= percent_sum <= 101.0


def test_profiler_details_include_fourier_and_eigensolve_diagnostics() -> None:
    profiler = SolverProfiler()
    profiler.enable_memory_tracking()
    run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _profiler=profiler,
    )
    summary = profiler.summary_dict()
    details = summary["details"]
    counts = details["counts"]
    timings = details["timings"]
    unique_counts = details["unique_counts"]

    assert counts["fourier_calls"] > 0
    assert counts["fourier_loop_iterations"] > 0
    assert counts["layer_operator_calls"] > 0
    assert counts["layer_eigensolve_cache_misses"] >= 0
    assert timings["fourier_exp"]["calls"] > 0
    assert timings["fourier_sum"]["calls"] > 0
    assert timings["layer_eigensolve_call"]["calls"] > 0
    assert unique_counts["layer_operator_unique"] > 0
    assert summary["peak_memory_bytes"] >= 0
    assert summary["derived_kpis"]["time_per_fourier_call_seconds"] > 0.0
    assert summary["derived_kpis"]["time_per_harmonic_seconds"] > 0.0


@pytest.mark.parametrize("backend_name", ["numba"])
def test_fourier_backend_matches_baseline(backend_name: str) -> None:
    baseline = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        backend="numpy",
    )
    candidate = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        backend=backend_name,
    )

    result = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        backend=backend_name,
    )
    baseline = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        backend="numpy",
    )

    assert result.selected_efficiency == pytest.approx(baseline.selected_efficiency, rel=1e-10, abs=1e-12)
    assert np.allclose(result.efficiency_all, baseline.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(result.diffraction_angle_all, baseline.diffraction_angle_all, rtol=1e-10, atol=1e-12)



