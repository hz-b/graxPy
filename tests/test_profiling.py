from __future__ import annotations

import inspect

import numpy as np
import pytest

from grax.gratings import LaminarGrating
from grax.rcwa_1d import res0, res1
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


def test_run_simulation_signature_hides_profiler() -> None:
    signature = inspect.signature(run_simulation)

    assert "_profiler" not in signature.parameters


def test_run_simulation_rejects_invalid_memory_mode() -> None:
    with pytest.raises(ValueError, match="memory_mode must be 'standard' or 'low_memory'"):
        run_simulation(
            grating=_grating(),
            energy_ev=200.0,
            grazing_angle_deg=4.0,
            fourier_orders=5,
            memory_mode="invalid",  # type: ignore[arg-type]
        )


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
    assert counts["layer_boundary_blocks_constructed"] > 0
    assert timings["fourier_exp"]["calls"] > 0
    assert timings["fourier_sum"]["calls"] > 0
    assert timings["layer_eigensolve_call"]["calls"] > 0
    assert unique_counts["layer_operator_unique"] > 0
    assert summary["peak_memory_bytes"] >= 0
    assert details["peaks"]["layer_boundary_block_temp_peak"] == pytest.approx(1.0)
    assert details["peaks"]["layer_boundary_block_bytes_peak"] > 0.0
    assert summary["derived_kpis"]["time_per_fourier_call_seconds"] > 0.0
    assert summary["derived_kpis"]["time_per_harmonic_seconds"] > 0.0


def test_low_memory_mode_matches_standard_result() -> None:
    standard = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        memory_mode="standard",
    )
    low_memory = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        memory_mode="low_memory",
    )

    assert low_memory.selected_efficiency == pytest.approx(standard.selected_efficiency, rel=1e-10, abs=1e-12)
    assert np.allclose(low_memory.efficiency_all, standard.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(low_memory.diffraction_angle_all, standard.diffraction_angle_all, rtol=1e-10, atol=1e-12)


def test_low_memory_build_textures_skips_dense_index_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    grating = _grating()

    def fail_dense_grid(**kwargs: object) -> np.ndarray:
        raise AssertionError("dense index grid should not be built")

    monkeypatch.setattr(grating, "_build_refractive_index_grid", fail_dense_grid)
    textures, profile = grating.build_textures(200.0, memory_mode="low_memory")

    assert len(textures) > 0
    assert len(profile[0]) == len(profile[1])


def test_low_memory_build_textures_compresses_consecutive_rows() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=1.0,
        z_resolution_nm=0.02,
    )
    standard_textures, standard_profile = grating.build_textures(200.0, memory_mode="standard")
    low_memory_textures, low_memory_profile = grating.build_textures(200.0, memory_mode="low_memory")

    assert len(low_memory_textures) <= len(standard_textures)
    assert len(low_memory_profile[0]) < len(standard_profile[0])


def test_res1_texture_conversion_cache_reuses_repeat_signatures() -> None:
    profiler = SolverProfiler()
    textures = [
        1.0 + 0.0j,
        1.0 + 0.0j,
        [np.asarray([100.0], dtype=float), np.asarray([1.0 + 0.0j], dtype=complex)],
        [np.asarray([100.0], dtype=float), np.asarray([1.0 + 0.0j], dtype=complex)],
    ]

    result = res1(
        wavelength=1239.8 / 200.0,
        period=2500.0,
        textures=textures,
        nn=5,
        beta0=0.5,
        parm=res0(1),
        _profiler=profiler,
    )
    summary = profiler.summary_dict()

    assert len(result.textures) == len(textures)
    assert result.textures[0] is result.textures[1]
    assert result.textures[2] is result.textures[3]
    assert summary["metadata"]["texture_conversion_cache_hits"] >= 2
    assert summary["metadata"]["texture_conversion_cache_misses"] == 2
    assert summary["details"]["counts"]["texture_conversion_cache_hits"] >= 2
    assert summary["details"]["counts"]["texture_conversion_cache_misses"] == 2


def test_low_memory_mode_reduces_peak_memory_for_fine_z_case() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=1.0,
        z_resolution_nm=0.02,
    )
    standard_profiler = SolverProfiler()
    standard_profiler.enable_memory_tracking()
    run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        memory_mode="standard",
        _profiler=standard_profiler,
    )
    low_memory_profiler = SolverProfiler()
    low_memory_profiler.enable_memory_tracking()
    run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        memory_mode="low_memory",
        _profiler=low_memory_profiler,
    )

    assert low_memory_profiler.summary_dict()["peak_memory_bytes"] < standard_profiler.summary_dict()["peak_memory_bytes"]


def test_incremental_cascade_matches_standard_mode_result_for_many_layers() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=1.0,
        z_resolution_nm=0.02,
    )
    baseline = run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        memory_mode="standard",
    )
    profiled = SolverProfiler()
    profiled.enable_memory_tracking()
    candidate = run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        memory_mode="standard",
        _profiler=profiled,
    )
    summary = profiled.summary_dict()

    assert candidate.selected_efficiency == pytest.approx(baseline.selected_efficiency, rel=1e-10, abs=1e-12)
    assert np.allclose(candidate.efficiency_all, baseline.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(candidate.diffraction_angle_all, baseline.diffraction_angle_all, rtol=1e-10, atol=1e-12)
    assert summary["details"]["counts"]["layer_boundary_blocks_constructed"] > 0
    assert summary["details"]["peaks"]["layer_boundary_block_temp_peak"] == pytest.approx(1.0)


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
