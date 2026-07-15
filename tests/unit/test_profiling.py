from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import grax
from grax.gratings import BaseGrating, LaminarGrating
from grax import rcwa_1d
from grax.rcwa_1d import res0, res1
from grax.simulation import core as simulation_core_module
from grax.simulation.batch import BatchSimulationRunner
from grax.simulation import run_simulation
from grax.simulation._profiling import SolverProfiler
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[2] / "validation" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")
PROFILING_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "profiling"
    / "profile_blazed_multilayer_case.py"
)
PROFILING_COMPARE_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "profiling"
    / "compare_blazed_multilayer_profiles.py"
)


def _grating() -> LaminarGrating:
    return LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=2.0,
        z_resolution_nm=0.2,
    )


class _FastGrating(BaseGrating):
    """Minimal grating for run_simulation logging tests."""

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a trivial one-period profile."""
        return np.asarray([0.0, self.period_nm]), np.asarray([0.0, 0.0])

    def profile_depth_nm(self) -> float:
        """Return zero profile depth."""
        return 0.0

    def build_textures(
        self,
        photon_energy_ev: float,
        *,
        n_inc: complex = 1.0 + 0.0j,
        _memory_mode: str = "low_memory",
    ) -> tuple[list[object], tuple[np.ndarray, np.ndarray]]:
        """Return a tiny texture/profile pair without material interpolation."""
        del photon_energy_ev, n_inc, _memory_mode
        return [1.0 + 0.0j], (np.asarray([1.0]), np.asarray([1]))


def _patch_fast_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the expensive RCWA stages with deterministic tiny stand-ins."""

    def fake_res1(*args: object, **kwargs: object) -> object:
        """Return a placeholder intermediate solver result."""
        del args, kwargs
        return object()

    def fake_res2(*args: object, **kwargs: object) -> SimpleNamespace:
        """Return a minimal reflected-efficiency result."""
        del args, kwargs
        return SimpleNamespace(
            inc_top_reflected=SimpleNamespace(
                order=np.asarray([-1]),
                efficiency=np.asarray([0.5]),
                theta=np.asarray([86.0]),
            )
        )

    monkeypatch.setattr(simulation_core_module, "res1", fake_res1)
    monkeypatch.setattr(simulation_core_module, "res2", fake_res2)


def _load_blazed_multilayer_profile_tool() -> object:
    """Load the profiling tool module without requiring tools to be a package."""
    module_name = "_grax_blazed_multilayer_profile_tool_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, PROFILING_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load blazed multilayer profiling tool.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_blazed_multilayer_compare_tool() -> object:
    """Load the profiling comparison tool without requiring tools to be a package."""

    module_name = "_grax_blazed_multilayer_compare_tool_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, PROFILING_COMPARE_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load blazed multilayer profiling comparison tool.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_test_blazed_multilayer_grating() -> object:
    """Return a coarse blazed multilayer grating for exact texture regression tests."""

    optical_constants_dir = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "blazed_multilayer"
        / "optical_constants"
    )
    silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    silicon.attrs["name"] = "Si"
    chromium = pd.read_csv(optical_constants_dir / "OC_Cr_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    chromium.attrs["name"] = "Cr"
    carbon = pd.read_csv(optical_constants_dir / "OC_C_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    carbon.attrs["name"] = "C"
    multilayer_stack = grax.MultilayerStack(
        substrate_material=silicon,
        material_a=chromium,
        material_b=carbon,
        d_period_nm=4.8,
        gamma=0.4,
        n_bilayers=6,
        top_material=carbon,
    )
    return grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=multilayer_stack,
        x_resolution_nm=2.0,
        z_resolution_nm=0.5,
    )


def _build_repeating_blazed_multilayer_grating() -> object:
    """Return a blazed multilayer grating with repeated layer-block signatures."""

    optical_constants_dir = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "blazed_multilayer"
        / "optical_constants"
    )
    silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    silicon.attrs["name"] = "Si"
    chromium = pd.read_csv(optical_constants_dir / "OC_Cr_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    chromium.attrs["name"] = "Cr"
    carbon = pd.read_csv(optical_constants_dir / "OC_C_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    carbon.attrs["name"] = "C"
    multilayer_stack = grax.MultilayerStack(
        substrate_material=silicon,
        material_a=chromium,
        material_b=carbon,
        d_period_nm=4.8,
        gamma=0.4,
        n_bilayers=20,
        top_material=carbon,
    )
    return grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=multilayer_stack,
        x_resolution_nm=1.0,
        z_resolution_nm=1.0,
    )


def _legacy_cascade_boundary_pair(
    left: np.ndarray,
    right: np.ndarray,
    basis_size: int,
    *,
    _profiler: SolverProfiler | None = None,
) -> np.ndarray:
    """Return the pre-optimization boundary-pair cascade result."""

    del _profiler
    l11 = left[:basis_size, :basis_size]
    l12 = left[:basis_size, basis_size:]
    l21 = left[basis_size:, :basis_size]
    l22 = left[basis_size:, basis_size:]
    r11 = right[:basis_size, :basis_size]
    r12 = right[:basis_size, basis_size:]
    r21 = right[basis_size:, :basis_size]
    r22 = right[basis_size:, basis_size:]

    matrix_to_solve = l22 - r11
    solved_l21 = np.linalg.solve(matrix_to_solve, l21)
    solved_r12 = np.linalg.solve(matrix_to_solve, r12)
    return np.block(
        [
            [
                l11 - l12 @ solved_l21,
                l12 @ solved_r12,
            ],
            [
                -r21 @ solved_l21,
                r22 + r21 @ solved_r12,
            ],
        ]
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


def test_run_simulation_public_default_matches_low_memory_escape_hatch() -> None:
    public_default = run_simulation(grating=_grating(), energy_ev=200.0, grazing_angle_deg=4.0, fourier_orders=5)
    low_memory = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _memory_mode="low_memory",
    )

    assert public_default.selected_efficiency == pytest.approx(low_memory.selected_efficiency, rel=1e-10, abs=1e-12)
    assert np.allclose(public_default.efficiency_all, low_memory.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(public_default.diffraction_angle_all, low_memory.diffraction_angle_all, rtol=1e-10, atol=1e-12)


def test_run_simulation_logs_peak_ram_when_available(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify successful simulations log peak RSS and RSS delta."""

    class FakeMemorySampler:
        """Deterministic memory sampler for log assertions."""

        peak_memory_bytes = 200 * 1024 * 1024
        memory_delta_bytes = 50 * 1024 * 1024

        def __enter__(self) -> FakeMemorySampler:
            """Enter the fake sampling context."""
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            """Exit the fake sampling context."""
            del exc_type, exc_value, traceback

    _patch_fast_solver(monkeypatch)
    monkeypatch.setattr(simulation_core_module, "PeakMemorySampler", FakeMemorySampler)

    with caplog.at_level(logging.INFO, logger="grax.simulation.core"):
        run_simulation(
            grating=_FastGrating(),
            energy_ev=500.0,
            grazing_angle_deg=14.176,
            fourier_orders=15,
        )

    assert (
        "Simulation completed at 500.00 eV, grazing=14.176 deg, "
        "peak_ram=200.00 MB, ram_delta=50.00 MB"
    ) in caplog.text


def test_run_simulation_skips_peak_ram_log_when_memory_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify unavailable RSS measurement does not produce a RAM log line."""

    class FakeUnavailableMemorySampler:
        """Fake sampler that represents platforms without RSS support."""

        peak_memory_bytes = None
        memory_delta_bytes = None

        def __enter__(self) -> FakeUnavailableMemorySampler:
            """Enter the fake sampling context."""
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            """Exit the fake sampling context."""
            del exc_type, exc_value, traceback

    _patch_fast_solver(monkeypatch)
    monkeypatch.setattr(simulation_core_module, "PeakMemorySampler", FakeUnavailableMemorySampler)

    with caplog.at_level(logging.INFO, logger="grax.simulation.core"):
        run_simulation(
            grating=_FastGrating(),
            energy_ev=500.0,
            grazing_angle_deg=14.176,
            fourier_orders=15,
        )

    assert "Running simulation at 500.00 eV" in caplog.text
    assert "peak_ram=" not in caplog.text


def test_run_simulation_signature_hides_profiler() -> None:
    signature = inspect.signature(run_simulation)

    assert "_profiler" not in signature.parameters
    assert "_memory_mode" not in signature.parameters


def test_run_simulation_rejects_public_memory_mode_keyword() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'memory_mode'"):
        run_simulation(
            grating=_grating(),
            energy_ev=200.0,
            grazing_angle_deg=4.0,
            fourier_orders=5,
            memory_mode="low_memory",  # type: ignore[arg-type]
        )


def test_run_simulation_rejects_invalid_private_memory_mode() -> None:
    with pytest.raises(ValueError, match="memory_mode must be 'low_memory' or 'legacy_dense'"):
        run_simulation(
            grating=_grating(),
            energy_ev=200.0,
            grazing_angle_deg=4.0,
            fourier_orders=5,
            _memory_mode="invalid",  # type: ignore[arg-type]
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


def test_profiler_live_stage_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verify optional profiler stage logging emits start and end events."""
    profiler = SolverProfiler(log_stages=True)

    with caplog.at_level(logging.INFO, logger="grax.simulation.profiling"):
        with profiler.record("outer_stage"):
            with profiler.record("inner_stage"):
                pass

    assert "stage start: outer_stage" in caplog.text
    assert "stage start: inner_stage" in caplog.text
    assert "stage end: inner_stage elapsed=" in caplog.text
    assert "stage end: outer_stage elapsed=" in caplog.text


def test_profiler_live_stage_logging_disabled_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify profiler stage logging remains opt-in."""
    profiler = SolverProfiler()

    with caplog.at_level(logging.INFO, logger="grax.simulation.profiling"):
        with profiler.record("quiet_stage"):
            pass

    assert "quiet_stage" not in caplog.text


def test_blazed_multilayer_profile_tool_parses_case_configuration() -> None:
    """Verify profiling tool CLI parsing without running the expensive case."""
    tool = _load_blazed_multilayer_profile_tool()
    parser = tool.build_arg_parser()
    args = parser.parse_args(
        [
            "--case-index",
            "3",
            "--x-resolution-nm",
            "0.02",
            "0.2",
            "--z-resolution-nm",
            "0.004",
            "0.4",
            "--fourier-orders",
            "11",
            "13",
            "--no-live-stage-log",
        ]
    )

    assert args.case_index == 3
    assert args.x_resolution_nm == pytest.approx([0.02, 0.2])
    assert args.z_resolution_nm == pytest.approx([0.004, 0.4])
    assert args.fourier_orders == [11, 13]
    assert args.no_live_stage_log is True


def test_blazed_multilayer_profile_tool_resolves_reference_case() -> None:
    """Verify case-index energy/angle resolution without running simulation."""
    tool = _load_blazed_multilayer_profile_tool()
    parser = tool.build_arg_parser()
    args = parser.parse_args(["--case-index", "1"])
    reference_data = pd.DataFrame(
        {
            "Energy": [500.0, 510.0],
            "Efficiency(GR)": [0.1, 0.2],
            "alpha": [14.176, 14.25],
        }
    )

    energy_ev, grazing_angle_deg = tool.resolve_case_parameters(args, reference_data)

    assert energy_ev == pytest.approx(510.0)
    assert grazing_angle_deg == pytest.approx(14.25)


def test_blazed_multilayer_profile_tool_writes_comparison_summary_name(tmp_path: Path) -> None:
    """Verify labeled matrix summary output naming for branch-to-branch comparison."""

    tool = _load_blazed_multilayer_profile_tool()
    fake_profiler = SolverProfiler()
    fake_profiler.set_metadata("texture_count", 12)
    fake_profiler.set_metadata("unique_texture_signatures", 10)
    fake_profiler.finalize()
    run = tool.ProfileRun(
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        fourier_orders=5,
        x_resolution_nm=0.1,
        z_resolution_nm=0.1,
        label="baseline",
        comparison_csv_name="custom_summary.csv",
        result=SimpleNamespace(selected_efficiency=0.5, selected_diffraction_angle_deg=75.0),
        profiler=fake_profiler,
    )

    summary_path = tool.write_matrix_summary(output_dir=tmp_path, runs=[run])

    assert summary_path.name == "custom_summary.csv"
    assert summary_path.exists()


def test_blazed_multilayer_compare_tool_builds_speedup_rows() -> None:
    """Verify baseline/candidate profiling summaries merge into speedup rows."""

    tool = _load_blazed_multilayer_compare_tool()
    baseline_rows = [
        {
            "label": "baseline",
            "energy_ev": "500.0",
            "grazing_angle_deg": "14.176",
            "fourier_orders": "5",
            "x_resolution_nm": "0.1",
            "z_resolution_nm": "0.1",
            "total_wall_seconds": "10.0",
            "texture_generation_seconds": "8.0",
            "fourier_coefficients_seconds": "1.0",
            "layer_propagation_cascade_seconds": "0.8",
            "profiled_exclusive_seconds": "9.9",
            "peak_memory_bytes": "1000",
            "texture_count": "100",
            "unique_texture_count": "100",
            "selected_efficiency": "0.25",
        }
    ]
    candidate_rows = [
        {
            "label": "candidate",
            "energy_ev": "500.0",
            "grazing_angle_deg": "14.176",
            "fourier_orders": "5",
            "x_resolution_nm": "0.1",
            "z_resolution_nm": "0.1",
            "total_wall_seconds": "5.0",
            "texture_generation_seconds": "3.0",
            "fourier_coefficients_seconds": "1.0",
            "layer_propagation_cascade_seconds": "0.8",
            "profiled_exclusive_seconds": "4.9",
            "peak_memory_bytes": "900",
            "texture_count": "100",
            "unique_texture_count": "100",
            "selected_efficiency": "0.25",
        }
    ]

    rows = tool.build_comparison_rows(baseline_rows, candidate_rows)

    assert len(rows) == 1
    assert rows[0]["total_wall_speedup"] == pytest.approx(2.0)
    assert rows[0]["texture_generation_speedup"] == pytest.approx(8.0 / 3.0)
    assert rows[0]["selected_efficiency_delta"] == pytest.approx(0.0)


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
    assert counts["layer_modal_matrices_calls"] > 0
    assert counts["layer_cascade_pair_calls"] > 0
    assert timings["fourier_exp"]["calls"] > 0
    assert timings["fourier_sum"]["calls"] > 0
    assert timings["layer_eigensolve_call"]["calls"] > 0
    assert timings["layer_modal_matrices_call"]["calls"] > 0
    assert unique_counts["layer_operator_unique"] > 0
    assert summary["peak_memory_bytes"] >= 0
    assert details["peaks"]["layer_boundary_block_temp_peak"] == pytest.approx(1.0)
    assert details["peaks"]["layer_boundary_block_bytes_peak"] > 0.0
    assert summary["derived_kpis"]["time_per_fourier_call_seconds"] > 0.0
    assert summary["derived_kpis"]["time_per_harmonic_seconds"] > 0.0


def test_low_memory_mode_matches_legacy_dense_result() -> None:
    legacy_dense = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _memory_mode="legacy_dense",
    )
    low_memory = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )

    assert low_memory.selected_efficiency == pytest.approx(legacy_dense.selected_efficiency, rel=1e-10, abs=1e-12)
    assert np.allclose(low_memory.efficiency_all, legacy_dense.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(low_memory.diffraction_angle_all, legacy_dense.diffraction_angle_all, rtol=1e-10, atol=1e-12)


def test_low_memory_build_textures_skips_dense_index_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    grating = _grating()

    def fail_dense_grid(**kwargs: object) -> np.ndarray:
        raise AssertionError("dense index grid should not be built")

    monkeypatch.setattr(grating, "_build_refractive_index_grid", fail_dense_grid)
    textures, profile = grating.build_textures(200.0, _memory_mode="low_memory")

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
    legacy_dense_textures, legacy_dense_profile = grating.build_textures(200.0, _memory_mode="legacy_dense")
    low_memory_textures, low_memory_profile = grating.build_textures(200.0, _memory_mode="low_memory")

    assert len(low_memory_textures) <= len(legacy_dense_textures)
    assert len(low_memory_profile[0]) < len(legacy_dense_profile[0])


def test_low_memory_multilayer_blazed_case_matches_legacy_dense_result() -> None:
    """Verify optimized multilayer low-memory path preserves legacy-dense physics."""

    grating = _build_test_blazed_multilayer_grating()
    legacy_dense = run_simulation(
        grating=grating,
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        fourier_orders=5,
        diffraction_order=2,
        _memory_mode="legacy_dense",
    )
    low_memory = run_simulation(
        grating=grating,
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        fourier_orders=5,
        diffraction_order=2,
        _memory_mode="low_memory",
    )

    assert low_memory.selected_efficiency == pytest.approx(
        legacy_dense.selected_efficiency,
        rel=1e-10,
        abs=1e-12,
    )
    assert np.allclose(low_memory.efficiency_all, legacy_dense.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        low_memory.diffraction_angle_all,
        legacy_dense.diffraction_angle_all,
        rtol=1e-10,
        atol=1e-12,
    )


def test_cascade_boundary_pair_matches_legacy_implementation() -> None:
    """Verify the optimized cascade pair algebra matches the legacy result."""

    rng = np.random.default_rng(1234)
    basis_size = 3
    matrix_size = 2 * basis_size
    left = rng.standard_normal((matrix_size, matrix_size)) + 1j * rng.standard_normal(
        (matrix_size, matrix_size)
    )
    right = rng.standard_normal((matrix_size, matrix_size)) + 1j * rng.standard_normal(
        (matrix_size, matrix_size)
    )
    right[:basis_size, :basis_size] = left[basis_size:, basis_size:] - np.eye(basis_size, dtype=complex)

    optimized = rcwa_1d._cascade_boundary_pair(left, right, basis_size)
    legacy = _legacy_cascade_boundary_pair(left, right, basis_size)

    assert np.allclose(optimized, legacy, rtol=1e-12, atol=1e-12)


def test_fused_modal_function_matrices_match_individual_solves() -> None:
    """Verify fused modal matrix construction preserves the existing algebra."""

    rng = np.random.default_rng(4321)
    matrix_size = 5
    eigenvectors = rng.standard_normal((matrix_size, matrix_size)) + 1j * rng.standard_normal(
        (matrix_size, matrix_size)
    )
    eigenvectors += np.eye(matrix_size, dtype=complex)
    first_modal_values = rng.standard_normal(matrix_size) + 1j * rng.standard_normal(matrix_size)
    second_modal_values = rng.standard_normal(matrix_size) + 1j * rng.standard_normal(matrix_size)

    first_fused, second_fused = rcwa_1d._modal_function_matrices(
        eigenvectors,
        first_modal_values,
        second_modal_values,
    )
    first_baseline = rcwa_1d._modal_function_matrix(eigenvectors, first_modal_values)
    second_baseline = rcwa_1d._modal_function_matrix(eigenvectors, second_modal_values)

    assert np.allclose(first_fused, first_baseline, rtol=1e-12, atol=1e-12)
    assert np.allclose(second_fused, second_baseline, rtol=1e-12, atol=1e-12)


def test_layer_propagation_cascade_reports_substage_timings() -> None:
    """Verify cascade profiling attributes the internal dense linear algebra work."""

    profiler = SolverProfiler()
    run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        _profiler=profiler,
    )
    stage_names = {stage["stage"] for stage in profiler.summary_dict()["stages"]}

    assert "layer_operator_build" in stage_names
    assert "layer_modal_values" in stage_names
    assert "layer_modal_matrices" in stage_names
    assert "layer_block_assembly" in stage_names
    assert "layer_block_cascade_pair" in stage_names
    assert "layer_cascade_pair_solve" in stage_names
    assert "layer_cascade_pair_multiply" in stage_names
    assert "layer_cascade_pair_assemble" in stage_names


def test_blazed_multilayer_optimized_cascade_matches_legacy_cascade() -> None:
    """Verify cascade optimization preserves blazed multilayer efficiencies."""

    grating = _build_test_blazed_multilayer_grating()
    optimized = run_simulation(
        grating=grating,
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        fourier_orders=5,
        diffraction_order=2,
        _memory_mode="low_memory",
    )

    original_cascade = rcwa_1d._cascade_boundary_pair
    try:
        rcwa_1d._cascade_boundary_pair = _legacy_cascade_boundary_pair
        legacy = run_simulation(
            grating=grating,
            energy_ev=500.0,
            grazing_angle_deg=14.176,
            fourier_orders=5,
            diffraction_order=2,
            _memory_mode="low_memory",
        )
    finally:
        rcwa_1d._cascade_boundary_pair = original_cascade

    assert optimized.selected_efficiency == pytest.approx(
        legacy.selected_efficiency,
        rel=1e-10,
        abs=1e-12,
    )
    assert np.allclose(optimized.efficiency_all, legacy.efficiency_all, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        optimized.diffraction_angle_all,
        legacy.diffraction_angle_all,
        rtol=1e-10,
        atol=1e-12,
    )


def test_blazed_multilayer_profile_uses_boundary_block_cache() -> None:
    """Verify repeated blazed multilayer slices reuse cached boundary blocks."""

    profiler = SolverProfiler()
    run_simulation(
        grating=_build_repeating_blazed_multilayer_grating(),
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        fourier_orders=5,
        diffraction_order=2,
        _memory_mode="low_memory",
        _profiler=profiler,
    )
    counts = profiler.summary_dict()["details"]["counts"]

    assert counts["layer_boundary_block_cache_hits"] > 0
    assert counts["layer_boundary_block_cache_misses"] > 0
    assert counts["layer_boundary_blocks_constructed"] > counts["layer_boundary_block_cache_misses"]


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
    legacy_dense_profiler = SolverProfiler()
    legacy_dense_profiler.enable_memory_tracking()
    run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        _memory_mode="legacy_dense",
        _profiler=legacy_dense_profiler,
    )
    low_memory_profiler = SolverProfiler()
    low_memory_profiler.enable_memory_tracking()
    run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
        _profiler=low_memory_profiler,
    )

    assert low_memory_profiler.summary_dict()["peak_memory_bytes"] < legacy_dense_profiler.summary_dict()["peak_memory_bytes"]


def test_incremental_cascade_matches_legacy_dense_result_for_many_layers() -> None:
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
        _memory_mode="legacy_dense",
    )
    profiled = SolverProfiler()
    profiled.enable_memory_tracking()
    candidate = run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=20,
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
    with pytest.warns(FutureWarning, match="deprecated"):
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
    with pytest.warns(FutureWarning, match="deprecated"):
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


def test_public_simulation_default_matches_explicit_numba() -> None:
    default_result = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )
    explicit_numba = run_simulation(
        grating=_grating(),
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        backend="numba",
    )

    assert default_result.selected_efficiency == pytest.approx(explicit_numba.selected_efficiency)
    assert np.allclose(default_result.efficiency_all, explicit_numba.efficiency_all)


def test_batch_runner_defaults_to_numba_backend() -> None:
    runner = BatchSimulationRunner()

    assert runner.backend == "numba"
