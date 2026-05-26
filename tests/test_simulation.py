from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from grax.gratings import BlazedGrating, LaminarGrating
from grax.materials import resolve_refractive_index
from grax.rcwa_1d import (
    DiffractionResult,
    _apply_debye_waller_roughness,
    _debye_waller_roughness_factor,
    debye_waller_roughness_diagnostics,
    res2,
)
from grax import simulation as simulation_module
from grax import peak_fitting as peak_fitting_module
from grax.simulation import (
    BatchSimulationResult,
    BatchSimulationRunner,
    CaseExecutionResult,
    MultilayerThetaSearchSweepResult,
    RCWASimulation,
    SingleSimulationResult,
    energy_angle_cases,
    estimate_multilayer_bragg_angle_deg,
    fixed_angle_cases,
    load_experimental_csv,
    multilayer_theta_search_cases,
    monochromator_cases,
    monochromator_grazing_angles_deg,
    plot_order_subset,
    run_multilayer_theta_search,
    run_multilayer_theta_search_sweep,
    run_simulation,
    write_all_orders_csv,
)
from grax.simulation import batch as simulation_batch_module
from grax.stacks import MultilayerStack
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[1] / "comparison_to_other_codes" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")
C = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_C_cxro.txt", "C")
CR = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Cr_cxro.txt", "Cr")
AU = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Au_cxro.txt", "Au")

EXAMPLE_SCRIPT_PATHS = [
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "single_simulation" / "single_simulation.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "fixed_angle_sweep" / "fixed_angle_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "monochromator_sweep" / "monochromator_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "energy_angle_sweep" / "energy_angle_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "multilayer_theta_search" / "multilayer_theta_search.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "batch_user_cases" / "batch_user_cases.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "blazed_multilayer_sweep" / "blazed_multilayer_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "blazed_multilayer_memory_comparison" / "blazed_multilayer_memory_comparison.py",
]
OPTIMIZER_EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "optimizer" / "optimizer_laminar"
)


def build_test_grating() -> LaminarGrating:
    """Return a reusable test grating."""

    return LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
    )


def build_laminar_example_grating(
    *,
    depth_nm: float = 14.9,
    x_resolution_nm: float = 1.0,
    z_resolution_nm: float = 1.0,
) -> LaminarGrating:
    """Return the laminar grating shape used by the public sweep examples."""

    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=depth_nm,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )


def build_monochromator_example_grating(
    *,
    x_resolution_nm: float = 1.0,
    z_resolution_nm: float = 1.0,
) -> BlazedGrating:
    """Return the blazed single-layer grating used by the public mono example."""

    return BlazedGrating(
        period_lpermm=600,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=30.0,
        blaze_angle_deg=0.75,
        anti_blaze_angle_deg=5.597,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )




def fake_single_result(
    *,
    energy_ev: float = 100.0,
    grazing_angle_deg: float = 4.0,
    orders: np.ndarray | None = None,
    selected_efficiency: float = 0.1,
) -> SingleSimulationResult:
    """Return a small typed single simulation result for batch tests."""

    order_values = np.asarray([-1, 0, 1], dtype=int) if orders is None else orders
    return SingleSimulationResult(
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        orders=order_values,
        selected_efficiency=selected_efficiency,
        selected_diffraction_angle_deg=2.0,
        efficiency_all=np.linspace(0.1, 0.3, order_values.size),
        diffraction_angle_all=np.linspace(1.0, 3.0, order_values.size),
        diffraction_order=1,
        fourier_orders=5,
    )


def build_multilayer_parity_grating() -> LaminarGrating:
    """Return the laminar multilayer grating used for Octave parity tests."""

    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=4,
            top_material=C,
        ),
        x_resolution_nm=20.0,
        z_resolution_nm=1.0,
    )


def build_multilayer_solver_regression_grating() -> LaminarGrating:
    """Return the full laminar multilayer grating that previously diverged."""

    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=40,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=0.1,
    )


def build_blazed_multilayer_angle_parity_grating() -> BlazedGrating:
    """Return the blazed multilayer grating used for angle-sweep parity tests."""

    return BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=3.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.0,
            gamma=0.4,
            n_bilayers=40,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=1.0,
    )


def test_debye_waller_roughness_damps_efficiencies_uniformly() -> None:
    result = DiffractionResult(
        order=np.asarray([-1, 0, 1], dtype=int),
        theta=np.asarray([1.0, 2.0, 3.0], dtype=float),
        efficiency=np.asarray([0.2, 0.4, 0.6], dtype=float),
        amplitude=np.asarray([1.0, 2.0, 3.0], dtype=complex),
    )
    damping = np.exp(-((4.0 * np.pi * 0.5 * 0.6 / 2.0) ** 2))

    damped = _apply_debye_waller_roughness(
        result,
        wavelength_nm=2.0,
        incidence_sine=0.6,
        roughness_sigma_nm=0.5,
    )
    unchanged = _apply_debye_waller_roughness(
        result,
        wavelength_nm=2.0,
        incidence_sine=0.6,
        roughness_sigma_nm=0.0,
    )

    assert np.allclose(damped.efficiency, result.efficiency * damping)
    assert np.allclose(damped.amplitude, result.amplitude)
    assert np.allclose(unchanged.efficiency, result.efficiency)
    assert _debye_waller_roughness_factor(
        wavelength_nm=2.0,
        incidence_sine=0.6,
        roughness_sigma_nm=None,
    ) == pytest.approx(1.0)


def test_debye_waller_roughness_diagnostics_reports_corrected_formula() -> None:
    diagnostics = debye_waller_roughness_diagnostics(
        sigma_nm=0.5,
        wavelength_nm=2.0,
        beta0=0.8,
        theta_surface_rad=0.25,
    )
    expected_argument = 4.0 * np.pi * 0.5 * 0.6 / 2.0

    assert diagnostics["sigma_nm"] == pytest.approx(0.5)
    assert diagnostics["wavelength_nm"] == pytest.approx(2.0)
    assert diagnostics["theta_surface_rad"] == pytest.approx(0.25)
    assert diagnostics["theta_normal_rad"] == pytest.approx((np.pi / 2.0) - 0.25)
    assert diagnostics["beta0"] == pytest.approx(0.8)
    assert diagnostics["sin_theta_used"] == pytest.approx(0.6)
    assert diagnostics["A"] == pytest.approx(expected_argument)
    assert diagnostics["A_squared"] == pytest.approx(expected_argument**2)
    assert diagnostics["damping_factor"] == pytest.approx(np.exp(-(expected_argument**2)))


def test_debye_waller_roughness_diagnostics_does_not_modify_efficiencies() -> None:
    efficiency = np.asarray([0.2, 0.4, 0.6], dtype=float)
    original_efficiency = efficiency.copy()

    debye_waller_roughness_diagnostics(
        sigma_nm=0.5,
        wavelength_nm=2.0,
        beta0=0.8,
    )

    assert np.allclose(efficiency, original_efficiency)


def test_res2_rejects_negative_roughness() -> None:
    with pytest.raises(ValueError, match="roughness_sigma_nm must be >= 0"):
        res2(None, ([], []), roughness_sigma_nm=-0.1)


def run_octave_laminar_multilayer_reference(tmp_path: Path) -> dict[str, np.ndarray]:
    """Run the default Octave laminar-multilayer reference fixture."""

    return run_octave_laminar_multilayer_reference_with_parameters(
        tmp_path,
        n_bilayers=4,
        z_resolution_nm=1.0,
        x_resolution_nm=20.0,
        fourier_orders=5,
    )


def run_octave_laminar_multilayer_reference_with_parameters(
    tmp_path: Path,
    *,
    n_bilayers: int,
    z_resolution_nm: float,
    x_resolution_nm: float,
    fourier_orders: int,
) -> dict[str, np.ndarray]:
    """Run the Octave laminar-multilayer reference fixture and load its outputs."""

    if shutil.which("octave-cli") is None:
        pytest.skip("octave-cli is not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "reticolo" / "tests" / "octave_laminar_multilayer_reference.m"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            "octave-cli",
            "--quiet",
            str(script_path),
            str(tmp_path),
            str(repo_root),
            str(n_bilayers),
            str(z_resolution_nm),
            str(x_resolution_nm),
            str(fourier_orders),
        ],
        check=True,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "x": np.loadtxt(tmp_path / "x.csv", delimiter=","),
        "z": np.loadtxt(tmp_path / "z.csv", delimiter=","),
        "surface": np.loadtxt(tmp_path / "surface.csv", delimiter=","),
        "material_id": np.loadtxt(tmp_path / "material_id.csv", delimiter=","),
        "solver": np.loadtxt(tmp_path / "solver.csv", delimiter=","),
    }


def run_octave_blazed_multilayer_angle_reference(
    tmp_path: Path,
    *,
    start_angle_index: int = 75,
    end_angle_index: int = 81,
) -> np.ndarray:
    """Run the Octave blazed-multilayer angle-sweep reference fixture."""

    if shutil.which("octave-cli") is None:
        pytest.skip("octave-cli is not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "reticolo" / "tests" / "octave_blazed_multilayer_angle_reference.m"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            "octave-cli",
            "--quiet",
            str(script_path),
            str(tmp_path),
            str(repo_root),
            str(start_angle_index),
            str(end_angle_index),
        ],
        check=True,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return np.loadtxt(tmp_path / "solver.csv", delimiter=",")


def test_rcwa_simulation_runs_for_multiple_energies() -> None:
    grating = build_test_grating()
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=25,
        grazing_angle_deg=4.0,
    )

    result = simulation.run([100.0, 150.0])

    assert np.allclose(result.energy_ev, np.array([100.0, 150.0]))
    assert result.orders.shape == (51,)
    assert result.efficiency.shape == (2,)
    assert result.diffraction_angle_deg.shape == (2,)
    assert result.efficiency_all.shape == (2, 51)
    assert result.diffraction_angle_all.shape == (2, 51)


def test_rcwa_simulation_runs_laminar_grating_end_to_end() -> None:
    grating = build_test_grating()
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=5,
        grazing_angle_deg=4.0,
    )

    result = simulation.run_single(100.0)

    assert result["orders"].shape == (11,)
    assert result["efficiency_all"].shape == (11,)
    assert result["diffraction_angle_all"].shape == (11,)


def test_rcwa_simulation_loads_experimental_data_and_plots_comparison(tmp_path: Path) -> None:
    grating = build_test_grating()
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=25,
        grazing_angle_deg=4.0,
    )
    result = simulation.run([100.0])
    experimental = load_experimental_csv(
        Path("data")
        / "Re__ELISA,_400l_mm_laminar_grating_from_HORIBA"
        / "lG400-HZB-ELISA_ascan-energy_alpha-4deg_1-order.csv"
    )

    output_path = tmp_path / "comparison.png"
    simulation.plot_against_experiment(result, experimental, output_path)

    assert experimental.shape[1] == 2
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_simulation_returns_typed_single_result() -> None:
    result = run_simulation(
        grating=build_test_grating(),
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )

    assert isinstance(result, SingleSimulationResult)
    assert result.energy_ev == pytest.approx(100.0)
    assert result.grazing_angle_deg == pytest.approx(4.0)
    assert result.orders.shape == (11,)
    assert result.efficiency_all.shape == (11,)
    assert result.diffraction_angle_all.shape == (11,)


def test_estimate_multilayer_bragg_angle_returns_finite_value() -> None:
    grating = build_blazed_multilayer_angle_parity_grating()

    angle = estimate_multilayer_bragg_angle_deg(grating=grating, energy_ev=2000.0)

    assert np.isfinite(angle)
    assert angle > 0.0


def test_run_multilayer_theta_search_selects_precise_peak_and_uses_selected_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_angle = 1.35
    calls: list[tuple[float, int]] = []

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        grazing_angle_deg = float(kwargs["grazing_angle_deg"])
        fourier_orders = int(kwargs["fourier_orders"])
        calls.append((grazing_angle_deg, fourier_orders))
        efficiency = max(0.0, 1.0 - ((grazing_angle_deg - target_angle) / 0.06) ** 2)
        return SingleSimulationResult(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=grazing_angle_deg,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=efficiency,
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([efficiency, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=fourier_orders,
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)

    result = run_multilayer_theta_search(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energy_ev=2000.0,
        diffraction_order=1,
        initial_grazing_angle_deg=1.3,
        rough_scan_half_width_deg=0.2,
        rough_scan_points=9,
        fine_scan_half_width_deg=0.05,
        fine_scan_points=11,
        rough_fourier_orders=5,
        final_fourier_orders=7,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
    )

    diagnostics = result.theta_search_diagnostics
    assert diagnostics is not None
    assert diagnostics.estimated_grazing_angle_deg == pytest.approx(1.3)
    assert diagnostics.rough_grazing_angles_deg.shape == (9,)
    assert diagnostics.precise_grazing_angles_deg.shape == (11,)
    assert result.grazing_angle_deg == pytest.approx(target_angle, abs=1e-6)
    assert diagnostics.selected_grazing_angle_deg == pytest.approx(result.grazing_angle_deg)
    assert calls[-1] == (pytest.approx(target_angle, abs=1e-6), 7)
    assert diagnostics.precise_fwhm_deg is not None
    assert diagnostics.precise_fwhm_deg > 0.0


def test_run_multilayer_theta_search_gauss_refines_between_sampled_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_angle = 1.353
    final_calls: list[tuple[float, int]] = []

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        grazing_angle_deg = float(kwargs["grazing_angle_deg"])
        fourier_orders = int(kwargs["fourier_orders"])
        final_calls.append((grazing_angle_deg, fourier_orders))
        efficiency = np.exp(-0.5 * ((grazing_angle_deg - target_angle) / 0.01) ** 2)
        return SingleSimulationResult(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=grazing_angle_deg,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=float(efficiency),
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([efficiency, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=fourier_orders,
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)

    result = run_multilayer_theta_search(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energy_ev=2000.0,
        diffraction_order=1,
        initial_grazing_angle_deg=1.35,
        rough_scan_half_width_deg=0.05,
        rough_scan_points=9,
        fine_scan_half_width_deg=0.01,
        fine_scan_points=5,
        rough_fourier_orders=5,
        final_fourier_orders=7,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
        precise_peak_selection_mode="gauss",
    )

    diagnostics = result.theta_search_diagnostics
    assert diagnostics is not None
    assert diagnostics.precise_peak_selection_mode_requested == "gauss"
    assert diagnostics.precise_peak_selection_mode_used == "gauss"
    assert diagnostics.precise_peak_fit_fallback_used is False
    assert diagnostics.precise_peak_fitted_center_deg == pytest.approx(target_angle, abs=5e-4)
    assert result.grazing_angle_deg == pytest.approx(target_angle, abs=5e-4)
    assert final_calls[-1] == (pytest.approx(target_angle, abs=5e-4), 7)


def test_run_multilayer_theta_search_voigt_fallbacks_to_gauss(monkeypatch: pytest.MonkeyPatch) -> None:
    target_angle = 1.353

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        grazing_angle_deg = float(kwargs["grazing_angle_deg"])
        efficiency = np.exp(-0.5 * ((grazing_angle_deg - target_angle) / 0.01) ** 2)
        return SingleSimulationResult(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=grazing_angle_deg,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=float(efficiency),
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([efficiency, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=int(kwargs["fourier_orders"]),
        )

    def always_fail_voigt(theta_deg: np.ndarray, efficiency_window: np.ndarray) -> tuple[float, float] | None:
        return None

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(peak_fitting_module, "_fit_voigt", always_fail_voigt)

    result = run_multilayer_theta_search(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energy_ev=2000.0,
        diffraction_order=1,
        initial_grazing_angle_deg=1.35,
        rough_scan_half_width_deg=0.05,
        rough_scan_points=9,
        fine_scan_half_width_deg=0.01,
        fine_scan_points=5,
        rough_fourier_orders=5,
        final_fourier_orders=7,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
        precise_peak_selection_mode="voigt",
    )

    diagnostics = result.theta_search_diagnostics
    assert diagnostics is not None
    assert diagnostics.precise_peak_selection_mode_requested == "voigt"
    assert diagnostics.precise_peak_selection_mode_used == "gauss"
    assert diagnostics.precise_peak_fit_fallback_used is True
    assert result.grazing_angle_deg == pytest.approx(target_angle, abs=5e-4)


def test_run_multilayer_theta_search_fit_falls_back_to_sampled_max_when_both_models_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_peak_angle = 1.35

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        grazing_angle_deg = float(kwargs["grazing_angle_deg"])
        efficiency = np.exp(-0.5 * ((grazing_angle_deg - sampled_peak_angle) / 0.01) ** 2)
        return SingleSimulationResult(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=grazing_angle_deg,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=float(efficiency),
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([efficiency, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=int(kwargs["fourier_orders"]),
        )

    def always_fail(
        theta_window_deg: np.ndarray,
        efficiency_window: np.ndarray,
    ) -> tuple[float, float] | None:
        return None

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(peak_fitting_module, "_fit_gaussian", always_fail)
    monkeypatch.setattr(peak_fitting_module, "_fit_voigt", always_fail)

    result = run_multilayer_theta_search(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energy_ev=2000.0,
        diffraction_order=1,
        initial_grazing_angle_deg=1.35,
        rough_scan_half_width_deg=0.05,
        rough_scan_points=9,
        fine_scan_half_width_deg=0.01,
        fine_scan_points=5,
        rough_fourier_orders=5,
        final_fourier_orders=7,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
        precise_peak_selection_mode="voigt",
    )

    diagnostics = result.theta_search_diagnostics
    assert diagnostics is not None
    assert diagnostics.precise_peak_selection_mode_requested == "voigt"
    assert diagnostics.precise_peak_selection_mode_used == "max"
    assert diagnostics.precise_peak_fit_fallback_used is True
    assert diagnostics.precise_peak_fitted_center_deg is None
    assert result.grazing_angle_deg == pytest.approx(sampled_peak_angle, abs=1e-6)


def test_peak_selection_uses_local_fit_window() -> None:
    theta_deg = np.linspace(0.0, 2.0, 41, dtype=float)
    efficiencies = 0.02 * np.exp(-0.5 * ((theta_deg - 0.35) / 0.03) ** 2)
    efficiencies += 1.0 * np.exp(-0.5 * ((theta_deg - 1.2) / 0.06) ** 2)

    selection = peak_fitting_module.select_peak_theta_from_scan(
        theta_deg,
        efficiencies,
        requested_mode="gauss",
    )

    assert selection.selected_theta_deg == pytest.approx(1.2, abs=5e-3)
    assert selection.fit_window_end_index - selection.fit_window_start_index < theta_deg.size


def test_precise_scan_fwhm_matches_synthetic_peak_width() -> None:
    theta = np.linspace(-1.0, 1.0, 101, dtype=float)
    sigma = 0.2
    efficiencies = np.exp(-0.5 * (theta / sigma) ** 2)

    fwhm = simulation_module._precise_scan_fwhm_deg(theta, efficiencies)

    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert fwhm is not None
    assert fwhm == pytest.approx(expected, rel=0.05)


def test_precise_scan_fwhm_returns_none_when_halfmax_not_bracketed() -> None:
    theta = np.asarray([0.0, 1.0, 2.0], dtype=float)
    efficiencies = np.asarray([1.0, 0.95, 0.9], dtype=float)

    fwhm = simulation_module._precise_scan_fwhm_deg(theta, efficiencies)

    assert fwhm is None


def test_adaptive_scan_half_widths_follow_lower_energy_fwhm_rule() -> None:
    rough, precise, source, source_energy, source_fwhm = simulation_module._adaptive_scan_half_widths(
        energy_ev=3000.0,
        initial_rough_half_width_deg=5.0,
        initial_precise_half_width_deg=2.0,
        completed_fwhm_by_energy={1000.0: 0.2, 2500.0: 0.15},
    )

    assert source == "from_lower_energy"
    assert source_energy == pytest.approx(2500.0)
    assert source_fwhm == pytest.approx(0.15)
    assert precise == pytest.approx(0.6)
    assert rough == pytest.approx(3.0)


def test_adaptive_scan_half_widths_fallback_to_initial_without_lower_energy() -> None:
    rough, precise, source, source_energy, source_fwhm = simulation_module._adaptive_scan_half_widths(
        energy_ev=3000.0,
        initial_rough_half_width_deg=5.0,
        initial_precise_half_width_deg=2.0,
        completed_fwhm_by_energy={3500.0: 0.1},
    )

    assert source == "initial"
    assert source_energy is None
    assert source_fwhm is None
    assert precise == pytest.approx(2.0)
    assert rough == pytest.approx(5.0)


def test_batch_runner_executes_generator_cases_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    runner = BatchSimulationRunner(default_fourier_orders=5)

    def case_generator() -> Iterator[dict[str, object]]:
        yield {
            "case_id": "case-1",
            "grating": grating,
            "energy_ev": 100.0,
            "grazing_angle_deg": 4.0,
            "label": "first",
        }
        yield {
            "case_id": "case-2",
            "grating": grating,
            "energy_ev": 150.0,
            "grazing_angle_deg": 4.5,
            "label": "second",
        }

    results = list(runner.run_cases(case_generator()))

    assert [case.case_id for case in results] == ["case-1", "case-2"]
    assert [case.label for case in results] == ["first", "second"]
    assert [case.status for case in results] == ["ok", "ok"]
    assert np.allclose([case.energy_ev for case in results], np.array([100.0, 150.0]))


def test_batch_runner_can_profile_peak_memory_and_preserve_memory_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []

    class FakeProfiler:
        def enable_memory_tracking(self) -> None:
            return None

        def finalize(self) -> None:
            return None

        def summary_dict(self) -> dict[str, object]:
            return {"peak_memory_bytes": 123456, "total_wall_seconds": 9.87}

    def fake_run_case_payload(payload: dict[str, object], *, diagnostic_callback: object = None) -> SingleSimulationResult:
        del diagnostic_callback
        payloads.append(payload)
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=float(payload["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_batch_module, "SolverProfiler", FakeProfiler)
    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)

    runner = BatchSimulationRunner()
    results = list(
        runner.run_cases(
            [
                {
                    "case_id": "case-1",
                    "grating": build_test_grating(),
                    "energy_ev": 100.0,
                    "grazing_angle_deg": 4.0,
                    "memory_mode": "low_memory",
                    "profile_memory": True,
                }
            ]
        )
    )

    assert len(payloads) == 1
    assert payloads[0]["memory_mode"] == "low_memory"
    assert results[0].peak_memory_bytes == 123456
    assert results[0].wall_seconds == pytest.approx(9.87)
    assert results[0].status == "ok"


def test_batch_runner_does_not_profile_memory_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_case_payload(payload: dict[str, object], *, diagnostic_callback: object = None) -> SingleSimulationResult:
        del diagnostic_callback
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=float(payload["grazing_angle_deg"]),
        )

    class ForbiddenProfiler:
        def __init__(self) -> None:
            raise AssertionError("Profiler should not be instantiated unless profile_memory=True.")

    monkeypatch.setattr(simulation_batch_module, "SolverProfiler", ForbiddenProfiler)
    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)

    runner = BatchSimulationRunner()
    results = list(
        runner.run_cases(
            [
                {
                    "case_id": "case-1",
                    "grating": build_test_grating(),
                    "energy_ev": 100.0,
                    "grazing_angle_deg": 4.0,
                }
            ]
        )
    )

    assert results[0].peak_memory_bytes is None


def test_batch_runner_rejects_resume_without_checkpoint_dir() -> None:
    with pytest.raises(ValueError, match="resume=True requires checkpoint_dir"):
        BatchSimulationRunner(resume=True, checkpoint_dir=None)


def test_batch_runner_resolves_max_workers_special_values() -> None:
    assert simulation_module._resolve_max_workers(None) == 1
    assert simulation_module._resolve_max_workers(1) == 1
    assert simulation_module._resolve_max_workers("all") == max(os.cpu_count() or 1, 1)
    assert simulation_module._resolve_max_workers("auto") == max((os.cpu_count() or 1) - 2, 1)


def test_batch_runner_auto_workers_calibration_respects_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(simulation_module, "AUTO_WORKER_MEMORY_RESERVE_BYTES", 2 * 1024**3)
    monkeypatch.setattr(simulation_module, "AUTO_WORKER_MEMORY_SAFETY_FACTOR", 1.0)
    monkeypatch.setattr(simulation_module, "_current_process_memory_bytes", lambda: 3 * 1024**3)

    assert simulation_module._calibrate_auto_max_workers_from_result(
        pending_case_count=10,
        available_memory_bytes=8 * 1024**3,
    ) == 2


def test_batch_runner_auto_workers_calibration_falls_back_to_cpu_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(simulation_module, "_current_process_memory_bytes", lambda: None)

    assert simulation_module._calibrate_auto_max_workers_from_result(
        pending_case_count=10,
        available_memory_bytes=8 * 1024**3,
    ) == 14


def test_batch_runner_rejects_invalid_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        BatchSimulationRunner(max_workers=0)
    with pytest.raises(ValueError, match="max_workers"):
        BatchSimulationRunner(max_workers="invalid")  # type: ignore[arg-type]


def test_batch_runner_rejects_parallel_subprocess_combination() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        BatchSimulationRunner(max_workers=2, execution_mode="subprocess")


def test_batch_runner_auto_generates_case_id_when_missing() -> None:
    runner = BatchSimulationRunner()
    results = list(
        runner.run_cases(
            [{"grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
        )
    )
    assert len(results) == 1
    assert results[0].case_id == "batch-00000000"


def test_batch_runner_applies_resolution_overrides_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[float, float]] = []

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        grating = kwargs["grating"]
        assert isinstance(grating, LaminarGrating)
        captured.append((grating.x_resolution_nm, grating.z_resolution_nm))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    original_x = grating.x_resolution_nm
    original_z = grating.z_resolution_nm

    runner = BatchSimulationRunner()
    list(
        runner.run_cases(
            [
                {
                    "case_id": "case-1",
                    "grating": grating,
                    "energy_ev": 100.0,
                    "grazing_angle_deg": 4.0,
                    "x_resolution_nm": 2.5,
                    "z_resolution_nm": 0.8,
                }
            ]
        )
    )

    assert captured == [(2.5, 0.8)]
    assert grating.x_resolution_nm == original_x
    assert grating.z_resolution_nm == original_z


def test_batch_runner_passes_roughness(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[float | None] = []

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        captured.append(kwargs["roughness_sigma_nm"])
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    runner = BatchSimulationRunner()
    list(
        runner.run_cases(
            [
                {
                    "case_id": "case-1",
                    "grating": build_test_grating(),
                    "energy_ev": 100.0,
                    "grazing_angle_deg": 4.0,
                    "roughness_sigma_nm": 0.5,
                }
            ]
        )
    )

    assert captured == [0.5]


def test_batch_runner_records_errors_in_continue_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        if kwargs["energy_ev"] == 150.0:
            raise RuntimeError("bad case")
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    runner = BatchSimulationRunner(on_error="continue")
    results = list(
        runner.run_cases(
            [
                {"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0},
                {"case_id": "case-2", "grating": grating, "energy_ev": 150.0, "grazing_angle_deg": 4.0},
            ]
        )
    )

    assert [case.status for case in results] == ["ok", "error"]
    assert results[1].error_message == "bad case"
    assert np.allclose([case.energy_ev for case in results if case.status == "ok"], np.array([100.0]))


def test_batch_runner_raises_in_fail_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        raise RuntimeError("stop")

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    runner = BatchSimulationRunner(on_error="fail_fast")

    with pytest.raises(RuntimeError, match="stop"):
        list(
            runner.run_cases(
                [{"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
            )
        )


def test_batch_runner_rejects_unsupported_material_input_in_fail_fast_mode() -> None:
    grating = LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        top_cap_material="DefinitelyMissingMaterial",
        top_cap_thickness_nm=0.7,
    )
    runner = BatchSimulationRunner(
        default_fourier_orders=5,
        on_error="fail_fast",
    )

    with pytest.raises(TypeError, match="Unsupported material input"):
        list(
            runner.run_cases(
                [{"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
            )
        )


def test_batch_runner_writes_jsonl_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[float] = []

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    cases = [
        {"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0},
        {"case_id": "case-2", "grating": grating, "energy_ev": 150.0, "grazing_angle_deg": 4.0},
    ]

    runner = BatchSimulationRunner(checkpoint_dir=tmp_path)
    first_results = list(runner.run_cases(iter(cases), metadata={"name": "test"}))
    second_runner = BatchSimulationRunner(checkpoint_dir=tmp_path, resume=True)
    second_results = list(second_runner.run_cases(iter(cases)))

    checkpoint_lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [case.case_id for case in first_results] == ["case-1", "case-2"]
    assert second_results == []
    assert calls == [100.0, 150.0]
    assert len(checkpoint_lines) == 2
    assert json.loads(checkpoint_lines[0])["case_id"] == "case-1"


def test_batch_runner_progress_updates_on_completion_not_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    updates: list[int] = []
    postfix_values: list[str] = []
    totals: list[int | None] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            totals.append(total)

        def update(self, value: int = 1) -> None:
            updates.append(value)

        def set_postfix_str(self, value: str) -> None:
            postfix_values.append(value)

        def close(self) -> None:
            return None

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    runner = BatchSimulationRunner(show_progress=True)

    results = list(
        runner.run_cases(
            [
                {"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0},
                {"case_id": "case-2", "grating": build_test_grating(), "energy_ev": 150.0, "grazing_angle_deg": 4.0},
            ]
        )
    )

    assert len(results) == 2
    assert totals == [2]
    assert updates == [1, 1]
    assert postfix_values == []

    checkpoint_path = tmp_path / "progress_resume_checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / "results.jsonl"
    checkpoint_file.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "index": 0,
                "label": None,
                "energy_ev": 100.0,
                "grazing_angle_deg": 4.0,
                "orders": [-1, 0, 1],
                "selected_efficiency": 0.1,
                "selected_diffraction_angle_deg": 2.0,
                "efficiency_all": [0.1, 0.2, 0.3],
                "diffraction_angle_all": [1.0, 2.0, 3.0],
                "status": "ok",
                "error_message": None,
                "case_data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    updates.clear()
    postfix_values.clear()
    totals.clear()
    resumed_runner = BatchSimulationRunner(show_progress=True, checkpoint_dir=checkpoint_path, resume=True)
    resumed_results = list(
        resumed_runner.run_cases(
            [
                {"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0},
                {"case_id": "case-2", "grating": build_test_grating(), "energy_ev": 150.0, "grazing_angle_deg": 4.0},
            ]
        )
    )
    assert [case.case_id for case in resumed_results] == ["case-2"]
    assert totals == [2]
    assert updates == [1, 1]
    assert postfix_values == []


def test_batch_runner_infers_progress_total_for_generator_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    totals: list[int | None] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            del desc, unit
            totals.append(total)

        def update(self, value: int = 1) -> None:
            del value

        def close(self) -> None:
            return None

    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    runner = BatchSimulationRunner(show_progress=True)
    cases = (
        {"case_id": f"case-{index}", "grating": build_test_grating(), "energy_ev": float(100 + index), "grazing_angle_deg": 4.0}
        for index in range(3)
    )

    results = list(runner.run_cases(cases))

    assert len(results) == 3
    assert totals == [3]


def test_batch_runner_parallel_executes_cases_and_writes_checkpoint(tmp_path: Path) -> None:
    grating = build_test_grating()
    runner = BatchSimulationRunner(
        default_fourier_orders=1,
        max_workers=2,
        checkpoint_dir=tmp_path,
        on_error="continue",
    )
    cases = [
        {"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0},
        {"case_id": "case-2", "grating": grating, "energy_ev": 120.0, "grazing_angle_deg": 4.0},
    ]

    results = list(runner.run_cases(cases, metadata={"name": "parallel"}))

    assert {case.case_id for case in results} == {"case-1", "case-2"}
    assert all(case.status == "ok" for case in results)
    checkpoint_lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 2
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resolved_max_workers"] == 2


def test_batch_runner_parallel_resume_skips_completed_cases(tmp_path: Path) -> None:
    grating = build_test_grating()
    checkpoint_path = tmp_path / "results.jsonl"
    checkpoint_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "index": 0,
                "label": None,
                "energy_ev": 100.0,
                "grazing_angle_deg": 4.0,
                "orders": [-1, 0, 1],
                "selected_efficiency": 0.1,
                "selected_diffraction_angle_deg": 2.0,
                "efficiency_all": [0.1, 0.2, 0.3],
                "diffraction_angle_all": [1.0, 2.0, 3.0],
                "status": "ok",
                "error_message": None,
                "case_data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = BatchSimulationRunner(
        default_fourier_orders=1,
        max_workers=2,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    cases = [
        {"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0},
        {"case_id": "case-2", "grating": grating, "energy_ev": 120.0, "grazing_angle_deg": 4.0},
    ]

    results = list(runner.run_cases(cases))

    assert [case.case_id for case in results] == ["case-2"]


def test_worker_initializer_forces_single_thread_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(variable, raising=False)

    simulation_module._worker_initializer()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_parallel_worker_is_top_level_and_spawn_compatible() -> None:
    assert simulation_module._parallel_worker_execute.__module__ == "grax.simulation"
    assert simulation_module._multiprocessing_start_method() in {"fork", "spawn"}


def test_rcwa_simulation_raises_for_non_physical_efficiency() -> None:
    simulation = RCWASimulation(grating=build_test_grating())

    with pytest.raises(ValueError, match="Non-physical reflected diffraction efficiency"):
        simulation._validate_reflected_efficiencies(
            photon_energy_ev=100.0,
            orders=np.asarray([-1, 0, 1], dtype=int),
            efficiency_all=np.asarray([1.2, 0.0, 0.0], dtype=float),
        )


def test_monochromator_helper_returns_monotonic_grazing_angles() -> None:
    energies = np.array([100.0, 150.0, 200.0], dtype=float)

    grazing_angles = monochromator_grazing_angles_deg(
        energies,
        period_lpermm=400,
        diffraction_order=1,
        cff=2.25,
    )

    assert grazing_angles.shape == (3,)
    assert np.all(np.isfinite(grazing_angles))
    assert np.all(np.diff(grazing_angles) < 0.0)


def test_lazy_case_helpers_yield_expected_cases() -> None:
    grating = build_test_grating()
    fixed = fixed_angle_cases(grating=grating, energies_ev=iter([100.0, 150.0]), grazing_angle_deg=4.0)
    first_fixed = next(fixed)
    assert first_fixed["case_id"] == "fixed-00000000"
    assert first_fixed["energy_ev"] == 100.0
    assert first_fixed["grazing_angle_deg"] == 4.0

    mono = monochromator_cases(grating=grating, energies_ev=iter([100.0]), diffraction_order=1, cff=2.25)
    first_mono = next(mono)
    expected_angle = monochromator_grazing_angles_deg(
        np.asarray([100.0]), period_lpermm=grating.period_lpermm, diffraction_order=1, cff=2.25
    )[0]
    assert first_mono["case_id"] == "mono-00000000"
    assert first_mono["grazing_angle_deg"] == pytest.approx(expected_angle)

    pairs = energy_angle_cases(grating=grating, energy_angle_pairs=iter([(100.0, 4.0), (150.0, 4.5)]))
    assert [next(pairs)["case_id"], next(pairs)["grazing_angle_deg"]] == ["pair-00000000", 4.5]

    theta_search = multilayer_theta_search_cases(grating=grating, energies_ev=iter([100.0, 150.0]))
    first_theta_search = next(theta_search)
    assert first_theta_search["case_id"] == "theta-search-00000000"
    assert first_theta_search["energy_ev"] == 100.0
    assert first_theta_search["workflow"] == "multilayer_theta_search"
    assert "grazing_angle_deg" not in first_theta_search


def test_case_helpers_reject_removed_public_override_arguments() -> None:
    grating = build_test_grating()
    with pytest.raises(TypeError, match="case_id_prefix"):
        list(
            fixed_angle_cases(
                grating=grating,
                energies_ev=[100.0],
                grazing_angle_deg=4.0,
                case_id_prefix="fixed",
            )
        )
    with pytest.raises(TypeError, match="case_defaults"):
        list(
            monochromator_cases(
                grating=grating,
                energies_ev=[100.0],
                case_defaults={"label": "x"},
            )
        )


def test_theta_search_sweep_rejects_removed_public_arguments() -> None:
    with pytest.raises(TypeError, match="case_id_prefix"):
        run_multilayer_theta_search_sweep(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[1800.0],
            output_dir=Path("unused"),
            case_id_prefix="theta",
        )
    with pytest.raises(TypeError, match="theta_retry_jitter_deg"):
        run_multilayer_theta_search_sweep(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[1800.0],
            output_dir=Path("unused"),
            theta_retry_jitter_deg=(0.002,),
        )


def test_run_multilayer_theta_search_rejects_removed_aliases() -> None:
    with pytest.raises(TypeError, match="fourier_orders"):
        run_multilayer_theta_search(  # type: ignore[call-arg]
            grating=build_blazed_multilayer_angle_parity_grating(),
            energy_ev=1800.0,
            fourier_orders=3,
        )
    with pytest.raises(TypeError, match="x_resolution_nm"):
        run_multilayer_theta_search(  # type: ignore[call-arg]
            grating=build_blazed_multilayer_angle_parity_grating(),
            energy_ev=1800.0,
            x_resolution_nm=1.0,
        )
    with pytest.raises(TypeError, match="z_resolution_nm"):
        run_multilayer_theta_search(  # type: ignore[call-arg]
            grating=build_blazed_multilayer_angle_parity_grating(),
            energy_ev=1800.0,
            z_resolution_nm=1.0,
        )
    with pytest.raises(TypeError, match="precise_scan_half_width_deg"):
        run_multilayer_theta_search(  # type: ignore[call-arg]
            grating=build_blazed_multilayer_angle_parity_grating(),
            energy_ev=1800.0,
            precise_scan_half_width_deg=0.1,
        )
    with pytest.raises(TypeError, match="precise_scan_points"):
        run_multilayer_theta_search(  # type: ignore[call-arg]
            grating=build_blazed_multilayer_angle_parity_grating(),
            energy_ev=1800.0,
            precise_scan_points=81,
        )


def test_result_export_helpers_write_expected_outputs(tmp_path: Path) -> None:
    result = BatchSimulationResult(
        cases=[
            CaseExecutionResult(
                case_id="case-1",
                index=0,
                label="case",
                energy_ev=100.0,
                grazing_angle_deg=4.0,
                orders=np.asarray([-1, 0, 1], dtype=int),
                selected_efficiency=0.1,
                selected_diffraction_angle_deg=2.0,
                efficiency_all=np.asarray([0.1, 0.2, 0.3], dtype=float),
                diffraction_angle_all=np.asarray([1.0, 2.0, 3.0], dtype=float),
                status="ok",
            ),
        ]
    )

    all_orders_path = tmp_path / "all_orders.csv"
    order_subset_plot_path = tmp_path / "subset.png"
    write_all_orders_csv(result, all_orders_path)
    plot_order_subset(
        result,
        order_subset_plot_path,
        diffraction_orders=[1, 2, 3],
        title="Orders 1-3",
    )

    all_orders_header = all_orders_path.read_text(encoding="utf-8").splitlines()[0]

    assert all_orders_header == "case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg"
    assert order_subset_plot_path.exists()


def test_batch_runner_live_plot_uses_requested_x_axis_and_order_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
            orders=np.asarray([-3, -2, -1, 0], dtype=int),
            selected_efficiency=0.3,
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    grating = build_test_grating()
    runner = BatchSimulationRunner(
        live_plot=True,
        live_plot_x_key="energy_ev",
        live_plot_order_count=2,
    )

    list(
        runner.run_cases(
            [
                {"case_id": "case-1", "grating": grating, "energy_ev": 100.0, "grazing_angle_deg": 4.0},
                {"case_id": "case-2", "grating": grating, "energy_ev": 150.0, "grazing_angle_deg": 5.0},
            ]
        )
    )

    assert runner._live_axis is not None
    lines = runner._live_axis.get_lines()
    assert len(lines) == 2
    assert np.allclose(lines[0].get_xdata(), np.array([100.0, 150.0]))
    assert np.allclose(lines[0].get_ydata(), np.array([0.23333333333333334, 0.23333333333333334]))
    assert np.allclose(lines[1].get_ydata(), np.array([0.16666666666666669, 0.16666666666666669]))
    plt.close("all")


def test_batch_runner_live_plot_can_overlay_reference_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
            selected_efficiency=0.3,
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    reference_data = np.asarray([[90.0, 0.2], [110.0, 0.25]], dtype=float)
    runner = BatchSimulationRunner(
        live_plot=True,
        live_plot_x_key="energy_ev",
        live_plot_order_count=1,
        live_plot_reference_data=reference_data,
    )

    list(
        runner.run_cases(
            [{"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
        )
    )

    assert runner._live_axis is not None
    lines = runner._live_axis.get_lines()
    assert len(lines) == 2
    assert np.allclose(lines[1].get_xdata(), reference_data[:, 0])
    assert np.allclose(lines[1].get_ydata(), reference_data[:, 1])
    plt.close("all")


def test_batch_runner_theta_search_writes_checkpoints_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.15 + (0.01 * len(calls)),
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    cases = list(
        multilayer_theta_search_cases(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[2000.0, 2200.0],
        )
    )

    runner = BatchSimulationRunner(checkpoint_dir=tmp_path)
    first_results = list(runner.run_cases(iter(cases), metadata={"name": "theta-search"}))
    resumed_runner = BatchSimulationRunner(checkpoint_dir=tmp_path, resume=True)
    resumed_results = list(resumed_runner.run_cases(iter(cases)))

    checkpoint_lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [case.case_id for case in first_results] == ["theta-search-00000000", "theta-search-00000001"]
    assert resumed_results == []
    assert calls == [2000.0, 2200.0]
    assert len(checkpoint_lines) == 2


def test_run_multilayer_theta_search_sweep_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        return SingleSimulationResult(
            energy_ev=energy_ev,
            grazing_angle_deg=1.2,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=0.35,
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([0.35, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=int(kwargs["final_fourier_orders"]),
            theta_search_diagnostics=diagnostics,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 2000.0],
        output_dir=tmp_path,
        diffraction_order=2,
        rough_fourier_orders=5,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert isinstance(sweep, MultilayerThetaSearchSweepResult)
    assert sweep.summary_csv_path.exists()
    assert sweep.all_orders_csv_path.exists()
    assert sweep.energy_efficiency_plot_path.exists()
    assert sweep.workflow_plot_path.exists()
    assert sweep.theta_scan_directory.exists()
    assert (sweep.theta_scan_directory / "theta_scan_1800eV.csv").exists()
    assert (sweep.theta_scan_directory / "theta_scan_1800eV.png").exists()
    assert (sweep.theta_scan_directory / "theta_scan_2000eV.csv").exists()
    assert (sweep.theta_scan_directory / "theta_scan_2000eV.png").exists()
    assert sweep.profile_plot_path is None
    assert sweep.stack_plot_path is None
    summary_header = sweep.summary_csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert summary_header == (
        "energy_ev,selected_grazing_angle_deg,selected_efficiency,precise_fwhm_deg,"
        "retry_triggered,retry_attempts,retry_status,selected_efficiency_is_exact_zero,"
        "selected_efficiency_below_retry_threshold,theta_tracking_center_mode,"
        "theta_tracking_auto_classification,theta_tracking_previous_energy_ev,"
        "theta_tracking_previous_grazing_angle_deg,theta_tracking_used_previous_theta,"
        "theta_tracking_bragg_fallback_triggered,theta_tracking_continuity_rejected,"
        "precise_peak_selection_mode_requested,precise_peak_selection_mode_used,"
        "precise_peak_fit_fallback_used,precise_peak_fitted_center_deg,precise_peak_fitted_fwhm_deg"
    )


def test_theta_search_diagnostics_fit_fields_round_trip_through_case_record() -> None:
    diagnostics = simulation_module.ThetaSearchDiagnostics(
        estimated_grazing_angle_deg=1.2,
        rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
        rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
        precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
        precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
        selected_grazing_angle_deg=1.2,
        selected_efficiency=0.35,
        precise_fwhm_deg=0.04,
        precise_peak_selection_mode_requested="voigt",
        precise_peak_selection_mode_used="gauss",
        precise_peak_fit_fallback_used=True,
        precise_peak_fitted_center_deg=1.201,
        precise_peak_fitted_fwhm_deg=0.038,
        precise_peak_fitted_theta_deg=np.asarray([1.18, 1.20, 1.22], dtype=float),
        precise_peak_fitted_efficiencies=np.asarray([0.32, 0.36, 0.31], dtype=float),
    )
    case = CaseExecutionResult(
        case_id="theta-search-00000000",
        index=0,
        label=None,
        energy_ev=1800.0,
        grazing_angle_deg=1.2,
        orders=np.asarray([-1, 0, 1], dtype=int),
        selected_efficiency=0.35,
        selected_diffraction_angle_deg=2.0,
        efficiency_all=np.asarray([0.35, 0.0, 0.0], dtype=float),
        diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
        status="ok",
        theta_search_diagnostics=diagnostics,
    )

    record = simulation_module._case_result_to_record(case)
    restored = simulation_module._case_result_from_record(record)

    assert restored.theta_search_diagnostics is not None
    assert restored.theta_search_diagnostics.precise_peak_selection_mode_requested == "voigt"
    assert restored.theta_search_diagnostics.precise_peak_selection_mode_used == "gauss"
    assert restored.theta_search_diagnostics.precise_peak_fit_fallback_used is True
    assert restored.theta_search_diagnostics.precise_peak_fitted_center_deg == pytest.approx(1.201)
    assert restored.theta_search_diagnostics.precise_peak_fitted_fwhm_deg == pytest.approx(0.038)
    assert restored.theta_search_diagnostics.precise_peak_fitted_theta_deg is not None
    assert restored.theta_search_diagnostics.precise_peak_fitted_efficiencies is not None
    assert np.allclose(restored.theta_search_diagnostics.precise_peak_fitted_theta_deg, [1.18, 1.20, 1.22])
    assert np.allclose(restored.theta_search_diagnostics.precise_peak_fitted_efficiencies, [0.32, 0.36, 0.31])


def test_case_execution_result_round_trip_preserves_peak_memory_bytes() -> None:
    case = CaseExecutionResult(
        case_id="case-1",
        index=0,
        label="case",
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        orders=np.asarray([-1, 0, 1], dtype=int),
        selected_efficiency=0.1,
        selected_diffraction_angle_deg=2.0,
        efficiency_all=np.asarray([0.1, 0.2, 0.3], dtype=float),
        diffraction_angle_all=np.asarray([1.0, 2.0, 3.0], dtype=float),
        status="ok",
        peak_memory_bytes=123456,
        wall_seconds=9.87,
    )

    record = simulation_module._case_result_to_record(case)
    restored = simulation_module._case_result_from_record(record)

    assert restored.peak_memory_bytes == 123456
    assert restored.wall_seconds == pytest.approx(9.87)


def test_public_examples_do_not_expose_quick_mode_flags() -> None:
    for example_path in EXAMPLE_SCRIPT_PATHS:
        source = example_path.read_text(encoding="utf-8")
        assert "--quick" not in source
        assert "quick_mode" not in source
        assert "Quick mode" not in source


def test_optimizer_example_assets_exist() -> None:
    expected_paths = [
        OPTIMIZER_EXAMPLE_ROOT / "0_fit_laminar_grating.py",
        OPTIMIZER_EXAMPLE_ROOT / "1_run_simulation_design_parameters.py",
        OPTIMIZER_EXAMPLE_ROOT / "2_run_simulation_fitted_parameters.py",
        OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py",
        OPTIMIZER_EXAMPLE_ROOT / "measured_alpha4deg_order1.csv",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_Si_cxro.txt",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_Pt_cxro.txt",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_C_cxro.txt",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing optimizer example asset: {path}"


def test_optimizer_example_scripts_compile() -> None:
    import py_compile

    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "0_fit_laminar_grating.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "1_run_simulation_design_parameters.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "2_run_simulation_fitted_parameters.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py"), doraise=True)


def test_optimizer_example_plot_uses_evaluation_energies() -> None:
    plot_source = (OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py").read_text(
        encoding="utf-8"
    )
    assert "evaluation_energies_ev" in plot_source
    assert "Optimization energies" in plot_source


def test_multilayer_theta_search_docs_use_grouped_canonical_arguments() -> None:
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "simulation"
        / "multilayer_theta_search"
        / "multilayer_theta_search.py"
    )
    tutorial_path = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "multilayer-theta-search.md"

    example_source = example_path.read_text(encoding="utf-8")
    tutorial_source = tutorial_path.read_text(encoding="utf-8")
    example_call = example_source.split("run_multilayer_theta_search_sweep(", maxsplit=1)[1].split(")\n", maxsplit=1)[0]
    tutorial_call = tutorial_source.split("run_multilayer_theta_search_sweep(", maxsplit=1)[1].split(")\n", maxsplit=1)[0]
    assert "run_multilayer_theta_search(" not in tutorial_source

    for call_block in (example_call, tutorial_call):
        assert call_block.index("multilayer_bragg_order") < call_block.index("rough_scan_half_width_deg")
        assert call_block.index("rough_scan_half_width_deg") < call_block.index("rough_fourier_orders")
        assert call_block.index("rough_fourier_orders") < call_block.index("rough_x_resolution_nm")
        assert call_block.index("rough_x_resolution_nm") < call_block.index("fine_scan_half_width_deg")
        assert call_block.index("fine_scan_half_width_deg") < call_block.index("fine_fourier_orders")
        assert call_block.index("fine_fourier_orders") < call_block.index("fine_x_resolution_nm")
        assert call_block.index("fine_x_resolution_nm") < call_block.index("final_fourier_orders")
        assert call_block.index("final_fourier_orders") < call_block.index("final_x_resolution_nm")
        assert call_block.index("final_x_resolution_nm") < call_block.index("precise_peak_selection_mode")
        assert "\n        fourier_orders=" not in call_block
        assert "\n        x_resolution_nm=" not in call_block
        assert "\n        z_resolution_nm=" not in call_block


def test_single_simulation_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    result = run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        diffraction_order=1,
        fourier_orders=3,
    )
    csv_path = tmp_path / "single_simulation.csv"
    profile_path = tmp_path / "single_simulation_profile.png"

    write_all_orders_csv(result, csv_path)
    grating.plot_profile(profile_path)

    assert result.selected_efficiency >= 0.0
    assert csv_path.exists()
    assert profile_path.exists()


def test_fixed_angle_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    cases = fixed_angle_cases(grating=grating, energies_ev=[200.0], grazing_angle_deg=4.0)
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "fixed_angle_all_orders.csv"
    orders_plot_path = tmp_path / "fixed_angle_orders_1_3.png"

    write_all_orders_csv(results, csv_path)
    plot_order_subset(results, orders_plot_path, diffraction_orders=[1, 2, 3], title="Fixed-angle parity")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()
    assert orders_plot_path.exists()


def test_monochromator_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_monochromator_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    cases = monochromator_cases(grating=grating, energies_ev=[200.0], diffraction_order=1, cff=2.25)
    runner = BatchSimulationRunner(default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "monochromator_all_orders.csv"
    orders_plot_path = tmp_path / "monochromator_orders_1_3.png"

    write_all_orders_csv(results, csv_path)
    plot_order_subset(results, orders_plot_path, diffraction_orders=[1, 2, 3], title="Monochromator parity")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()
    assert orders_plot_path.exists()


def test_blazed_multilayer_sweep_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.0,
            gamma=0.4,
            n_bilayers=50,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=1.0,
    )
    cases = monochromator_cases(grating=grating, energies_ev=[500.0], diffraction_order=1, cff=2.25)
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "blazed_multilayer_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


def test_blazed_multilayer_memory_comparison_example_structure() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "simulation"
        / "blazed_multilayer_memory_comparison"
        / "blazed_multilayer_memory_comparison.py"
    )

    source = script_path.read_text(encoding="utf-8")

    assert "rp.MultilayerStack(" in source
    assert "rp.BlazedGrating(" in source
    assert "rp.monochromator_cases(" in source
    assert "rp.BatchSimulationRunner(" in source
    assert "show_progress=True" in source
    assert 'sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))' in source
    assert '"memory_mode": "standard"' in source
    assert '"memory_mode": "low_memory"' in source
    assert 'profile_memory": True' in source
    assert "blazed_multilayer_memory_comparison.csv" in source
    assert "blazed_multilayer_memory_comparison.png" in source
    assert "blazed_multilayer_profile.png" in source
    assert "multilayer_stack_schematic.png" in source


def test_energy_angle_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_blazed_multilayer_angle_parity_grating()
    cases = energy_angle_cases(grating=grating, energy_angle_pairs=[(1800.0, 8.0)])
    runner = BatchSimulationRunner(default_diffraction_order=2, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "energy_angle_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


def test_multilayer_theta_search_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_blazed_multilayer_angle_parity_grating()
    sweep = run_multilayer_theta_search_sweep(
        grating=grating,
        energies_ev=[1800.0],
        output_dir=tmp_path,
        diffraction_order=2,
        multilayer_bragg_order=1,
        rough_scan_half_width_deg=0.5,
        rough_scan_points=21,
        fine_scan_half_width_deg=0.1,
        fine_scan_points=21,
        rough_fourier_orders=3,
        fine_fourier_orders=3,
        final_fourier_orders=3,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        fine_x_resolution_nm=1.0,
        fine_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
        precise_peak_selection_mode="voigt",
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-3,
        max_zero_efficiency_retries=1,
        show_progress=False,
        live_plot=False,
        on_error="fail_fast",
        save_profile_plot=False,
        save_stack_plot=False,
        backend="numba",
    )

    assert len(sweep.batch_result.cases) == 1
    assert sweep.batch_result.cases[0].status == "ok"
    assert sweep.summary_csv_path.exists()
    assert sweep.theta_scan_directory.exists()


def test_batch_user_cases_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(depth_nm=14.9, x_resolution_nm=1.0, z_resolution_nm=1.0)
    grazing_angle_deg = float(
        monochromator_grazing_angles_deg(
            [1000.0],
            period_lpermm=grating.period_lpermm,
            diffraction_order=1,
            cff=2.25,
        )[0]
    )
    user_cases = [
        {
            "case_id": "user-laminar-depth-015",
            "label": "Laminar grating at depth 14.9 nm",
            "grating": grating,
            "energy_ev": 1000.0,
            "grazing_angle_deg": grazing_angle_deg,
            "diffraction_order": 1,
            "depth_nm": 14.9,
        }
    ]
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(user_cases))
    csv_path = tmp_path / "batch_user_cases_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


def test_multilayer_theta_search_sweep_accumulates_elapsed_time_across_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        return SingleSimulationResult(
            energy_ev=energy_ev,
            grazing_angle_deg=1.2,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=0.35,
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([0.35, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=int(kwargs["final_fourier_orders"]),
            theta_search_diagnostics=diagnostics,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)

    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        resume=False,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    metadata_path = tmp_path / "checkpoints" / "metadata.json"
    first_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first_metadata["cumulative_elapsed_seconds"] = 12.0
    metadata_path.write_text(json.dumps(first_metadata, indent=2), encoding="utf-8")
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert first.current_run_elapsed_seconds >= 0.0
    assert first.total_elapsed_seconds == pytest.approx(first.current_run_elapsed_seconds)
    assert resumed.current_run_elapsed_seconds >= 0.0
    assert resumed.total_elapsed_seconds >= 12.0
    assert metadata["last_run_elapsed_seconds"] == pytest.approx(resumed.current_run_elapsed_seconds)
    assert metadata["cumulative_elapsed_seconds"] == pytest.approx(resumed.total_elapsed_seconds)


def test_multilayer_theta_search_sweep_retries_on_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_count = {"value": 0}

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        call_count["value"] += 1
        selected_efficiency = 5e-5 if call_count["value"] == 1 else 2e-4
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=selected_efficiency,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-4,
        max_zero_efficiency_retries=1,
    )

    assert call_count["value"] == 2
    assert len(sweep.batch_result.cases) == 1
    result = sweep.batch_result.cases[0]
    assert result.retry_triggered is True
    assert result.retry_attempts == 1
    assert result.retry_status == "recovered"
    assert result.selected_efficiency_below_retry_threshold is False


def test_batch_runner_theta_retry_uses_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_run_case_payload(
        payload: dict[str, object],
        *,
        diagnostic_callback: object = None,
    ) -> SingleSimulationResult:
        del diagnostic_callback
        calls.append(float(payload["energy_ev"]))
        selected_efficiency = 5e-5 if len(calls) == 1 else 2e-4
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=selected_efficiency,
        )

    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)
    cases = list(
        multilayer_theta_search_cases(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[2000.0],
        )
    )
    runner = BatchSimulationRunner(
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-4,
        max_zero_efficiency_retries=1,
    )
    results = list(runner.run_cases(cases))

    assert len(calls) == 2
    assert len(results) == 1
    result = results[0]
    assert result.retry_triggered is True
    assert result.retry_attempts == 1
    assert result.retry_status == "recovered"
    assert result.selected_efficiency_below_retry_threshold is False


def test_retry_selected_efficiency_threshold_validation() -> None:
    with pytest.raises(ValueError, match="retry_selected_efficiency_threshold"):
        BatchSimulationRunner(retry_selected_efficiency_threshold=-1.0)
    with pytest.raises(ValueError, match="retry_selected_efficiency_threshold"):
        run_multilayer_theta_search_sweep(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[1800.0],
            output_dir=Path("unused"),
            retry_selected_efficiency_threshold=-1.0,
        )


def test_batch_runner_removed_constructor_args_raise_type_error() -> None:
    with pytest.raises(TypeError, match="total_cases"):
        BatchSimulationRunner(total_cases=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="live_theta_scan_plot"):
        BatchSimulationRunner(live_theta_scan_plot=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="max_total_reflected_efficiency"):
        BatchSimulationRunner(max_total_reflected_efficiency=2.0)  # type: ignore[call-arg]


def test_batch_runner_passes_min_reflected_efficiency_to_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def fake_run_case_payload(
        payload: dict[str, object],
        *,
        diagnostic_callback: object = None,
    ) -> SingleSimulationResult:
        del diagnostic_callback
        payloads.append(payload)
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=float(payload["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)
    runner = BatchSimulationRunner(min_reflected_efficiency=-0.125)

    list(
        runner.run_cases(
            [{"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
        )
    )

    assert len(payloads) == 1
    assert payloads[0]["min_efficiency"] == pytest.approx(-0.125)
    assert payloads[0]["max_total_reflected_efficiency"] == pytest.approx(1.05)


def test_multilayer_theta_search_auto_tracks_previous_for_dense_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        return fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles == [None, pytest.approx(0.5)]
    second = sweep.batch_result.cases[1]
    assert second.theta_tracking_center_mode == "tracked_previous"
    assert second.theta_tracking_auto_classification == "auto_dense"
    assert second.theta_tracking_used_previous_theta is True
    assert second.theta_tracking_previous_energy_ev == pytest.approx(1800.0)
    assert second.theta_tracking_previous_grazing_angle_deg == pytest.approx(0.5)


def test_multilayer_theta_search_auto_uses_bragg_for_sparse_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0, 2300.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles[-1] is None
    last = sweep.batch_result.cases[-1]
    assert last.theta_tracking_center_mode == "bragg"
    assert last.theta_tracking_auto_classification == "auto_sparse"
    assert last.theta_tracking_used_previous_theta is False


def test_multilayer_theta_search_tracking_override_uses_previous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        return fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 2300.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_tracking_energy_step_ev=1000.0,
    )

    assert initial_angles == [None, pytest.approx(0.5)]
    second = sweep.batch_result.cases[1]
    assert second.theta_tracking_center_mode == "tracked_previous"
    assert second.theta_tracking_auto_classification == "auto_dense"


def test_multilayer_theta_search_tracked_branch_falls_back_to_bragg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[float, float | None]] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        initial = None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        calls.append((energy_ev, initial))
        if energy_ev == 1800.0:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.50, selected_efficiency=0.4)
        if initial is not None:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.70, selected_efficiency=5e-5)
        return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.49, selected_efficiency=0.3)

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        retry_on_selected_efficiency_zero=False,
        retry_selected_efficiency_threshold=1e-4,
    )

    assert calls == [(1800.0, None), (1802.0, pytest.approx(0.5)), (1802.0, None)]
    second = sweep.batch_result.cases[1]
    assert second.grazing_angle_deg == pytest.approx(0.49)
    assert second.selected_efficiency == pytest.approx(0.3)
    assert second.theta_tracking_center_mode == "bragg"
    assert second.theta_tracking_bragg_fallback_triggered is True


def test_multilayer_theta_search_continuity_guard_rejects_upward_jump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        initial = None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        if energy_ev == 1800.0:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.50, selected_efficiency=0.4)
        if initial is not None:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.60, selected_efficiency=0.39)
        return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.49, selected_efficiency=0.38)

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        retry_on_selected_efficiency_zero=False,
    )

    second = sweep.batch_result.cases[1]
    assert second.grazing_angle_deg == pytest.approx(0.49)
    assert second.theta_tracking_bragg_fallback_triggered is True
    assert second.theta_tracking_continuity_rejected is True


def test_multilayer_theta_search_bragg_mode_matches_legacy_centering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="bragg",
    )

    assert initial_angles == [None, None]
    assert sweep.batch_result.cases[1].theta_tracking_center_mode == "bragg"


def test_multilayer_theta_search_parallel_auto_uses_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=3,
    )

    assert len(submitted_payloads) == 3
    assert submitted_payloads[0].get("initial_grazing_angle_deg") is None


def test_multilayer_theta_search_parallel_auto_calibrates_first_case_before_pool_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []
    submitted_payloads: list[dict[str, object]] = []
    calibrated_inputs: list[tuple[int, int | None]] = []
    executor_max_workers: list[int] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del mp_context, initializer
            executor_max_workers.append(max_workers)

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    def fake_calibrate_auto_max_workers_from_result(
        *,
        pending_case_count: int,
        available_memory_bytes: int | None,
    ) -> int:
        calibrated_inputs.append((pending_case_count, available_memory_bytes))
        return 2

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(
        simulation_module,
        "_calibrate_auto_max_workers_from_result",
        fake_calibrate_auto_max_workers_from_result,
    )
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )

    assert calls == [1800.0]
    assert [float(payload["energy_ev"]) for payload in submitted_payloads] == [1802.0, 1804.0]
    assert calibrated_inputs == [(3, 123456789)]
    assert executor_max_workers == [2]


def test_multilayer_theta_search_parallel_auto_falls_back_when_no_lower_result_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert submitted_payloads[1].get("initial_grazing_angle_deg") is None


def test_multilayer_theta_search_parallel_auto_uses_available_lower_result_for_later_dense_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object], order: int) -> None:
            self._result = result
            self.order = order

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers
            self.counter = 0

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            self.counter += 1
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5 if float(payload["energy_ev"]) == 1800.0 else 0.49,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result}, self.counter)

    def fake_as_completed(futures: object) -> Iterator[FakeFuture]:
        future_list = list(futures)
        future_list.sort(key=lambda future: future.order)
        return iter(future_list)

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", fake_as_completed)

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert submitted_payloads[2].get("initial_grazing_angle_deg") == pytest.approx(0.5)


def test_multilayer_theta_search_parallel_auto_voigt_calibration_runs_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []
    submitted_payloads: list[dict[str, object]] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_peak_selection_mode_used="voigt",
            precise_peak_fitted_theta_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_peak_fitted_efficiencies=np.asarray([0.30, 0.36, 0.32], dtype=float),
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del max_workers, mp_context, initializer

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=1.2,
                    selected_efficiency=0.35,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(simulation_module, "_calibrate_auto_max_workers_from_result", lambda **kwargs: 2)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        precise_peak_selection_mode="voigt",
        max_workers="auto",
    )

    assert calls == [1800.0]
    assert [float(payload["energy_ev"]) for payload in submitted_payloads] == [1802.0, 1804.0]
    assert [case.energy_ev for case in sweep.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]


def test_multilayer_theta_search_progress_updates_only_on_completed_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    updates: list[int] = []
    postfixes: list[str] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit

        def update(self, value: int = 1) -> None:
            updates.append(value)

        def set_postfix_str(self, value: str) -> None:
            postfixes.append(value)

        def close(self) -> None:
            return None

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=True,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert updates == [1, 1, 1]
    assert postfixes
    assert any("active=" in value and "queued=" in value and "done=" in value for value in postfixes)


def test_multilayer_theta_search_sweep_resume_skips_completed_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
            precise_peak_selection_mode_used="voigt",
            precise_peak_fitted_theta_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_peak_fitted_efficiencies=np.asarray([0.30, 0.36, 0.32], dtype=float),
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert len(first.batch_result.cases) == 2
    assert calls == [1800.0, 1802.0, 1804.0]
    assert [case.energy_ev for case in resumed.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]
    assert (resumed.theta_scan_directory / "theta_scan_1800eV.png").exists()
    assert (resumed.theta_scan_directory / "theta_scan_1802eV.png").exists()
    assert (resumed.theta_scan_directory / "theta_scan_1804eV.png").exists()


def test_multilayer_theta_search_sweep_resume_rebuilds_previous_theta_tracking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=theta,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )
    initial_angles.clear()
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles == [pytest.approx(0.5)]
    second = [case for case in resumed.batch_result.cases if case.energy_ev == 1802.0][0]
    assert second.theta_tracking_previous_energy_ev == pytest.approx(1800.0)
    assert second.theta_tracking_previous_grazing_angle_deg == pytest.approx(0.5)


def test_multilayer_theta_search_sweep_auto_resume_does_not_rerun_completed_calibration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del max_workers, mp_context, initializer

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(simulation_module, "_calibrate_auto_max_workers_from_result", lambda **kwargs: 2)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    checkpoint_dir = tmp_path / "checkpoints"
    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        resume=False,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )

    assert len(first.batch_result.cases) == 2
    assert calls == [1800.0, 1804.0]
    assert [case.energy_ev for case in resumed.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]


def test_multilayer_theta_search_sweep_resume_progress_preloads_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    updates: list[int] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit

        def update(self, value: int = 1) -> None:
            updates.append(value)

        def set_postfix_str(self, value: str) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    updates.clear()
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=True,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert updates == [1, 1]


def test_multilayer_theta_search_sweep_resume_ignores_malformed_checkpoint_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "results.jsonl").write_text('{"broken": \n', encoding="utf-8")
    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "run",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert calls == [1800.0]


def test_laminar_multilayer_geometry_matches_octave_reference(tmp_path: Path) -> None:
    octave_reference = run_octave_laminar_multilayer_reference(tmp_path)
    grating = build_multilayer_parity_grating()
    stack = grating.resolved_stack()

    x_grid = grating._build_x_grid(num_periods=1)
    z_grid = grating._build_solver_z_grid(stack)
    surface = grating._surface_profile_on_grid(x_grid, num_periods=1)

    assert np.allclose(x_grid, octave_reference["x"])
    assert np.allclose(z_grid, octave_reference["z"])
    assert np.allclose(surface, octave_reference["surface"])


def test_laminar_multilayer_material_map_matches_octave_reference(tmp_path: Path) -> None:
    octave_reference = run_octave_laminar_multilayer_reference(tmp_path)
    grating = build_multilayer_parity_grating()
    stack = grating.resolved_stack()
    assert isinstance(stack, MultilayerStack)

    x_grid = grating._build_x_grid(num_periods=1)
    z_grid = grating._build_solver_z_grid(stack)
    surface = grating._surface_profile_on_grid(x_grid, num_periods=1)
    index_grid = grating._build_refractive_index_grid(
        x_grid=x_grid,
        z_grid=z_grid,
        surface=surface,
        coating_stack=stack,
        photon_energy_ev=1000.0,
        n_inc=1.0 + 0.0j,
    )

    material_id = np.full(index_grid.shape, 0, dtype=int)
    n_cr = resolve_refractive_index(CR, 1000.0)
    n_c = resolve_refractive_index(C, 1000.0)
    material_id[np.isclose(index_grid, n_cr)] = 1
    material_id[np.isclose(index_grid, n_c)] = 2
    material_id[np.isclose(index_grid, 1.0 + 0.0j)] = 3

    assert np.array_equal(material_id, octave_reference["material_id"].astype(int))


def test_laminar_multilayer_solver_matches_octave_reference(tmp_path: Path) -> None:
    octave_reference = run_octave_laminar_multilayer_reference(tmp_path)
    grating = build_multilayer_parity_grating()
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=5,
        grazing_angle_deg=4.0,
        max_reflected_efficiency=2.0,
        max_total_reflected_efficiency=2.0,
    )

    python_result = simulation.run_single(1000.0)
    octave_orders = octave_reference["solver"][:, 0].astype(int)

    for order, octave_theta, octave_efficiency in octave_reference["solver"]:
        match = np.where(python_result["orders"] == int(order))[0]
        assert match.size == 1
        idx = int(match[0])
        assert python_result["diffraction_angle_all"][idx] == pytest.approx(90.0 - octave_theta, abs=1e-9)
        assert python_result["efficiency_all"][idx] == pytest.approx(octave_efficiency, abs=1e-3)

    assert set(octave_orders).issubset(set(python_result["orders"].tolist()))


def test_laminar_multilayer_solver_stays_physical_for_full_example(tmp_path: Path) -> None:
    octave_reference = run_octave_laminar_multilayer_reference_with_parameters(
        tmp_path,
        n_bilayers=40,
        z_resolution_nm=0.1,
        x_resolution_nm=1.0,
        fourier_orders=5,
    )
    grating = build_multilayer_solver_regression_grating()
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=5,
        grazing_angle_deg=4.0,
    )

    python_result = simulation.run_single(1000.0)

    assert float(np.max(python_result["efficiency_all"])) <= 1.05
    assert float(np.min(python_result["efficiency_all"])) >= -1e-8
    assert float(np.sum(python_result["efficiency_all"])) <= 1.05

    for order, octave_theta, octave_efficiency in octave_reference["solver"]:
        matches = np.where(python_result["orders"] == int(order))[0]
        if matches.size == 0:
            continue
        idx = int(matches[0])
        assert python_result["diffraction_angle_all"][idx] == pytest.approx(90.0 - octave_theta, abs=1e-9)
        assert python_result["efficiency_all"][idx] == pytest.approx(octave_efficiency, abs=2e-3)


def test_blazed_multilayer_angle_sweep_matches_reticolo_v9_reference(tmp_path: Path) -> None:
    octave_reference = np.atleast_2d(run_octave_blazed_multilayer_angle_reference(tmp_path))
    grating = build_blazed_multilayer_angle_parity_grating()
    expected_python_orders = np.arange(-5, 6, dtype=int)
    expected_reticolo_orders = np.arange(-5, 2, dtype=int)

    for grazing_angle_deg in np.arange(75, 82, dtype=float) / 10.0:
        simulation = RCWASimulation(
            grating=grating,
            diffraction_order=1,
            fourier_orders=5,
            grazing_angle_deg=float(grazing_angle_deg),
            max_reflected_efficiency=2.0,
            max_total_reflected_efficiency=2.0,
        )
        python_result = simulation.run_single(500.0)
        assert np.array_equal(python_result["orders"], expected_python_orders)

        octave_rows = octave_reference[np.isclose(octave_reference[:, 0], grazing_angle_deg)]
        assert np.array_equal(octave_rows[:, 1].astype(int), expected_reticolo_orders)
        for _, order, octave_theta, octave_efficiency in octave_rows:
            match = np.where(python_result["orders"] == int(order))[0]
            assert match.size == 1
            idx = int(match[0])
            assert python_result["diffraction_angle_all"][idx] == pytest.approx(90.0 - octave_theta, abs=1e-6)
            assert python_result["efficiency_all"][idx] == pytest.approx(octave_efficiency, abs=2e-3)


def test_blazed_single_layer_200ev_matches_reticolo_reference_more_closely() -> None:
    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.729,
        anti_blaze_angle_deg=5.597,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=30.0,
        x_resolution_nm=1.0,
        z_resolution_nm=0.1,
    )
    grazing_angle_deg = float(
        monochromator_grazing_angles_deg(
            np.asarray([200.0], dtype=float),
            period_lpermm=600,
            diffraction_order=1,
            cff=2.25,
        )[0]
    )
    simulation = RCWASimulation(
        grating=grating,
        diffraction_order=1,
        fourier_orders=20,
        grazing_angle_deg=grazing_angle_deg,
        max_reflected_efficiency=2.0,
        max_total_reflected_efficiency=2.0,
    )

    python_result = simulation.run_single(200.0)
    order_index = int(np.where(python_result["orders"] == -1)[0][0])

    assert python_result["diffraction_angle_all"][order_index] == pytest.approx(5.517321, abs=1e-6)
    assert python_result["efficiency_all"][order_index] == pytest.approx(0.115515, abs=3e-3)
