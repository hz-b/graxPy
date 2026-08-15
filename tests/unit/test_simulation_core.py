from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from grax import RoughnessSpec
from grax import peak_fitting as peak_fitting_module
from grax import simulation as simulation_module
from grax.gratings import LaminarGrating
from grax.rcwa_1d import (
    DiffractionResult,
    _apply_debye_waller_roughness,
    _debye_waller_roughness_factor,
    debye_waller_roughness_diagnostics,
    res2,
)
from grax.simulation import (
    BatchSimulationResult,
    BatchSimulationRunner,
    CaseExecutionResult,
    MultilayerThetaSearchSweepResult,
    GratingSimulation,
    SingleSimulationResult,
    efficiency_for_order,
    energy_angle_cases,
    estimate_multilayer_bragg_angle_deg,
    fixed_angle_cases,
    load_experimental_csv,
    monochromator_cases,
    monochromator_grazing_angles_deg,
    multilayer_theta_search_cases,
    plot_order_subset,
    run_multilayer_theta_search,
    run_multilayer_theta_search_sweep,
    run_simulation,
    write_all_orders_csv,
)
from grax.simulation import batch as simulation_batch_module
from grax.simulation import core as simulation_core_module
from tests.simulation_helpers import (
    C,
    CR,
    PT,
    SI,
    build_blazed_multilayer_angle_parity_grating,
    build_test_grating,
    fake_single_result,
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


def test_per_layer_debye_waller_combines_sigmas_in_quadrature() -> None:
    from grax.stacks import LayerSpec, assemble_custom_stack

    stack = assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=[
            LayerSpec(material=CR, thickness_nm=2.0, roughness_sigma_nm=0.3),
            LayerSpec(material=C, thickness_nm=3.0, roughness_sigma_nm=0.4),
        ],
    )
    grating = LaminarGrating(
        substrate_material=SI,
        coating_stack=stack,
        roughness=RoughnessSpec(kind="debye-waller", sigma_nm=0.0),
    )

    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )

    # interfaces: [substrate=0.0, top-of-Cr=0.3, top-of-C=0.4] -> quadrature.
    expected = float(np.sqrt(0.0**2 + 0.3**2 + 0.4**2))
    assert result.roughness_sigma_nm == pytest.approx(expected)


def test_debye_waller_without_per_layer_overrides_keeps_single_sigma() -> None:
    grating = build_test_grating()
    grating.roughness = RoughnessSpec(kind="debye-waller", sigma_nm=0.5)

    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )

    assert result.roughness_sigma_nm == pytest.approx(0.5)


def test_rcwa_simulation_runs_for_multiple_energies() -> None:
    grating = build_test_grating()
    simulation = GratingSimulation(
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
    simulation = GratingSimulation(
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
    simulation = GratingSimulation(
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


def test_run_simulation_num_supercells_one_matches_baseline() -> None:
    grating_default = build_test_grating()
    grating_explicit = LaminarGrating(
        substrate_material=grating_default.substrate_material,
        layer_material=grating_default.layer_material,
        layer_thickness_nm=grating_default.layer_thickness_nm,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.0, num_supercells=1, num_realizations=1),
    )

    result_default = run_simulation(
        grating=grating_default,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )
    result_explicit = run_simulation(
        grating=grating_explicit,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )

    assert result_explicit.num_supercells == 1
    assert np.allclose(result_default.orders, result_explicit.orders)
    assert np.allclose(result_default.efficiency_all, result_explicit.efficiency_all, atol=1e-6)


def test_run_simulation_num_supercells_produces_fractional_orders() -> None:
    base_grating = build_test_grating()
    grating = LaminarGrating(
        substrate_material=base_grating.substrate_material,
        layer_material=base_grating.layer_material,
        layer_thickness_nm=base_grating.layer_thickness_nm,
        x_resolution_nm=2.0,
        roughness=RoughnessSpec(
            kind="random-interface",
            sigma_nm=0.3,
            seed=1,
            correlation_length_nm=100.0,
            num_supercells=3,
        ),
    )

    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )

    assert result.num_supercells == 3
    spacing = np.diff(np.sort(result.orders))
    assert np.allclose(spacing, spacing[0])
    assert spacing[0] == pytest.approx(1.0 / 3.0)
    efficiency = efficiency_for_order(result.orders, result.efficiency_all, diffraction_order=1)
    assert np.isfinite(efficiency)


def test_run_simulation_warns_when_effective_fourier_orders_is_large() -> None:
    grating = LaminarGrating(
        substrate_material=build_test_grating().substrate_material,
        layer_material=build_test_grating().layer_material,
        layer_thickness_nm=build_test_grating().layer_thickness_nm,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.1, num_supercells=4, num_realizations=1),
    )

    with pytest.warns(UserWarning, match="effective Fourier orders"):
        run_simulation(
            grating=grating,
            energy_ev=500.0,
            grazing_angle_deg=1.0,
            fourier_orders=15,
        )


def test_run_simulation_num_realizations_default_and_debye_waller_result_fields() -> None:
    debye_grating = LaminarGrating(
        substrate_material=build_test_grating().substrate_material,
        layer_material=build_test_grating().layer_material,
        layer_thickness_nm=build_test_grating().layer_thickness_nm,
        roughness=RoughnessSpec(kind="debye-waller", sigma_nm=0.5),
    )
    no_roughness_result = run_simulation(
        grating=build_test_grating(),
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )
    debye_result = run_simulation(
        grating=debye_grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )

    assert no_roughness_result.num_realizations == 1
    assert debye_result.num_realizations == 1


def test_run_simulation_averages_efficiency_across_realizations() -> None:
    base_grating = build_test_grating()
    averaged_grating = LaminarGrating(
        substrate_material=base_grating.substrate_material,
        layer_material=base_grating.layer_material,
        layer_thickness_nm=base_grating.layer_thickness_nm,
        x_resolution_nm=2.0,
        roughness=RoughnessSpec(
            kind="random-interface",
            sigma_nm=0.3,
            seed=123,
            correlation_length_nm=100.0,
            num_realizations=4,
        ),
    )

    averaged_result = run_simulation(
        grating=averaged_grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=3,
    )

    assert averaged_result.num_realizations == 4

    realization_seeds = averaged_grating.roughness.realization_seeds()
    assert len(realization_seeds) == 4

    individual_efficiencies = []
    for realization_seed in realization_seeds:
        realization_grating = LaminarGrating(
            substrate_material=base_grating.substrate_material,
            layer_material=base_grating.layer_material,
            layer_thickness_nm=base_grating.layer_thickness_nm,
            x_resolution_nm=2.0,
            roughness=RoughnessSpec(
                kind="random-interface",
                sigma_nm=0.3,
                seed=realization_seed,
                correlation_length_nm=100.0,
                num_realizations=1,
            ),
        )
        realization_result = run_simulation(
            grating=realization_grating,
            energy_ev=100.0,
            grazing_angle_deg=4.0,
            fourier_orders=3,
        )
        assert np.allclose(realization_result.orders, averaged_result.orders)
        individual_efficiencies.append(realization_result.efficiency_all)

    expected_mean = np.mean(individual_efficiencies, axis=0)
    assert np.allclose(averaged_result.efficiency_all, expected_mean)
    assert averaged_result.selected_efficiency == pytest.approx(
        expected_mean[np.where(np.isclose(averaged_result.orders, -1.0))[0][0]]
    )


def test_run_simulation_num_realizations_one_matches_direct_solve() -> None:
    base_grating = build_test_grating()
    grating = LaminarGrating(
        substrate_material=base_grating.substrate_material,
        layer_material=base_grating.layer_material,
        layer_thickness_nm=base_grating.layer_thickness_nm,
        roughness=RoughnessSpec(kind="random-interface", sigma_nm=0.2, seed=7, num_realizations=1),
    )

    result_a = run_simulation(grating=grating, energy_ev=100.0, grazing_angle_deg=4.0, fourier_orders=3)
    result_b = run_simulation(grating=grating, energy_ev=100.0, grazing_angle_deg=4.0, fourier_orders=3)

    assert result_a.num_realizations == 1
    assert np.allclose(result_a.efficiency_all, result_b.efficiency_all)


def test_write_all_orders_csv_formats_fractional_and_integer_orders(tmp_path: Path) -> None:
    result = SingleSimulationResult(
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        orders=np.array([-1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0]),
        selected_efficiency=0.1,
        selected_diffraction_angle_deg=5.0,
        efficiency_all=np.array([0.1, 0.2, 0.3, 0.2, 0.1]),
        diffraction_angle_all=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        diffraction_order=1,
        fourier_orders=3,
        num_supercells=3,
    )
    csv_path = tmp_path / "supercell_all_orders.csv"

    write_all_orders_csv(result, csv_path)

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    order_cells = [row.split(",")[3] for row in rows[1:]]
    assert order_cells == ["-1", "-0.3333333333333333", "0", "0.3333333333333333", "1"]


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
                    "_memory_mode": "low_memory",
                    "profile_memory": True,
                }
            ]
        )
    )

    assert len(payloads) == 1
    assert payloads[0]["_memory_mode"] == "low_memory"
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
    monkeypatch.setattr(simulation_module, "_peak_process_memory_bytes", lambda: 1 * 1024**3)

    assert simulation_module._calibrate_auto_max_workers_from_result(
        pending_case_count=10,
        available_memory_bytes=8 * 1024**3,
    ) == 2


def test_batch_runner_auto_workers_calibration_uses_peak_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When the per-solve peak RSS exceeds the steady-state RSS, the peak must
    # drive worker sizing so large-supercell solves do not oversubscribe RAM.
    monkeypatch.setattr(simulation_module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(simulation_module, "AUTO_WORKER_MEMORY_RESERVE_BYTES", 2 * 1024**3)
    monkeypatch.setattr(simulation_module, "AUTO_WORKER_MEMORY_SAFETY_FACTOR", 1.0)
    monkeypatch.setattr(simulation_module, "_current_process_memory_bytes", lambda: 1 * 1024**3)
    monkeypatch.setattr(simulation_module, "_peak_process_memory_bytes", lambda: 3 * 1024**3)

    # usable = 8 - 2 = 6 GiB; 6 // 3 (peak) = 2, not 6 // 1 (steady) = 6.
    assert simulation_module._calibrate_auto_max_workers_from_result(
        pending_case_count=10,
        available_memory_bytes=8 * 1024**3,
    ) == 2


def test_batch_runner_auto_workers_calibration_falls_back_to_cpu_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(simulation_module, "_current_process_memory_bytes", lambda: None)
    monkeypatch.setattr(simulation_module, "_peak_process_memory_bytes", lambda: None)

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


def test_run_simulation_uses_solver_roughness_from_grating(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[float | None] = []

    class FakeEfficiencies:
        inc_top_reflected = DiffractionResult(
            order=np.asarray([-1, 0, 1], dtype=int),
            theta=np.asarray([1.0, 2.0, 3.0], dtype=float),
            efficiency=np.asarray([0.2, 0.1, 0.05], dtype=float),
            amplitude=np.asarray([1.0, 1.0, 1.0], dtype=complex),
        )

    def fake_res2(
        *args: object,
        roughness_sigma_nm: float | None = None,
        **kwargs: object,
    ) -> FakeEfficiencies:
        del args, kwargs
        captured.append(roughness_sigma_nm)
        return FakeEfficiencies()

    grating = build_test_grating()
    grating.roughness = RoughnessSpec(kind="debye-waller", sigma_nm=0.5)
    fake_profile = (np.asarray([0.0]), np.asarray([0]))
    monkeypatch.setattr(grating, "build_textures", lambda *args, **kwargs: ([], fake_profile))
    monkeypatch.setattr(simulation_core_module, "res0", lambda *args, **kwargs: object())
    monkeypatch.setattr(simulation_core_module, "res1", lambda *args, **kwargs: object())
    monkeypatch.setattr(simulation_core_module, "res2", fake_res2)

    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=1,
    )

    assert captured == [0.5]
    assert result.roughness_sigma_nm == pytest.approx(0.5)


def test_run_simulation_keeps_random_interface_roughness_out_of_res2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[float | None] = []

    class FakeEfficiencies:
        inc_top_reflected = DiffractionResult(
            order=np.asarray([-1, 0, 1], dtype=int),
            theta=np.asarray([1.0, 2.0, 3.0], dtype=float),
            efficiency=np.asarray([0.2, 0.1, 0.05], dtype=float),
            amplitude=np.asarray([1.0, 1.0, 1.0], dtype=complex),
        )

    def fake_res2(
        *args: object,
        roughness_sigma_nm: float | None = None,
        **kwargs: object,
    ) -> FakeEfficiencies:
        del args, kwargs
        captured.append(roughness_sigma_nm)
        return FakeEfficiencies()

    grating = build_test_grating()
    grating.roughness = RoughnessSpec(kind="random-interface", sigma_nm=0.5, num_realizations=1)
    fake_profile = (np.asarray([0.0]), np.asarray([0]))
    monkeypatch.setattr(grating, "build_textures", lambda *args, **kwargs: ([], fake_profile))
    monkeypatch.setattr(simulation_core_module, "res0", lambda *args, **kwargs: object())
    monkeypatch.setattr(simulation_core_module, "res1", lambda *args, **kwargs: object())
    monkeypatch.setattr(simulation_core_module, "res2", fake_res2)

    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=1,
    )

    assert captured == [None]
    assert result.roughness_sigma_nm is None


def test_run_simulation_rejects_explicit_and_grating_roughness() -> None:
    grating = build_test_grating()
    grating.roughness = RoughnessSpec(kind="debye-waller", sigma_nm=0.5)

    with pytest.raises(ValueError, match="either on the grating or as roughness_sigma_nm"):
        run_simulation(
            grating=grating,
            energy_ev=100.0,
            grazing_angle_deg=4.0,
            roughness_sigma_nm=0.5,
        )


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
        top_cap_material="Xx",
        top_cap_thickness_nm=0.7,
    )
    runner = BatchSimulationRunner(
        default_fourier_orders=5,
        on_error="fail_fast",
    )

    with pytest.raises(ValueError, match="not available"):
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
    simulation = GratingSimulation(grating=build_test_grating())

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
    monkeypatch.setattr(runner, "close_live_plot", lambda: None)

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
    monkeypatch.setattr(runner, "close_live_plot", lambda: None)

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


def test_batch_runner_closes_live_plot_after_run_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_simulation(**kwargs: object) -> SingleSimulationResult:
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=float(kwargs["grazing_angle_deg"]),
            selected_efficiency=0.3,
        )

    monkeypatch.setattr(simulation_module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(simulation_batch_module, "_refresh_interactive_figure", lambda figure: None)
    runner = BatchSimulationRunner(live_plot=True, live_plot_order_count=1)
    closed_figures = []
    original_close = simulation_batch_module.plt.close

    def close_spy(figure: object) -> None:
        closed_figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(simulation_batch_module.plt, "close", close_spy)

    results = list(
        runner.run_cases(
            [{"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
        )
    )

    assert len(results) == 1
    assert len(closed_figures) == 1
    assert runner._live_figure is None
    assert runner._live_axis is None
    assert runner._live_x_values == []
    assert runner._live_y_values == {1: []}


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


def test_case_execution_result_round_trip_preserves_fractional_orders() -> None:
    # Supercell roughness produces fractional physical orders; the parallel
    # batch path serializes results through these records, which previously
    # truncated orders to int (collapsing e.g. -1.2 and -1.4 onto -1).
    fractional_orders = np.asarray([-1.4, -1.2, -1.0, -0.8, 0.0, 1.0], dtype=float)
    case = CaseExecutionResult(
        case_id="case-1",
        index=0,
        label="case",
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        orders=fractional_orders,
        selected_efficiency=0.1,
        selected_diffraction_angle_deg=2.0,
        efficiency_all=np.linspace(0.1, 0.6, fractional_orders.size),
        diffraction_angle_all=np.linspace(1.0, 6.0, fractional_orders.size),
        status="ok",
    )

    record = simulation_module._case_result_to_record(case)
    restored = simulation_module._case_result_from_record(record)

    assert np.allclose(restored.orders, fractional_orders)
    assert np.unique(restored.orders).size == fractional_orders.size


