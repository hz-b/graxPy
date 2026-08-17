from __future__ import annotations

import ast
import inspect
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
import py_compile

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from grax import RoughnessSpec
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
from grax.simulation import core as simulation_core_module
from grax.simulation import (
    BatchSimulationResult,
    BatchSimulationRunner,
    CaseExecutionResult,
    MultilayerThetaSearchSweepResult,
    GratingSimulation,
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
from tests.simulation_helpers import (
    AU,
    C,
    CR,
    EXAMPLE_SCRIPT_PATHS,
    OPTICAL_CONSTANTS_DIR,
    OPTIMIZER_EXAMPLE_ROOT,
    PT,
    SI,
    build_blazed_multilayer_angle_parity_grating,
    build_laminar_example_grating,
    build_monochromator_example_grating,
    build_multilayer_parity_grating,
    build_multilayer_solver_regression_grating,
    build_test_grating,
    fake_single_result,
    run_octave_blazed_multilayer_angle_reference,
    run_octave_laminar_multilayer_reference,
    run_octave_laminar_multilayer_reference_with_parameters,
)


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
    simulation = GratingSimulation(
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
    simulation = GratingSimulation(
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
        simulation = GratingSimulation(
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
    simulation = GratingSimulation(
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


# ── TM (p-polarization) tests ─────────────────────────────────────────────────


def test_tm_smoke_run_simulation_returns_valid_result() -> None:
    """TM run_simulation returns a SingleSimulationResult with non-negative efficiency."""
    grating = build_test_grating()
    result = run_simulation(
        grating=grating,
        energy_ev=100.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
        polarization="p",
    )

    assert isinstance(result, SingleSimulationResult)
    assert result.polarization == "p"
    assert result.selected_efficiency >= 0.0
    assert np.isfinite(result.selected_efficiency)


def test_tm_parity_reticolo_exemple1_1d() -> None:
    """TM T0 matches RETICOLO exemple1_1D.m to within 1e-4.

    Geometry: λ=6µm, period=10µm, height=20µm, n_top=1, n_bottom=1.5,
    grating texture: ridges n=1.5 at [0,1µm] and [9,10µm], groove n=1.0 at [1,9µm].
    Normal incidence, 25 Fourier orders, TM (pol=-1).
    RETICOLO reference: T0=0.9111790287.
    """
    from grax.rcwa_1d import res0, res1, res2

    grating_texture = [
        np.array([1000.0, 9000.0]),
        np.array([1.5, 1.0], dtype=complex),
    ]
    parm = res0(-1)
    aa = res1(6000.0, 10000.0, [1.0, 1.5, grating_texture], 25, 0.0, parm)
    result = res2(aa, ([0.0, 20000.0, 0.0], [0, 2, 1]))

    idx0 = int(np.where(result.inc_top_reflected.order == 0)[0][0])
    T0 = result.inc_top_transmitted.efficiency[idx0]
    R0 = result.inc_top_reflected.efficiency[idx0]
    sum_rt = float(
        np.sum(result.inc_top_reflected.efficiency)
        + np.sum(result.inc_top_transmitted.efficiency)
    )

    assert T0 == pytest.approx(0.9111790287, abs=1e-4)
    assert sum_rt == pytest.approx(1.0, abs=1e-6)


def test_tm_fresnel_homogeneous_layer_energy_conservation() -> None:
    """TM homogeneous layer gives correct Fresnel R and energy conservation."""
    from grax.rcwa_1d import res0, res1, res2

    parm = res0(-1)
    aa = res1(6000.0, 10000.0, [1.0, 1.5, 1.5], 3, 0.0, parm)
    result = res2(aa, ([0.0, 20000.0, 0.0], [0, 2, 1]))

    idx0 = int(np.where(result.inc_top_reflected.order == 0)[0][0])
    R0 = result.inc_top_reflected.efficiency[idx0]
    T0 = result.inc_top_transmitted.efficiency[idx0]
    sum_rt = float(
        np.sum(result.inc_top_reflected.efficiency)
        + np.sum(result.inc_top_transmitted.efficiency)
    )

    k0 = 2 * np.pi / 6000.0
    kz_top = k0 * 1.0
    kz_bot = k0 * 1.5
    r_fresnel = (kz_top / 1.0**2 - kz_bot / 1.5**2) / (kz_top / 1.0**2 + kz_bot / 1.5**2)
    R_fresnel = float(np.abs(r_fresnel) ** 2)

    assert R0 == pytest.approx(R_fresnel, abs=1e-6)
    assert sum_rt == pytest.approx(1.0, abs=1e-6)


def test_te_parity_reticolo_exemple1_1d_regression() -> None:
    """TE T0 continues to match RETICOLO exemple1_1D.m after TM changes.

    Sanity check that the a1*a2 operator fix did not disturb the TE path.
    """
    from grax.rcwa_1d import res0, res1, res2

    grating_texture = [
        np.array([1000.0, 9000.0]),
        np.array([1.5, 1.0], dtype=complex),
    ]
    parm = res0(1)
    aa = res1(6000.0, 10000.0, [1.0, 1.5, grating_texture], 25, 0.0, parm)
    result = res2(aa, ([0.0, 20000.0, 0.0], [0, 2, 1]))

    idx0 = int(np.where(result.inc_top_reflected.order == 0)[0][0])
    T0 = result.inc_top_transmitted.efficiency[idx0]
    sum_rt = float(
        np.sum(result.inc_top_reflected.efficiency)
        + np.sum(result.inc_top_transmitted.efficiency)
    )

    assert T0 == pytest.approx(0.5050632035, abs=1e-4)
    assert sum_rt == pytest.approx(1.0, abs=1e-6)
