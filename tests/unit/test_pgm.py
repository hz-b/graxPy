# ruff: noqa: D103
"""Reference cases carried from the colleague's SX700 working-range project."""

import threading

import numpy as np
import pytest

import grax
from grax.pgm import (
    PGMCancelled,
    SX700Parameters,
    calculate_optical_efficiency,
    calculate_working_range,
    dashboard_title,
    geometry_for_state,
    parameter_blocks,
    side_view_geometry,
)
from grax.pgm.grating import _iter_grating_points, grating_efficiency_single
from grax.pgm.optics import mirror_reflectivity_array


def test_sx700_geometry_reference():
    """The graxpy-backend geometry gives the same angles and beam throughput."""
    p = SX700Parameters(groove_density_l_mm=400, diffraction_order=1)
    point = geometry_for_state(200, 2.5, p)
    assert point is not None
    assert point["alpha_deg"] == pytest.approx(88.2402439406153)
    assert point["beta_deg"] == pytest.approx(-85.59696907300781)
    assert point["theta_deg"] == pytest.approx(86.91860650681156)
    assert point["throughput_percent"] == pytest.approx(100)
    assert point["valid"] is True


@pytest.mark.parametrize(
    ("stack", "scan", "reference", "tolerance"),
    [
        (
            grax.SingleLayerStack(
                substrate_material="Si", layer_material="Ir", layer_thickness_nm=30
            ),
            dict(
                groove_density_l_mm=400,
                diffraction_order=1,
                n_energy=6,
                n_c=4,
                energy_min_eV=200,
                energy_max_eV=800,
                c_min=2,
                c_max=4,
            ),
            [
                [0.5854594, 0.6015672, 0.6266455, 0.6516156, 0.6782576, 0.6929693],
                [0.6346554, 0.6494104, 0.6726607, 0.6957684, 0.7203467, 0.7340548],
                [0.6594994, 0.6735059, 0.6956798, 0.7176898, 0.7410577, 0.7541349],
                [0.6746367, 0.6881667, 0.7096382, 0.7309328, 0.7535145, 0.7661725],
            ],
            0.002,
        ),
        (
            grax.MultilayerStack(
                substrate_material="Si",
                material_a="Cr",
                material_b="C",
                d_period_nm=4.8,
                gamma=2 / 4.8,
                n_bilayers=60,
            ),
            dict(
                groove_density_l_mm=2400,
                diffraction_order=2,
                n_energy=6,
                n_c=4,
                energy_min_eV=1000,
                energy_max_eV=6000,
                c_min=1.5,
                c_max=4,
            ),
            [
                [0.0076504, 0.000935, 0.0001793, 0.0004813, 0.0001158, 0.0000106],
                [0.0045317, 0.009162, 0.0071426, 0.0003827, 0.0005789, 0.0002553],
                [0.007495, 0.003505, 0.0175033, 0.0061627, 0.0013098, 0.0006525],
                [0.0102803, 0.0033179, 0.0137865, 0.0676314, 0.0025818, 0.0017435],
            ],
            0.006,
        ),
    ],
)
def test_existing_grax_mirror_matches_colleague_reference_within_material_difference(
    stack,
    scan,
    reference,
    tolerance,
):
    """GraxPy's optical constants and roughness model differ slightly from the old mirror."""
    p = SX700Parameters(**scan, mirror_stack=stack)
    _, _, geometry = calculate_working_range(p)
    actual = mirror_reflectivity_array(geometry, p)
    np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=0)


def test_pgm_validation_and_grating_density():
    grating = grax.BlazedGrating(
        period_lpermm=400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=None,
        substrate_material="Pt",
        layer_material="Pt",
        layer_thickness_nm=1,
    )
    assert SX700Parameters(grating=grating).groove_density_l_mm == 400
    with pytest.raises(ValueError, match="contradicts"):
        SX700Parameters(grating=grating, groove_density_l_mm=2400)
    with pytest.raises(ValueError, match="s polarization"):
        SX700Parameters(polarization="p")
    with pytest.raises(ValueError, match="maximums"):
        SX700Parameters(c_min=3, c_max=2)


@pytest.mark.smoke
def test_pgm_grating_matches_original_solver_reference():
    """The colleague's validated 400 l/mm sawtooth case was 31.845% efficient."""
    grating = grax.BlazedGrating(
        period_lpermm=400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=None,
        substrate_material="Pt",
        layer_material="Pt",
        layer_thickness_nm=1,
        x_resolution_nm=0.5,
        z_resolution_nm=0.1,
    )
    p = SX700Parameters(grating=grating, diffraction_order=1, rcwa_harmonics=25)
    from grax.pgm.geometry import _cff_from_grating_grazing

    cff = _cff_from_grating_grazing(200, 4, p)
    assert grating_efficiency_single(200, cff, p) == pytest.approx(0.31845, abs=0.001)


def _dashboard_optics():
    grating = grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=0.9,
        coating_stack=grax.MultilayerStack(
            substrate_material="Si",
            material_a="Cr",
            material_b="C",
            d_period_nm=4.8,
            gamma=2.0 / 4.8,
            n_bilayers=40,
            top_material="C",
        ),
    )
    mirror = grax.MultilayerStack(
        substrate_material="Si",
        material_a="Cr",
        material_b="C",
        d_period_nm=4.8,
        gamma=2.0 / 4.8,
        n_bilayers=60,
        top_material="Cr",
    )
    return grating, mirror


def test_live_callbacks_report_base_maps_then_shrinking_unsolved_nodes(monkeypatch):
    grating, mirror = _dashboard_optics()
    base, unsolved = [], []
    p = SX700Parameters(
        grating=grating,
        mirror_stack=mirror,
        n_energy=6,
        n_c=5,
        rcwa_energy_nodes=10,
        rcwa_cff_nodes=10,
        rcwa_use_cache=False,
        rcwa_convergence_check=False,
        maps_ready_callback=lambda *args: base.append(args),
        nodes_callback=lambda e, c, eta: unsolved.append(int(np.isnan(eta).sum())),
    )
    monkeypatch.setattr(
        "grax.pgm.optics._iter_grating_points",
        lambda _p, points: ((index, 0.25) for index in range(len(points))),
    )
    energies, cff, geometry = calculate_working_range(p)
    calculate_optical_efficiency(p, energies, cff, geometry)
    assert len(base) == 1
    assert base[0][3].shape == (5, 6)
    assert unsolved == sorted(unsolved, reverse=True)
    assert unsolved[0] == 99
    assert unsolved[-1] == 0


def test_side_view_uses_the_state_geometry():
    p = SX700Parameters(groove_density_l_mm=2400, diffraction_order=2)
    view = side_view_geometry(1000, 2.25, p)
    state = geometry_for_state(1000, 2.25, p)
    assert view["hit_points"]["x"] == pytest.approx(
        [state["mirror_x_rays_mm"][1], state["grating_x_rays_mm"][1]]
    )
    assert view["hit_points"]["y"][0] == pytest.approx(-p.beam_offset_mm)
    assert len(view["central_ray"]) == 3
    assert len(view["edge_rays"]) == 6
    assert geometry_for_state(1.0, 2.25, p) is None
    assert side_view_geometry(1.0, 2.25, p) is None


def test_dashboard_text_describes_the_optics():
    grating, mirror = _dashboard_optics()
    p = SX700Parameters(grating=grating, mirror_stack=mirror)
    assert dashboard_title(p).endswith("2400 l/mm   |   Mirror Cr/C ML   |   Grating Cr/C ML")
    sx700, optics = parameter_blocks(p)
    assert "Premirror L        600 mm" in sx700
    assert "Mirror ML periods  60" in optics
    assert "Grating ML periods 40" in optics
    assert "RCWA orders        19" in optics


def test_parallel_grating_solves_receive_the_stop_event_and_pid_callback(monkeypatch):
    grating, mirror = _dashboard_optics()
    received = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def run_cases(self, cases):
            return iter(())

    monkeypatch.setattr("grax.pgm.grating.grax.BatchSimulationRunner", FakeRunner)
    monkeypatch.setattr("grax.pgm.grating.grax.energy_angle_cases", lambda **kwargs: [])
    stop = threading.Event()
    callback = lambda pids: None  # noqa: E731
    p = SX700Parameters(
        grating=grating,
        mirror_stack=mirror,
        rcwa_workers=2,
        stop_event=stop,
        worker_pids_callback=callback,
    )
    assert list(_iter_grating_points(p, [(1000.0, 2.25), (1100.0, 2.25)])) == []
    assert received["max_workers"] == 2
    assert received["stop_event"] is stop
    assert received["on_worker_pids_changed"] is callback

    # Stopping ends the solve instead of returning a partial node set.
    stop.set()
    with pytest.raises(PGMCancelled):
        list(_iter_grating_points(p, [(1000.0, 2.25), (1100.0, 2.25)]))
