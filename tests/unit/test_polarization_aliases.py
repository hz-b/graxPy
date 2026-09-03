"""Tests for the TE/TM polarization aliases and theta-search polarization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import grax
from grax.simulation import (
    BatchSimulationRunner,
    GratingSimulation,
    fixed_angle_cases,
    monochromator_cases,
    multilayer_theta_search_cases,
    run_multilayer_theta_search,
    run_simulation,
)
from grax.simulation.core import POLARIZATION_ALIASES, normalize_polarization
from grax.simulation.serialization import _case_result_from_record
from tests.simulation_helpers import C, CR, PT, SI


def _grating() -> grax.LaminarGrating:
    """Return a coarse coated laminar grating."""

    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=4.0,
        z_resolution_nm=1.0,
    )


def _multilayer_grating() -> grax.BlazedGrating:
    """Return a coarse blazed grating on a Cr/C multilayer stack."""

    return grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=grax.MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=4.8,
            gamma=0.4,
            n_bilayers=8,
            top_material=C,
        ),
        x_resolution_nm=2.0,
        z_resolution_nm=0.5,
    )


_THETA_SEARCH_KWARGS = dict(
    diffraction_order=2,
    initial_grazing_angle_deg=14.176,
    rough_scan_half_width_deg=0.4,
    rough_scan_points=5,
    rough_fourier_orders=2,
    rough_x_resolution_nm=4.0,
    rough_z_resolution_nm=2.0,
    fine_scan_half_width_deg=0.1,
    fine_scan_points=5,
    fine_fourier_orders=2,
    fine_x_resolution_nm=4.0,
    fine_z_resolution_nm=2.0,
    final_fourier_orders=3,
    final_x_resolution_nm=4.0,
    final_z_resolution_nm=2.0,
    validate_physical_results=False,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [("s", "s"), ("S", "s"), ("te", "s"), ("TE", "s"), ("  Te  ", "s"),
     ("p", "p"), ("P", "p"), ("tm", "p"), ("TM", "p"), (" tM ", "p")],
)
def test_normalize_polarization_maps_every_accepted_spelling(
    spelling: str, canonical: str
) -> None:
    """Verify each accepted spelling canonicalizes, ignoring case and whitespace."""

    assert normalize_polarization(spelling) == canonical


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "x", "tem", "s-pol", "1", "transverse electric"])
def test_normalize_polarization_rejects_anything_else(value: str) -> None:
    """Verify unknown spellings raise rather than silently picking a default."""

    with pytest.raises(ValueError, match="polarization must be one of"):
        normalize_polarization(value)


@pytest.mark.unit
def test_polarization_aliases_cover_both_states_only() -> None:
    """Verify the alias table stays deliberately small."""

    assert set(POLARIZATION_ALIASES.values()) == {"s", "p"}
    assert set(POLARIZATION_ALIASES) == {"s", "te", "p", "tm"}


@pytest.mark.unit
@pytest.mark.parametrize(("alias", "canonical"), [("TE", "s"), ("TM", "p")])
def test_run_simulation_alias_is_bit_identical_to_canonical(
    alias: str, canonical: str
) -> None:
    """Verify an alias is a pure rename, not a different calculation.

    Bit-identical rather than close: the alias resolves before any physics
    happens, so anything other than an exact match would mean the two spellings
    took different paths.
    """

    common = dict(
        grating=_grating(),
        energy_ev=300.0,
        grazing_angle_deg=4.0,
        fourier_orders=8,
        validate_physical_results=False,
    )
    alias_result = run_simulation(**common, polarization=alias)
    canonical_result = run_simulation(**common, polarization=canonical)

    assert alias_result.polarization == canonical
    assert np.array_equal(alias_result.efficiency_all, canonical_result.efficiency_all)
    assert alias_result.selected_efficiency == canonical_result.selected_efficiency


@pytest.mark.unit
def test_s_and_p_still_differ() -> None:
    """Guard the alias test above: s and p must not be accidentally equal."""

    common = dict(
        grating=_grating(),
        energy_ev=300.0,
        grazing_angle_deg=4.0,
        fourier_orders=8,
        validate_physical_results=False,
    )
    assert run_simulation(**common, polarization="s").selected_efficiency != (
        run_simulation(**common, polarization="p").selected_efficiency
    )


@pytest.mark.unit
def test_grating_simulation_accepts_an_alias() -> None:
    """Verify the compatibility wrapper canonicalizes too."""

    simulation = GratingSimulation(
        grating=_grating(),
        fourier_orders=6,
        polarization="TM",
        validate_physical_results=False,
    )
    assert simulation.run_single(300.0)["efficiency"] == pytest.approx(
        GratingSimulation(
            grating=_grating(),
            fourier_orders=6,
            polarization="p",
            validate_physical_results=False,
        ).run_single(300.0)["efficiency"]
    )


@pytest.mark.unit
def test_batch_runner_canonicalizes_runner_and_case_polarization(tmp_path: Path) -> None:
    """Verify aliases work as a runner setting and as a per-case override."""

    runner = BatchSimulationRunner(
        fourier_orders=6,
        polarization="TM",
        checkpoint_dir=tmp_path / "checkpoints",
        on_error="fail_fast",
    )
    assert runner.polarization == "p"

    cases = list(
        fixed_angle_cases(
            grating=_grating(),
            energies_ev=[300.0],
            grazing_angle_deg=4.0,
            polarization="TE",
        )
    )
    results = list(runner.run_cases(cases))

    assert results[0].polarization == "s"

    records = [
        json.loads(line)
        for line in (tmp_path / "checkpoints" / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert _case_result_from_record(records[0]).polarization == "s"


@pytest.mark.unit
def test_case_builders_accept_aliases() -> None:
    """Verify the case builders pass an alias through for the runner to resolve."""

    monochromator = list(
        monochromator_cases(
            grating=_grating(), energies_ev=[300.0], cff=2.25, polarization="TM"
        )
    )
    theta_search = list(
        multilayer_theta_search_cases(
            grating=_multilayer_grating(), energies_ev=[500.0], polarization="TE"
        )
    )

    assert monochromator[0]["polarization"] == "TM"
    assert theta_search[0]["polarization"] == "TE"


@pytest.mark.unit
def test_parameter_study_rejects_a_bad_polarization_at_the_entry_point() -> None:
    """Verify the parameter study validates rather than failing three frames down."""

    with pytest.raises(ValueError, match="polarization must be one of"):
        grax.run_parameter_study(
            grating=_grating(),
            energies_ev=[300.0],
            grazing_angle_deg=4.0,
            polarization="tem",
            save_csv=False,
            show_progress=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["rcwa", "neviere"])
def test_theta_search_polarization_reaches_the_solver(solver: str) -> None:
    """Verify the theta search actually solves in the requested polarization.

    This workflow had no polarization argument at all, so every search ran the
    ``run_simulation`` default. Asserting only that ``p`` runs would pass even if
    the value were stored and ignored; the check that matters is that ``p``
    selects a different angle and efficiency from ``s``.
    """

    s_result = run_multilayer_theta_search(
        grating=_multilayer_grating(), energy_ev=500.0, solver=solver,
        polarization="s", **_THETA_SEARCH_KWARGS,
    )
    p_result = run_multilayer_theta_search(
        grating=_multilayer_grating(), energy_ev=500.0, solver=solver,
        polarization="p", **_THETA_SEARCH_KWARGS,
    )

    assert s_result.polarization == "s"
    assert p_result.polarization == "p"
    assert p_result.selected_efficiency != s_result.selected_efficiency
    assert p_result.grazing_angle_deg != s_result.grazing_angle_deg


@pytest.mark.unit
def test_theta_search_alias_is_bit_identical_to_canonical() -> None:
    """Verify TM reaches the theta search as exactly p."""

    alias = run_multilayer_theta_search(
        grating=_multilayer_grating(), energy_ev=500.0,
        polarization="TM", **_THETA_SEARCH_KWARGS,
    )
    canonical = run_multilayer_theta_search(
        grating=_multilayer_grating(), energy_ev=500.0,
        polarization="p", **_THETA_SEARCH_KWARGS,
    )

    assert alias.polarization == "p"
    assert alias.grazing_angle_deg == canonical.grazing_angle_deg
    assert alias.selected_efficiency == canonical.selected_efficiency


@pytest.mark.unit
def test_theta_search_batch_workflow_honours_polarization() -> None:
    """Verify the batch payload for this workflow carries polarization.

    That payload is built separately from the plain-case one and previously had
    no polarization key at all, which is the same shape of gap that let ``solver``
    go missing from it.
    """

    cases = list(
        multilayer_theta_search_cases(
            grating=_multilayer_grating(),
            energies_ev=[500.0],
            diffraction_order=2,
            rough_scan_half_width_deg=0.4,
            rough_scan_points=5,
            rough_fourier_orders=2,
            rough_x_resolution_nm=4.0,
            rough_z_resolution_nm=2.0,
            fine_scan_half_width_deg=0.1,
            fine_scan_points=5,
            fine_fourier_orders=2,
            fine_x_resolution_nm=4.0,
            fine_z_resolution_nm=2.0,
            final_fourier_orders=3,
            final_x_resolution_nm=4.0,
            final_z_resolution_nm=2.0,
            polarization="TM",
        )
    )
    runner = BatchSimulationRunner(fourier_orders=3, on_error="fail_fast")
    results = list(runner.run_cases(cases))

    assert results[0].polarization == "p"


@pytest.mark.unit
def test_theta_search_cases_carry_requested_solver_into_the_batch() -> None:
    """A case-level solver from the generator must beat the runner default."""

    common = dict(
        grating=_multilayer_grating(),
        energies_ev=[500.0],
        diffraction_order=2,
        rough_scan_half_width_deg=0.4,
        rough_scan_points=5,
        rough_fourier_orders=2,
        rough_x_resolution_nm=4.0,
        rough_z_resolution_nm=2.0,
        fine_scan_half_width_deg=0.1,
        fine_scan_points=5,
        fine_fourier_orders=2,
        fine_x_resolution_nm=4.0,
        fine_z_resolution_nm=2.0,
        final_fourier_orders=3,
        final_x_resolution_nm=4.0,
        final_z_resolution_nm=2.0,
    )
    cases = list(multilayer_theta_search_cases(solver="neviere", **common))
    assert cases[0]["solver"] == "neviere"

    # Runner default is rcwa; the case must override it.
    runner = BatchSimulationRunner(fourier_orders=3, solver="rcwa", on_error="fail_fast")
    results = list(runner.run_cases(cases))

    assert results[0].solver == "neviere"
