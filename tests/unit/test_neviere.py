"""Unit tests for the Nevière differential-method solver."""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import grax
from grax import rcwa_1d
from grax.gratings import BlazedGrating, LaminarGrating, ProfileGrating
from grax.simulation import (
    BatchSimulationRunner,
    energy_angle_cases,
    fixed_angle_cases,
    monochromator_cases,
    multilayer_theta_search_cases,
    run_multilayer_theta_search,
    run_simulation,
)
from grax.simulation.serialization import _case_result_from_record
from grax.solvers import res0, res1, res2, res2_dm
from grax.solvers.common import prepare_layer_stack, propagating_energy_balance
from grax.solvers.neviere import (
    NeviereOptions,
    build_grating_epsilon_sampler,
    coerce_neviere_options,
)
from grax.stacks import MultilayerStack
from tests.optical_constants import OpticalConstantsTable
from tests.simulation_helpers import AU, C, CR, PT, SI

# A purely real refractive index (beta = 0) so the structure is lossless and the
# propagating orders must sum to exactly one.
LOSSLESS = OpticalConstantsTable(
    energy_ev=np.asarray([1.0, 1.0e5], dtype=float),
    delta=np.asarray([0.02, 0.02], dtype=float),
    beta=np.zeros(2, dtype=float),
    name="LosslessDielectric",
)
LOSSLESS_INDEX = 1.0 - 0.02

# Deviations between the two solvers are pure Runge-Kutta truncation error and
# measure ~1e-11 at production resolution; these bounds sit well above that
# noise floor while still catching any real divergence.
SOLVER_PARITY_ATOL = 1e-8
SOLVER_PARITY_RTOL = 1e-6


def _laminar_grating() -> LaminarGrating:
    """Return a coarse coated laminar grating."""

    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        top_cap_material=C,
        top_cap_thickness_nm=0.7,
        x_resolution_nm=4.0,
        z_resolution_nm=1.0,
    )


def _blazed_grating() -> BlazedGrating:
    """Return a coarse coated blazed grating."""

    return BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.729,
        anti_blaze_angle_deg=5.597,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=30.0,
        x_resolution_nm=4.0,
        z_resolution_nm=1.0,
    )


def _multilayer_grating() -> BlazedGrating:
    """Return a coarse blazed grating on a Cr/C multilayer stack."""

    return BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=MultilayerStack(
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


def _sinusoidal_grating(
    *,
    depth_nm: float = 10.0,
    material: object = None,
    period_lpermm: int = 2000,
    z_resolution_nm: float = 0.25,
) -> ProfileGrating:
    """Return a sinusoidal surface-relief grating built from explicit points."""

    resolved_material = SI if material is None else material
    period_nm = 1e6 / period_lpermm
    x_points_nm = np.linspace(0.0, period_nm, 33)
    z_points_nm = depth_nm * 0.5 * (1.0 - np.cos(2.0 * np.pi * x_points_nm / period_nm))
    return ProfileGrating(
        period_lpermm=period_lpermm,
        x_points_nm=x_points_nm,
        z_points_nm=z_points_nm,
        substrate_material=resolved_material,
        layer_material=resolved_material,
        layer_thickness_nm=5.0,
        x_resolution_nm=2.0,
        z_resolution_nm=z_resolution_nm,
    )


def _solve_pair(
    grating: object,
    *,
    energy_ev: float,
    grazing_angle_deg: float,
    polarization: str,
    fourier_orders: int,
    options: NeviereOptions | None = None,
):
    """Return the reflected/transmitted results from both solvers for one case."""

    parm = res0(1 if polarization == "s" else -1)
    textures, profile = grating.build_textures(energy_ev, n_inc=1.0 + 0.0j)
    aa = res1(
        1239.8 / energy_ev,
        grating.period_nm,
        textures,
        fourier_orders,
        np.sin(np.deg2rad(90.0 - grazing_angle_deg)),
        parm,
        _fourier_backend="numba",
    )
    return aa, profile, res2(aa, profile, parm), res2_dm(aa, profile, parm, options=options)


def _assert_solvers_agree(
    grating: object,
    *,
    energy_ev: float,
    grazing_angle_deg: float,
    polarization: str,
    fourier_orders: int,
) -> None:
    """Assert both solvers return the same all-order efficiencies and angles."""

    common = dict(
        grating=grating,
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=fourier_orders,
        polarization=polarization,
        validate_physical_results=False,
    )
    rcwa_result = run_simulation(**common, solver="rcwa")
    neviere_result = run_simulation(**common, solver="neviere")

    assert np.array_equal(rcwa_result.orders, neviere_result.orders)
    assert np.allclose(
        rcwa_result.efficiency_all,
        neviere_result.efficiency_all,
        atol=SOLVER_PARITY_ATOL,
        rtol=SOLVER_PARITY_RTOL,
    )
    assert np.allclose(
        rcwa_result.diffraction_angle_all,
        neviere_result.diffraction_angle_all,
        atol=1e-9,
    )
    assert rcwa_result.solver == "rcwa"
    assert neviere_result.solver == "neviere"


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_neviere_matches_rcwa_for_laminar_grating(polarization: str) -> None:
    """Verify the differential method reproduces RCWA on a coated laminar grating."""

    _assert_solvers_agree(
        _laminar_grating(),
        energy_ev=300.0,
        grazing_angle_deg=4.0,
        polarization=polarization,
        fourier_orders=10,
    )


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_neviere_matches_rcwa_for_blazed_grating(polarization: str) -> None:
    """Verify the differential method reproduces RCWA on a blazed grating."""

    _assert_solvers_agree(
        _blazed_grating(),
        energy_ev=500.0,
        grazing_angle_deg=1.5,
        polarization=polarization,
        fourier_orders=8,
    )


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_neviere_matches_rcwa_for_sinusoidal_grating(polarization: str) -> None:
    """Verify the differential method reproduces RCWA on a sinusoidal profile."""

    _assert_solvers_agree(
        _sinusoidal_grating(),
        energy_ev=300.0,
        grazing_angle_deg=5.0,
        polarization=polarization,
        fourier_orders=8,
    )


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_neviere_matches_rcwa_for_multilayer_stack(polarization: str) -> None:
    """Verify the differential method reproduces RCWA on a blazed multilayer stack."""

    _assert_solvers_agree(
        _multilayer_grating(),
        energy_ev=500.0,
        grazing_angle_deg=14.176,
        polarization=polarization,
        fourier_orders=6,
    )


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_neviere_conserves_energy_for_lossless_dielectric_grating(polarization: str) -> None:
    """Verify propagating reflected and transmitted orders sum to one without absorption."""

    grating = _sinusoidal_grating(depth_nm=12.0, material=LOSSLESS)
    aa, profile, _, neviere = _solve_pair(
        grating,
        energy_ev=500.0,
        grazing_angle_deg=30.0,
        polarization=polarization,
        fourier_orders=12,
        options=NeviereOptions(step_phase=0.005),
    )
    n_top, n_bottom, _ = prepare_layer_stack(aa, profile)
    balance = propagating_energy_balance(
        neviere.inc_top_reflected,
        neviere.inc_top_transmitted,
        wavelength=aa.wavelength,
        period=aa.period,
        beta0=aa.beta0,
        n_top=n_top,
        n_bottom=n_bottom,
    )

    assert balance["reflected"] > 0.0
    assert balance["transmitted"] > 0.0
    assert balance["total"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("grazing_angle_deg", [30.0, 60.0])
def test_neviere_reproduces_analytic_fresnel_in_the_flat_limit(
    polarization: str,
    grazing_angle_deg: float,
) -> None:
    """Verify a zero-depth grating returns the analytic single-interface reflectivity.

    This pins the absolute normalization of the solver, independently of RCWA:
    with no groove the structure is a plain vacuum/substrate interface, so the
    zeroth order must equal the Fresnel reflectivity and every other order must
    vanish.
    """

    energy_ev = 500.0
    k0 = 2.0 * np.pi / (1239.8 / energy_ev)
    kx = k0 * np.sin(np.deg2rad(90.0 - grazing_angle_deg))
    kz_vacuum = np.sqrt(complex(k0**2 - kx**2))
    kz_substrate = np.sqrt(complex((k0 * LOSSLESS_INDEX) ** 2 - kx**2))
    if polarization == "s":
        reflection = (kz_vacuum - kz_substrate) / (kz_vacuum + kz_substrate)
    else:
        substrate_admittance = kz_substrate / LOSSLESS_INDEX**2
        reflection = (kz_vacuum - substrate_admittance) / (kz_vacuum + substrate_admittance)
    expected_efficiency = float(abs(reflection) ** 2)

    result = run_simulation(
        grating=_sinusoidal_grating(depth_nm=0.0, material=LOSSLESS),
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        fourier_orders=6,
        polarization=polarization,
        solver="neviere",
        validate_physical_results=False,
    )

    zeroth_index = int(np.argmin(np.abs(result.orders)))
    other_efficiencies = np.delete(result.efficiency_all, zeroth_index)
    assert result.efficiency_all[zeroth_index] == pytest.approx(expected_efficiency, rel=1e-9)
    assert np.max(np.abs(other_efficiencies)) < 1e-20


@pytest.mark.unit
@pytest.mark.parametrize(
    "roughness_kwargs",
    [
        pytest.param({"roughness_sigma_nm": 0.5}, id="solver-debye-waller"),
        pytest.param(
            {"roughness": grax.RoughnessSpec(kind="debye-waller", sigma_nm=0.5)},
            id="grating-debye-waller",
        ),
        pytest.param(
            {
                "roughness": grax.RoughnessSpec(
                    kind="random-interface",
                    sigma_nm=0.3,
                    num_supercells=2,
                    num_realizations=2,
                    seed=7,
                )
            },
            id="random-interface-supercell",
        ),
    ],
)
def test_neviere_matches_rcwa_through_the_roughness_paths(
    roughness_kwargs: dict[str, object],
) -> None:
    """Verify roughness reaches both solvers identically.

    Roughness is applied around the solve rather than inside it, but the
    supercell path also changes the period, the order grid and the realization
    averaging, so it is worth pinning that the differential method threads
    through all of it unchanged.
    """

    grating_kwargs = dict(roughness_kwargs)
    solver_kwargs: dict[str, object] = {}
    if "roughness_sigma_nm" in grating_kwargs:
        solver_kwargs["roughness_sigma_nm"] = grating_kwargs.pop("roughness_sigma_nm")

    common = dict(
        grating=dataclasses.replace(_laminar_grating(), **grating_kwargs),
        energy_ev=300.0,
        grazing_angle_deg=4.0,
        fourier_orders=6,
        polarization="p",
        validate_physical_results=False,
        **solver_kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rcwa_result = run_simulation(**common, solver="rcwa")
        neviere_result = run_simulation(**common, solver="neviere")

    assert np.array_equal(rcwa_result.orders, neviere_result.orders)
    assert np.allclose(
        rcwa_result.efficiency_all,
        neviere_result.efficiency_all,
        atol=SOLVER_PARITY_ATOL,
        rtol=SOLVER_PARITY_RTOL,
    )
    assert neviere_result.num_supercells == rcwa_result.num_supercells
    assert neviere_result.num_realizations == rcwa_result.num_realizations


@pytest.mark.unit
def test_neviere_converges_to_rcwa_as_step_phase_shrinks() -> None:
    """Verify the residual against RCWA is Runge-Kutta truncation error."""

    grating = _laminar_grating()
    parm = res0(-1)
    textures, profile = grating.build_textures(300.0, n_inc=1.0 + 0.0j)
    aa = res1(
        1239.8 / 300.0,
        grating.period_nm,
        textures,
        10,
        np.sin(np.deg2rad(86.0)),
        parm,
        _fourier_backend="numba",
    )
    reference = np.real(res2(aa, profile, parm).inc_top_reflected.efficiency)

    deviations = []
    for step_phase in (0.4, 0.2, 0.1):
        efficiency = np.real(
            res2_dm(
                aa,
                profile,
                parm,
                options=NeviereOptions(step_phase=step_phase),
            ).inc_top_reflected.efficiency
        )
        deviations.append(float(np.max(np.abs(efficiency - reference))))

    assert deviations[0] > deviations[1] > deviations[2]
    # Fourth-order accuracy: halving the step should cut the error by ~16.
    assert deviations[1] < deviations[0] / 8.0
    assert deviations[2] < deviations[1] / 8.0


@pytest.mark.unit
def test_continuous_sampling_is_independent_of_z_resolution() -> None:
    """Verify continuous sampling ignores the staircase the RCWA path depends on.

    The staircase solvers converge towards the continuous answer as
    ``z_resolution_nm`` shrinks, while continuous sampling reads the true profile
    and so returns the same efficiencies at every z resolution.
    """

    results = []
    staircase_deviations = []
    for z_resolution_nm in (1.0, 0.25):
        grating = _sinusoidal_grating(depth_nm=10.0, z_resolution_nm=z_resolution_nm)
        common = dict(
            grating=grating,
            energy_ev=300.0,
            grazing_angle_deg=5.0,
            fourier_orders=8,
            polarization="p",
            validate_physical_results=False,
        )
        continuous = run_simulation(
            **common,
            solver="neviere",
            solver_options=NeviereOptions(z_sampling="continuous", sample_phase=0.05),
        )
        staircase = run_simulation(**common, solver="rcwa")
        results.append(np.asarray(continuous.efficiency_all, dtype=float))
        staircase_deviations.append(
            float(np.max(np.abs(staircase.efficiency_all - continuous.efficiency_all)))
        )

    assert np.allclose(results[0], results[1], atol=1e-12)
    # The staircase result moves towards the continuous one as z refines.
    assert staircase_deviations[1] < staircase_deviations[0] / 2.0


@pytest.mark.unit
def test_continuous_sampling_matches_a_finely_sliced_staircase() -> None:
    """Verify continuous sampling agrees with RCWA once the staircase is converged."""

    energy_ev = 300.0
    fine = run_simulation(
        grating=_sinusoidal_grating(depth_nm=10.0, z_resolution_nm=0.01),
        energy_ev=energy_ev,
        grazing_angle_deg=5.0,
        fourier_orders=8,
        polarization="p",
        solver="rcwa",
        validate_physical_results=False,
    )
    continuous = run_simulation(
        grating=_sinusoidal_grating(depth_nm=10.0, z_resolution_nm=1.0),
        energy_ev=energy_ev,
        grazing_angle_deg=5.0,
        fourier_orders=8,
        polarization="p",
        solver="neviere",
        solver_options=NeviereOptions(z_sampling="continuous", sample_phase=0.01),
        validate_physical_results=False,
    )

    assert np.allclose(fine.efficiency_all, continuous.efficiency_all, atol=1e-4)


@pytest.mark.unit
def test_continuous_sampling_requires_an_epsilon_sampler() -> None:
    """Verify res2_dm rejects continuous sampling without a permittivity sampler."""

    grating = _laminar_grating()
    parm = res0(1)
    textures, profile = grating.build_textures(300.0, n_inc=1.0 + 0.0j)
    aa = res1(1239.8 / 300.0, grating.period_nm, textures, 4, 0.99, parm,
              _fourier_backend="numba")

    with pytest.raises(ValueError, match="requires an epsilon_sampler"):
        res2_dm(aa, profile, parm, options=NeviereOptions(z_sampling="continuous"))


@pytest.mark.unit
def test_epsilon_sampler_reads_the_profile_between_solver_rows() -> None:
    """Verify the continuous sampler resolves depths the z grid never lands on."""

    grating = _sinusoidal_grating(depth_nm=10.0, z_resolution_nm=1.0)
    parm = res0(1)
    textures, _ = grating.build_textures(300.0, n_inc=1.0 + 0.0j)
    aa = res1(1239.8 / 300.0, grating.period_nm, textures, 6, 0.996, parm,
              _fourier_backend="numba")
    sampler = build_grating_epsilon_sampler(
        grating,
        photon_energy_ev=300.0,
        period_nm=grating.period_nm,
        orders=aa.orders,
    )

    # Two depths inside the groove region, half a solver row apart.
    first = sampler(6.25)
    second = sampler(6.75)

    assert not np.allclose(first.epsilon_fourier, second.epsilon_fourier)


@pytest.mark.unit
def test_neviere_options_reject_invalid_settings() -> None:
    """Verify option validation covers each numerical setting."""

    with pytest.raises(ValueError, match="z_sampling"):
        NeviereOptions(z_sampling="staircase")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_phase"):
        NeviereOptions(step_phase=0.0)
    with pytest.raises(ValueError, match="block_phase"):
        NeviereOptions(block_phase=-1.0)
    with pytest.raises(ValueError, match="sample_phase"):
        NeviereOptions(sample_phase=-1.0)
    with pytest.raises(ValueError, match="max_step_nm"):
        NeviereOptions(max_step_nm=0.0)
    with pytest.raises(ValueError, match="max_steps_per_layer"):
        NeviereOptions(max_steps_per_layer=0)
    with pytest.raises(ValueError, match="energy_balance_tolerance"):
        NeviereOptions(energy_balance_tolerance=0.0)


@pytest.mark.unit
def test_coerce_neviere_options_accepts_none_mapping_and_instance() -> None:
    """Verify option coercion covers the shapes callers and case dicts provide."""

    assert coerce_neviere_options(None) == NeviereOptions()
    assert coerce_neviere_options({"step_phase": 0.01}).step_phase == pytest.approx(0.01)
    instance = NeviereOptions(block_phase=1.0)
    assert coerce_neviere_options(instance) is instance
    with pytest.raises(TypeError, match="NeviereOptions"):
        coerce_neviere_options(0.5)  # type: ignore[arg-type]


@pytest.mark.unit
def test_neviere_options_round_trip_through_to_dict() -> None:
    """Verify options serialize to a JSON-compatible mapping and back."""

    options = NeviereOptions(z_sampling="continuous", step_phase=0.01, max_step_nm=0.2)
    restored = coerce_neviere_options(json.loads(json.dumps(options.to_dict())))

    assert restored == options


@pytest.mark.unit
def test_energy_balance_tolerance_raises_on_a_violated_balance() -> None:
    """Verify the internal sanity check fires when the balance exceeds its bound."""

    grating = _sinusoidal_grating(depth_nm=10.0, material=LOSSLESS)
    parm = res0(1)
    textures, profile = grating.build_textures(500.0, n_inc=1.0 + 0.0j)
    aa = res1(
        1239.8 / 500.0,
        grating.period_nm,
        textures,
        8,
        np.sin(np.deg2rad(60.0)),
        parm,
        _fourier_backend="numba",
    )

    # A lossless structure sits at exactly 1.0, so any bound below that must fire.
    with pytest.raises(ValueError, match="energy balance"):
        res2_dm(aa, profile, parm, options=NeviereOptions(energy_balance_tolerance=0.5))

    # The same solve passes under a physical bound.
    res2_dm(aa, profile, parm, options=NeviereOptions(energy_balance_tolerance=1.05))


@pytest.mark.unit
def test_run_simulation_rejects_an_unknown_solver() -> None:
    """Verify the solver name is validated before any work happens."""

    with pytest.raises(ValueError, match="solver must be one of"):
        run_simulation(
            grating=_laminar_grating(),
            energy_ev=300.0,
            grazing_angle_deg=4.0,
            fourier_orders=4,
            solver="differential",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_batch_runner_records_and_overrides_the_solver(tmp_path: Path) -> None:
    """Verify the runner default, per-case override, and checkpoint round-trip."""

    grating = _laminar_grating()
    cases = list(
        fixed_angle_cases(
            grating=grating,
            energies_ev=[300.0, 400.0],
            grazing_angle_deg=4.0,
            polarization="p",
        )
    )
    checkpoint_dir = tmp_path / "checkpoints"
    runner = BatchSimulationRunner(
        fourier_orders=6,
        solver="neviere",
        checkpoint_dir=checkpoint_dir,
        on_error="fail_fast",
    )
    results = list(runner.run_cases(cases))

    assert [result.solver for result in results] == ["neviere", "neviere"]

    records = [
        json.loads(line)
        for line in (checkpoint_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    restored = [_case_result_from_record(record) for record in records]
    assert [result.solver for result in restored] == ["neviere", "neviere"]

    mixed = [dict(cases[0], solver="rcwa"), dict(cases[1], solver="neviere")]
    mixed_runner = BatchSimulationRunner(fourier_orders=6, on_error="fail_fast")
    assert [result.solver for result in mixed_runner.run_cases(mixed)] == ["rcwa", "neviere"]


@pytest.mark.unit
def test_batch_runner_rejects_an_unknown_solver() -> None:
    """Verify the runner validates its default solver at construction time."""

    with pytest.raises(ValueError, match="solver must be one of"):
        BatchSimulationRunner(solver="dm")


@pytest.mark.unit
def test_rcwa_1d_alias_module_still_exposes_the_historical_names() -> None:
    """Verify the compatibility shim covers every name the repo used to import."""

    historical_names = [
        "ArrayLike",
        "BoundaryBlockCache",
        "DiffractionResult",
        "EigenCache",
        "FourierBackend",
        "Parameters",
        "Res1Result",
        "Res2Result",
        "Texture1D",
        "_angles_from_kx",
        "_apply_debye_waller_roughness",
        "_cascade_boundary_pair",
        "_convolution_matrix",
        "_debye_waller_roughness_factor",
        "_kz_branch",
        "_kz_branch_array",
        "_layer_boundary_block",
        "_modal_function_matrices",
        "_modal_function_matrix",
        "_piecewise_fourier_coefficients",
        "_solve_te_stack",
        "_solve_tm_stack",
        "debye_waller_roughness_diagnostics",
        "res0",
        "res1",
        "res2",
        "safe_linalg_solve",
    ]

    missing = [name for name in historical_names if not hasattr(rcwa_1d, name)]
    assert missing == []
    assert rcwa_1d.res2 is res2
    assert rcwa_1d.res2_dm is res2_dm
    assert grax.NeviereOptions is NeviereOptions


@pytest.mark.unit
def test_multilayer_theta_search_honours_the_requested_solver() -> None:
    """Verify the theta-search workflow does not silently fall back to RCWA.

    The batch runner builds a separate payload for this workflow. That payload
    carried ``backend`` but not ``solver``, so a runner configured for the
    differential method quietly computed with RCWA instead: no error, no warning,
    and a result that correctly reported the solver that actually ran rather than
    the one that was asked for.
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
        )
    )
    runner = BatchSimulationRunner(
        fourier_orders=3,
        solver="neviere",
        on_error="fail_fast",
    )

    results = list(runner.run_cases(cases))

    assert [result.solver for result in results] == ["neviere"]


@pytest.mark.unit
def test_multilayer_theta_search_agrees_across_solvers() -> None:
    """Verify both solvers select the same angle and efficiency for a theta search."""

    search_kwargs = dict(
        grating=_multilayer_grating(),
        energy_ev=500.0,
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
    rcwa_result = run_multilayer_theta_search(**search_kwargs, solver="rcwa")
    neviere_result = run_multilayer_theta_search(**search_kwargs, solver="neviere")

    assert neviere_result.solver == "neviere"
    assert rcwa_result.solver == "rcwa"
    assert neviere_result.grazing_angle_deg == pytest.approx(
        rcwa_result.grazing_angle_deg, abs=1e-9
    )
    assert neviere_result.selected_efficiency == pytest.approx(
        rcwa_result.selected_efficiency,
        abs=SOLVER_PARITY_ATOL,
        rel=SOLVER_PARITY_RTOL,
    )


@pytest.mark.unit
def test_solver_options_round_trip_through_a_checkpoint(tmp_path: Path) -> None:
    """Verify a checkpointed differential-method run records its integration settings.

    The solver name alone does not pin the result: the same "neviere" label
    covers every ``step_phase`` and both sampling modes. Recording the options
    makes a resumed or archived run reproducible.
    """

    options = NeviereOptions(step_phase=0.01, block_phase=1.5)
    checkpoint_dir = tmp_path / "checkpoints"
    runner = BatchSimulationRunner(
        fourier_orders=5,
        solver="neviere",
        solver_options=options,
        checkpoint_dir=checkpoint_dir,
        on_error="fail_fast",
    )
    cases = list(
        fixed_angle_cases(
            grating=_laminar_grating(),
            energies_ev=[300.0],
            grazing_angle_deg=4.0,
        )
    )

    results = list(runner.run_cases(cases))
    assert results[0].solver_options == options.to_dict()

    records = [
        json.loads(line)
        for line in (checkpoint_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    restored = _case_result_from_record(records[0])

    assert restored.solver_options == options.to_dict()
    assert NeviereOptions(**restored.solver_options) == options


@pytest.mark.unit
def test_rcwa_results_record_no_solver_options() -> None:
    """Verify the RCWA path leaves solver_options unset rather than inventing one."""

    result = run_simulation(
        grating=_laminar_grating(),
        energy_ev=300.0,
        grazing_angle_deg=4.0,
        fourier_orders=5,
    )

    assert result.solver == "rcwa"
    assert result.solver_options is None


@pytest.mark.unit
def test_grating_simulation_replaces_the_rcwa_specific_name() -> None:
    """Verify the renamed wrapper is the only spelling and drives both solvers.

    ``RCWASimulation`` described only one of the two solvers it can now run.
    This is a deliberate breaking rename with no alias.
    """

    assert not hasattr(grax.simulation, "RCWASimulation")
    assert "RCWASimulation" not in dir(grax.simulation)

    grating = _laminar_grating()
    rcwa = grax.simulation.GratingSimulation(
        grating=grating,
        fourier_orders=5,
        polarization="p",
        validate_physical_results=False,
    ).run_single(300.0)
    neviere = grax.simulation.GratingSimulation(
        grating=grating,
        fourier_orders=5,
        polarization="p",
        solver="neviere",
        validate_physical_results=False,
    ).run_single(300.0)

    assert np.allclose(
        rcwa["efficiency_all"],
        neviere["efficiency_all"],
        atol=SOLVER_PARITY_ATOL,
        rtol=SOLVER_PARITY_RTOL,
    )


# --------------------------------------------------------------------------
# Coverage for the entry points the examples use.
#
# Each of these is exercised by an example script, so each must be shown to
# work with either solver. Without this section a reader could swap --solver on
# an example and hit an untested path.
# --------------------------------------------------------------------------


def _selected_efficiencies(results) -> np.ndarray:
    """Return the selected-order efficiency of every successful case."""

    return np.asarray(
        [result.selected_efficiency for result in results if result.status == "ok"],
        dtype=float,
    )


def _run_cases_with(solver: str, cases, **runner_kwargs):
    """Run one case iterable through the batch runner with one solver."""

    runner = BatchSimulationRunner(
        fourier_orders=6,
        on_error="fail_fast",
        solver=solver,
        **runner_kwargs,
    )
    return list(runner.run_cases(cases))


@pytest.mark.unit
def test_monochromator_cases_agree_across_solvers() -> None:
    """Verify the monochromator case builder works with either solver.

    ``monochromator_cases`` is the case builder the examples use most, and its
    grazing angle is solved per energy from the cff condition rather than given,
    so it is worth confirming both solvers see the same geometry.
    """

    energies_ev = [200.0, 400.0, 600.0]

    def build():
        return monochromator_cases(
            grating=_laminar_grating(),
            energies_ev=energies_ev,
            diffraction_order=1,
            cff=2.25,
            polarization="p",
        )

    rcwa_results = _run_cases_with("rcwa", build())
    neviere_results = _run_cases_with("neviere", build())

    assert [result.grazing_angle_deg for result in rcwa_results] == [
        result.grazing_angle_deg for result in neviere_results
    ]
    assert np.allclose(
        _selected_efficiencies(rcwa_results),
        _selected_efficiencies(neviere_results),
        atol=SOLVER_PARITY_ATOL,
        rtol=SOLVER_PARITY_RTOL,
    )
    assert {result.solver for result in neviere_results} == {"neviere"}


@pytest.mark.unit
def test_energy_angle_cases_agree_across_solvers() -> None:
    """Verify the explicit energy-angle case builder works with either solver."""

    pairs = [(300.0, 4.0), (500.0, 3.0), (800.0, 2.0)]

    def build():
        return energy_angle_cases(
            grating=_laminar_grating(),
            energy_angle_pairs=pairs,
            polarization="p",
        )

    rcwa_results = _run_cases_with("rcwa", build())
    neviere_results = _run_cases_with("neviere", build())

    assert np.allclose(
        _selected_efficiencies(rcwa_results),
        _selected_efficiencies(neviere_results),
        atol=SOLVER_PARITY_ATOL,
        rtol=SOLVER_PARITY_RTOL,
    )


@pytest.mark.unit
def test_parameter_study_agrees_across_solvers() -> None:
    """Verify the convergence parameter study runs on either solver.

    ``run_parameter_study`` drives ``GratingSimulation`` rather than the batch
    runner, so it is a separate path to the solver from every other entry point.
    """

    common = dict(
        grating=_laminar_grating(),
        energies_ev=[300.0],
        grazing_angle_deg=4.0,
        polarization="p",
        fourier_orders_values=[4, 6],
        x_resolution_values=[8.0, 4.0],
        z_resolution_values=[2.0, 1.0],
        save_csv=False,
        show_progress=False,
    )
    rcwa_study = grax.run_parameter_study(**common, solver="rcwa")
    neviere_study = grax.run_parameter_study(**common, solver="neviere")

    for energy_result_rcwa, energy_result_neviere in zip(
        rcwa_study.results, neviere_study.results
    ):
        for parameter in ("fourier_orders", "x_resolution_nm", "z_resolution_nm"):
            assert np.allclose(
                energy_result_rcwa.sweeps[parameter].efficiencies,
                energy_result_neviere.sweeps[parameter].efficiencies,
                atol=SOLVER_PARITY_ATOL,
                rtol=SOLVER_PARITY_RTOL,
            )


@pytest.mark.unit
def test_theta_search_sweep_agrees_across_solvers(tmp_path: Path) -> None:
    """Verify the multi-energy theta-search sweep runs on either solver.

    The sweep has its own runner-settings construction separate from
    ``BatchSimulationRunner``; that is where ``solver`` was previously dropped.
    """

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
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    rcwa_sweep = grax.run_multilayer_theta_search_sweep(
        **common, output_dir=tmp_path / "rcwa", solver="rcwa"
    )
    neviere_sweep = grax.run_multilayer_theta_search_sweep(
        **common, output_dir=tmp_path / "neviere", solver="neviere"
    )

    rcwa_case = rcwa_sweep.batch_result.cases[0]
    neviere_case = neviere_sweep.batch_result.cases[0]
    assert neviere_case.solver == "neviere"
    assert rcwa_case.solver == "rcwa"
    assert neviere_case.grazing_angle_deg == pytest.approx(
        rcwa_case.grazing_angle_deg, abs=1e-9
    )
    assert neviere_case.selected_efficiency == pytest.approx(
        rcwa_case.selected_efficiency,
        abs=SOLVER_PARITY_ATOL,
        rel=SOLVER_PARITY_RTOL,
    )
    # The sweep builds its own CaseExecutionResult rather than reusing the
    # runner's, and used to copy only the diagnostics across. solver and
    # solver_options fell back to their dataclass defaults, so a neviere sweep
    # reported itself as an rcwa one while computing the right answer -- the two
    # results above differ by ~5e-14, so the solve was correct and only the
    # provenance was wrong.
    assert neviere_case.solver_options is not None
    assert rcwa_case.solver_options is None
    # This workflow does not expose polarization, so it is always s.
    assert neviere_case.polarization == "s"


@pytest.mark.unit
def test_custom_stack_agrees_across_solvers() -> None:
    """Verify a hand-assembled layer stack works with either solver.

    ``assemble_custom_stack`` produces a different stack class from
    ``MultilayerStack``, and therefore a different texture-building path.
    """

    layers_bottom_up = [
        grax.LayerSpec(material=PT, thickness_nm=3.0),
        grax.LayerSpec(material=CR, thickness_nm=4.0),
        grax.LayerSpec(material=C, thickness_nm=5.0),
    ]
    stack = grax.assemble_custom_stack(
        substrate_material=SI,
        layers_bottom_up=layers_bottom_up,
        top_cap_material=C,
        top_cap_thickness_nm=1.0,
    )
    grating = BlazedGrating(
        period_lpermm=1200,
        blaze_angle_deg=1.2,
        anti_blaze_angle_deg=4.0,
        coating_stack=stack,
        x_resolution_nm=4.0,
        z_resolution_nm=1.0,
    )

    _assert_solvers_agree(
        grating,
        energy_ev=400.0,
        grazing_angle_deg=3.0,
        polarization="p",
        fourier_orders=6,
    )


@pytest.mark.unit
def test_afm_grating_agrees_across_solvers() -> None:
    """Verify a grating built from AFM data works with either solver.

    ``AFMGrating`` subclasses ``ProfileGrating``, so this covers the
    construction path rather than a new solver path, but the examples build
    gratings this way and the swap has to hold for them too.
    """

    period_nm = 1e6 / 600
    x_points_nm = np.linspace(0.0, period_nm, 65)
    z_points_nm = 8.0 * (x_points_nm / period_nm)
    grating = grax.AFMGrating(
        period_lpermm=600,
        x_points_nm=x_points_nm,
        z_points_nm=z_points_nm,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=20.0,
        x_resolution_nm=4.0,
        z_resolution_nm=1.0,
    )

    _assert_solvers_agree(
        grating,
        energy_ev=400.0,
        grazing_angle_deg=3.0,
        polarization="p",
        fourier_orders=6,
    )


@pytest.mark.unit
def test_output_helpers_accept_either_solver_result(tmp_path: Path) -> None:
    """Verify the export and plotting helpers work on either solver's results.

    These are solver-agnostic by design; the check is that a differential-method
    result carries everything they need, since every example ends by calling them.
    """

    cases = list(
        fixed_angle_cases(
            grating=_laminar_grating(),
            energies_ev=[300.0, 400.0],
            grazing_angle_deg=4.0,
            polarization="p",
        )
    )
    for solver in ("rcwa", "neviere"):
        results = _run_cases_with(solver, list(cases))

        csv_path = tmp_path / f"all_orders_{solver}.csv"
        grax.write_all_orders_csv(results, csv_path)
        assert csv_path.read_text(encoding="utf-8").count("\n") > 2

        plot_path = tmp_path / f"orders_{solver}.png"
        grax.plot_order_subset(results, plot_path, diffraction_orders=[1, 2], title=solver)
        assert plot_path.exists()

        efficiency = grax.efficiency_for_order(
            results[0].orders, results[0].efficiency_all, diffraction_order=1
        )
        assert np.isfinite(efficiency)
