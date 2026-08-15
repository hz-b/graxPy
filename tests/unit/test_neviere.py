"""Unit tests for the Nevière differential-method solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import grax
from grax import rcwa_1d
from grax.gratings import BlazedGrating, LaminarGrating, ProfileGrating
from grax.simulation import BatchSimulationRunner, fixed_angle_cases, run_simulation
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
            neviere_options=NeviereOptions(z_sampling="continuous", block_phase=0.1),
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
        neviere_options=NeviereOptions(z_sampling="continuous", block_phase=0.05),
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
        default_fourier_orders=6,
        default_solver="neviere",
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
    mixed_runner = BatchSimulationRunner(default_fourier_orders=6, on_error="fail_fast")
    assert [result.solver for result in mixed_runner.run_cases(mixed)] == ["rcwa", "neviere"]


@pytest.mark.unit
def test_batch_runner_rejects_an_unknown_default_solver() -> None:
    """Verify the runner validates its default solver at construction time."""

    with pytest.raises(ValueError, match="solver must be one of"):
        BatchSimulationRunner(default_solver="dm")


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
