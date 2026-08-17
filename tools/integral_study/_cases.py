"""Shared case definitions and solver drivers for the integral-method study.

The integral solver in :mod:`grax.solvers.integral` is not reachable through
``grax.run_simulation``: it is deliberately unwired while its practicality for
graxPy's X-ray regime is being established. This module therefore drives all
three solvers at the ``res2`` level, through the same grating objects, so the
comparison is of numerics only and not of two different geometries.

The cases are the benchmark matrix from the study plan. They are parametrized by
period and photon energy so that ``d / lambda`` -- the quantity that decides
whether a boundary-integral method is affordable -- can be swept independently of
the profile shape.

Nothing here writes to the package or changes solver defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import grax
from grax.simulation._memory import PeakMemorySampler
from grax.solvers import res0, res1, res2, res2_dm
from grax.solvers.integral import IntegralOptions, res2_im

__all__ = [
    "IntegralCase",
    "SolverRun",
    "build_cases",
    "case_by_name",
    "fresnel_reflectance",
    "run_integral",
    "run_reference",
]

#: Photon-energy to wavelength conversion used throughout graxPy.
_EV_NM = 1239.8


@dataclass(frozen=True)
class IntegralCase:
    """One comparison case.

    Attributes:
        name: Short identifier, matching the benchmark matrix in the plan.
        description: One-line human-readable summary.
        grating: Grating carrying the profile and the material stack.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        polarization: ``1`` for TE (s), ``-1`` for TM (p).
        fourier_orders: Truncation for the two reference solvers.
        reported_orders: Half-width of the order range the comparison covers.
        interfaces: Number of material interfaces, for reporting.
    """

    name: str
    description: str
    grating: Any
    energy_ev: float
    grazing_angle_deg: float
    polarization: int
    fourier_orders: int
    reported_orders: int
    interfaces: int

    @property
    def wavelength_nm(self) -> float:
        """Return the vacuum wavelength in nanometers."""

        return _EV_NM / float(self.energy_ev)

    @property
    def period_nm(self) -> float:
        """Return the grating period in nanometers."""

        return float(self.grating.period_nm)

    @property
    def period_over_wavelength(self) -> float:
        """Return ``d / lambda``, the difficulty parameter for this study."""

        return self.period_nm / self.wavelength_nm

    @property
    def beta0(self) -> float:
        """Return the in-plane direction cosine the solvers take."""

        return float(np.sin(np.deg2rad(90.0 - self.grazing_angle_deg)))

    @property
    def orders(self) -> np.ndarray:
        """Return the signed order range used for the comparison."""

        return np.arange(-self.reported_orders, self.reported_orders + 1, dtype=float)

    def at(self, *, energy_ev: float) -> IntegralCase:
        """Return a copy of this case at a different photon energy.

        Only the energy is varied here. Changing the period has to rebuild the
        profile as well, because ``ProfileGrating`` carries explicit sample
        points spanning one period; use :func:`build_sinusoid_case` for that.

        Args:
            energy_ev: Replacement photon energy.

        Returns:
            A new case; the original is untouched.
        """

        return replace(self, energy_ev=float(energy_ev))


@dataclass(frozen=True)
class SolverRun:
    """One solver's result for one case.

    Attributes:
        solver: ``"rcwa"``, ``"neviere"`` or ``"integral"``.
        efficiency: Reflected efficiency per signed order.
        orders: The signed orders, aligned with ``efficiency``.
        energy_balance: Sum of reflected and transmitted propagating efficiency.
            Below one for absorbing materials, which is physical, so it is a
            sanity indicator rather than a pass/fail test.
        seconds: Wall-clock time for the solve.
        peak_memory_bytes: Peak process RSS during the solve, when available.
        unknowns: Boundary unknown count for the integral solver, else ``None``.
    """

    solver: str
    efficiency: np.ndarray
    orders: np.ndarray
    energy_balance: float
    seconds: float
    peak_memory_bytes: int | None
    unknowns: int | None = None

    def order(self, index: int) -> float:
        """Return the efficiency of one signed order.

        Args:
            index: Signed diffraction order.

        Returns:
            Reflected efficiency.

        Raises:
            KeyError: If the order is outside the computed range.
        """

        match = np.nonzero(np.isclose(self.orders, float(index)))[0]
        if match.size != 1:
            raise KeyError(f"Order {index} is not in this result.")
        return float(np.real(self.efficiency[int(match[0])]))


def _profile_grating(
    *,
    period_lpermm: float,
    x_points_nm: np.ndarray,
    z_points_nm: np.ndarray,
    substrate: Any,
    layer: Any = None,
    layer_thickness_nm: float = 0.0,
    top_cap: Any = None,
    top_cap_thickness_nm: float = 0.0,
) -> Any:
    """Return a ProfileGrating with the requested stack.

    Args:
        period_lpermm: Period in lines per millimeter.
        x_points_nm: Profile x coordinates for one period.
        z_points_nm: Profile heights for one period.
        substrate: Substrate material.
        layer: Optional coating material.
        layer_thickness_nm: Coating thickness; ``0`` means no coating.
        top_cap: Optional cap material.
        top_cap_thickness_nm: Cap thickness.

    Returns:
        Configured grating.
    """

    return grax.ProfileGrating(
        period_lpermm=period_lpermm,
        x_points_nm=np.asarray(x_points_nm, dtype=float),
        z_points_nm=np.asarray(z_points_nm, dtype=float),
        substrate_material=substrate,
        layer_material=layer if layer is not None else substrate,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=top_cap,
        top_cap_thickness_nm=top_cap_thickness_nm,
        z_resolution_nm=0.05,
        x_resolution_nm=0.5,
    )


def _sinusoid(
    period_nm: float, depth_nm: float, samples: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    """Return one period of a sinusoidal profile.

    Args:
        period_nm: Period in nanometers.
        depth_nm: Peak-to-trough depth in nanometers.
        samples: Number of sample points.

    Returns:
        Positions and heights.
    """

    x = np.linspace(0.0, period_nm, samples)
    z = 0.5 * depth_nm * (1.0 - np.cos(2.0 * np.pi * x / period_nm))
    return x, z


def build_sinusoid_case(
    *,
    period_nm: float,
    depth_nm: float = 20.0,
    energy_ev: float = 100.0,
    grazing_angle_deg: float = 4.0,
    polarization: int = -1,
    fourier_orders: int = 40,
    reported_orders: int = 3,
    samples: int = 257,
) -> IntegralCase:
    """Return a shallow-sinusoid case at an arbitrary period.

    This is the vehicle for the ``d / lambda`` scaling study. Sweeping the
    *period* at fixed wavelength and fixed depth holds ``h / lambda`` constant
    while ``d / lambda`` grows, so the grating gets shallower relative to its
    period and the excited order count stays bounded. That is the regime real
    X-ray gratings occupy -- ``h / d`` between 0.006 and 0.024 across graxPy's
    validation suite -- and it isolates the quantity under test.

    Sweeping the energy instead would grow ``h / lambda`` at the same time,
    exciting more orders and confounding the two effects.

    The defaults matter for the study to mean anything. At 40 degrees grazing on
    Si every order of this grating sits between 2e-6 and 4e-4, so an absolute
    tolerance of 1e-4 -- the level the plan gates on, and the level that matters
    for the real validation cases -- would permit 25 to 100 percent relative
    error and the measurement would be meaningless. At 4 degrees grazing in TM,
    matching the validation suite, order zero carries 13 percent and the first
    orders a few times 1e-3, so the same absolute tolerance is a real test.

    Args:
        period_nm: Grating period in nanometers.
        depth_nm: Peak-to-trough depth in nanometers.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        polarization: ``1`` for TE, ``-1`` for TM.
        fourier_orders: Truncation for the reference solvers.
        reported_orders: Half-width of the compared order range.
        samples: Profile sample count for one period.

    Returns:
        The case.
    """

    x, z = _sinusoid(float(period_nm), float(depth_nm), samples)
    return IntegralCase(
        name=f"sinusoid_d{period_nm:g}nm",
        description=f"Shallow sinusoid on Si, period {period_nm:g} nm, depth {depth_nm:g} nm",
        grating=_profile_grating(
            period_lpermm=1e6 / float(period_nm),
            x_points_nm=x,
            z_points_nm=z,
            substrate="Si",
        ),
        energy_ev=float(energy_ev),
        grazing_angle_deg=float(grazing_angle_deg),
        polarization=int(polarization),
        fourier_orders=int(fourier_orders),
        reported_orders=int(reported_orders),
        interfaces=1,
    )


def build_cases() -> dict[str, IntegralCase]:
    """Return the benchmark cases keyed by name.

    Periods and energies are the *defaults*; the scaling study varies them
    through :meth:`IntegralCase.at`. Materials are the bundled Henke tables, so
    these exercise the same optical-constant path the production solvers use.

    Returns:
        Mapping of case name to case.
    """

    cases: dict[str, IntegralCase] = {}

    # B1: flat interface. The only case with a closed-form answer.
    cases["B1_flat"] = IntegralCase(
        name="B1_flat",
        description="Flat Si interface, analytic Fresnel reference",
        grating=_profile_grating(
            period_lpermm=1e6 / 100.0,
            x_points_nm=np.array([0.0, 100.0]),
            z_points_nm=np.array([0.0, 0.0]),
            substrate="Si",
        ),
        energy_ev=100.0,
        grazing_angle_deg=30.0,
        polarization=1,
        fourier_orders=10,
        reported_orders=2,
        interfaces=1,
    )

    # B2: the workhorse for the scaling study. Smooth, shallow, no corners.
    period_nm = 100.0
    x, z = _sinusoid(period_nm, 5.0)
    cases["B2_sinusoid"] = IntegralCase(
        name="B2_sinusoid",
        description="Shallow sinusoid on Si, smooth profile",
        grating=_profile_grating(
            period_lpermm=1e6 / period_nm,
            x_points_nm=x,
            z_points_nm=z,
            substrate="Si",
        ),
        energy_ev=100.0,
        grazing_angle_deg=40.0,
        polarization=1,
        fourier_orders=40,
        reported_orders=3,
        interfaces=1,
    )

    # B3: corners with moderate slope.
    cases["B3_laminar_15deg"] = IntegralCase(
        name="B3_laminar_15deg",
        description="Laminar Si, 15 degree sidewalls",
        grating=grax.LaminarGrating(
            period_lpermm=1e6 / 100.0,
            width_to_period_ratio=0.6,
            depth_nm=5.0,
            left_wall_angle_deg=15.0,
            right_wall_angle_deg=15.0,
            substrate_material="Si",
            layer_material="Si",
            layer_thickness_nm=0.0,
            z_resolution_nm=0.05,
            x_resolution_nm=0.5,
        ),
        energy_ev=100.0,
        grazing_angle_deg=40.0,
        polarization=1,
        fourier_orders=40,
        reported_orders=3,
        interfaces=1,
    )

    # B4: exactly vertical walls make the profile a non-graph.
    cases["B4_laminar_90deg"] = IntegralCase(
        name="B4_laminar_90deg",
        description="Laminar Si, vertical sidewalls (non-graph profile)",
        grating=grax.LaminarGrating(
            period_lpermm=1e6 / 100.0,
            width_to_period_ratio=0.5,
            depth_nm=5.0,
            left_wall_angle_deg=90.0,
            right_wall_angle_deg=90.0,
            substrate_material="Si",
            layer_material="Si",
            layer_thickness_nm=0.0,
            z_resolution_nm=0.05,
            x_resolution_nm=0.5,
        ),
        energy_ev=100.0,
        grazing_angle_deg=40.0,
        polarization=1,
        fourier_orders=40,
        reported_orders=3,
        interfaces=1,
    )

    # B5: sharp apex.
    cases["B5_blazed"] = IntegralCase(
        name="B5_blazed",
        description="Blazed Si with anti-blaze facet",
        grating=grax.BlazedGrating(
            period_lpermm=1e6 / 100.0,
            blaze_angle_deg=2.0,
            anti_blaze_angle_deg=8.0,
            substrate_material="Si",
            layer_material="Si",
            layer_thickness_nm=0.0,
            z_resolution_nm=0.05,
            x_resolution_nm=0.5,
        ),
        energy_ev=100.0,
        grazing_angle_deg=40.0,
        polarization=-1,
        fourier_orders=40,
        reported_orders=3,
        interfaces=1,
    )

    # B6: three interfaces, with a cap far thinner than the corrugation depth.
    x6, z6 = _sinusoid(100.0, 5.0)
    cases["B6_coated"] = IntegralCase(
        name="B6_coated",
        description="Shallow sinusoid, Si + 10 nm Pt + 0.7 nm C cap",
        grating=_profile_grating(
            period_lpermm=1e6 / 100.0,
            x_points_nm=x6,
            z_points_nm=z6,
            substrate="Si",
            layer="Pt",
            layer_thickness_nm=10.0,
            top_cap="C",
            top_cap_thickness_nm=0.7,
        ),
        energy_ev=100.0,
        grazing_angle_deg=40.0,
        polarization=-1,
        fourier_orders=40,
        reported_orders=3,
        interfaces=3,
    )

    return cases


def case_by_name(name: str) -> IntegralCase:
    """Return one case by name.

    Args:
        name: Case identifier.

    Returns:
        The case.

    Raises:
        KeyError: If the name is unknown.
    """

    cases = build_cases()
    if name not in cases:
        raise KeyError(f"Unknown case {name!r}. Available: {', '.join(sorted(cases))}")
    return cases[name]


def run_reference(case: IntegralCase, solver: str) -> SolverRun:
    """Run one of the two Fourier solvers on a case.

    Args:
        case: The case to solve.
        solver: ``"rcwa"`` or ``"neviere"``.

    Returns:
        The result.

    Raises:
        ValueError: If the solver name is not one of the two references.
    """

    if solver not in ("rcwa", "neviere"):
        raise ValueError(f"run_reference takes 'rcwa' or 'neviere', got {solver!r}.")

    with PeakMemorySampler() as sampler:
        start = time.perf_counter()
        textures, profile = case.grating.build_textures(case.energy_ev, n_inc=1.0 + 0.0j)
        parm = res0(case.polarization)
        aa = res1(
            case.wavelength_nm,
            case.period_nm,
            textures,
            case.fourier_orders,
            case.beta0,
            parm,
        )
        result = res2(aa, profile, parm) if solver == "rcwa" else res2_dm(aa, profile, parm)
        seconds = time.perf_counter() - start

    orders = np.asarray(result.inc_top_reflected.order, dtype=float)
    keep = np.isin(orders, case.orders)
    return SolverRun(
        solver=solver,
        efficiency=np.real(result.inc_top_reflected.efficiency)[keep],
        orders=orders[keep],
        energy_balance=_energy_balance(result),
        seconds=seconds,
        peak_memory_bytes=sampler.peak_memory_bytes,
    )


def run_integral(
    case: IntegralCase,
    *,
    boundary_points: int | str = "auto",
    **option_overrides: Any,
) -> SolverRun:
    """Run the boundary-integral solver on a case.

    Args:
        case: The case to solve.
        boundary_points: Panels per interface, or ``"auto"``.
        **option_overrides: Further :class:`~grax.solvers.integral.IntegralOptions`
            fields, for the parameter sweeps in the scaling study.

    Returns:
        The result.
    """

    options = IntegralOptions(boundary_points=boundary_points, **option_overrides)
    with PeakMemorySampler() as sampler:
        start = time.perf_counter()
        result = res2_im(
            grating=case.grating,
            wavelength_nm=case.wavelength_nm,
            period_nm=case.period_nm,
            orders=case.orders,
            beta0=case.beta0,
            polarization=case.polarization,
            photon_energy_ev=case.energy_ev,
            options=options,
        )
        seconds = time.perf_counter() - start

    resolved = options.resolved_boundary_points(
        period_nm=case.period_nm,
        wavelength_nm=case.wavelength_nm,
        orders=case.reported_orders,
    )
    return SolverRun(
        solver="integral",
        efficiency=np.real(result.inc_top_reflected.efficiency),
        orders=np.asarray(result.inc_top_reflected.order, dtype=float),
        energy_balance=_energy_balance(result),
        seconds=seconds,
        peak_memory_bytes=sampler.peak_memory_bytes,
        unknowns=resolved,
    )


def _energy_balance(result: Any) -> float:
    """Return the summed reflected and transmitted efficiency.

    Args:
        result: A ``Res2Result``.

    Returns:
        The sum over all reported orders.
    """

    return float(np.sum(np.real(result.inc_top_reflected.efficiency))) + float(
        np.sum(np.real(result.inc_top_transmitted.efficiency))
    )


def fresnel_reflectance(
    *, n_above: complex, n_below: complex, beta0: float, polarization: int
) -> float:
    """Return the analytic order-zero reflectance of a flat interface.

    Args:
        n_above: Refractive index of the incident medium.
        n_below: Refractive index of the substrate.
        beta0: In-plane direction cosine, normalised to the vacuum wavenumber.
        polarization: ``1`` for TE, ``-1`` for TM.

    Returns:
        Reflectance ``|r|^2``.
    """

    beta_above = _branch(np.sqrt(complex(n_above) ** 2 - beta0**2 + 0j))
    beta_below = _branch(np.sqrt(complex(n_below) ** 2 - beta0**2 + 0j))
    if polarization == 1:
        amplitude = (beta_above - beta_below) / (beta_above + beta_below)
    else:
        above = beta_above / complex(n_above) ** 2
        below = beta_below / complex(n_below) ** 2
        amplitude = (above - below) / (above + below)
    return float(abs(amplitude) ** 2)


def _branch(value: complex) -> complex:
    """Return the ``Im >= 0`` square-root branch used throughout graxPy."""

    if value.imag < 0 or (abs(value.imag) < 1e-15 and value.real < 0):
        return -value
    return value


def max_deviation(left: SolverRun, right: SolverRun) -> float:
    """Return the largest absolute efficiency difference over shared orders.

    Args:
        left: One result.
        right: Another result.

    Returns:
        Maximum absolute difference.

    Raises:
        ValueError: If the two results share no orders.
    """

    shared = np.intersect1d(left.orders, right.orders)
    if shared.size == 0:
        raise ValueError("The two results share no diffraction orders.")
    return max(abs(left.order(int(order)) - right.order(int(order))) for order in shared)
