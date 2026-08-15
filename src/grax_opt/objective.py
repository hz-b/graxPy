"""Objective evaluation for Ax-based grating optimization."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional
import warnings

import numpy as np

from grax import BatchSimulationRunner, monochromator_grazing_angles_deg
from grax.simulation import _resolve_max_workers as _resolve_simulation_max_workers

from .data import MeasurementData, sample_measurement_data
from .evaluation import build_evaluation_cases

LossFunction = Callable[[np.ndarray, np.ndarray], float]
BuildGratingFunction = Callable[[Mapping[str, float]], object]
ResolveSolverParametersFunction = Callable[[Mapping[str, float]], Dict[str, Optional[float]]]


def _evaluation_schedule(
    config: Any,
    measurement: MeasurementData,
    *,
    grating_period_lpermm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned energy and angle arrays for one trial evaluation."""

    explicit_angles = getattr(config, "evaluation_grazing_angles_deg", [])
    if len(explicit_angles) > 0:
        evaluation_energies_ev, grazing_angles, _evaluation_mode = build_evaluation_cases(
            config.evaluation_energies_ev,
            explicit_angles,
        )
        assert grazing_angles is not None
        return evaluation_energies_ev, grazing_angles
    if getattr(config, "angle_mode", "fixed") == "fixed":
        evaluation_energies_ev = measurement.energy_ev
        grazing_angles = np.full(
            evaluation_energies_ev.shape,
            float(config.grazing_angle_deg),
            dtype=float,
        )
        return evaluation_energies_ev, grazing_angles
    evaluation_energies_ev = measurement.energy_ev
    grazing_angles = monochromator_grazing_angles_deg(
        evaluation_energies_ev,
        period_lpermm=grating_period_lpermm,
        diffraction_order=config.diffraction_order,
        cff=config.cff,
    )
    return evaluation_energies_ev, grazing_angles


def _trial_max_workers(config: Any) -> int:
    """Return the effective trial-level worker count for optimizer evaluation."""

    return int(_resolve_simulation_max_workers(getattr(config, "max_workers", None)))


class _BatchCaseFailure(RuntimeError):
    """Raised when a trial's batch evaluation contains a failed case.

    Carries the runner's resolved worker count so callers can still report it
    on the failure-penalty path instead of falling back to a generic default.
    """

    def __init__(self, case_id: str, status: str, resolved_max_workers: int) -> None:
        super().__init__(f"Optimizer trial batch case {case_id} failed with status={status}.")
        self.resolved_max_workers = resolved_max_workers


def _warn_if_numpy_backend_requested(backend: str, *, stacklevel: int = 3) -> None:
    """Warn when callers explicitly request the deprecated NumPy backend."""

    if str(backend).lower() != "numpy":
        return
    warnings.warn(
        "backend='numpy' is deprecated and will be removed in a future version. "
        "Use backend='numba' or rely on the default numba backend instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def mean_squared_error(
    measured_efficiency: np.ndarray,
    simulated_efficiency: np.ndarray,
) -> float:
    """Return the mean squared error between measured and simulated data."""

    residual = np.asarray(simulated_efficiency, dtype=float) - np.asarray(
        measured_efficiency,
        dtype=float,
    )
    return float(np.mean(residual**2))


def build_evaluation_measurement(
    config: Any,
    measurement: MeasurementData,
) -> MeasurementData:
    """Return the measurement data used for objective evaluation."""

    evaluation_energies_ev, _evaluation_angles_deg, _evaluation_mode = build_evaluation_cases(
        config.evaluation_energies_ev,
        getattr(config, "evaluation_grazing_angles_deg", []),
    )
    return sample_measurement_data(measurement, evaluation_energies_ev)


def simulate_efficiency_curve_with_metadata(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> tuple[np.ndarray, int]:
    """Simulate the selected diffraction-order efficiency.

    Args:
        config: Optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        measurement: Energy grid and target efficiencies used for evaluation.
        backend: Fourier coefficient backend to use for the simulation.
        build_grating_fn: Optional hook that builds a grating from the trial
            parameter mapping.
        resolve_solver_parameters_fn: Optional hook that resolves solver
            parameters from the trial parameter mapping.

    Returns:
        Simulated efficiency values on the measurement grid.

    Note:
        The electromagnetic solver comes from ``config.solver`` (and
        ``config.solver_options``), alongside ``diffraction_order`` and
        ``fourier_orders``, rather than from a separate argument. ``backend``
        stays an argument because it is resolved from ``"auto"`` by the caller.
    """

    if build_grating_fn is None:
        raise RuntimeError("build_grating_fn is required for measurement-fit optimizer execution.")
    if resolve_solver_parameters_fn is None:
        raise RuntimeError(
            "resolve_solver_parameters_fn is required for measurement-fit optimizer execution."
        )
    grating = build_grating_fn(trial_parameters)
    solver_parameters = resolve_solver_parameters_fn(trial_parameters)
    evaluation_energies_ev, grazing_angles = _evaluation_schedule(
        config,
        measurement,
        grating_period_lpermm=float(grating.period_lpermm),
    )
    cases: list[dict[str, object]] = []
    for index, (energy_ev, grazing_angle_deg) in enumerate(zip(evaluation_energies_ev, grazing_angles)):
        cases.append(
            {
                "case_id": f"trial_eval_{index}",
                "grating": grating,
                "energy_ev": float(energy_ev),
                "grazing_angle_deg": float(grazing_angle_deg),
                "diffraction_order": int(config.diffraction_order),
                "fourier_orders": int(config.fourier_orders),
                "roughness_sigma_nm": solver_parameters["roughness_sigma_nm"],
            }
        )

    runner = BatchSimulationRunner(
        diffraction_order=int(config.diffraction_order),
        fourier_orders=int(config.fourier_orders),
        max_workers=getattr(config, "max_workers", None),
        validate_physical_results=bool(config.validate_physical_results),
        backend=backend,
        solver=str(getattr(config, "solver", "rcwa")),
        solver_options=getattr(config, "solver_options", None),
    )
    trial_results = list(runner.run_cases(cases))
    efficiencies = np.empty(len(cases), dtype=float)
    for result in trial_results:
        if result.status != "ok":
            raise _BatchCaseFailure(result.case_id, result.status, int(runner.resolved_max_workers))
        efficiencies[int(result.index)] = float(result.selected_efficiency)

    return efficiencies, int(runner.resolved_max_workers)


def simulate_efficiency_curve(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> np.ndarray:
    """Simulate the selected diffraction-order efficiency."""

    efficiencies, _resolved_max_workers = simulate_efficiency_curve_with_metadata(
        config,
        trial_parameters,
        measurement,
        backend=backend,
        build_grating_fn=build_grating_fn,
        resolve_solver_parameters_fn=resolve_solver_parameters_fn,
    )
    return efficiencies


def evaluate_trial(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    loss_function: LossFunction | None = None,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> float:
    """Evaluate one Ax trial and return a scalar loss.

    Any simulation failure is converted into a finite penalty so the optimizer
    can continue exploring the search space.

    Args:
        config: Optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        measurement: Energy grid and target efficiencies used for evaluation.
        loss_function: Optional custom loss function.
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Optional hook that builds a grating from the trial
            parameter mapping.
        resolve_solver_parameters_fn: Optional hook that resolves solver
            parameters from the trial parameter mapping.

    Returns:
        Scalar loss value for the candidate.
    """

    _warn_if_numpy_backend_requested(backend, stacklevel=2)
    selected_loss_function = loss_function or mean_squared_error
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    try:
        simulate_kwargs: dict[str, object] = {
            "backend": backend,
        }
        if build_grating_fn is not None:
            simulate_kwargs["build_grating_fn"] = build_grating_fn
        if resolve_solver_parameters_fn is not None:
            simulate_kwargs["resolve_solver_parameters_fn"] = resolve_solver_parameters_fn
        simulated_efficiency, _resolved_max_workers = simulate_efficiency_curve_with_metadata(
            config,
            trial_parameters,
            evaluation_measurement,
            **simulate_kwargs,
        )
    except Exception:
        return float(config.failure_penalty)
    return float(selected_loss_function(evaluation_measurement.efficiency, simulated_efficiency))


def evaluate_trial_with_metadata(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    loss_function: LossFunction | None = None,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> tuple[float, int]:
    """Evaluate one Ax trial and return loss plus resolved worker count."""

    _warn_if_numpy_backend_requested(backend, stacklevel=2)
    selected_loss_function = loss_function or mean_squared_error
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    try:
        simulated_efficiency, resolved_max_workers = simulate_efficiency_curve_with_metadata(
            config,
            trial_parameters,
            evaluation_measurement,
            backend=backend,
            build_grating_fn=build_grating_fn,
            resolve_solver_parameters_fn=resolve_solver_parameters_fn,
        )
    except _BatchCaseFailure as error:
        return float(config.failure_penalty), int(error.resolved_max_workers)
    except Exception:
        return float(config.failure_penalty), _trial_max_workers(config)
    return (
        float(selected_loss_function(evaluation_measurement.efficiency, simulated_efficiency)),
        int(resolved_max_workers),
    )
