"""Objective evaluation for Ax-based grating optimization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Dict, Mapping, Optional
import logging
import warnings

import numpy as np

from grax import BatchSimulationRunner, monochromator_grazing_angles_deg
from grax.simulation import _resolve_max_workers as _resolve_simulation_max_workers

from .data import MeasurementData, sample_measurement_data
from .evaluation import build_evaluation_cases

module_logger = logging.getLogger(__name__)

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
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Optional hook that builds a grating from the trial
            parameter mapping.
        resolve_solver_parameters_fn: Optional hook that resolves solver
            parameters from the trial parameter mapping.

    Returns:
        Simulated efficiency values on the measurement grid.
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
        default_diffraction_order=int(config.diffraction_order),
        default_fourier_orders=int(config.fourier_orders),
        max_workers=getattr(config, "max_workers", None),
        validate_physical_results=bool(config.validate_physical_results),
        backend=backend,
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


def reduce_joint_losses(
    per_measurement_losses: Mapping[str, float],
    *,
    reduction: str = "mean",
    weights: Mapping[str, float] | None = None,
    point_counts: Mapping[str, int] | None = None,
) -> float:
    """Combine per-measurement losses into one joint objective value.

    Args:
        per_measurement_losses: Loss for each measurement, keyed by label.
        reduction: One of ``"mean"``, ``"sum"``, ``"pooled"``, or ``"weighted"``.
        weights: Explicit per-measurement weights, required for ``"weighted"``.
        point_counts: Evaluation point count per measurement, required for
            ``"pooled"``.

    Returns:
        The reduced joint loss.

    Raises:
        ValueError: If the losses are empty, the reduction is unknown, or the
            inputs required by the chosen reduction are missing.
    """

    if len(per_measurement_losses) == 0:
        raise ValueError("per_measurement_losses must not be empty.")

    labels = list(per_measurement_losses)
    losses = np.asarray([float(per_measurement_losses[label]) for label in labels], dtype=float)

    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        weight_values = np.ones(len(labels), dtype=float)
    elif reduction == "pooled":
        if point_counts is None:
            raise ValueError("point_counts is required for the 'pooled' reduction.")
        weight_values = np.asarray(
            [float(point_counts[label]) for label in labels],
            dtype=float,
        )
    elif reduction == "weighted":
        if weights is None:
            raise ValueError("weights is required for the 'weighted' reduction.")
        weight_values = np.asarray([float(weights[label]) for label in labels], dtype=float)
    else:
        raise ValueError(
            "joint_loss_reduction must be one of 'mean', 'sum', 'pooled', or 'weighted'."
        )

    weight_total = float(np.sum(weight_values))
    if weight_total <= 0.0:
        raise ValueError("Joint loss reduction weights must sum to a positive value.")
    return float(np.sum(weight_values * losses) / weight_total)


def simulate_joint_efficiency_curves_with_metadata(
    config: Any,
    trial_parameters: Mapping[str, float],
    joint_measurements: Sequence[Any],
    *,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Simulate efficiency curves for every measurement in one flat batch.

    All measurements are evaluated in a single :class:`BatchSimulationRunner`
    batch so trial-level ``max_workers`` parallelizes across angles as well as
    energies. Results are reassembled by ``result.index`` because the parallel
    runner yields in completion order rather than input order.

    Args:
        config: Joint optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        joint_measurements: Prepared per-angle measurements to evaluate.
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Hook that builds a grating from the trial parameters.
        resolve_solver_parameters_fn: Hook that resolves solver parameters.

    Returns:
        A mapping of measurement label to simulated efficiencies, and the
        runner's resolved worker count.

    Raises:
        RuntimeError: If the required build hooks are missing.
        _BatchCaseFailure: If any case fails or no result is returned for a case.
    """

    if build_grating_fn is None:
        raise RuntimeError("build_grating_fn is required for joint optimizer execution.")
    if resolve_solver_parameters_fn is None:
        raise RuntimeError(
            "resolve_solver_parameters_fn is required for joint optimizer execution."
        )
    grating = build_grating_fn(trial_parameters)
    solver_parameters = resolve_solver_parameters_fn(trial_parameters)

    cases: list[dict[str, object]] = []
    case_slots: list[tuple[str, int]] = []
    for joint_measurement in joint_measurements:
        label = str(joint_measurement.label)
        for point_index, energy_ev in enumerate(joint_measurement.evaluation_energies_ev):
            cases.append(
                {
                    "case_id": f"trial_eval_{label}_{point_index}",
                    "grating": grating,
                    "energy_ev": float(energy_ev),
                    "grazing_angle_deg": float(joint_measurement.grazing_angle_deg),
                    "diffraction_order": int(config.diffraction_order),
                    "fourier_orders": int(config.fourier_orders),
                    "roughness_sigma_nm": solver_parameters["roughness_sigma_nm"],
                }
            )
            case_slots.append((label, point_index))

    runner = BatchSimulationRunner(
        default_diffraction_order=int(config.diffraction_order),
        default_fourier_orders=int(config.fourier_orders),
        max_workers=getattr(config, "max_workers", None),
        validate_physical_results=bool(config.validate_physical_results),
        backend=backend,
    )

    simulated: dict[str, np.ndarray] = {
        str(joint_measurement.label): np.full(
            len(joint_measurement.evaluation_energies_ev),
            np.nan,
            dtype=float,
        )
        for joint_measurement in joint_measurements
    }
    filled = np.zeros(len(cases), dtype=bool)
    for result in runner.run_cases(cases):
        if result.status != "ok":
            raise _BatchCaseFailure(result.case_id, result.status, int(runner.resolved_max_workers))
        flat_index = int(result.index)
        label, point_index = case_slots[flat_index]
        simulated[label][point_index] = float(result.selected_efficiency)
        filled[flat_index] = True

    if not bool(filled.all()):
        missing_index = int(np.argmin(filled))
        raise _BatchCaseFailure(
            str(cases[missing_index]["case_id"]),
            "missing",
            int(runner.resolved_max_workers),
        )

    return simulated, int(runner.resolved_max_workers)


def evaluate_joint_trial_with_metadata(
    config: Any,
    trial_parameters: Mapping[str, float],
    joint_measurements: Sequence[Any],
    *,
    loss_function: LossFunction | None = None,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> tuple[float, dict[str, float], dict[str, np.ndarray], int]:
    """Evaluate one joint multi-angle trial.

    Args:
        config: Joint optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        joint_measurements: Prepared per-angle measurements to evaluate.
        loss_function: Optional custom per-measurement loss function.
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Hook that builds a grating from the trial parameters.
        resolve_solver_parameters_fn: Hook that resolves solver parameters.

    Returns:
        The joint loss, the per-measurement losses, the simulated curves, and
        the resolved worker count. On failure the penalty is reported for the
        joint loss and every measurement, with empty simulated curves.
    """

    _warn_if_numpy_backend_requested(backend, stacklevel=2)
    selected_loss_function = loss_function or mean_squared_error
    labels = [str(joint_measurement.label) for joint_measurement in joint_measurements]

    try:
        simulated, resolved_max_workers = simulate_joint_efficiency_curves_with_metadata(
            config,
            trial_parameters,
            joint_measurements,
            backend=backend,
            build_grating_fn=build_grating_fn,
            resolve_solver_parameters_fn=resolve_solver_parameters_fn,
        )
    except _BatchCaseFailure as error:
        module_logger.warning(
            "Joint optimizer trial penalized: %s (resolved_max_workers=%s).",
            error,
            error.resolved_max_workers,
        )
        penalty = float(config.failure_penalty)
        return penalty, {label: penalty for label in labels}, {}, int(error.resolved_max_workers)
    except Exception as error:
        module_logger.warning(
            "Joint optimizer trial penalized by %s: %s.",
            type(error).__name__,
            error,
        )
        penalty = float(config.failure_penalty)
        return penalty, {label: penalty for label in labels}, {}, _trial_max_workers(config)

    per_measurement_losses = {
        str(joint_measurement.label): float(
            selected_loss_function(
                np.asarray(joint_measurement.evaluation_efficiency, dtype=float),
                simulated[str(joint_measurement.label)],
            )
        )
        for joint_measurement in joint_measurements
    }
    joint_loss = reduce_joint_losses(
        per_measurement_losses,
        reduction=str(config.joint_loss_reduction),
        weights={
            str(joint_measurement.label): float(joint_measurement.weight)
            for joint_measurement in joint_measurements
        },
        point_counts={
            str(joint_measurement.label): len(joint_measurement.evaluation_energies_ev)
            for joint_measurement in joint_measurements
        },
    )
    return joint_loss, per_measurement_losses, simulated, int(resolved_max_workers)


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


def evaluate_trial_curve_with_metadata(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    loss_function: LossFunction | None = None,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> tuple[float, int, np.ndarray | None]:
    """Evaluate one Ax trial and also return its simulated curve.

    Returning the curve lets callers plot the best fit without re-running the
    simulation afterwards.

    Args:
        config: Optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        measurement: Energy grid and target efficiencies used for evaluation.
        loss_function: Optional custom loss function.
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Hook that builds a grating from the trial parameters.
        resolve_solver_parameters_fn: Hook that resolves solver parameters.

    Returns:
        The loss, the resolved worker count, and the simulated efficiencies.
        The curve is ``None`` when the trial was penalized.
    """

    _warn_if_numpy_backend_requested(backend, stacklevel=3)
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
        module_logger.warning(
            "Optimizer trial penalized: %s (resolved_max_workers=%s).",
            error,
            error.resolved_max_workers,
        )
        return float(config.failure_penalty), int(error.resolved_max_workers), None
    except Exception as error:
        module_logger.warning(
            "Optimizer trial penalized by %s: %s.",
            type(error).__name__,
            error,
        )
        return float(config.failure_penalty), _trial_max_workers(config), None
    return (
        float(selected_loss_function(evaluation_measurement.efficiency, simulated_efficiency)),
        int(resolved_max_workers),
        simulated_efficiency,
    )


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
    """Evaluate one Ax trial and return loss plus resolved worker count.

    Args:
        config: Optimization configuration describing the simulation setup.
        trial_parameters: Ax trial parameters for the current candidate.
        measurement: Energy grid and target efficiencies used for evaluation.
        loss_function: Optional custom loss function.
        backend: RCWA backend to use for the simulation.
        build_grating_fn: Hook that builds a grating from the trial parameters.
        resolve_solver_parameters_fn: Hook that resolves solver parameters.

    Returns:
        The loss and the resolved worker count.
    """

    loss, resolved_max_workers, _simulated_efficiency = evaluate_trial_curve_with_metadata(
        config,
        trial_parameters,
        measurement,
        loss_function=loss_function,
        backend=backend,
        build_grating_fn=build_grating_fn,
        resolve_solver_parameters_fn=resolve_solver_parameters_fn,
    )
    return loss, resolved_max_workers
