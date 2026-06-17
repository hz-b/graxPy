"""Objective evaluation for Ax-based grating optimization."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional
import warnings

import numpy as np

from grax import monochromator_grazing_angles_deg, run_simulation

from .data import MeasurementData, sample_measurement_data
from .evaluation import build_evaluation_cases

LossFunction = Callable[[np.ndarray, np.ndarray], float]
BuildGratingFunction = Callable[[Mapping[str, float]], object]
ResolveSolverParametersFunction = Callable[[Mapping[str, float]], Dict[str, Optional[float]]]


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


def simulate_efficiency_curve(
    config: Any,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    backend: str,
    build_grating_fn: BuildGratingFunction | None = None,
    resolve_solver_parameters_fn: ResolveSolverParametersFunction | None = None,
) -> np.ndarray:
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
    explicit_angles = getattr(config, "evaluation_grazing_angles_deg", [])
    if len(explicit_angles) > 0:
        evaluation_energies_ev, grazing_angles, _evaluation_mode = build_evaluation_cases(
            config.evaluation_energies_ev,
            explicit_angles,
        )
    elif getattr(config, "angle_mode", "fixed") == "fixed":
        evaluation_energies_ev = measurement.energy_ev
        grazing_angles = np.full(
            evaluation_energies_ev.shape,
            float(config.grazing_angle_deg),
            dtype=float,
        )
    else:
        evaluation_energies_ev = measurement.energy_ev
        grazing_angles = monochromator_grazing_angles_deg(
            evaluation_energies_ev,
            period_lpermm=float(grating.period_lpermm),
            diffraction_order=config.diffraction_order,
            cff=config.cff,
        )
    efficiencies: list[float] = []
    for energy_ev, grazing_angle_deg in zip(evaluation_energies_ev, grazing_angles):
        result = run_simulation(
            grating=grating,
            energy_ev=float(energy_ev),
            grazing_angle_deg=float(grazing_angle_deg),
            diffraction_order=config.diffraction_order,
            fourier_orders=config.fourier_orders,
            validate_physical_results=config.validate_physical_results,
            roughness_sigma_nm=solver_parameters["roughness_sigma_nm"],
            backend=backend,
        )
        efficiencies.append(float(result.selected_efficiency))

    return np.asarray(efficiencies, dtype=float)


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
        simulated_efficiency = simulate_efficiency_curve(
            config,
            trial_parameters,
            evaluation_measurement,
            **simulate_kwargs,
        )
    except Exception:
        return float(config.failure_penalty)
    return float(selected_loss_function(evaluation_measurement.efficiency, simulated_efficiency))
