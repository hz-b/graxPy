"""Objective evaluation for Ax-based grating optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import numpy as np

from grax import monochromator_grazing_angles_deg, run_simulation

from .config import BlazedAxConfig, LaminarAxConfig
from .data import MeasurementData, sample_measurement_data
from .model import build_grating, resolve_solver_parameters

LossFunction = Callable[[np.ndarray, np.ndarray], float]


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


LOSS_FUNCTIONS: dict[str, LossFunction] = {
    "mse": mean_squared_error,
}


def build_evaluation_measurement(
    config: BlazedAxConfig | LaminarAxConfig,
    measurement: MeasurementData,
) -> MeasurementData:
    """Return the measurement data used for objective evaluation."""

    return sample_measurement_data(measurement, config.evaluation_energies_ev)


def simulate_efficiency_curve(
    config: BlazedAxConfig | LaminarAxConfig,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    backend: str,
) -> np.ndarray:
    """Simulate the selected diffraction-order efficiency on the measurement grid."""

    grating = build_grating(config, trial_parameters)
    solver_parameters = resolve_solver_parameters(config, trial_parameters)
    if isinstance(config, LaminarAxConfig) and config.angle_mode == "fixed":
        grazing_angles = np.full(
            measurement.energy_ev.shape,
            float(config.grazing_angle_deg),
            dtype=float,
        )
    else:
        grazing_angles = monochromator_grazing_angles_deg(
            measurement.energy_ev,
            period_lpermm=float(grating.period_lpermm),
            diffraction_order=config.diffraction_order,
            cff=config.cff,
        )
    efficiencies: list[float] = []
    for energy_ev, grazing_angle_deg in zip(measurement.energy_ev, grazing_angles):
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
    config: BlazedAxConfig | LaminarAxConfig,
    trial_parameters: Mapping[str, float],
    measurement: MeasurementData,
    *,
    loss_function: LossFunction | None = None,
    backend: str,
) -> float:
    """Evaluate one Ax trial and return a scalar loss.

    Any simulation failure is converted into a finite penalty so the optimizer
    can continue exploring the search space.
    """

    selected_loss_function = loss_function or LOSS_FUNCTIONS[config.loss_name]
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    try:
        simulated_efficiency = simulate_efficiency_curve(
            config,
            trial_parameters,
            evaluation_measurement,
            backend=backend,
        )
    except Exception:
        return float(config.failure_penalty)
    return float(selected_loss_function(evaluation_measurement.efficiency, simulated_efficiency))
