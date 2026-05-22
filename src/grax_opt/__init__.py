"""Standalone optimization helpers for grax."""

from .convergence import (
    SimulationConvergenceConfig,
    SimulationConvergenceEnergyResult,
    SimulationConvergenceResult,
    optimize_simulation_convergence,
)
from .config import ParameterBounds
from .data import MeasurementData, load_measurement_data, sample_measurement_data
from .dynamic import (
    DynamicOptimizationConfig as MeasurementFitConfig,
    build_dynamic_ax_parameters,
    optimize_dynamic as _optimize_to_measurements,
    resolve_dynamic_trial_parameters,
)
from .objective import build_evaluation_measurement, evaluate_trial
from .optimize import OptimizationResult, TrialRecord, json_safe_grating_parameters

def optimize_to_measurements(config):
    """Run the measurement-fit optimizer with the public renamed entrypoint.

    Args:
        config: Measurement-fit configuration or plain mapping.

    Returns:
        Optimization result bundle with persisted artifacts.
    """

    return _optimize_to_measurements(config)


__all__ = [
    "MeasurementFitConfig",
    "MeasurementData",
    "OptimizationResult",
    "ParameterBounds",
    "SimulationConvergenceConfig",
    "SimulationConvergenceEnergyResult",
    "SimulationConvergenceResult",
    "TrialRecord",
    "build_evaluation_measurement",
    "build_dynamic_ax_parameters",
    "evaluate_trial",
    "json_safe_grating_parameters",
    "load_measurement_data",
    "sample_measurement_data",
    "optimize_simulation_convergence",
    "optimize_to_measurements",
    "resolve_dynamic_trial_parameters",
]
