"""Standalone optimization helpers for grax."""

from .config import ParameterBounds
from .data import MeasurementData, load_measurement_data, sample_measurement_data
from .dynamic import (
    MeasurementFitConfig,
    build_measurement_fit_ax_parameters,
    optimize_to_measurements as _optimize_to_measurements,
    resolve_measurement_fit_trial_parameters,
)
from .objective import build_evaluation_measurement, evaluate_trial
from .optimize import OptimizationResult, TrialRecord, json_safe_grating_parameters

def optimize_to_measurements(config):
    """Run the measurement-fit optimizer with the public renamed entrypoint.

    Args:
        config: Measurement-fit configuration or plain mapping with the same
            required keys as :func:`grax_opt.dynamic.optimize_to_measurements`.

    Returns:
        OptimizationResult: Result bundle with persisted artifacts.
    """

    return _optimize_to_measurements(config)


__all__ = [
    "MeasurementFitConfig",
    "MeasurementData",
    "OptimizationResult",
    "ParameterBounds",
    "TrialRecord",
    "build_evaluation_measurement",
    "build_measurement_fit_ax_parameters",
    "evaluate_trial",
    "json_safe_grating_parameters",
    "load_measurement_data",
    "sample_measurement_data",
    "optimize_to_measurements",
    "resolve_measurement_fit_trial_parameters",
]
