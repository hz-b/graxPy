"""Standalone optimization helpers for grax."""

from .config import ParameterBounds
from .data import MeasurementData, load_measurement_data, sample_measurement_data
from .dynamic import (
    MeasurementFitConfig,
    build_measurement_fit_ax_parameters,
    optimize_to_measurements as _optimize_to_measurements,
    resolve_measurement_fit_trial_parameters,
)
from .joint import (
    JointMeasurement,
    JointMeasurementFitConfig,
    JointOptimizationResult,
    MeasurementSpec,
    optimize_to_joint_measurements as _optimize_to_joint_measurements,
)
from .objective import build_evaluation_measurement, evaluate_trial, reduce_joint_losses
from .optimize import OptimizationResult, TrialRecord, json_safe_grating_parameters

def optimize_to_measurements(config):
    """Run the measurement-fit optimizer with the public renamed entrypoint.

    Args:
        config: Spec mapping describing the measurement-fit optimization run.

    Returns:
        OptimizationResult: Result bundle with persisted artifacts.
    """

    return _optimize_to_measurements(config)


def optimize_to_joint_measurements(config):
    """Fit one parameter set jointly against several measured curves.

    Each measurement is recorded at its own fixed grazing angle and keeps its
    own energy grid.

    Args:
        config: Spec mapping describing the joint optimization run.

    Returns:
        JointOptimizationResult: Result bundle with persisted artifacts.
    """

    return _optimize_to_joint_measurements(config)


__all__ = [
    "JointMeasurement",
    "JointMeasurementFitConfig",
    "JointOptimizationResult",
    "MeasurementSpec",
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
    "optimize_to_joint_measurements",
    "optimize_to_measurements",
    "reduce_joint_losses",
    "resolve_measurement_fit_trial_parameters",
]
