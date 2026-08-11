"""Measurement-fit optimization helpers for user-defined grating problems."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

import numpy as np
from grax.simulation import _resolve_max_workers as _resolve_simulation_max_workers

from .config import ParameterBounds
from .evaluation import normalize_evaluation_selection
from .data import MeasurementData, load_measurement_data
from .objective import build_evaluation_measurement, simulate_efficiency_curve
from .checkpoint import OptimizerCheckpointSession
from .loop import TrialEvaluation, TrialLoopState, run_ax_trial_loop
from .optimize import (
    OptimizationResult,
    TrialRecord,
    _atomic_write_text,
    _describe_optimizer_compute_context,
    _evaluate_candidate_batch,
    _import_ax_client as _import_ax_client_from_optimize,
    _import_objective_properties,
    _patch_torch_fork_rng_for_cpu_only,
    _resolve_optimizer_backend,
    _save_best_fit_plot,
    _save_loss_history_plot,
    _write_trial_history_csv,
    json_safe_grating_parameters,
)

BuildGratingFunction = Callable[[Mapping[str, float]], object]
ResolveSolverParametersFunction = Callable[[Mapping[str, float]], Dict[str, Optional[float]]]


def _coerce_parameter_bounds(
    bounds: ParameterBounds | Sequence[float],
) -> ParameterBounds:
    """Convert raw bound values into a typed bounds object."""

    if isinstance(bounds, ParameterBounds):
        return bounds
    if len(bounds) != 2:
        raise ValueError("Parameter bounds must contain exactly two values.")
    return ParameterBounds(lower=float(bounds[0]), upper=float(bounds[1]))


def _normalize_parameter_bounds(
    parameter_bounds: Mapping[str, ParameterBounds | Sequence[float]],
) -> dict[str, ParameterBounds]:
    """Normalize a parameter-bounds mapping."""

    normalized: dict[str, ParameterBounds] = {}
    for name, bounds in parameter_bounds.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Parameter names must be non-empty strings.")
        normalized[name] = _coerce_parameter_bounds(bounds)
    if not normalized:
        raise ValueError("parameter_bounds must be provided and non-empty.")
    return normalized


def _normalize_constraints(equality_constraints: Mapping[str, str]) -> dict[str, str]:
    """Normalize equality-constraint mappings."""

    normalized: dict[str, str] = {}
    for target, source in equality_constraints.items():
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Constraint target names must be non-empty strings.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Constraint source names must be non-empty strings.")
        if target == source:
            raise ValueError("Equality constraints must tie two different parameters.")
        if target in normalized:
            raise ValueError(f"Duplicate equality constraint target: {target!r}.")
        normalized[target] = source
    return normalized


def _validate_constraint_graph(
    *,
    parameter_bounds: Mapping[str, ParameterBounds],
    equality_constraints: Mapping[str, str],
) -> None:
    """Validate equality constraints against available parameter names."""

    missing_targets = sorted(
        name for name in equality_constraints if name not in parameter_bounds
    )
    if missing_targets:
        raise ValueError(
            "Equality-constraint targets must be present in parameter_bounds: "
            f"{missing_targets}"
        )
    missing_sources = sorted(
        name for name in equality_constraints.values() if name not in parameter_bounds
    )
    if missing_sources:
        raise ValueError(
            "Equality-constraint sources must be present in parameter_bounds: "
            f"{missing_sources}"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        """Depth-first walk of the constraint graph."""

        if name in visited:
            return
        if name in visiting:
            raise ValueError("Equality constraints must not contain cycles.")
        visiting.add(name)
        source = equality_constraints.get(name)
        if source is not None:
            visit(source)
        visiting.remove(name)
        visited.add(name)

    for target in equality_constraints:
        visit(target)


def _default_solver_parameter_resolver(
    resolved_parameters: Mapping[str, float],
) -> dict[str, float | None]:
    """Resolve default solver parameters from the fully expanded parameter set."""

    roughness_sigma_nm = resolved_parameters.get("roughness_sigma_nm")
    return {
        "roughness_sigma_nm": (
            None if roughness_sigma_nm is None else float(roughness_sigma_nm)
        ),
    }


@dataclass(frozen=True)
class MeasurementFitConfig:
    """Run configuration for measurement-fit optimization.

    Attributes:
        build_grating: Callable that builds a grating from resolved parameters.
        parameter_bounds: Mapping of parameter names to optimization bounds.
        measurement_path: Path to two-column measurement data.
        output_dir: Directory where optimizer artifacts are written.
        angle_mode: Evaluation geometry mode (``"fixed"`` or ``"cff"``).
        grazing_angle_deg: Fixed grazing angle used when ``angle_mode="fixed"``.
        cff: Constant-focus factor used when ``angle_mode="cff"``.
        diffraction_order: Diffraction order to optimize.
        fourier_orders: Fourier truncation order for RCWA evaluations.
        roughness_sigma_nm: Optional fixed roughness sigma in nanometers.
        validate_physical_results: Whether to validate physical output ranges.
        total_trials: Total Ax trials to run.
        batch_size: Number of candidates proposed and evaluated per batch.
        random_seed: Optional random seed for deterministic Ax behavior.
        equality_constraints: Mapping of tied parameter names to their source
            parameter names.
        objective_name: Objective metric name reported to Ax.
        experiment_name: Experiment label persisted in optimizer artifacts.
        failure_penalty: Finite penalty assigned to failed simulations.
        objective_sem: Reported objective standard error for Ax.
        enable_early_stopping: Whether early stopping logic is enabled.
        early_stopping_patience: Allowed non-improving trials before stopping.
        early_stopping_min_relative_improvement: Relative improvement threshold.
        early_stopping_warmup_trials: Trials to complete before early-stop checks.
        save_best_fit_plot: Whether to write ``best_fit.png``.
        save_loss_plot: Whether to write ``optimization_loss_history.png``.
        backend: Requested compute backend.
        evaluation_energies_ev: Discrete energies used for objective evaluation.
        evaluation_grazing_angles_deg: Optional grazing angles (deg) used for
            explicit energy-angle evaluation pairs.
        max_workers: Optional trial-level multiprocessing worker count used for
            objective evaluation through ``BatchSimulationRunner``.
        solver_parameter_resolver: Optional callable that resolves solver
            parameters from the fully expanded parameter dictionary.
        resume: Whether to continue a previous run from its checkpoint.
            ``total_trials`` is cumulative, so raising it extends the run.
        checkpoint_dir: Checkpoint directory. Defaults to ``output_dir/checkpoint``.
        checkpoint_interval: Number of trials between checkpoint flushes.
    """

    build_grating: BuildGratingFunction
    parameter_bounds: Mapping[str, ParameterBounds | Sequence[float]]
    measurement_path: Path
    output_dir: Path
    angle_mode: str = "fixed"
    grazing_angle_deg: float = 4.0
    cff: float = 2.5
    diffraction_order: int = 1
    fourier_orders: int = 20
    roughness_sigma_nm: float | None = None
    validate_physical_results: bool = True
    total_trials: int = 20
    batch_size: int = 1
    random_seed: int | None = None
    equality_constraints: Mapping[str, str] = field(default_factory=dict)
    objective_name: str = "loss"
    experiment_name: str = "measurement_fit"
    failure_penalty: float = 1.0e6
    objective_sem: float = 1.0e-6
    enable_early_stopping: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_relative_improvement: float = 5.0e-3
    early_stopping_warmup_trials: int = 8
    save_best_fit_plot: bool = True
    save_loss_plot: bool = True
    backend: str = "auto"
    evaluation_energies_ev: list[float] = field(default_factory=list)
    evaluation_grazing_angles_deg: list[float] = field(default_factory=list)
    max_workers: int | str | None = None
    solver_parameter_resolver: ResolveSolverParametersFunction | None = None
    resume: bool = False
    checkpoint_dir: Path | None = None
    checkpoint_interval: int = 1

    def __post_init__(self) -> None:
        """Normalize paths, bounds, and validation settings."""

        object.__setattr__(self, "measurement_path", Path(self.measurement_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.checkpoint_dir is not None:
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(
            self,
            "parameter_bounds",
            _normalize_parameter_bounds(self.parameter_bounds),
        )
        object.__setattr__(
            self,
            "equality_constraints",
            _normalize_constraints(self.equality_constraints),
        )

        if not callable(self.build_grating):
            raise ValueError("build_grating must be callable.")
        if self.angle_mode not in {"fixed", "cff"}:
            raise ValueError("angle_mode must be either 'fixed' or 'cff'.")
        if self.angle_mode == "fixed" and self.grazing_angle_deg <= 0.0:
            raise ValueError("grazing_angle_deg must be > 0 for fixed angle mode.")
        if self.cff <= 0.0:
            raise ValueError("cff must be > 0.")
        if self.diffraction_order <= 0:
            raise ValueError("diffraction_order must be > 0.")
        if self.fourier_orders <= 0:
            raise ValueError("fourier_orders must be > 0.")
        if self.roughness_sigma_nm is not None and self.roughness_sigma_nm < 0.0:
            raise ValueError("roughness_sigma_nm must be >= 0 when provided.")
        if self.total_trials <= 0:
            raise ValueError("total_trials must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0.")
        resolved_max_workers = _resolve_simulation_max_workers(self.max_workers)
        if resolved_max_workers > 1 and self.batch_size > 1:
            raise ValueError(
                "batch_size > 1 cannot be combined with optimizer max_workers > 1. "
                "Use trial-level multiprocessing or candidate batching, but not both."
            )
        if self.failure_penalty <= 0.0:
            raise ValueError("failure_penalty must be > 0.")
        if not np.isfinite(self.objective_sem) or self.objective_sem <= 0.0:
            raise ValueError("objective_sem must be finite and > 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if (
            not np.isfinite(self.early_stopping_min_relative_improvement)
            or self.early_stopping_min_relative_improvement < 0.0
        ):
            raise ValueError(
                "early_stopping_min_relative_improvement must be finite and >= 0."
            )
        if self.early_stopping_warmup_trials < 0:
            raise ValueError("early_stopping_warmup_trials must be >= 0.")
        if self.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be one of: auto, numba, numpy.")
        selection = normalize_evaluation_selection(
            self.evaluation_energies_ev,
            self.evaluation_grazing_angles_deg,
        )
        object.__setattr__(self, "evaluation_energies_ev", selection.evaluation_energies_ev)
        object.__setattr__(
            self,
            "evaluation_grazing_angles_deg",
            selection.evaluation_grazing_angles_deg,
        )
        if self.solver_parameter_resolver is not None and not callable(
            self.solver_parameter_resolver
        ):
            raise ValueError("solver_parameter_resolver must be callable when provided.")
        _validate_constraint_graph(
            parameter_bounds=self.parameter_bounds,
            equality_constraints=self.equality_constraints,
        )
        if len(build_free_parameter_names(self)) == 0:
            raise ValueError(
                "At least one free parameter must remain after applying equality constraints."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "MeasurementFitConfig":
        """Build a measurement-fit optimizer config from a plain mapping.

        Args:
            mapping: Plain mapping with the same keys accepted by the dataclass
                constructor.

        Returns:
            Normalized measurement-fit optimizer configuration.
        """

        config = dict(mapping)
        if "build_grating" not in config:
            raise ValueError("build_grating must be provided in the measurement-fit spec.")
        parameter_bounds = config.pop("parameter_bounds", None)
        if parameter_bounds is None:
            raise ValueError("parameter_bounds must be provided in the measurement-fit spec.")
        equality_constraints = config.pop("equality_constraints", {})
        if equality_constraints is None:
            equality_constraints = {}
        measurement_fit_config = cls(
            build_grating=config.pop("build_grating"),
            parameter_bounds=parameter_bounds,
            equality_constraints=equality_constraints,
            measurement_path=config.pop("measurement_path"),
            output_dir=config.pop("output_dir"),
            angle_mode=config.pop("angle_mode", "fixed"),
            grazing_angle_deg=config.pop("grazing_angle_deg", 4.0),
            cff=config.pop("cff", 2.5),
            diffraction_order=config.pop("diffraction_order", 1),
            fourier_orders=config.pop("fourier_orders", 20),
            roughness_sigma_nm=config.pop("roughness_sigma_nm", None),
            validate_physical_results=config.pop("validate_physical_results", True),
            total_trials=config.pop("total_trials", 20),
            batch_size=config.pop("batch_size", 1),
            random_seed=config.pop("random_seed", None),
            objective_name=config.pop("objective_name", "loss"),
            experiment_name=config.pop("experiment_name", "measurement_fit"),
            failure_penalty=config.pop("failure_penalty", 1.0e6),
            objective_sem=config.pop("objective_sem", 1.0e-6),
            enable_early_stopping=config.pop("enable_early_stopping", False),
            early_stopping_patience=config.pop("early_stopping_patience", 8),
            early_stopping_min_relative_improvement=config.pop(
                "early_stopping_min_relative_improvement",
                5.0e-3,
            ),
            early_stopping_warmup_trials=config.pop("early_stopping_warmup_trials", 8),
            save_best_fit_plot=config.pop("save_best_fit_plot", True),
            save_loss_plot=config.pop("save_loss_plot", True),
            backend=config.pop("backend", "auto"),
            evaluation_energies_ev=list(config.pop("evaluation_energies_ev", [])),
            evaluation_grazing_angles_deg=list(
                config.pop("evaluation_grazing_angles_deg", [])
            ),
            max_workers=config.pop("max_workers", None),
            solver_parameter_resolver=config.pop("solver_parameter_resolver", None),
            resume=bool(config.pop("resume", False)),
            checkpoint_dir=config.pop("checkpoint_dir", None),
            checkpoint_interval=int(config.pop("checkpoint_interval", 1)),
        )
        if config:
            raise ValueError(f"Unexpected measurement-fit spec keys: {sorted(config)}")
        return measurement_fit_config


def build_free_parameter_names(config: MeasurementFitConfig) -> list[str]:
    """Return the names of parameters optimized directly by Ax.

    Args:
        config: Measurement-fit optimizer configuration.

    Returns:
        Parameter names that remain free variables in Ax.
    """

    constrained_targets = set(config.equality_constraints)
    return [
        name
        for name in config.parameter_bounds
        if name not in constrained_targets
    ]


def build_measurement_fit_ax_parameters(config: MeasurementFitConfig) -> list[dict[str, object]]:
    """Build Ax parameter definitions for a measurement-fit optimization config.

    Args:
        config: Measurement-fit optimizer configuration.

    Returns:
        Ax parameter definitions for the free variables.
    """

    return [
        {
            "name": name,
            "type": "range",
            "bounds": bounds.as_list(),
            "value_type": "float",
        }
        for name, bounds in config.parameter_bounds.items()
        if name not in config.equality_constraints
    ]


def resolve_measurement_fit_trial_parameters(
    config: MeasurementFitConfig,
    trial_parameters: Mapping[str, float],
) -> dict[str, float]:
    """Expand a free-parameter Ax arm into the full constrained parameter set.

    Args:
        config: Measurement-fit optimizer configuration.
        trial_parameters: Free parameter values proposed by Ax.

    Returns:
        Full parameter dictionary with equality constraints applied.
    """

    free_parameter_names = build_free_parameter_names(config)
    expected_names = set(free_parameter_names)
    provided_names = set(trial_parameters)
    missing_names = sorted(expected_names - provided_names)
    extra_names = sorted(provided_names - expected_names)
    if missing_names or extra_names:
        message_parts: list[str] = []
        if missing_names:
            message_parts.append(f"missing parameters: {missing_names}")
        if extra_names:
            message_parts.append(f"unexpected parameters: {extra_names}")
        raise ValueError("Invalid trial parameters: " + "; ".join(message_parts))

    resolved: dict[str, float] = {
        name: float(trial_parameters[name]) for name in free_parameter_names
    }

    def resolve(name: str) -> float:
        """Resolve one constrained parameter recursively."""

        if name in resolved:
            return resolved[name]
        if name not in config.equality_constraints:
            raise ValueError(
                f"Parameter {name!r} is not available in the measurement-fit spec."
            )
        source = config.equality_constraints[name]
        value = float(resolve(source))
        resolved[name] = value
        return value

    for name in config.parameter_bounds:
        resolve(name)

    for name, bounds in config.parameter_bounds.items():
        value = float(resolved[name])
        if value < bounds.lower or value > bounds.upper:
            raise ValueError(
                f"Resolved value for {name!r}={value} is outside bounds "
                f"[{bounds.lower}, {bounds.upper}]."
            )
    return resolved


def _resolve_measurement_fit_solver_parameters(
    config: MeasurementFitConfig,
    resolved_parameters: Mapping[str, float],
) -> dict[str, float | None]:
    """Resolve solver parameters for a measurement-fit optimization trial."""

    if config.solver_parameter_resolver is not None:
        return dict(config.solver_parameter_resolver(resolved_parameters))
    return _default_solver_parameter_resolver(resolved_parameters)


def _create_ax_client_for_measurement_fit_config(config: MeasurementFitConfig):
    """Create and initialize an Ax client for one measurement-fit optimization run."""

    _patch_torch_fork_rng_for_cpu_only()
    print(_describe_optimizer_compute_context())
    AxClient = _import_ax_client_from_optimize()
    client_kwargs: dict[str, object] = {}
    client_signature = inspect.signature(AxClient)
    if config.random_seed is not None and "random_seed" in client_signature.parameters:
        client_kwargs["random_seed"] = config.random_seed
    ax_client = AxClient(**client_kwargs)

    create_signature = inspect.signature(ax_client.create_experiment)
    create_kwargs: dict[str, object] = {
        "parameters": build_measurement_fit_ax_parameters(config),
        "name": config.experiment_name,
    }
    if "objective_name" in create_signature.parameters:
        create_kwargs["objective_name"] = config.objective_name
        if "minimize" in create_signature.parameters:
            create_kwargs["minimize"] = True
    elif "objectives" in create_signature.parameters:
        ObjectiveProperties = _import_objective_properties()
        create_kwargs["objectives"] = {
            config.objective_name: ObjectiveProperties(minimize=True),
        }
    else:
        raise RuntimeError("Unsupported Ax client create_experiment signature.")
    ax_client.create_experiment(**create_kwargs)
    return ax_client


def _write_measurement_fit_result_json(
    *,
    config: MeasurementFitConfig,
    best_parameters: dict[str, float],
    best_grating_parameters: dict[str, object],
    best_solver_parameters: dict[str, float | None],
    best_loss: float,
    stopped_early: bool,
    completed_trials: int,
    early_stop_reason: str | None,
    backend_requested: str,
    backend_effective: str,
    optimizer_requested_max_workers: int | str | None,
    optimizer_resolved_max_workers: int,
    output_path: Path,
) -> None:
    """Write the best measurement-fit optimization result to JSON."""

    payload = {
        "optimization_mode": "measurement_fit",
        "experiment_name": config.experiment_name,
        "objective_name": config.objective_name,
        "measurement_path": str(config.measurement_path),
        "evaluation_mode": (
            "energy_angle_pairs"
            if len(config.evaluation_grazing_angles_deg) > 0
            else "energy_only"
        ),
        "evaluation_case_mode": (
            "energy_angle_pairs"
            if len(config.evaluation_grazing_angles_deg) > 0
            else "energy_only"
        ),
        "evaluation_energies_ev": list(config.evaluation_energies_ev),
        "evaluation_grazing_angles_deg": list(config.evaluation_grazing_angles_deg),
        "parameter_bounds": {
            name: bounds.as_list() for name, bounds in config.parameter_bounds.items()
        },
        "equality_constraints": dict(config.equality_constraints),
        "best_loss": best_loss,
        "best_parameters": best_parameters,
        "best_grating_parameters": json_safe_grating_parameters(best_grating_parameters),
        "best_solver_parameters": best_solver_parameters,
        "stopped_early": bool(stopped_early),
        "completed_trials": int(completed_trials),
        "early_stop_reason": early_stop_reason,
        "backend_requested": backend_requested,
        "backend_effective": backend_effective,
        "optimizer_execution_strategy": "trial_batch_runner",
        "optimizer_requested_max_workers": optimizer_requested_max_workers,
        "optimizer_resolved_max_workers": int(optimizer_resolved_max_workers),
        "diffraction_order": int(config.diffraction_order),
        "fourier_orders": int(config.fourier_orders),
        "angle_mode": config.angle_mode,
    }
    if config.angle_mode == "fixed":
        payload["grazing_angle_deg"] = float(config.grazing_angle_deg)
    else:
        payload["cff"] = float(config.cff)
    _atomic_write_text(output_path, json.dumps(payload, indent=2))


def _persist_measurement_fit_optimizer_artifacts(
    *,
    config: MeasurementFitConfig,
    evaluation_measurement: MeasurementData,
    best_parameters: dict[str, float],
    best_grating_parameters: dict[str, object],
    best_solver_parameters: dict[str, float | None],
    best_loss: float,
    trial_records: list[TrialRecord],
    stopped_early: bool,
    completed_trials: int,
    early_stop_reason: str | None,
    backend_requested: str,
    backend_effective: str,
    optimizer_requested_max_workers: int | str | None,
    optimizer_resolved_max_workers: int,
    best_simulated_efficiency: np.ndarray | None = None,
    write_heavy_artifacts: bool = True,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Persist measurement-fit optimizer artifacts and return their paths.

    Args:
        config: Measurement-fit configuration for the run.
        evaluation_measurement: Measurement sampled onto the evaluation grid.
        best_parameters: Best free parameters found so far.
        best_grating_parameters: Best resolved grating parameters.
        best_solver_parameters: Best resolved solver parameters.
        best_loss: Best objective value found so far.
        trial_records: Completed trial records.
        stopped_early: Whether early stopping ended the run.
        completed_trials: Number of trials successfully evaluated.
        early_stop_reason: Human-readable early-stopping reason, or ``None``.
        backend_requested: Backend requested by the caller.
        backend_effective: Backend actually used.
        optimizer_requested_max_workers: Worker count requested by the caller.
        optimizer_resolved_max_workers: Worker count actually used.
        best_simulated_efficiency: Cached best-fit curve. When ``None`` the
            curve is re-simulated so the plot can still be written.
        write_heavy_artifacts: Whether to rewrite the plots.

    Returns:
        Paths to the JSON summary, trial-history CSV, best-fit plot, and
        loss-history plot. Optional paths are ``None`` when disabled.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = config.output_dir / "best_result.json"
    trial_history_csv_path = config.output_dir / "trial_history.csv"
    best_fit_plot_path = (
        config.output_dir / "best_fit.png" if config.save_best_fit_plot else None
    )
    loss_history_plot_path = (
        config.output_dir / "optimization_loss_history.png"
        if config.save_loss_plot
        else None
    )
    _write_measurement_fit_result_json(
        config=config,
        best_parameters=best_parameters,
        best_grating_parameters=best_grating_parameters,
        best_solver_parameters=best_solver_parameters,
        best_loss=best_loss,
        stopped_early=stopped_early,
        completed_trials=completed_trials,
        early_stop_reason=early_stop_reason,
        backend_requested=backend_requested,
        backend_effective=backend_effective,
        optimizer_requested_max_workers=optimizer_requested_max_workers,
        optimizer_resolved_max_workers=optimizer_resolved_max_workers,
        output_path=result_json_path,
    )
    _write_trial_history_csv(trial_records, trial_history_csv_path)
    if best_fit_plot_path is not None and write_heavy_artifacts and best_parameters:
        simulated_efficiency = best_simulated_efficiency
        if simulated_efficiency is None:
            simulated_efficiency = simulate_efficiency_curve(
                config,
                best_parameters,
                evaluation_measurement,
                backend=backend_effective,
                build_grating_fn=lambda trial_parameters: config.build_grating(
                    resolve_measurement_fit_trial_parameters(config, trial_parameters)
                ),
                resolve_solver_parameters_fn=lambda trial_parameters: _resolve_measurement_fit_solver_parameters(
                    config,
                    resolve_measurement_fit_trial_parameters(config, trial_parameters),
                ),
            )
        _save_best_fit_plot(
            measurement=evaluation_measurement,
            simulated_efficiency=simulated_efficiency,
            output_path=best_fit_plot_path,
        )
    if loss_history_plot_path is not None and write_heavy_artifacts and trial_records:
        _save_loss_history_plot(
            trial_records=trial_records,
            output_path=loss_history_plot_path,
            stopped_early=stopped_early,
        )
    return result_json_path, trial_history_csv_path, best_fit_plot_path, loss_history_plot_path


def optimize_to_measurements(
    config: MeasurementFitConfig | Mapping[str, object],
) -> OptimizationResult:
    """Run Ax optimization for a measurement-fit grating configuration.

    Args:
        config: Spec mapping describing the measurement-fit optimization run.
            Required keys:
            build_grating: Callable that builds the grating from the resolved
                parameter dictionary for one trial.
            parameter_bounds: Mapping of parameter names to lower/upper bounds.
                These names define the free optimization variables.
            measurement_path: Path to the measured two-column dataset.
            output_dir: Directory where optimizer artifacts are written.
            evaluation_energies_ev: Non-empty list of positive energies used
                for objective evaluation.
            Optional keys include ``angle_mode``, ``grazing_angle_deg``, ``cff``,
            ``evaluation_grazing_angles_deg``, ``equality_constraints``,
            ``diffraction_order``, ``fourier_orders``, ``total_trials``,
            ``batch_size``, ``random_seed``, ``backend``,
            ``enable_early_stopping``, ``early_stopping_patience``,
            ``early_stopping_min_relative_improvement``,
            ``early_stopping_warmup_trials``, ``save_best_fit_plot``, and
            ``save_loss_plot``.

    Returns:
        OptimizationResult: Result bundle containing the best parameters, the
        resolved grating parameters, and artifact paths written to disk.
    """

    if not isinstance(config, MeasurementFitConfig):
        config = MeasurementFitConfig.from_mapping(config)

    backend_effective = _resolve_optimizer_backend(config.backend)
    measurement = load_measurement_data(config.measurement_path)
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    state = TrialLoopState()
    state.resolved_max_workers = _resolve_simulation_max_workers(config.max_workers)

    checkpoint = OptimizerCheckpointSession(
        config=config,
        backend_requested=config.backend,
        backend_effective=backend_effective,
    )
    ax_client = checkpoint.restore_or_create_ax_client(
        lambda run_config: _create_ax_client_for_measurement_fit_config(run_config),
        state,
    )
    if checkpoint.resumed and state.best_parameters:
        state.best_grating_parameters = dict(
            resolve_measurement_fit_trial_parameters(config, state.best_parameters)
        )
        state.best_solver_parameters = dict(
            _resolve_measurement_fit_solver_parameters(config, state.best_grating_parameters)
        )

    build_grating_fn = lambda trial_parameters: config.build_grating(
        resolve_measurement_fit_trial_parameters(config, trial_parameters)
    )
    resolve_solver_parameters_fn = lambda trial_parameters: _resolve_measurement_fit_solver_parameters(
        config,
        resolve_measurement_fit_trial_parameters(config, trial_parameters),
    )

    def evaluate_candidates(candidates) -> list[TrialEvaluation]:
        """Evaluate one batch of measurement-fit candidates.

        Args:
            candidates: Candidate ``(trial_index, parameters)`` pairs.

        Returns:
            One evaluation per candidate, ordered by trial index.
        """

        evaluated = _evaluate_candidate_batch(
            list(candidates),
            config=config,
            measurement=measurement,
            backend_effective=backend_effective,
            build_grating_fn=build_grating_fn,
            resolve_solver_parameters_fn=resolve_solver_parameters_fn,
        )
        return [
            TrialEvaluation(
                trial_index=int(trial_index),
                parameters=dict(parameters),
                loss=float(loss),
                resolved_max_workers=int(trial_resolved_max_workers),
                extras=(
                    {}
                    if simulated_efficiency is None
                    else {"simulated_efficiency": simulated_efficiency}
                ),
            )
            for (
                trial_index,
                parameters,
                loss,
                trial_resolved_max_workers,
                simulated_efficiency,
            ) in evaluated
        ]

    def on_trial_completed(*, evaluation: TrialEvaluation, state: TrialLoopState, improved: bool) -> None:
        """Refresh derived best-fit state and rewrite optimizer artifacts.

        Args:
            evaluation: Evaluation for the trial that just completed.
            state: Mutable loop state to update.
            improved: Whether this trial produced a new best loss.
        """

        if improved:
            state.best_grating_parameters = dict(
                resolve_measurement_fit_trial_parameters(config, evaluation.parameters)
            )
            state.best_solver_parameters = dict(
                _resolve_measurement_fit_solver_parameters(
                    config,
                    state.best_grating_parameters,
                )
            )
        _persist_measurement_fit_optimizer_artifacts(
            config=config,
            evaluation_measurement=evaluation_measurement,
            best_parameters=state.best_parameters,
            best_grating_parameters=state.best_grating_parameters,
            best_solver_parameters=state.best_solver_parameters,
            best_loss=state.best_loss,
            trial_records=state.trial_records,
            stopped_early=state.stopped_early,
            completed_trials=state.completed_trials,
            early_stop_reason=state.early_stop_reason,
            backend_requested=config.backend,
            backend_effective=backend_effective,
            optimizer_requested_max_workers=config.max_workers,
            optimizer_resolved_max_workers=state.resolved_max_workers,
            best_simulated_efficiency=state.best_extras.get("simulated_efficiency"),
            write_heavy_artifacts=improved,
        )
        checkpoint.record_trial(state=state, ax_client=ax_client)

    with checkpoint:
        run_ax_trial_loop(
            ax_client=ax_client,
            config=config,
            state=state,
            evaluate_candidates=evaluate_candidates,
            on_trial_completed=on_trial_completed,
        )
        checkpoint.persist(state=state, ax_client=ax_client)

    if not state.trial_records:
        raise RuntimeError("Optimization produced no completed trials.")

    if not state.best_parameters:
        state.best_parameters = dict(state.trial_records[-1].parameters)
        state.best_grating_parameters = resolve_measurement_fit_trial_parameters(
            config,
            state.best_parameters,
        )
        state.best_solver_parameters = _resolve_measurement_fit_solver_parameters(
            config,
            state.best_grating_parameters,
        )
        state.best_loss = float(state.trial_records[-1].loss)

    result_paths = _persist_measurement_fit_optimizer_artifacts(
        config=config,
        evaluation_measurement=evaluation_measurement,
        best_parameters=state.best_parameters,
        best_grating_parameters=state.best_grating_parameters,
        best_solver_parameters=state.best_solver_parameters,
        best_loss=state.best_loss,
        trial_records=state.trial_records,
        stopped_early=state.stopped_early,
        completed_trials=state.completed_trials,
        early_stop_reason=state.early_stop_reason,
        backend_requested=config.backend,
        backend_effective=backend_effective,
        optimizer_requested_max_workers=config.max_workers,
        optimizer_resolved_max_workers=state.resolved_max_workers,
        best_simulated_efficiency=state.best_extras.get("simulated_efficiency"),
        write_heavy_artifacts=True,
    )

    return OptimizationResult(
        best_parameters=state.best_parameters,
        best_grating_parameters=state.best_grating_parameters,
        best_loss=state.best_loss,
        measurement_path=config.measurement_path,
        result_json_path=result_paths[0],
        trial_history_csv_path=result_paths[1],
        best_fit_plot_path=result_paths[2],
        loss_history_plot_path=result_paths[3],
        trial_records=state.trial_records,
        stopped_early=state.stopped_early,
        completed_trials=state.completed_trials,
        early_stop_reason=state.early_stop_reason,
    )


DynamicOptimizationConfig = MeasurementFitConfig
build_dynamic_ax_parameters = build_measurement_fit_ax_parameters
resolve_dynamic_trial_parameters = resolve_measurement_fit_trial_parameters
_resolve_dynamic_solver_parameters = _resolve_measurement_fit_solver_parameters
_create_ax_client_for_dynamic_config = _create_ax_client_for_measurement_fit_config
_write_dynamic_result_json = _write_measurement_fit_result_json
_persist_dynamic_optimizer_artifacts = _persist_measurement_fit_optimizer_artifacts
optimize_dynamic = optimize_to_measurements
