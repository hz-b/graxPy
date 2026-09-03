"""Joint multi-angle measurement-fit optimization.

Fits one parameter set simultaneously against several measured curves recorded
at different grazing angles. Each measurement keeps its own energy grid, and the
per-measurement losses are combined into a single joint objective.
"""

from __future__ import annotations

import csv
import inspect
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from grax import normalize_polarization
from grax.simulation import _resolve_max_workers as _resolve_simulation_max_workers

from .checkpoint import OptimizerCheckpointSession
from .config import ParameterBounds
from .data import load_measurement_data, sample_measurement_data
from .dynamic import (
    _default_solver_parameter_resolver,
    _normalize_constraints,
    _normalize_parameter_bounds,
    _validate_constraint_graph,
    build_free_parameter_names,
    build_measurement_fit_ax_parameters,
    resolve_measurement_fit_trial_parameters,
)
from .loop import TrialEvaluation, TrialLoopState, run_ax_trial_loop
from .objective import evaluate_joint_trial_with_metadata
from .optimize import (
    TrialRecord,
    _atomic_write_text,
    _describe_optimizer_compute_context,
    _import_ax_client,
    _import_objective_properties,
    _patch_torch_fork_rng_for_cpu_only,
    _resolve_optimizer_backend,
    _save_loss_history_plot,
    _write_trial_history_csv,
    json_safe_grating_parameters,
)

JOINT_LOSS_REDUCTIONS = frozenset({"mean", "sum", "pooled", "weighted"})


@dataclass(frozen=True)
class MeasurementSpec:
    """One measured curve and the conditions it was recorded under.

    Every condition defaults to ``None``, meaning "inherit the run-level value
    from :class:`JointMeasurementFitConfig`". Set one to fit a curve taken under
    conditions that differ from the rest: a different grazing angle, a different
    angle mode, a different diffraction order, or a different polarization.

    Purely numerical settings -- ``fourier_orders``, ``solver``,
    ``solver_options``, ``backend``, ``max_workers`` -- stay run-level, because
    they describe how the simulation is computed rather than what was measured.

    Attributes:
        measurement_path: Path to the measured two-column dataset.
        grazing_angle_deg: Fixed grazing angle, used when the resolved angle
            mode is ``"fixed"``.
        angle_mode: ``"fixed"`` or ``"cff"`` for this measurement.
        cff: Fixed-focus constant, used when the resolved angle mode is
            ``"cff"``.
        diffraction_order: Diffraction order this curve was recorded in.
        polarization: Polarization this curve was recorded in, as ``s``/``p``
            or the equivalent ``TE``/``TM``.
        evaluation_energies_ev: Energies used for evaluation. When empty, the
            measurement's own energy grid is used.
        measurement_efficiency: Optional measured efficiencies to use directly
            instead of interpolating the file. Supply this when the values were
            prepared upstream, for example by smoothing or downsampling.
        weight: Relative weight used by the ``"weighted"`` joint reduction.
        label: Identifier used in artifacts. Defaults to a description of the
            measurement's own conditions.
    """

    measurement_path: Path
    grazing_angle_deg: float | None = None
    angle_mode: str | None = None
    cff: float | None = None
    diffraction_order: int | None = None
    polarization: str | None = None
    evaluation_energies_ev: list[float] = field(default_factory=list)
    measurement_efficiency: list[float] | None = None
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        """Normalize paths and validate the measurement definition."""

        object.__setattr__(self, "measurement_path", Path(self.measurement_path))
        object.__setattr__(
            self,
            "evaluation_energies_ev",
            [float(energy_ev) for energy_ev in self.evaluation_energies_ev],
        )
        if self.measurement_efficiency is not None:
            object.__setattr__(
                self,
                "measurement_efficiency",
                [float(value) for value in self.measurement_efficiency],
            )
        if self.angle_mode is not None:
            if self.angle_mode not in {"fixed", "cff"}:
                raise ValueError("angle_mode must be one of: fixed, cff.")
        if self.polarization is not None:
            object.__setattr__(self, "polarization", normalize_polarization(self.polarization))
        if self.diffraction_order is not None:
            object.__setattr__(self, "diffraction_order", int(self.diffraction_order))
        if self.label is None:
            object.__setattr__(self, "label", self._default_label())

        if self.grazing_angle_deg is not None and float(self.grazing_angle_deg) <= 0.0:
            raise ValueError("grazing_angle_deg must be > 0.")
        if self.cff is not None and float(self.cff) <= 0.0:
            raise ValueError("cff must be > 0.")
        if any(energy_ev <= 0.0 for energy_ev in self.evaluation_energies_ev):
            raise ValueError("evaluation_energies_ev values must be > 0.")
        if not float(self.weight) > 0.0:
            raise ValueError("weight must be > 0.")
        if self.measurement_efficiency is not None:
            if len(self.measurement_efficiency) == 0:
                raise ValueError("measurement_efficiency must not be empty when provided.")
            if (
                len(self.evaluation_energies_ev) > 0
                and len(self.measurement_efficiency) != len(self.evaluation_energies_ev)
            ):
                raise ValueError(
                    "measurement_efficiency must have the same length as evaluation_energies_ev."
                )

    def _default_label(self) -> str:
        """Return a label describing whichever conditions this spec sets.

        Returns:
            A label built from the conditions given, falling back to the
            measurement file stem when the spec inherits everything.
        """

        parts: list[str] = []
        if self.grazing_angle_deg is not None:
            parts.append(f"alpha{float(self.grazing_angle_deg):g}deg")
        if self.cff is not None:
            parts.append(f"cff{float(self.cff):g}".replace(".", "p"))
        if self.diffraction_order is not None:
            parts.append(f"order{int(self.diffraction_order)}")
        if self.polarization is not None:
            parts.append(str(self.polarization))
        if not parts:
            return Path(self.measurement_path).stem
        return "_".join(parts)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> MeasurementSpec:
        """Build a measurement spec from a plain mapping.

        Args:
            mapping: Spec mapping describing one measurement.

        Returns:
            The normalized measurement spec.

        Raises:
            ValueError: If required keys are missing or unexpected keys remain.
        """

        spec = dict(mapping)
        if "measurement_path" not in spec:
            raise ValueError("Each joint measurement requires 'measurement_path'.")
        evaluation_energies_ev = spec.pop("evaluation_energies_ev", None)
        measurement_efficiency = spec.pop("measurement_efficiency", None)
        grazing_angle_deg = spec.pop("grazing_angle_deg", None)
        cff = spec.pop("cff", None)
        diffraction_order = spec.pop("diffraction_order", None)
        instance = cls(
            measurement_path=Path(spec.pop("measurement_path")),  # type: ignore[arg-type]
            grazing_angle_deg=(None if grazing_angle_deg is None else float(grazing_angle_deg)),
            angle_mode=spec.pop("angle_mode", None),  # type: ignore[arg-type]
            cff=(None if cff is None else float(cff)),
            diffraction_order=(None if diffraction_order is None else int(diffraction_order)),
            polarization=spec.pop("polarization", None),  # type: ignore[arg-type]
            evaluation_energies_ev=(
                [] if evaluation_energies_ev is None else list(evaluation_energies_ev)
            ),
            measurement_efficiency=(
                None if measurement_efficiency is None else list(measurement_efficiency)
            ),
            weight=float(spec.pop("weight", 1.0)),
            label=spec.pop("label", None),  # type: ignore[arg-type]
        )
        if spec:
            raise ValueError(f"Unexpected joint measurement keys: {sorted(spec)}")
        return instance


@dataclass(frozen=True)
class JointMeasurement:
    """A measurement spec resolved onto its evaluation grid and conditions.

    Every condition is resolved here: the per-measurement override if the spec
    set one, the run-level value otherwise. Downstream code reads these fields
    and never consults the run-level defaults again.

    Attributes:
        label: Identifier used in artifacts.
        measurement_path: Path the measurement was loaded from.
        angle_mode: Resolved angle mode, ``"fixed"`` or ``"cff"``.
        grazing_angle_deg: Resolved grazing angle, for ``angle_mode="fixed"``.
        cff: Resolved fixed-focus constant, for ``angle_mode="cff"``.
        diffraction_order: Resolved diffraction order.
        polarization: Resolved polarization, canonicalized to ``s`` or ``p``.
        evaluation_energies_ev: Energies used for evaluation.
        evaluation_efficiency: Measured efficiencies on the evaluation grid.
        weight: Relative weight for the ``"weighted"`` joint reduction.
    """

    label: str
    measurement_path: Path
    angle_mode: str
    grazing_angle_deg: float | None
    cff: float | None
    diffraction_order: int
    polarization: str
    evaluation_energies_ev: np.ndarray
    evaluation_efficiency: np.ndarray
    weight: float


def prepare_joint_measurements(
    measurements: Sequence[MeasurementSpec],
    *,
    angle_mode: str = "fixed",
    grazing_angle_deg: float | None = None,
    cff: float | None = None,
    diffraction_order: int = 1,
    polarization: str = "s",
) -> list[JointMeasurement]:
    """Resolve every measurement spec onto its evaluation grid and conditions.

    Each condition is taken from the spec when it sets one and from the
    run-level default otherwise, so the resolved measurements are self-contained.

    Args:
        measurements: Normalized measurement specs.
        angle_mode: Run-level angle mode inherited by specs that omit one.
        grazing_angle_deg: Run-level grazing angle inherited by specs that omit one.
        cff: Run-level fixed-focus constant inherited by specs that omit one.
        diffraction_order: Run-level diffraction order inherited by specs that omit one.
        polarization: Run-level polarization inherited by specs that omit one.

    Returns:
        One resolved measurement per spec, in the same order.

    Raises:
        ValueError: If supplied efficiencies do not match the resolved grid, or
            if a measurement resolves to an angle mode without the value that
            mode needs.
    """

    prepared: list[JointMeasurement] = []
    for spec in measurements:
        resolved_angle_mode = spec.angle_mode if spec.angle_mode is not None else str(angle_mode)
        resolved_angle = (
            spec.grazing_angle_deg if spec.grazing_angle_deg is not None else grazing_angle_deg
        )
        resolved_cff = spec.cff if spec.cff is not None else cff
        resolved_order = (
            spec.diffraction_order
            if spec.diffraction_order is not None
            else int(diffraction_order)
        )
        resolved_polarization = (
            spec.polarization if spec.polarization is not None else normalize_polarization(polarization)
        )
        if resolved_angle_mode == "fixed" and resolved_angle is None:
            raise ValueError(
                f"Measurement {spec.label!r} resolves to angle_mode='fixed' but no "
                "grazing_angle_deg was given on the measurement or the run."
            )
        if resolved_angle_mode == "cff" and resolved_cff is None:
            raise ValueError(
                f"Measurement {spec.label!r} resolves to angle_mode='cff' but no cff was "
                "given on the measurement or the run."
            )
        measurement = load_measurement_data(spec.measurement_path)
        if len(spec.evaluation_energies_ev) > 0:
            energies = np.asarray(spec.evaluation_energies_ev, dtype=float)
            if spec.measurement_efficiency is None:
                sampled = sample_measurement_data(measurement, energies)
                efficiencies = np.asarray(sampled.efficiency, dtype=float)
            else:
                efficiencies = np.asarray(spec.measurement_efficiency, dtype=float)
        else:
            energies = np.asarray(measurement.energy_ev, dtype=float)
            if spec.measurement_efficiency is None:
                efficiencies = np.asarray(measurement.efficiency, dtype=float)
            else:
                efficiencies = np.asarray(spec.measurement_efficiency, dtype=float)
                if efficiencies.shape != energies.shape:
                    raise ValueError(
                        "measurement_efficiency must have the same length as the measurement "
                        f"energy grid for {spec.label!r}."
                    )
        # Null out the value the resolved angle mode does not use, so artifacts and
        # the resume fingerprint never imply a fixed angle was applied to a cff
        # measurement, or the reverse.
        if resolved_angle_mode == "cff":
            resolved_angle = None
        else:
            resolved_cff = None
        prepared.append(
            JointMeasurement(
                label=str(spec.label),
                measurement_path=spec.measurement_path,
                angle_mode=resolved_angle_mode,
                grazing_angle_deg=(None if resolved_angle is None else float(resolved_angle)),
                cff=(None if resolved_cff is None else float(resolved_cff)),
                diffraction_order=int(resolved_order),
                polarization=str(resolved_polarization),
                evaluation_energies_ev=energies,
                evaluation_efficiency=efficiencies,
                weight=float(spec.weight),
            )
        )
    return prepared


@dataclass(frozen=True)
class JointMeasurementFitConfig:
    """Configuration for a joint measurement fit.

    Fits one parameter set against several measured curves at once. The curves
    need not differ only by grazing angle: ``angle_mode``, ``grazing_angle_deg``,
    ``cff``, ``diffraction_order`` and ``polarization`` are run-level defaults
    that any individual :class:`MeasurementSpec` may override, so a single fit
    can span angles, angle modes, diffraction orders and polarizations.

    Attributes:
        build_grating: Callable building the grating from resolved parameters.
        parameter_bounds: Mapping of parameter names to lower/upper bounds.
        output_dir: Directory where optimizer artifacts are written.
        measurements: Measurement specs to fit jointly.
        angle_mode: Default angle mode, ``"fixed"`` or ``"cff"``.
        grazing_angle_deg: Default grazing angle for ``angle_mode="fixed"``.
        cff: Default fixed-focus constant for ``angle_mode="cff"``.
        polarization: Default polarization, as ``s``/``p`` or ``TE``/``TM``.
        solver: Electromagnetic solver used for every trial evaluation,
            ``"rcwa"`` (default) or ``"neviere"``. Unlike ``backend`` there is no
            ``"auto"`` mode: the two solvers are different methods, not
            interchangeable implementations of one.
        solver_options: Integration settings for ``solver="neviere"``, as a
            mapping matching :class:`grax.NeviereOptions`.
        diffraction_order: Default diffraction order selected for evaluation.
        fourier_orders: Fourier orders used by the solver.
        roughness_sigma_nm: Optional roughness passed to the solver.
        validate_physical_results: Whether to validate simulated results.
        total_trials: Cumulative trial budget across resumed runs.
        batch_size: Number of candidates generated per Ax batch.
        random_seed: Optional Ax random seed.
        equality_constraints: Mapping of tied parameter names to their source.
        objective_name: Ax objective name.
        experiment_name: Ax experiment name.
        failure_penalty: Loss reported for failed trials.
        objective_sem: Standard error reported with each observation.
        enable_early_stopping: Whether early stopping is active.
        early_stopping_patience: Non-improving trials tolerated before stopping.
        early_stopping_min_relative_improvement: Minimum relative gain counted
            as an improvement.
        early_stopping_warmup_trials: Trials completed before stopping applies.
        joint_loss_reduction: How per-measurement losses are combined.
        save_best_fit_plot: Whether to write the multi-panel best-fit plot.
        save_loss_plot: Whether to write the loss-history plot.
        save_comparison_csv: Whether to write the long-form comparison CSV.
        backend: Requested Fourier coefficient backend.
        max_workers: Trial-level worker count for the batch runner.
        solver_parameter_resolver: Optional solver-parameter hook.
        resume: Whether to resume from a previous checkpoint.
        checkpoint_dir: Checkpoint directory. Defaults to ``output_dir/checkpoint``.
        checkpoint_interval: Trials between checkpoint flushes.
    """

    build_grating: Any
    parameter_bounds: Mapping[str, ParameterBounds | Sequence[float]]
    output_dir: Path
    measurements: Sequence[MeasurementSpec | Mapping[str, object]] = field(default_factory=list)
    angle_mode: str = "fixed"
    grazing_angle_deg: float | None = None
    cff: float | None = None
    polarization: str = "s"
    solver: str = "rcwa"
    solver_options: dict[str, object] | None = None
    diffraction_order: int = 1
    fourier_orders: int = 20
    roughness_sigma_nm: float | None = None
    validate_physical_results: bool = True
    total_trials: int = 20
    batch_size: int = 1
    random_seed: int | None = None
    equality_constraints: Mapping[str, str] = field(default_factory=dict)
    objective_name: str = "joint_loss"
    experiment_name: str = "joint_measurement_fit"
    failure_penalty: float = 1.0e6
    objective_sem: float = 1.0e-6
    enable_early_stopping: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_relative_improvement: float = 5.0e-3
    early_stopping_warmup_trials: int = 8
    joint_loss_reduction: str = "mean"
    save_best_fit_plot: bool = True
    save_loss_plot: bool = True
    save_comparison_csv: bool = True
    backend: str = "auto"
    max_workers: int | str | None = None
    solver_parameter_resolver: Any = None
    resume: bool = False
    checkpoint_dir: Path | None = None
    checkpoint_interval: int = 1

    def __post_init__(self) -> None:
        """Normalize paths, bounds, measurements, and validate settings."""

        object.__setattr__(self, "output_dir", Path(self.output_dir))
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
        object.__setattr__(
            self,
            "measurements",
            [
                item if isinstance(item, MeasurementSpec) else MeasurementSpec.from_mapping(item)
                for item in self.measurements
            ],
        )
        object.__setattr__(self, "polarization", normalize_polarization(self.polarization))
        if self.checkpoint_dir is not None:
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

        if not callable(self.build_grating):
            raise ValueError("build_grating must be callable.")
        if len(self.measurements) == 0:
            raise ValueError("measurements must be provided and non-empty.")
        labels = [str(spec.label) for spec in self.measurements]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "measurements must have unique labels. Labels default to the conditions a "
                "measurement sets, so give an explicit 'label' when two measurements share them."
            )
        if self.angle_mode not in {"fixed", "cff"}:
            raise ValueError("angle_mode must be one of: fixed, cff.")
        if self.solver not in {"rcwa", "neviere"}:
            raise ValueError("solver must be one of: rcwa, neviere.")
        if self.grazing_angle_deg is not None and self.grazing_angle_deg <= 0.0:
            raise ValueError("grazing_angle_deg must be > 0.")
        if self.cff is not None and self.cff <= 0.0:
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
        if self.joint_loss_reduction not in JOINT_LOSS_REDUCTIONS:
            raise ValueError(
                "joint_loss_reduction must be one of 'mean', 'sum', 'pooled', or 'weighted'."
            )
        if self.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be one of 'auto', 'numba', or 'numpy'.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0.")
        if self.solver_parameter_resolver is not None and not callable(
            self.solver_parameter_resolver
        ):
            raise ValueError("solver_parameter_resolver must be callable when provided.")

        _validate_constraint_graph(
            parameter_bounds=self.parameter_bounds,
            equality_constraints=self.equality_constraints,
        )
        if not build_free_parameter_names(self):
            raise ValueError("At least one parameter must remain free for optimization.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> JointMeasurementFitConfig:
        """Build a joint configuration from a plain spec mapping.

        Args:
            mapping: Spec mapping describing the joint optimization run.

        Returns:
            The normalized joint configuration.

        Raises:
            ValueError: If required keys are missing or unexpected keys remain.
        """

        config = dict(mapping)
        if "build_grating" not in config:
            raise ValueError("Joint measurement-fit spec requires 'build_grating'.")
        parameter_bounds = config.pop("parameter_bounds", None)
        if parameter_bounds is None:
            raise ValueError("Joint measurement-fit spec requires 'parameter_bounds'.")
        measurements = config.pop("measurements", None)
        if measurements is None:
            raise ValueError("Joint measurement-fit spec requires 'measurements'.")
        equality_constraints = config.pop("equality_constraints", None) or {}

        instance = cls(
            build_grating=config.pop("build_grating"),
            parameter_bounds=parameter_bounds,  # type: ignore[arg-type]
            output_dir=config.pop("output_dir"),  # type: ignore[arg-type]
            measurements=list(measurements),  # type: ignore[arg-type]
            angle_mode=str(config.pop("angle_mode", "fixed")),
            grazing_angle_deg=config.pop("grazing_angle_deg", None),  # type: ignore[arg-type]
            cff=config.pop("cff", None),  # type: ignore[arg-type]
            polarization=str(config.pop("polarization", "s")),
            solver=str(config.pop("solver", "rcwa")),
            solver_options=config.pop("solver_options", None),  # type: ignore[arg-type]
            diffraction_order=int(config.pop("diffraction_order", 1)),
            fourier_orders=int(config.pop("fourier_orders", 20)),
            roughness_sigma_nm=config.pop("roughness_sigma_nm", None),  # type: ignore[arg-type]
            validate_physical_results=bool(config.pop("validate_physical_results", True)),
            total_trials=int(config.pop("total_trials", 20)),
            batch_size=int(config.pop("batch_size", 1)),
            random_seed=config.pop("random_seed", None),  # type: ignore[arg-type]
            equality_constraints=equality_constraints,  # type: ignore[arg-type]
            objective_name=str(config.pop("objective_name", "joint_loss")),
            experiment_name=str(config.pop("experiment_name", "joint_measurement_fit")),
            failure_penalty=float(config.pop("failure_penalty", 1.0e6)),
            objective_sem=float(config.pop("objective_sem", 1.0e-6)),
            enable_early_stopping=bool(config.pop("enable_early_stopping", False)),
            early_stopping_patience=int(config.pop("early_stopping_patience", 8)),
            early_stopping_min_relative_improvement=float(
                config.pop("early_stopping_min_relative_improvement", 5.0e-3)
            ),
            early_stopping_warmup_trials=int(config.pop("early_stopping_warmup_trials", 8)),
            joint_loss_reduction=str(config.pop("joint_loss_reduction", "mean")),
            save_best_fit_plot=bool(config.pop("save_best_fit_plot", True)),
            save_loss_plot=bool(config.pop("save_loss_plot", True)),
            save_comparison_csv=bool(config.pop("save_comparison_csv", True)),
            backend=str(config.pop("backend", "auto")),
            max_workers=config.pop("max_workers", None),  # type: ignore[arg-type]
            solver_parameter_resolver=config.pop("solver_parameter_resolver", None),
            resume=bool(config.pop("resume", False)),
            checkpoint_dir=config.pop("checkpoint_dir", None),  # type: ignore[arg-type]
            checkpoint_interval=int(config.pop("checkpoint_interval", 1)),
        )
        if config:
            raise ValueError(f"Unexpected joint measurement-fit spec keys: {sorted(config)}")
        return instance


@dataclass(frozen=True)
class JointOptimizationResult:
    """Result bundle returned by the joint multi-angle optimizer.

    Attributes:
        best_parameters: Best free parameters returned by Ax.
        best_grating_parameters: Best resolved grating parameters.
        best_loss: Best joint objective value found.
        per_measurement_best_losses: Per-measurement losses for the best trial.
        measurements: Resolved measurements used for the fit.
        result_json_path: Path to the persisted JSON summary.
        trial_history_csv_path: Path to the per-trial history CSV.
        best_fit_plot_path: Path to the multi-panel best-fit plot, if written.
        loss_history_plot_path: Path to the loss-history plot, if written.
        comparison_csv_path: Path to the long-form comparison CSV, if written.
        trial_records: Per-trial records including per-measurement losses.
        stopped_early: Whether early stopping ended the run.
        completed_trials: Number of trials successfully evaluated.
        early_stop_reason: Human-readable early-stopping reason, or ``None``.
    """

    best_parameters: dict[str, float]
    best_grating_parameters: dict[str, object]
    best_loss: float
    per_measurement_best_losses: dict[str, float]
    measurements: list[JointMeasurement]
    result_json_path: Path
    trial_history_csv_path: Path
    best_fit_plot_path: Path | None
    loss_history_plot_path: Path | None
    comparison_csv_path: Path | None
    trial_records: list[TrialRecord]
    stopped_early: bool
    completed_trials: int
    early_stop_reason: str | None


def _resolve_joint_solver_parameters(
    config: JointMeasurementFitConfig,
    resolved_parameters: Mapping[str, float],
) -> dict[str, float | None]:
    """Resolve solver parameters for one joint trial.

    Args:
        config: Joint optimization configuration.
        resolved_parameters: Fully expanded grating parameters.

    Returns:
        Solver parameters for the trial.
    """

    if config.solver_parameter_resolver is not None:
        return dict(config.solver_parameter_resolver(resolved_parameters))
    return _default_solver_parameter_resolver(resolved_parameters)


def _create_ax_client_for_joint_config(config: JointMeasurementFitConfig) -> Any:
    """Create and configure an Ax client for a joint optimization run.

    Args:
        config: Joint optimization configuration.

    Returns:
        A configured Ax client with the experiment created.

    Raises:
        RuntimeError: If the installed Ax client API is unsupported.
    """

    _patch_torch_fork_rng_for_cpu_only()
    print(_describe_optimizer_compute_context())
    ax_client_cls = _import_ax_client()

    client_kwargs: dict[str, object] = {}
    client_signature = inspect.signature(ax_client_cls)
    if config.random_seed is not None and "random_seed" in client_signature.parameters:
        client_kwargs["random_seed"] = config.random_seed
    ax_client = ax_client_cls(**client_kwargs)

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
        objective_properties = _import_objective_properties()
        create_kwargs["objectives"] = {config.objective_name: objective_properties(minimize=True)}
    else:
        raise RuntimeError("Unsupported Ax client create_experiment signature.")
    ax_client.create_experiment(**create_kwargs)
    return ax_client


def describe_measurement_conditions(measurement: JointMeasurement) -> str:
    """Return a one-line description of the conditions a measurement resolved to.

    Args:
        measurement: Resolved measurement to describe.

    Returns:
        A compact description covering angle, diffraction order and polarization,
        for plot titles and console output.
    """

    if measurement.angle_mode == "cff":
        geometry = f"cff = {float(measurement.cff):g}"
    else:
        geometry = f"alpha = {float(measurement.grazing_angle_deg):g} deg"
    return (
        f"{measurement.label}: {geometry}, order {int(measurement.diffraction_order)}, "
        f"{measurement.polarization}-pol"
    )


def _save_joint_best_fit_plot(
    *,
    measurements: Sequence[JointMeasurement],
    simulated_by_label: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save one measurement-vs-simulation panel per angle.

    Args:
        measurements: Resolved measurements used for the fit.
        simulated_by_label: Best-fit simulated curves keyed by label.
        output_path: Destination image path.
    """

    figure, axes = plt.subplots(
        len(measurements),
        1,
        figsize=(10, 4.0 * len(measurements)),
        squeeze=False,
    )
    for axis, measurement in zip(axes[:, 0], measurements, strict=True):
        axis.plot(
            measurement.evaluation_energies_ev,
            measurement.evaluation_efficiency,
            "o-",
            linewidth=1.0,
            label="Measurement",
        )
        simulated = simulated_by_label.get(measurement.label)
        if simulated is not None:
            axis.plot(
                measurement.evaluation_energies_ev,
                simulated,
                "s-",
                linewidth=1.0,
                label="Best fit",
            )
        axis.set_xlabel("Photon Energy (eV)")
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title(f"Joint Best Fit -- {describe_measurement_conditions(measurement)}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _write_joint_comparison_csv(
    *,
    measurements: Sequence[JointMeasurement],
    simulated_by_label: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    """Write a long-form measured-versus-simulated comparison table.

    Each row carries the conditions its measurement resolved to, since those can
    differ from one measurement to the next.

    Args:
        measurements: Resolved measurements used for the fit.
        simulated_by_label: Best-fit simulated curves keyed by label.
        output_path: Destination CSV path.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "label",
            "angle_mode",
            "grazing_angle_deg",
            "cff",
            "diffraction_order",
            "polarization",
            "energy_ev",
            "measured_efficiency",
            "simulated_efficiency",
        ]
    )
    for measurement in measurements:
        simulated = simulated_by_label.get(measurement.label)
        for point_index, energy_ev in enumerate(measurement.evaluation_energies_ev):
            writer.writerow(
                [
                    measurement.label,
                    measurement.angle_mode,
                    "" if measurement.grazing_angle_deg is None else measurement.grazing_angle_deg,
                    "" if measurement.cff is None else measurement.cff,
                    int(measurement.diffraction_order),
                    measurement.polarization,
                    float(energy_ev),
                    float(measurement.evaluation_efficiency[point_index]),
                    "" if simulated is None else float(simulated[point_index]),
                ]
            )
    _atomic_write_text(output_path, buffer.getvalue())


def _write_joint_result_json(
    *,
    config: JointMeasurementFitConfig,
    measurements: Sequence[JointMeasurement],
    state: TrialLoopState,
    backend_requested: str,
    backend_effective: str,
    output_path: Path,
) -> None:
    """Write the joint optimizer JSON summary.

    Args:
        config: Joint optimization configuration.
        measurements: Resolved measurements used for the fit.
        state: Loop state carrying the best-so-far results.
        backend_requested: Backend requested by the caller.
        backend_effective: Backend actually used.
        output_path: Destination JSON path.
    """

    payload: dict[str, object] = {
        "optimization_mode": "joint_measurement_fit",
        "experiment_name": config.experiment_name,
        "objective_name": config.objective_name,
        "joint_loss_reduction": config.joint_loss_reduction,
        "measurements": [
            {
                "label": measurement.label,
                "angle_mode": measurement.angle_mode,
                "grazing_angle_deg": measurement.grazing_angle_deg,
                "cff": measurement.cff,
                "diffraction_order": measurement.diffraction_order,
                "polarization": measurement.polarization,
                "measurement_path": str(measurement.measurement_path),
                "evaluation_energies_ev": [
                    float(energy_ev) for energy_ev in measurement.evaluation_energies_ev
                ],
                "point_count": int(len(measurement.evaluation_energies_ev)),
                "weight": measurement.weight,
            }
            for measurement in measurements
        ],
        "parameter_bounds": {
            name: [bounds.lower, bounds.upper] for name, bounds in config.parameter_bounds.items()
        },
        "equality_constraints": dict(config.equality_constraints),
        "best_loss": state.best_loss,
        "per_measurement_best_losses": dict(state.best_extras.get("per_measurement_losses", {})),
        "best_parameters": dict(state.best_parameters),
        "best_grating_parameters": json_safe_grating_parameters(state.best_grating_parameters),
        "best_solver_parameters": dict(state.best_solver_parameters),
        "stopped_early": state.stopped_early,
        "completed_trials": state.completed_trials,
        "early_stop_reason": state.early_stop_reason,
        "backend_requested": backend_requested,
        "backend_effective": backend_effective,
        "solver": str(getattr(config, "solver", "rcwa")),
        "solver_options": getattr(config, "solver_options", None),
        "optimizer_execution_strategy": "trial_batch_runner",
        "optimizer_requested_max_workers": config.max_workers,
        "optimizer_resolved_max_workers": state.resolved_max_workers,
        "angle_mode": config.angle_mode,
        "polarization": config.polarization,
        "diffraction_order": config.diffraction_order,
        "fourier_orders": config.fourier_orders,
    }
    _atomic_write_text(output_path, json.dumps(payload, indent=2))


def _persist_joint_optimizer_artifacts(
    *,
    config: JointMeasurementFitConfig,
    measurements: Sequence[JointMeasurement],
    state: TrialLoopState,
    backend_requested: str,
    backend_effective: str,
    write_heavy_artifacts: bool,
) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    """Persist joint optimizer artifacts and return their paths.

    Args:
        config: Joint optimization configuration.
        measurements: Resolved measurements used for the fit.
        state: Loop state carrying the best-so-far results.
        backend_requested: Backend requested by the caller.
        backend_effective: Backend actually used.
        write_heavy_artifacts: Whether to rewrite plots and the comparison CSV.

    Returns:
        Paths to the JSON summary, trial-history CSV, best-fit plot, loss-history
        plot, and comparison CSV. Optional paths are ``None`` when disabled.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = config.output_dir / "best_result.json"
    trial_history_csv_path = config.output_dir / "trial_history.csv"
    best_fit_plot_path = config.output_dir / "best_fit.png" if config.save_best_fit_plot else None
    loss_history_plot_path = (
        config.output_dir / "optimization_loss_history.png" if config.save_loss_plot else None
    )
    comparison_csv_path = (
        config.output_dir / "best_fit_comparison.csv" if config.save_comparison_csv else None
    )

    _write_joint_result_json(
        config=config,
        measurements=measurements,
        state=state,
        backend_requested=backend_requested,
        backend_effective=backend_effective,
        output_path=result_json_path,
    )
    _write_trial_history_csv(state.trial_records, trial_history_csv_path)

    simulated_by_label = state.best_extras.get("simulated_by_label", {})
    if write_heavy_artifacts:
        if best_fit_plot_path is not None and simulated_by_label:
            _save_joint_best_fit_plot(
                measurements=measurements,
                simulated_by_label=simulated_by_label,
                output_path=best_fit_plot_path,
            )
        if comparison_csv_path is not None and simulated_by_label:
            _write_joint_comparison_csv(
                measurements=measurements,
                simulated_by_label=simulated_by_label,
                output_path=comparison_csv_path,
            )
        if loss_history_plot_path is not None and state.trial_records:
            _save_loss_history_plot(
                trial_records=state.trial_records,
                output_path=loss_history_plot_path,
                stopped_early=state.stopped_early,
            )

    return (
        result_json_path,
        trial_history_csv_path,
        best_fit_plot_path,
        loss_history_plot_path,
        comparison_csv_path,
    )


def optimize_to_joint_measurements(
    config: JointMeasurementFitConfig | Mapping[str, object],
) -> JointOptimizationResult:
    """Fit one parameter set jointly against several measured curves.

    Each measurement is recorded at its own fixed grazing angle and keeps its own
    energy grid. Every trial evaluates all measurements in a single batch, and the
    per-measurement losses are combined using ``joint_loss_reduction``.

    Args:
        config: Joint configuration, or a spec mapping describing the run.
            Required keys: ``build_grating``, ``parameter_bounds``,
            ``output_dir``, and ``measurements``.

    Returns:
        JointOptimizationResult: Result bundle with persisted artifact paths.

    Raises:
        RuntimeError: If the optimization produced no completed trials.
    """

    if not isinstance(config, JointMeasurementFitConfig):
        config = JointMeasurementFitConfig.from_mapping(config)

    backend_effective = _resolve_optimizer_backend(config.backend)
    measurements = prepare_joint_measurements(
        config.measurements,
        angle_mode=config.angle_mode,
        grazing_angle_deg=config.grazing_angle_deg,
        cff=config.cff,
        diffraction_order=config.diffraction_order,
        polarization=config.polarization,
    )

    state = TrialLoopState()
    state.resolved_max_workers = _resolve_simulation_max_workers(config.max_workers)

    checkpoint = OptimizerCheckpointSession(
        config=config,
        backend_requested=config.backend,
        backend_effective=backend_effective,
    )
    ax_client = checkpoint.restore_or_create_ax_client(
        lambda run_config: _create_ax_client_for_joint_config(run_config),
        state,
    )
    if checkpoint.resumed and state.best_parameters:
        state.best_grating_parameters = dict(
            resolve_measurement_fit_trial_parameters(config, state.best_parameters)
        )
        state.best_solver_parameters = dict(
            _resolve_joint_solver_parameters(config, state.best_grating_parameters)
        )

    build_grating_fn = lambda trial_parameters: config.build_grating(
        resolve_measurement_fit_trial_parameters(config, trial_parameters)
    )
    resolve_solver_parameters_fn = lambda trial_parameters: _resolve_joint_solver_parameters(
        config,
        resolve_measurement_fit_trial_parameters(config, trial_parameters),
    )

    def evaluate_candidates(candidates) -> list[TrialEvaluation]:
        """Evaluate one batch of joint candidates.

        Args:
            candidates: Candidate ``(trial_index, parameters)`` pairs.

        Returns:
            One evaluation per candidate.
        """

        evaluations: list[TrialEvaluation] = []
        for trial_index, parameters in candidates:
            (
                joint_loss,
                per_measurement_losses,
                simulated_by_label,
                resolved_max_workers,
            ) = evaluate_joint_trial_with_metadata(
                config,
                parameters,
                measurements,
                backend=backend_effective,
                build_grating_fn=build_grating_fn,
                resolve_solver_parameters_fn=resolve_solver_parameters_fn,
            )
            extras: dict[str, Any] = {
                "per_measurement_losses": per_measurement_losses,
                "simulated_by_label": simulated_by_label,
            }
            for label, loss in per_measurement_losses.items():
                extras[f"loss_{label}"] = float(loss)
            evaluations.append(
                TrialEvaluation(
                    trial_index=int(trial_index),
                    parameters=dict(parameters),
                    loss=float(joint_loss),
                    resolved_max_workers=int(resolved_max_workers),
                    extras=extras,
                )
            )
        return evaluations

    def on_trial_completed(
        *, evaluation: TrialEvaluation, state: TrialLoopState, improved: bool
    ) -> None:
        """Refresh derived best state and rewrite joint artifacts.

        Args:
            evaluation: Evaluation for the trial that just completed.
            state: Mutable loop state to update.
            improved: Whether this trial produced a new best joint loss.
        """

        if improved:
            state.best_grating_parameters = dict(
                resolve_measurement_fit_trial_parameters(config, evaluation.parameters)
            )
            state.best_solver_parameters = dict(
                _resolve_joint_solver_parameters(config, state.best_grating_parameters)
            )
        _persist_joint_optimizer_artifacts(
            config=config,
            measurements=measurements,
            state=state,
            backend_requested=config.backend,
            backend_effective=backend_effective,
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
        raise RuntimeError("Joint optimization produced no completed trials.")

    if not state.best_parameters:
        state.best_parameters = dict(state.trial_records[-1].parameters)
        state.best_grating_parameters = dict(
            resolve_measurement_fit_trial_parameters(config, state.best_parameters)
        )
        state.best_solver_parameters = dict(
            _resolve_joint_solver_parameters(config, state.best_grating_parameters)
        )
        state.best_loss = float(state.trial_records[-1].loss)

    result_paths = _persist_joint_optimizer_artifacts(
        config=config,
        measurements=measurements,
        state=state,
        backend_requested=config.backend,
        backend_effective=backend_effective,
        write_heavy_artifacts=True,
    )

    return JointOptimizationResult(
        best_parameters=state.best_parameters,
        best_grating_parameters=state.best_grating_parameters,
        best_loss=state.best_loss,
        per_measurement_best_losses=dict(state.best_extras.get("per_measurement_losses", {})),
        measurements=measurements,
        result_json_path=result_paths[0],
        trial_history_csv_path=result_paths[1],
        best_fit_plot_path=result_paths[2],
        loss_history_plot_path=result_paths[3],
        comparison_csv_path=result_paths[4],
        trial_records=state.trial_records,
        stopped_early=state.stopped_early,
        completed_trials=state.completed_trials,
        early_stop_reason=state.early_stop_reason,
    )
