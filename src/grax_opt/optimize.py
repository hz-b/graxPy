"""Top-level Ax optimization loop for grating fitting."""

from __future__ import annotations

import concurrent.futures
import csv
import inspect
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grax.materials import material_label

from .config import BlazedAxConfig, LaminarAxConfig
from .data import MeasurementData, load_measurement_data
from .model import build_ax_parameters, resolve_grating_parameters, resolve_solver_parameters
from .objective import build_evaluation_measurement, evaluate_trial, simulate_efficiency_curve


def _is_cuda_usable() -> bool:
    """Return whether PyTorch can safely use CUDA on this machine."""

    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except RuntimeError:
        return False


def _patch_torch_fork_rng_for_cpu_only() -> None:
    """Prevent torch.random.fork_rng() from probing CUDA on CPU-only hosts.

    AxClient uses torch.random.fork_rng() internally when generating BoTorch
    trials. On hosts where PyTorch is built with CUDA support but the installed
    NVIDIA driver is too old, the default implementation raises while trying to
    initialize CUDA even though the optimization itself is CPU-only.
    """

    if _is_cuda_usable():
        return

    try:
        import torch
    except ImportError:
        return

    original_fork_rng = torch.random.fork_rng
    if getattr(original_fork_rng, "_grax_cpu_only_patch", False):
        return

    def cpu_only_fork_rng(*args, **kwargs):
        if not args and "devices" not in kwargs:
            kwargs["devices"] = []
        return original_fork_rng(*args, **kwargs)

    cpu_only_fork_rng._grax_cpu_only_patch = True  # type: ignore[attr-defined]
    torch.random.fork_rng = cpu_only_fork_rng


def _detect_cpu_model() -> str:
    """Best-effort CPU model detection."""

    cpu_model = platform.processor().strip()
    if cpu_model:
        return cpu_model
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                model = value.strip()
                if model:
                    return model
    return "unknown"


def _build_optimizer_compute_banner(
    *,
    mode: str,
    model: str,
    torch_version: str,
    torch_cuda_version: str,
) -> str:
    """Format startup compute context for optimizer runs."""

    return (
        f"Optimizer compute: {mode} | model={model} "
        f"| torch={torch_version} | cuda={torch_cuda_version}"
    )


def _describe_optimizer_compute_context() -> str:
    """Resolve optimizer compute context and return printable banner."""

    try:
        import torch
    except ImportError:
        return _build_optimizer_compute_banner(
            mode="CPU",
            model=_detect_cpu_model(),
            torch_version="not-installed",
            torch_cuda_version="unavailable",
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    cuda_version_raw = getattr(torch.version, "cuda", None)
    torch_cuda_version = str(cuda_version_raw) if cuda_version_raw is not None else "unavailable"
    if _is_cuda_usable():
        try:
            gpu_model = str(torch.cuda.get_device_name(0))
        except RuntimeError:
            gpu_model = "unknown"
        return _build_optimizer_compute_banner(
            mode="GPU",
            model=gpu_model,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
        )
    return _build_optimizer_compute_banner(
        mode="CPU",
        model=_detect_cpu_model(),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
    )


def _is_numba_available() -> bool:
    """Return whether numba can be imported."""

    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_optimizer_backend(requested_backend: str) -> str:
    """Resolve requested optimizer backend to an executable backend."""

    normalized_backend = str(requested_backend).lower()
    numba_available = _is_numba_available()
    if normalized_backend == "auto":
        return "numba" if numba_available else "numpy"
    if normalized_backend == "numba":
        if numba_available:
            return "numba"
        print("Requested optimizer backend 'numba' not available; falling back to 'numpy'.")
        return "numpy"
    return "numpy"


def _resolve_batch_worker_count(batch_size: int) -> int:
    """Return effective worker count for one candidate batch."""

    return max(1, min(int(batch_size), int(os.cpu_count() or 1)))


def _evaluate_candidate_worker(
    candidate: tuple[int, dict[str, float]],
    *,
    config: BlazedAxConfig | LaminarAxConfig,
    measurement: MeasurementData,
    backend_effective: str,
) -> tuple[int, dict[str, float], float]:
    """Evaluate one optimizer candidate and return trial index, params, and loss."""

    trial_index, parameters = candidate
    loss = float(evaluate_trial(config, parameters, measurement, backend=backend_effective))
    return int(trial_index), dict(parameters), float(loss)


def _evaluate_candidate_batch(
    candidates: list[tuple[int, dict[str, float]]],
    *,
    config: BlazedAxConfig | LaminarAxConfig,
    measurement: MeasurementData,
    backend_effective: str,
) -> list[tuple[int, dict[str, float], float]]:
    """Evaluate a candidate batch, optionally in parallel."""

    if len(candidates) <= 1:
        return [
            _evaluate_candidate_worker(
                candidates[0],
                config=config,
                measurement=measurement,
                backend_effective=backend_effective,
            )
        ]

    worker_count = _resolve_batch_worker_count(len(candidates))
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _evaluate_candidate_worker,
                candidate,
                config=config,
                measurement=measurement,
                backend_effective=backend_effective,
            )
            for candidate in candidates
        ]
        evaluated = [future.result() for future in futures]
    return sorted(evaluated, key=lambda item: int(item[0]))


@dataclass(frozen=True)
class TrialRecord:
    """Summary of one completed Ax trial."""

    trial_index: int
    loss: float
    parameters: dict[str, float]


@dataclass(frozen=True)
class OptimizationResult:
    """Result bundle returned by optimizer entrypoints."""

    best_parameters: dict[str, float]
    best_grating_parameters: dict[str, object]
    best_loss: float
    measurement_path: Path
    result_json_path: Path
    trial_history_csv_path: Path
    best_fit_plot_path: Path | None
    loss_history_plot_path: Path | None
    trial_records: list[TrialRecord]
    stopped_early: bool
    completed_trials: int
    early_stop_reason: str | None


def _import_ax_optimize():
    """Import the Ax optimize entrypoint lazily."""

    try:
        from ax import optimize as ax_optimize
    except ImportError:
        try:
            from ax.service.managed_loop import optimize as ax_optimize
        except ImportError as exc:
            raise ImportError(
                "Ax is not installed. Install the optional dependency with `pip install .[opt]`."
            ) from exc
    return ax_optimize


def _import_ax_client():
    """Import the Ax client entrypoint lazily."""

    try:
        from ax.service.ax_client import AxClient
    except ImportError as exc:
        raise ImportError(
            "Ax is not installed. Install the optional dependency with `pip install .[opt]`."
        ) from exc
    return AxClient


def _import_objective_properties():
    """Import Ax objective properties for newer Ax client APIs."""

    from ax.service.utils.instantiation import ObjectiveProperties

    return ObjectiveProperties


def _import_max_parallelism_exception():
    """Import Ax max-parallelism exception lazily."""

    try:
        from ax.exceptions.generation_strategy import MaxParallelismReachedException
    except ImportError:
        return None
    return MaxParallelismReachedException


def _import_data_required_exception():
    """Import Ax data-required exception lazily."""

    try:
        from ax.exceptions.core import DataRequiredError
    except ImportError:
        return None
    return DataRequiredError


def _build_ax_optimize_kwargs(
    config: BlazedAxConfig | LaminarAxConfig,
    measurement: MeasurementData,
) -> dict[str, object]:
    """Build keyword arguments for the Ax optimize function."""

    ax_optimize = _import_ax_optimize()
    backend_effective = _resolve_optimizer_backend(config.backend)

    def evaluation_function(parameterization: dict[str, float]) -> dict[str, tuple[float, float]]:
        loss = evaluate_trial(config, parameterization, measurement, backend=backend_effective)
        return {config.objective_name: (loss, config.objective_sem)}

    kwargs: dict[str, object] = {
        "parameters": build_ax_parameters(config),
        "experiment_name": config.experiment_name,
        "objective_name": config.objective_name,
        "evaluation_function": evaluation_function,
        "minimize": True,
        "total_trials": config.total_trials,
    }
    signature = inspect.signature(ax_optimize)
    if config.random_seed is not None and "random_seed" in signature.parameters:
        kwargs["random_seed"] = config.random_seed
    return kwargs


def _extract_trial_records(experiment) -> list[TrialRecord]:
    """Extract trial history from an Ax experiment."""

    trial_records: list[TrialRecord] = []
    for trial in sorted(experiment.trials.values(), key=lambda item: item.index):
        arm = getattr(trial, "arm", None)
        parameters = {}
        if arm is not None:
            parameters = {name: float(value) for name, value in arm.parameters.items()}
        objective_mean = getattr(trial, "objective_mean", None)
        loss = float(objective_mean) if objective_mean is not None else float("nan")
        trial_records.append(
            TrialRecord(
                trial_index=int(trial.index),
                loss=loss,
                parameters=parameters,
            )
        )
    return trial_records


def _write_trial_history_csv(
    trial_records: list[TrialRecord],
    output_path: Path,
) -> None:
    """Write per-trial optimization history to CSV."""

    parameter_names: list[str] = []
    for record in trial_records:
        for name in record.parameters:
            if name not in parameter_names:
                parameter_names.append(name)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_index", "loss", *parameter_names])
        for record in trial_records:
            writer.writerow(
                [
                    record.trial_index,
                    record.loss,
                    *[record.parameters.get(name, "") for name in parameter_names],
                ]
            )


def json_safe_grating_parameters(parameters: dict[str, object]) -> dict[str, object]:
    """Return grating parameters with material objects converted to labels."""
    serializable: dict[str, object] = {}
    for name, value in parameters.items():
        if name.endswith("_material"):
            serializable[name] = None if value is None else material_label(value)
        else:
            serializable[name] = value
    return serializable


def _write_result_json(
    *,
    config: BlazedAxConfig | LaminarAxConfig,
    best_parameters: dict[str, float],
    best_grating_parameters: dict[str, object],
    best_loss: float,
    stopped_early: bool,
    completed_trials: int,
    early_stop_reason: str | None,
    backend_requested: str,
    backend_effective: str,
    output_path: Path,
) -> None:
    """Write the best optimization result to JSON."""

    payload = {
        "experiment_name": config.experiment_name,
        "objective_name": config.objective_name,
        "measurement_path": str(config.measurement_path),
        "evaluation_mode": "discrete",
        "evaluation_energies_ev": list(config.evaluation_energies_ev),
        "best_loss": best_loss,
        "best_parameters": best_parameters,
        "best_grating_parameters": json_safe_grating_parameters(best_grating_parameters),
        "best_solver_parameters": resolve_solver_parameters(config, best_parameters),
        "stopped_early": bool(stopped_early),
        "completed_trials": int(completed_trials),
        "early_stop_reason": early_stop_reason,
        "backend_requested": backend_requested,
        "backend_effective": backend_effective,
    }
    if isinstance(config, LaminarAxConfig):
        payload["angle_mode"] = config.angle_mode
        if config.angle_mode == "fixed":
            payload["grazing_angle_deg"] = config.grazing_angle_deg
        else:
            payload["cff"] = config.cff
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_best_fit_plot(
    *,
    measurement: MeasurementData,
    simulated_efficiency: np.ndarray,
    output_path: Path,
) -> None:
    """Save a measurement-vs-simulation overlay plot for the best fit."""

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        measurement.energy_ev,
        measurement.efficiency,
        "o-",
        linewidth=1.0,
        label="Measurement",
    )
    axis.plot(measurement.energy_ev, simulated_efficiency, "s-", linewidth=1.0, label="Best fit")
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title("Blazed Grating Optimization Best Fit")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_loss_history_plot(
    *,
    trial_records: list[TrialRecord],
    output_path: Path,
    stopped_early: bool,
) -> None:
    """Save trial-loss and running-best history."""

    trial_indices = np.asarray([record.trial_index for record in trial_records], dtype=float)
    losses = np.asarray([record.loss for record in trial_records], dtype=float)
    running_best = np.minimum.accumulate(losses)
    positive_mask = np.isfinite(losses) & (losses > 0.0)
    positive_running_best = np.isfinite(running_best) & (running_best > 0.0)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(trial_indices, losses, "o-", linewidth=1.0, label="Trial loss")
    axis.plot(trial_indices, running_best, "s-", linewidth=1.0, label="Running best")
    if np.any(positive_mask) and np.any(positive_running_best):
        axis.set_yscale("log")
    if stopped_early and trial_indices.size > 0:
        stop_trial_index = float(trial_indices[-1])
        axis.axvline(
            stop_trial_index,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="Early stop",
        )
    axis.set_xlabel("Trial")
    axis.set_ylabel("Loss")
    axis.set_title("Optimization Loss History")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _create_ax_client_for_config(config: BlazedAxConfig | LaminarAxConfig):
    """Create and initialize an Ax client for one optimization run."""

    _patch_torch_fork_rng_for_cpu_only()
    print(_describe_optimizer_compute_context())
    AxClient = _import_ax_client()
    client_kwargs: dict[str, object] = {}
    client_signature = inspect.signature(AxClient)
    if config.random_seed is not None and "random_seed" in client_signature.parameters:
        client_kwargs["random_seed"] = config.random_seed
    ax_client = AxClient(**client_kwargs)

    create_signature = inspect.signature(ax_client.create_experiment)
    create_kwargs: dict[str, object] = {
        "parameters": build_ax_parameters(config),
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


def _complete_ax_trial(
    *,
    ax_client,
    config: BlazedAxConfig | LaminarAxConfig,
    trial_index: int,
    loss: float,
) -> None:
    """Submit one completed trial result back to Ax."""

    raw_data = {config.objective_name: (float(loss), float(config.objective_sem))}
    complete_signature = inspect.signature(ax_client.complete_trial)
    if "raw_data" in complete_signature.parameters:
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        return
    if "data" in complete_signature.parameters:
        ax_client.complete_trial(trial_index=trial_index, data=raw_data)
        return
    raise RuntimeError("Unsupported Ax client complete_trial signature.")


def _is_significant_improvement(
    *,
    previous_best_loss: float,
    new_loss: float,
    min_relative_improvement: float,
) -> bool:
    """Return whether a new loss improves enough to reset early-stop patience."""

    if not np.isfinite(previous_best_loss):
        return True
    if new_loss >= previous_best_loss:
        return False
    relative_improvement = (previous_best_loss - new_loss) / max(abs(previous_best_loss), 1.0e-12)
    return bool(relative_improvement >= min_relative_improvement)


def _persist_optimizer_artifacts(
    *,
    config: BlazedAxConfig | LaminarAxConfig,
    evaluation_measurement: MeasurementData,
    best_parameters: dict[str, float],
    best_grating_parameters: dict[str, object],
    best_loss: float,
    trial_records: list[TrialRecord],
    result_json_path: Path,
    trial_history_csv_path: Path,
    best_fit_plot_path: Path | None,
    loss_history_plot_path: Path | None,
    stopped_early: bool,
    completed_trials: int,
    early_stop_reason: str | None,
    backend_effective: str,
) -> None:
    """Rewrite all optimizer artifacts from the current optimizer state."""

    _write_result_json(
        config=config,
        best_parameters=best_parameters,
        best_grating_parameters=best_grating_parameters,
        best_loss=best_loss,
        stopped_early=stopped_early,
        completed_trials=completed_trials,
        early_stop_reason=early_stop_reason,
        backend_requested=config.backend,
        backend_effective=backend_effective,
        output_path=result_json_path,
    )
    _write_trial_history_csv(trial_records, trial_history_csv_path)

    if best_fit_plot_path is not None:
        simulated_efficiency = simulate_efficiency_curve(
            config,
            best_parameters,
            evaluation_measurement,
            backend=backend_effective,
        )
        _save_best_fit_plot(
            measurement=evaluation_measurement,
            simulated_efficiency=simulated_efficiency,
            output_path=best_fit_plot_path,
        )
    if loss_history_plot_path is not None:
        _save_loss_history_plot(
            trial_records=trial_records,
            output_path=loss_history_plot_path,
            stopped_early=stopped_early,
        )


def _optimize_grating(config: BlazedAxConfig | LaminarAxConfig) -> OptimizationResult:
    """Run Ax optimization for a grating config."""

    measurement = load_measurement_data(config.measurement_path)
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    backend_effective = _resolve_optimizer_backend(config.backend)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = config.output_dir / "best_result.json"
    trial_history_csv_path = config.output_dir / "trial_history.csv"
    best_fit_plot_path = config.output_dir / "best_fit.png" if config.save_best_fit_plot else None
    loss_history_plot_path = (
        config.output_dir / "optimization_loss_history.png" if config.save_loss_plot else None
    )

    ax_client = _create_ax_client_for_config(config)
    trial_records: list[TrialRecord] = []
    best_parameters_float: dict[str, float] = {}
    best_grating_parameters = resolve_grating_parameters(config, best_parameters_float)
    best_loss = float("inf")
    completed_trials = 0
    stopped_early = False
    early_stop_reason: str | None = None
    consecutive_non_improving_trials = 0
    max_parallelism_exception_type = _import_max_parallelism_exception()
    data_required_exception_type = _import_data_required_exception()

    while completed_trials < int(config.total_trials):
        remaining_trials = int(config.total_trials) - completed_trials
        target_batch_size = min(int(config.batch_size), remaining_trials)
        candidates: list[tuple[int, dict[str, float]]] = []
        while len(candidates) < target_batch_size:
            try:
                raw_parameters, trial_index = ax_client.get_next_trial()
            except Exception as error:
                is_max_parallelism_error = (
                    max_parallelism_exception_type is not None
                    and isinstance(error, max_parallelism_exception_type)
                )
                is_data_required_error = (
                    data_required_exception_type is not None
                    and isinstance(error, data_required_exception_type)
                )
                if is_max_parallelism_error or is_data_required_error:
                    reason = "max_parallelism" if is_max_parallelism_error else "data_required"
                    if len(candidates) == 0:
                        raise RuntimeError(
                            "Ax blocked candidate generation before any candidate could be "
                            f"proposed (target_batch_size={target_batch_size}, reason={reason})."
                        ) from error
                    print(
                        "Ax generation clamp: "
                        f"reason={reason}, requested_batch={target_batch_size}, "
                        f"generated_batch={len(candidates)}"
                    )
                    break
                raise
            parameters = {name: float(value) for name, value in raw_parameters.items()}
            candidates.append((int(trial_index), parameters))

        evaluated_candidates = _evaluate_candidate_batch(
            candidates,
            config=config,
            measurement=measurement,
            backend_effective=backend_effective,
        )

        for trial_index, parameters, loss in evaluated_candidates:
            _complete_ax_trial(
                ax_client=ax_client,
                config=config,
                trial_index=int(trial_index),
                loss=float(loss),
            )
            trial_records.append(
                TrialRecord(
                    trial_index=int(trial_index),
                    loss=float(loss),
                    parameters=dict(parameters),
                )
            )
            completed_trials = len(trial_records)

            improved_enough = _is_significant_improvement(
                previous_best_loss=best_loss,
                new_loss=float(loss),
                min_relative_improvement=float(config.early_stopping_min_relative_improvement),
            )
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_parameters_float = dict(parameters)
                best_grating_parameters = resolve_grating_parameters(config, best_parameters_float)

            if completed_trials > int(config.early_stopping_warmup_trials):
                if improved_enough:
                    consecutive_non_improving_trials = 0
                else:
                    consecutive_non_improving_trials += 1
                if (
                    config.enable_early_stopping
                    and consecutive_non_improving_trials >= int(config.early_stopping_patience)
                ):
                    stopped_early = True
                    early_stop_reason = (
                        "loss plateau after "
                        f"{completed_trials} trials; patience={config.early_stopping_patience}, "
                        f"warmup={config.early_stopping_warmup_trials}, "
                        "min_relative_improvement="
                        f"{config.early_stopping_min_relative_improvement:.6g}"
                    )
                    break

        _persist_optimizer_artifacts(
            config=config,
            evaluation_measurement=evaluation_measurement,
            best_parameters=best_parameters_float,
            best_grating_parameters=best_grating_parameters,
            best_loss=best_loss,
            trial_records=trial_records,
            result_json_path=result_json_path,
            trial_history_csv_path=trial_history_csv_path,
            best_fit_plot_path=best_fit_plot_path,
            loss_history_plot_path=loss_history_plot_path,
            stopped_early=stopped_early,
            completed_trials=completed_trials,
            early_stop_reason=early_stop_reason,
            backend_effective=backend_effective,
        )
        if stopped_early:
            break

    return OptimizationResult(
        best_parameters=best_parameters_float,
        best_grating_parameters=best_grating_parameters,
        best_loss=best_loss,
        measurement_path=measurement.source_path,
        result_json_path=result_json_path,
        trial_history_csv_path=trial_history_csv_path,
        best_fit_plot_path=best_fit_plot_path,
        loss_history_plot_path=loss_history_plot_path,
        trial_records=trial_records,
        stopped_early=stopped_early,
        completed_trials=completed_trials,
        early_stop_reason=early_stop_reason,
    )


def optimize_blazed(config: BlazedAxConfig) -> OptimizationResult:
    """Run Ax optimization for a blazed grating.

    Args:
        config: Blazed optimizer configuration including initial grating
            geometry, optimization bounds, trial count, and output directory.

    Returns:
        OptimizationResult with the best parameters, resolved grating
        parameters, best loss, and persisted artifact paths.

    Raises:
        ImportError: If Ax is not installed (install with ``pip install .[opt]``).
        ValueError: If ``config`` fails dataclass validation.
    """

    return _optimize_grating(config)


def optimize_laminar(config: LaminarAxConfig) -> OptimizationResult:
    """Run Ax optimization for a laminar grating.

    Args:
        config: Laminar optimizer configuration including initial grating
            geometry, optimization bounds, evaluation mode, trial count, and
            output directory.

    Returns:
        OptimizationResult with the best parameters, resolved grating
        parameters, best loss, and persisted artifact paths.

    Raises:
        ImportError: If Ax is not installed (install with ``pip install .[opt]``).
        ValueError: If ``config`` fails dataclass validation.
    """

    return _optimize_grating(config)
