"""Shared optimizer runtime helpers used by measurement-fit optimization."""

from __future__ import annotations

import concurrent.futures
import csv
import inspect
import io
import os
import platform
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from grax.materials import material_label

from .data import MeasurementData
from .objective import evaluate_trial_curve_with_metadata


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
    """Prevent torch.random.fork_rng() from probing CUDA on CPU-only hosts."""

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
        if numba_available:
            return "numba"
        raise RuntimeError(
            "Optimizer backend 'auto' resolved to numba, but numba is not installed. "
            "Reinstall the standard graxpy package so the required numba dependency is available."
        )
    if normalized_backend == "numba":
        if numba_available:
            return "numba"
        raise RuntimeError(
            "Requested optimizer backend 'numba' but numba is not installed. "
            "Reinstall the standard graxpy package so the required numba dependency is available."
        )
    if normalized_backend == "numpy":
        warnings.warn(
            "backend='numpy' is deprecated and will be removed in a future version. "
            "Use backend='numba' or rely on the default numba backend instead.",
            FutureWarning,
            stacklevel=2,
        )
        return "numpy"
    return "numpy"


def _resolve_batch_worker_count(batch_size: int) -> int:
    """Return effective worker count for one candidate batch."""

    return max(1, min(int(batch_size), int(os.cpu_count() or 1)))


def _evaluate_candidate_worker(
    candidate: tuple[int, dict[str, float]],
    *,
    config: Any,
    measurement: MeasurementData,
    backend_effective: str,
    build_grating_fn=None,
    resolve_solver_parameters_fn=None,
) -> tuple[int, dict[str, float], float, int, Any]:
    """Evaluate one optimizer candidate.

    Args:
        candidate: Candidate ``(trial_index, parameters)`` pair.
        config: Optimization configuration describing the simulation setup.
        measurement: Measurement data used for evaluation.
        backend_effective: Fourier coefficient backend to use for the simulation.
        build_grating_fn: Optional grating-build hook.
        resolve_solver_parameters_fn: Optional solver-parameter hook.

    Returns:
        The trial index, parameters, loss, resolved worker count, and the
        simulated curve for the candidate.
    """

    trial_index, parameters = candidate
    evaluate_kwargs: dict[str, object] = {
        "backend": backend_effective,
    }
    if build_grating_fn is not None:
        evaluate_kwargs["build_grating_fn"] = build_grating_fn
    if resolve_solver_parameters_fn is not None:
        evaluate_kwargs["resolve_solver_parameters_fn"] = resolve_solver_parameters_fn
    loss, resolved_max_workers, simulated_efficiency = evaluate_trial_curve_with_metadata(
        config,
        parameters,
        measurement,
        **evaluate_kwargs,
    )
    return (
        int(trial_index),
        dict(parameters),
        float(loss),
        int(resolved_max_workers),
        simulated_efficiency,
    )


def _evaluate_candidate_batch(
    candidates: list[tuple[int, dict[str, float]]],
    *,
    config: Any,
    measurement: MeasurementData,
    backend_effective: str,
    build_grating_fn=None,
    resolve_solver_parameters_fn=None,
) -> list[tuple[int, dict[str, float], float, int, Any]]:
    """Evaluate a candidate batch, optionally in parallel."""

    if len(candidates) <= 1:
        return [
            _evaluate_candidate_worker(
                candidates[0],
                config=config,
                measurement=measurement,
                backend_effective=backend_effective,
                build_grating_fn=build_grating_fn,
                resolve_solver_parameters_fn=resolve_solver_parameters_fn,
            )
        ]

    worker_count = _resolve_batch_worker_count(len(candidates))
    executor_cls: type[concurrent.futures.Executor]
    if build_grating_fn is None and resolve_solver_parameters_fn is None:
        executor_cls = concurrent.futures.ProcessPoolExecutor
    else:
        executor_cls = concurrent.futures.ThreadPoolExecutor
    with executor_cls(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _evaluate_candidate_worker,
                candidate,
                config=config,
                measurement=measurement,
                backend_effective=backend_effective,
                build_grating_fn=build_grating_fn,
                resolve_solver_parameters_fn=resolve_solver_parameters_fn,
            )
            for candidate in candidates
        ]
        evaluated = [future.result() for future in futures]
    return sorted(evaluated, key=lambda item: int(item[0]))


@dataclass(frozen=True)
class TrialRecord:
    """Summary of one completed Ax trial.

    Attributes:
        trial_index: Ax trial index for the completed trial.
        loss: Objective value observed for the trial.
        parameters: Free parameter values evaluated for the trial.
        extras: Optional scalar diagnostics such as per-measurement losses.
    """

    trial_index: int
    loss: float
    parameters: dict[str, float]
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    """Result bundle returned by optimizer entrypoints.

    Attributes:
        best_parameters: Best free parameters returned by Ax.
        best_grating_parameters: Best resolved grating parameters after applying
            equality constraints.
        best_loss: Best objective value found during the optimization run.
        measurement_path: Path to the measurement data used for the fit.
        result_json_path: Path to the persisted JSON summary for the run.
        trial_history_csv_path: Path to the persisted per-trial history CSV.
        best_fit_plot_path: Path to the best-fit plot when plotting is enabled,
            otherwise ``None``.
        loss_history_plot_path: Path to the loss-history plot when plotting is
            enabled, otherwise ``None``.
        trial_records: Per-trial records including trial index, loss, and free
            parameter values.
        stopped_early: Whether early stopping ended the run before exhausting
            ``total_trials``.
        completed_trials: Number of trials successfully evaluated.
        early_stop_reason: Human-readable reason for early stopping, or
            ``None`` when the run completed normally.
    """

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


def _atomic_write_text(output_path: Path, text: str) -> None:
    """Write text to a path atomically so a crash cannot truncate the file.

    Args:
        output_path: Destination path to replace.
        text: File contents to write.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_trial_history_csv(
    trial_records: list[TrialRecord],
    output_path: Path,
) -> None:
    """Write per-trial optimization history to CSV.

    Args:
        trial_records: Completed trial records to serialize.
        output_path: Destination CSV path.
    """

    parameter_names: list[str] = []
    for record in trial_records:
        for name in record.parameters:
            if name not in parameter_names:
                parameter_names.append(name)

    extra_names: list[str] = []
    for record in trial_records:
        for name in record.extras:
            if name not in extra_names:
                extra_names.append(name)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["trial_index", "loss", *parameter_names, *extra_names])
    for record in trial_records:
        writer.writerow(
            [
                record.trial_index,
                record.loss,
                *[record.parameters.get(name, "") for name in parameter_names],
                *[record.extras.get(name, "") for name in extra_names],
            ]
        )
    _atomic_write_text(output_path, buffer.getvalue())


def json_safe_grating_parameters(parameters: dict[str, object]) -> dict[str, object]:
    """Return grating parameters with material objects converted to labels."""

    serializable: dict[str, object] = {}
    for name, value in parameters.items():
        if name.endswith("_material"):
            serializable[name] = None if value is None else material_label(value)
        else:
            serializable[name] = value
    return serializable


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
    axis.set_title("Measurement-Fit Optimizer Best Fit")
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
    if losses.size > 0 and np.isfinite(losses[0]) and losses[0] > 0.0:
        outlier_ceiling = float(losses[0]) * 10.0
        if np.any(losses[np.isfinite(losses)] > outlier_ceiling):
            axis.set_ylim(top=outlier_ceiling)
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


def _complete_ax_trial(
    *,
    ax_client,
    config: Any,
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
