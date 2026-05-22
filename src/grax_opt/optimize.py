"""Shared optimizer runtime helpers used by dynamic optimization."""

from __future__ import annotations

import concurrent.futures
import csv
import inspect
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from grax.materials import material_label

from .data import MeasurementData
from .objective import evaluate_trial


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
    config: Any,
    measurement: MeasurementData,
    backend_effective: str,
    build_grating_fn=None,
    resolve_solver_parameters_fn=None,
) -> tuple[int, dict[str, float], float]:
    """Evaluate one optimizer candidate and return trial index, params, and loss."""

    trial_index, parameters = candidate
    evaluate_kwargs: dict[str, object] = {
        "backend": backend_effective,
    }
    if build_grating_fn is not None:
        evaluate_kwargs["build_grating_fn"] = build_grating_fn
    if resolve_solver_parameters_fn is not None:
        evaluate_kwargs["resolve_solver_parameters_fn"] = resolve_solver_parameters_fn
    loss = float(evaluate_trial(config, parameters, measurement, **evaluate_kwargs))
    return int(trial_index), dict(parameters), float(loss)


def _evaluate_candidate_batch(
    candidates: list[tuple[int, dict[str, float]]],
    *,
    config: Any,
    measurement: MeasurementData,
    backend_effective: str,
    build_grating_fn=None,
    resolve_solver_parameters_fn=None,
) -> list[tuple[int, dict[str, float], float]]:
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
    axis.set_title("Dynamic Optimizer Best Fit")
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
