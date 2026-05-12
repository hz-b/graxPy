"""Top-level Ax optimization loop for grating fitting."""

from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grax.materials import material_label

from .config import BlazedAxConfig, LaminarAxConfig
from .data import MeasurementData, load_measurement_data
from .model import build_ax_parameters, resolve_grating_parameters, resolve_solver_parameters
from .objective import build_evaluation_measurement, evaluate_trial, simulate_efficiency_curve


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
    trial_records: list[TrialRecord]


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


def _build_ax_optimize_kwargs(
    config: BlazedAxConfig | LaminarAxConfig,
    measurement: MeasurementData,
) -> dict[str, object]:
    """Build keyword arguments for the Ax optimize function."""

    ax_optimize = _import_ax_optimize()

    def evaluation_function(parameterization: dict[str, float]) -> dict[str, tuple[float, float]]:
        loss = evaluate_trial(config, parameterization, measurement)
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


def _optimize_grating(config: BlazedAxConfig | LaminarAxConfig) -> OptimizationResult:
    """Run Ax optimization for a grating config."""

    measurement = load_measurement_data(config.measurement_path)
    evaluation_measurement = build_evaluation_measurement(config, measurement)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ax_optimize = _import_ax_optimize()
    kwargs = _build_ax_optimize_kwargs(config, measurement)
    best_parameters, values, experiment, _model = ax_optimize(**kwargs)
    mean_values, _covariances = values
    best_loss = float(mean_values[config.objective_name])

    best_parameters_float = {name: float(value) for name, value in best_parameters.items()}
    best_grating_parameters = resolve_grating_parameters(config, best_parameters_float)
    trial_records = _extract_trial_records(experiment)

    result_json_path = config.output_dir / "best_result.json"
    trial_history_csv_path = config.output_dir / "trial_history.csv"
    best_fit_plot_path = config.output_dir / "best_fit.png" if config.save_best_fit_plot else None

    _write_result_json(
        config=config,
        best_parameters=best_parameters_float,
        best_grating_parameters=best_grating_parameters,
        best_loss=best_loss,
        output_path=result_json_path,
    )
    _write_trial_history_csv(trial_records, trial_history_csv_path)

    if best_fit_plot_path is not None:
        simulated_efficiency = simulate_efficiency_curve(
            config,
            best_parameters_float,
            evaluation_measurement,
        )
        _save_best_fit_plot(
            measurement=evaluation_measurement,
            simulated_efficiency=simulated_efficiency,
            output_path=best_fit_plot_path,
        )

    return OptimizationResult(
        best_parameters=best_parameters_float,
        best_grating_parameters=best_grating_parameters,
        best_loss=best_loss,
        measurement_path=measurement.source_path,
        result_json_path=result_json_path,
        trial_history_csv_path=trial_history_csv_path,
        best_fit_plot_path=best_fit_plot_path,
        trial_records=trial_records,
    )


def optimize_blazed(config: BlazedAxConfig) -> OptimizationResult:
    """Run Ax optimization for a blazed grating."""

    return _optimize_grating(config)


def optimize_laminar(config: LaminarAxConfig) -> OptimizationResult:
    """Run Ax optimization for a laminar grating."""

    return _optimize_grating(config)
