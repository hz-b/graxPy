"""Benchmark trial-level optimizer multiprocessing for the laminar example."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from grax import LaminarGrating
from grax_opt import build_evaluation_measurement, load_measurement_data, optimize_to_measurements
from grax_opt.objective import evaluate_trial_with_metadata

from example_config import (
    angle_mode,
    cff,
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    grazing_angle_deg,
    layer_thickness_nm,
    measurement_path,
    optical_constants_dir,
    optimizer_backend,
    period_lpermm,
    random_seed,
    width_to_period_ratio,
    depth_nm,
    left_wall_angle_deg,
    right_wall_angle_deg,
    top_cap_thickness_nm,
    x_resolution_nm,
    z_resolution_nm,
)


def _load_material(path: Path, name: str) -> pd.DataFrame:
    material = pd.read_csv(
        path,
        skiprows=1,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    material.attrs["name"] = name
    return material


silicon = _load_material(optical_constants_dir / "n_Si_cxro.txt", "Si")
platinum = _load_material(optical_constants_dir / "n_Pt_cxro.txt", "Pt")
carbon = _load_material(optical_constants_dir / "n_C_cxro.txt", "C")


def build_grating(parameters: dict[str, float]) -> LaminarGrating:
    """Build the laminar grating used for benchmark trials."""

    return LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


def _spec(*, output_dir: Path, total_trials: int, max_workers: int | str | None) -> dict[str, object]:
    """Return a benchmark-friendly measurement-fit spec."""

    return {
        "build_grating": build_grating,
        "parameter_bounds": {
            "width_to_period_ratio": (0.5, 0.8),
            "depth_nm": (13.9, 15.9),
            "left_wall_angle_deg": (5.0, 20.0),
            "right_wall_angle_deg": (5.0, 20.0),
            "top_cap_thickness_nm": (0.3, 2.0),
        },
        "measurement_path": measurement_path,
        "output_dir": output_dir,
        "angle_mode": angle_mode,
        "grazing_angle_deg": grazing_angle_deg,
        "cff": cff,
        "diffraction_order": diffraction_order,
        "fourier_orders": fourier_orders,
        "validate_physical_results": True,
        "total_trials": total_trials,
        "batch_size": 1,
        "random_seed": random_seed,
        "experiment_name": "laminar_fit_benchmark",
        "save_best_fit_plot": False,
        "save_loss_plot": False,
        "enable_early_stopping": False,
        "evaluation_energies_ev": list(evaluation_energies_ev),
        "backend": optimizer_backend,
        "max_workers": max_workers,
    }


def _trial_parameters() -> dict[str, float]:
    """Return one representative laminar candidate for single-trial timing."""

    return {
        "width_to_period_ratio": float(width_to_period_ratio),
        "depth_nm": float(depth_nm),
        "left_wall_angle_deg": float(left_wall_angle_deg),
        "right_wall_angle_deg": float(right_wall_angle_deg),
        "top_cap_thickness_nm": float(top_cap_thickness_nm),
    }


def _build_grating_fn(trial_parameters: dict[str, float]) -> LaminarGrating:
    return build_grating(trial_parameters)


def _resolve_solver_parameters_fn(_trial_parameters: dict[str, float]) -> dict[str, float | None]:
    return {"roughness_sigma_nm": None}


def run_single_trial_benchmark(*, max_workers: int | str | None, output_dir: Path) -> dict[str, object]:
    """Measure one objective evaluation with a fixed candidate."""

    spec = _spec(output_dir=output_dir, total_trials=1, max_workers=max_workers)
    measurement = load_measurement_data(measurement_path)
    evaluation_measurement = build_evaluation_measurement(
        type("BenchmarkConfig", (), spec)(),
        measurement,
    )
    started = time.perf_counter()
    loss, resolved_max_workers = evaluate_trial_with_metadata(
        type("BenchmarkConfig", (), spec)(),
        _trial_parameters(),
        evaluation_measurement,
        backend=str(spec["backend"]),
        build_grating_fn=_build_grating_fn,
        resolve_solver_parameters_fn=_resolve_solver_parameters_fn,
    )
    wall_seconds = time.perf_counter() - started
    return {
        "mode": "single_trial",
        "requested_max_workers": max_workers,
        "resolved_max_workers": resolved_max_workers,
        "loss": loss,
        "wall_seconds": wall_seconds,
    }


def run_full_optimization_benchmark(
    *,
    max_workers: int | str | None,
    total_trials: int,
    output_dir: Path,
) -> dict[str, object]:
    """Measure the end-to-end optimizer runtime."""

    spec = _spec(output_dir=output_dir, total_trials=total_trials, max_workers=max_workers)
    started = time.perf_counter()
    result = optimize_to_measurements(spec)
    wall_seconds = time.perf_counter() - started
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    return {
        "mode": "full_optimization",
        "requested_max_workers": max_workers,
        "resolved_max_workers": payload["optimizer_resolved_max_workers"],
        "completed_trials": result.completed_trials,
        "best_loss": result.best_loss,
        "wall_seconds": wall_seconds,
    }


def _normalize_workers(raw: str) -> int | str | None:
    """Parse CLI worker selection into optimizer config values."""

    if raw == "none":
        return None
    if raw in {"auto", "all"}:
        return raw
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single-trial", "full"), default="full")
    parser.add_argument("--serial-workers", default="1")
    parser.add_argument("--parallel-workers", default="auto")
    parser.add_argument("--total-trials", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "benchmark_optimizer_multiprocessing",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    serial_workers = _normalize_workers(args.serial_workers)
    parallel_workers = _normalize_workers(args.parallel_workers)

    if args.mode == "single-trial":
        serial_summary = run_single_trial_benchmark(
            max_workers=serial_workers,
            output_dir=output_dir / "single_trial_serial",
        )
        parallel_summary = run_single_trial_benchmark(
            max_workers=parallel_workers,
            output_dir=output_dir / "single_trial_parallel",
        )
    else:
        serial_summary = run_full_optimization_benchmark(
            max_workers=serial_workers,
            total_trials=args.total_trials,
            output_dir=output_dir / "full_serial",
        )
        parallel_summary = run_full_optimization_benchmark(
            max_workers=parallel_workers,
            total_trials=args.total_trials,
            output_dir=output_dir / "full_parallel",
        )

    speedup = serial_summary["wall_seconds"] / parallel_summary["wall_seconds"]
    summary = {
        "mode": args.mode,
        "serial": serial_summary,
        "parallel": parallel_summary,
        "speedup_ratio": speedup,
    }
    summary_path = output_dir / f"{args.mode.replace('-', '_')}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
