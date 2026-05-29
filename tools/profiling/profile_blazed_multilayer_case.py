"""Profile one blazed multilayer simulation case with live stage logging."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import grax as rp  # noqa: E402
from grax.simulation._profiling import SolverProfiler  # noqa: E402

BLAZED_MULTILAYER_DIR = PROJECT_ROOT / "comparison_to_other_codes" / "blazed_multilayer"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "blazed_multilayer_case"
DEFAULT_FOURIER_ORDERS = [5, 10, 15]
DEFAULT_RESOLUTIONS_NM = [0.05, 0.1, 1.0]
DEFAULT_BACKEND = "numba"


@dataclass(frozen=True)
class ProfileRun:
    """One completed profiling run and its configuration."""

    energy_ev: float
    grazing_angle_deg: float
    fourier_orders: int
    x_resolution_nm: float
    z_resolution_nm: float
    label: str
    comparison_csv_name: str | None
    result: rp.SingleSimulationResult
    profiler: SolverProfiler


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the profiling tool."""
    parser = argparse.ArgumentParser(
        description="Profile one blazed multilayer simulation case with live stage timing."
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Row index from the DiffraMod energy/angle table to profile.",
    )
    parser.add_argument(
        "--energy-ev",
        type=float,
        help="Override photon energy in eV. Must be used with --grazing-angle-deg.",
    )
    parser.add_argument(
        "--grazing-angle-deg",
        type=float,
        help="Override grazing angle in degrees. Must be used with --energy-ev.",
    )
    parser.add_argument(
        "--x-resolution-nm",
        type=float,
        nargs="+",
        default=DEFAULT_RESOLUTIONS_NM,
        help="Horizontal profile discretization values in nm.",
    )
    parser.add_argument(
        "--z-resolution-nm",
        type=float,
        nargs="+",
        default=DEFAULT_RESOLUTIONS_NM,
        help="Vertical profile discretization values in nm.",
    )
    parser.add_argument(
        "--fourier-orders",
        type=int,
        nargs="+",
        default=DEFAULT_FOURIER_ORDERS,
        help="Fourier truncation order values on one side of zero.",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Run label used in output filenames and matrix summary rows.",
    )
    parser.add_argument(
        "--comparison-csv-name",
        type=str,
        default=None,
        help="Optional override for the matrix summary CSV filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-configuration reports and matrix summary.",
    )
    parser.add_argument(
        "--no-live-stage-log",
        action="store_true",
        help="Disable live stage start/end logging.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI argument combinations."""
    has_energy = args.energy_ev is not None
    has_angle = args.grazing_angle_deg is not None
    if has_energy != has_angle:
        raise ValueError("--energy-ev and --grazing-angle-deg must be provided together.")
    if args.case_index < 0:
        raise ValueError("--case-index must be >= 0.")
    if any(float(value) <= 0.0 for value in args.x_resolution_nm):
        raise ValueError("all --x-resolution-nm values must be > 0.")
    if any(float(value) <= 0.0 for value in args.z_resolution_nm):
        raise ValueError("all --z-resolution-nm values must be > 0.")
    if any(int(value) < 1 for value in args.fourier_orders):
        raise ValueError("all --fourier-orders values must be >= 1.")
    if not str(args.label).strip():
        raise ValueError("--label must not be empty.")
    if getattr(args, "backend", DEFAULT_BACKEND) != DEFAULT_BACKEND:
        raise ValueError("The profiling tool requires the numba backend.")


def _load_materials() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load optical constants used by the blazed multilayer footprint."""
    optical_constants_dir = BLAZED_MULTILAYER_DIR / "optical_constants"
    silicon = pd.read_csv(
        optical_constants_dir / "OC_Si_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    silicon.attrs["name"] = "Si"
    chromium = pd.read_csv(
        optical_constants_dir / "OC_Cr_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    chromium.attrs["name"] = "Cr"
    carbon = pd.read_csv(
        optical_constants_dir / "OC_C_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    carbon.attrs["name"] = "C"
    return silicon, chromium, carbon


def load_reference_data() -> pd.DataFrame:
    """Load the DiffraMod energy/angle table for case selection."""
    reference_data = pd.read_csv(
        BLAZED_MULTILAYER_DIR / "simulation" / "DiffractMod_CrC_d4.8_N60.dat",
        sep=r"\s+",
        engine="python",
    )
    reference_data = reference_data[["Energy", "Efficiency(GR)", "alpha"]].copy()
    reference_data = reference_data.apply(pd.to_numeric, errors="coerce").dropna()
    return reference_data.reset_index(drop=True)


def resolve_case_parameters(
    args: argparse.Namespace,
    reference_data: pd.DataFrame,
) -> tuple[float, float]:
    """Return energy and grazing angle selected by CLI arguments."""
    if args.energy_ev is not None and args.grazing_angle_deg is not None:
        return float(args.energy_ev), float(args.grazing_angle_deg)
    if args.case_index >= len(reference_data):
        raise ValueError(
            f"--case-index {args.case_index} is out of range for "
            f"{len(reference_data)} reference rows."
        )
    row = reference_data.iloc[int(args.case_index)]
    return float(row["Energy"]), float(row["alpha"])


def build_blazed_multilayer_grating(
    *,
    x_resolution_nm: float,
    z_resolution_nm: float,
) -> rp.BlazedGrating:
    """Return the blazed multilayer grating used by the comparison sweep."""
    silicon, chromium, carbon = _load_materials()
    multilayer_stack = rp.MultilayerStack(
        substrate_material=silicon,
        material_a=chromium,
        material_b=carbon,
        d_period_nm=4.8,
        gamma=0.4,
        n_bilayers=60,
        top_material=carbon,
    )
    return rp.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=multilayer_stack,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )


def configure_profiler(
    *,
    args: argparse.Namespace,
    energy_ev: float,
    grazing_angle_deg: float,
    fourier_orders: int,
    x_resolution_nm: float,
    z_resolution_nm: float,
) -> SolverProfiler:
    """Return a configured profiler for one blazed multilayer run."""
    profiler = SolverProfiler(log_stages=not args.no_live_stage_log)
    profiler.enable_memory_tracking()
    profiler.set_metadata("profile_case", "blazed_multilayer_single_case")
    profiler.set_metadata("case_index", int(args.case_index))
    profiler.set_metadata("energy_ev", float(energy_ev))
    profiler.set_metadata("grazing_angle_deg", float(grazing_angle_deg))
    profiler.set_metadata("fourier_orders", int(fourier_orders))
    profiler.set_metadata("x_resolution_nm", float(x_resolution_nm))
    profiler.set_metadata("z_resolution_nm", float(z_resolution_nm))
    profiler.set_metadata("backend", DEFAULT_BACKEND)
    profiler.set_metadata("python_version", sys.version.split()[0])
    profiler.set_metadata("numpy_version", np.__version__)
    return profiler


def _json_default(value: object) -> Any:
    """Return JSON-compatible values for NumPy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _run_label(*, fourier_orders: int, x_resolution_nm: float, z_resolution_nm: float) -> str:
    """Return a filesystem-safe label for one profile configuration."""
    x_label = f"{x_resolution_nm:g}".replace(".", "p")
    z_label = f"{z_resolution_nm:g}".replace(".", "p")
    return f"fo{fourier_orders}_x{x_label}_z{z_label}"


def _sanitize_label(label: str) -> str:
    """Return a filesystem-safe run label."""

    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)


def _stage_seconds(summary: dict[str, object], stage_name: str) -> float:
    """Return exclusive seconds for one named stage from a profiler summary."""

    for stage in summary["stages"]:
        if stage["stage"] == stage_name:
            return float(stage["seconds_exclusive"])
    return 0.0


def write_run_outputs(
    *,
    output_dir: Path,
    run: ProfileRun,
) -> tuple[Path, Path]:
    """Write human-readable and JSON outputs for one profiling run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label = _run_label(
        fourier_orders=run.fourier_orders,
        x_resolution_nm=run.x_resolution_nm,
        z_resolution_nm=run.z_resolution_nm,
    )
    run_label = _sanitize_label(run.label)
    report_path = output_dir / f"profile_report_{run_label}_{label}.txt"
    summary_path = output_dir / f"profile_summary_{run_label}_{label}.json"
    report_path.write_text(run.profiler.format_report(), encoding="utf-8")
    summary = run.profiler.summary_dict()
    summary["label"] = run.label
    summary["result"] = {
        "selected_efficiency": float(run.result.selected_efficiency),
        "selected_diffraction_angle_deg": float(run.result.selected_diffraction_angle_deg),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return report_path, summary_path


def write_matrix_summary(*, output_dir: Path, runs: list[ProfileRun]) -> Path:
    """Write a CSV summary across all profile configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = _sanitize_label(runs[0].label) if runs else "run"
    summary_filename = runs[0].comparison_csv_name or f"profile_matrix_summary_{run_label}.csv"
    summary_path = output_dir / summary_filename
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "energy_ev",
                "grazing_angle_deg",
                "fourier_orders",
                "x_resolution_nm",
                "z_resolution_nm",
                "total_wall_seconds",
                "texture_generation_seconds",
                "fourier_coefficients_seconds",
                "layer_propagation_cascade_seconds",
                "profiled_exclusive_seconds",
                "peak_memory_bytes",
                "texture_count",
                "unique_texture_count",
                "selected_efficiency",
            ]
        )
        for run in runs:
            summary = run.profiler.summary_dict()
            metadata = summary["metadata"]
            writer.writerow(
                [
                    run.label,
                    run.energy_ev,
                    run.grazing_angle_deg,
                    run.fourier_orders,
                    run.x_resolution_nm,
                    run.z_resolution_nm,
                    summary["total_wall_seconds"],
                    _stage_seconds(summary, "texture_generation"),
                    _stage_seconds(summary, "fourier_coefficients"),
                    _stage_seconds(summary, "layer_propagation_cascade"),
                    summary["profiled_exclusive_seconds"],
                    summary["peak_memory_bytes"],
                    metadata.get("texture_count"),
                    metadata.get("unique_texture_signatures"),
                    run.result.selected_efficiency,
                ]
            )
    return summary_path


def run_profile_cases(args: argparse.Namespace) -> list[ProfileRun]:
    """Run the one-energy Fourier/resolution profile matrix."""
    _validate_args(args)
    reference_data = load_reference_data()
    energy_ev, grazing_angle_deg = resolve_case_parameters(args, reference_data)
    runs: list[ProfileRun] = []
    logger = logging.getLogger("grax.simulation.profiling")
    for fourier_orders, x_resolution_nm, z_resolution_nm in product(
        args.fourier_orders,
        args.x_resolution_nm,
        args.z_resolution_nm,
    ):
        logger.info(
            "profile configuration start: energy=%.6g eV grazing=%.6f deg "
            "fourier_orders=%s x_resolution_nm=%.6g z_resolution_nm=%.6g",
            energy_ev,
            grazing_angle_deg,
            int(fourier_orders),
            float(x_resolution_nm),
            float(z_resolution_nm),
        )
        profiler = configure_profiler(
            args=args,
            energy_ev=energy_ev,
            grazing_angle_deg=grazing_angle_deg,
            fourier_orders=int(fourier_orders),
            x_resolution_nm=float(x_resolution_nm),
            z_resolution_nm=float(z_resolution_nm),
        )
        grating = build_blazed_multilayer_grating(
            x_resolution_nm=float(x_resolution_nm),
            z_resolution_nm=float(z_resolution_nm),
        )
        result = rp.run_simulation(
            grating=grating,
            energy_ev=energy_ev,
            grazing_angle_deg=grazing_angle_deg,
            diffraction_order=2,
            fourier_orders=int(fourier_orders),
            _profiler=profiler,
            backend=DEFAULT_BACKEND,
        )
        runs.append(
            ProfileRun(
                energy_ev=energy_ev,
                grazing_angle_deg=grazing_angle_deg,
                fourier_orders=int(fourier_orders),
                x_resolution_nm=float(x_resolution_nm),
                z_resolution_nm=float(z_resolution_nm),
                label=str(args.label),
                comparison_csv_name=args.comparison_csv_name,
                result=result,
                profiler=profiler,
            )
        )
    return runs


def main(argv: list[str] | None = None) -> int:
    """Run the profiling command-line tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    runs = run_profile_cases(args)
    for run in runs:
        report_path, summary_path = write_run_outputs(output_dir=Path(args.output_dir), run=run)
        print(run.profiler.format_report())
        print(f"Selected efficiency: {run.result.selected_efficiency:.6g}")
        print(f"Saved report: {report_path}")
        print(f"Saved summary: {summary_path}")
    matrix_summary_path = write_matrix_summary(output_dir=Path(args.output_dir), runs=runs)
    print(f"Saved matrix summary: {matrix_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
