"""Reproducible runtime benchmarks for the RCWA and Neviere solvers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .gratings import BaseGrating, BlazedGrating, LaminarGrating
from .simulation._profiling import SolverProfiler
from .simulation.batch import BatchSimulationRunner
from .simulation.cases import monochromator_cases
from .simulation.core import run_simulation
from .stacks import MultilayerStack

Solver = Literal["rcwa", "neviere"]
Difficulty = Literal["easy", "difficult"]


@dataclass(frozen=True)
class BenchmarkPreset:
    """Numerical settings for one benchmark difficulty."""

    name: Difficulty
    fourier_orders: int
    x_resolution_nm: float
    z_resolution_nm: float


PRESETS = {
    "easy": BenchmarkPreset("easy", 10, 1.0, 1.0),
    "difficult": BenchmarkPreset("difficult", 25, 0.1, 0.1),
}


@dataclass(frozen=True)
class BenchmarkCase:
    """A named grating factory and monochromator settings."""

    name: str
    grating_factory: object
    diffraction_order: int
    grazing_angle_deg: float
    polarization: str

    def build_grating(self, preset: BenchmarkPreset) -> BaseGrating:
        return self.grating_factory(preset)  # type: ignore[operator]


@dataclass
class TimingRecord:
    case: str
    difficulty: Difficulty
    execution_mode: str
    solver: Solver
    energy_ev: float
    fourier_orders: int
    x_resolution_nm: float
    z_resolution_nm: float
    samples_seconds: list[float]
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    stddev_seconds: float
    efficiency: float | None
    error_message: str = ""
    profiler: dict[str, object] | None = None


def default_cases() -> tuple[BenchmarkCase, ...]:
    """Return the three validation-inspired monochromator benchmark cases."""

    def laminar(p: BenchmarkPreset) -> LaminarGrating:
        return LaminarGrating(period_lpermm=400, width_to_period_ratio=0.67,
                              depth_nm=14.9, left_wall_angle_deg=15.0,
                              right_wall_angle_deg=15.0, substrate_material="Si",
                              layer_material="Pt", layer_thickness_nm=28.77,
                              x_resolution_nm=p.x_resolution_nm, z_resolution_nm=p.z_resolution_nm)

    def blazed(p: BenchmarkPreset) -> BlazedGrating:
        return BlazedGrating(period_lpermm=600, blaze_angle_deg=0.729,
                             anti_blaze_angle_deg=5.597, substrate_material="Si",
                             layer_material="Au", layer_thickness_nm=30.0,
                             x_resolution_nm=p.x_resolution_nm, z_resolution_nm=p.z_resolution_nm)

    def multilayer(p: BenchmarkPreset) -> BlazedGrating:
        return BlazedGrating(period_lpermm=2400, blaze_angle_deg=1.37,
                             anti_blaze_angle_deg=3.25,
                             coating_stack=MultilayerStack(substrate_material="Si",
                                 material_a="Cr", material_b="C", d_period_nm=4.8,
                                 gamma=0.4, n_bilayers=20, top_material="C"),
                             x_resolution_nm=p.x_resolution_nm, z_resolution_nm=p.z_resolution_nm)

    return (BenchmarkCase("laminar", laminar, 1, 4.0, "p"),
            BenchmarkCase("blazed", blazed, 1, 4.0, "p"),
            BenchmarkCase("blazed_multilayer", multilayer, 2, 4.0, "p"))


def benchmark_energies(energies_ev: list[float] | np.ndarray | None = None,
                       *, count: int = 100) -> np.ndarray:
    """Return an energy grid, rejecting more than 100 points."""
    if count < 1 or count > 100:
        raise ValueError("count must be between 1 and 100")
    if energies_ev is None:
        return np.linspace(50.0, 2000.0, count)
    values = np.asarray(energies_ev, dtype=float)
    if values.size > 100:
        raise ValueError("energy sweeps may contain at most 100 points")
    return values


def run_solver_benchmark(*, energies_ev=None, repeats: int = 3, warmups: int = 1,
                         output_dir: str | Path | None = None,
                         show_progress: bool = True, multiprocessing: bool = False,
                         max_workers: int | str | None = "auto") -> list[TimingRecord]:
    """Run all cases and presets for both solvers and optionally export results."""
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups must be non-negative")
    energies = benchmark_energies(energies_ev, count=10 if energies_ev is None and not multiprocessing else 100)
    if multiprocessing:
        records = _run_multiprocessing_benchmark(
            energies, repeats=repeats, warmups=warmups, max_workers=max_workers,
            show_progress=show_progress,
        )
        if output_dir is not None:
            export_benchmark(records, output_dir)
        return records
    records: list[TimingRecord] = []
    configurations = [(case, difficulty, preset, solver)
                      for case in default_cases() for difficulty, preset in PRESETS.items()
                      for solver in ("rcwa", "neviere")]
    iterator = tqdm(configurations, desc="Serial solver benchmark", unit="configuration",
                    disable=not show_progress)
    for case, difficulty, preset, solver in iterator:
        grating = case.build_grating(preset)
        energy_iterator = tqdm(
            enumerate(energies), total=len(energies),
            desc=f"{case.name}/{difficulty}/{solver}", unit="energy", leave=False,
            disable=not show_progress,
        )
        for energy_index, energy in energy_iterator:
            if energy_index == 0:
                for _ in range(warmups):
                    run_simulation(
                        grating=grating, energy_ev=float(energy),
                        grazing_angle_deg=case.grazing_angle_deg,
                        diffraction_order=case.diffraction_order,
                        fourier_orders=preset.fourier_orders,
                        polarization=case.polarization, solver=solver,
                        solver_options={"z_sampling": "textures"},
                        validate_physical_results=False,
                    )
            samples: list[float] = []
            result = None
            profiler_summary = None
            error = ""
            try:
                for _ in range(repeats):
                    profiler = SolverProfiler()
                    started = perf_counter()
                    result = run_simulation(
                        grating=grating, energy_ev=float(energy),
                        grazing_angle_deg=case.grazing_angle_deg,
                        diffraction_order=case.diffraction_order,
                        fourier_orders=preset.fourier_orders,
                        polarization=case.polarization, solver=solver,
                        solver_options={"z_sampling": "textures"},
                        validate_physical_results=False, _profiler=profiler,
                    )
                    samples.append(perf_counter() - started)
                    profiler_summary = profiler.summary_dict()
            except Exception as exc:
                error = str(exc)
            records.append(TimingRecord(
                case.name, difficulty, "serial", solver, float(energy), preset.fourier_orders,
                preset.x_resolution_nm, preset.z_resolution_nm, samples,
                float(np.median(samples)) if samples else float("nan"),
                float(np.min(samples)) if samples else float("nan"),
                float(np.max(samples)) if samples else float("nan"),
                float(np.std(samples)) if samples else float("nan"),
                None if result is None else float(result.selected_efficiency), error,
                profiler_summary,
            ))
        energy_iterator.close()
    if output_dir is not None:
        export_benchmark(records, output_dir)
    return records

def _run_multiprocessing_benchmark(energies: np.ndarray, *, repeats: int, warmups: int,
                                   max_workers: int | str | None,
                                   show_progress: bool) -> list[TimingRecord]:
    """Run complete monochromator sweeps through the multiprocessing batch runner."""
    records: list[TimingRecord] = []
    configurations = [(case, difficulty, preset, solver)
                      for case in default_cases() for difficulty, preset in PRESETS.items()
                      for solver in ("rcwa", "neviere")]
    iterator = tqdm(configurations, desc="Multiprocessing solver benchmark",
                    unit="configuration", disable=not show_progress)
    for case, difficulty, preset, solver in iterator:
        grating = case.build_grating(preset)
        cases = list(monochromator_cases(
            grating=grating, energies_ev=energies,
            diffraction_order=case.diffraction_order, cff=2.25,
            polarization=case.polarization,
        ))
        for _ in range(warmups):
            list(BatchSimulationRunner(
                solver=solver, fourier_orders=preset.fourier_orders,
                max_workers=max_workers, show_progress=False, on_error="continue",
                backend="numba", solver_options={"z_sampling": "textures"},
            ).run_cases(cases))
        samples: list[float] = []
        latest = []
        for _ in range(repeats):
            started = perf_counter()
            latest = list(BatchSimulationRunner(
                solver=solver, fourier_orders=preset.fourier_orders,
                max_workers=max_workers, show_progress=False, on_error="continue",
                backend="numba", solver_options={"z_sampling": "textures"},
            ).run_cases(cases))
            samples.append(perf_counter() - started)
        successful = [item for item in latest if item.status == "ok"]
        per_point = [sample / len(energies) for sample in samples]
        records.append(TimingRecord(
            case.name, difficulty, "multiprocessing", solver, float("nan"), preset.fourier_orders,
            preset.x_resolution_nm, preset.z_resolution_nm, per_point,
            float(np.median(per_point)), float(np.min(per_point)),
            float(np.max(per_point)), float(np.std(per_point)),
            None if not successful else float(np.mean([x.selected_efficiency for x in successful])),
            "" if len(successful) == len(energies) else "one or more points failed",
            None,
        ))
    return records


def export_benchmark(records: list[TimingRecord], output_dir: str | Path) -> None:
    """Write JSON, CSV, and summary plots for benchmark records."""
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    modes = sorted({record.execution_mode for record in records})
    for mode in modes:
        mode_records = [record for record in records if record.execution_mode == mode]
        suffix = mode
        (path / f"solver_runtime_benchmark_{suffix}.json").write_text(
            json.dumps([asdict(record) for record in mode_records], indent=2, allow_nan=True), encoding="utf-8")
        with (path / f"solver_runtime_benchmark_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["case", "difficulty", "execution_mode", "solver", "energy_ev", "fourier_orders",
                  "x_resolution_nm", "z_resolution_nm", "median_seconds", "minimum_seconds",
                  "maximum_seconds", "stddev_seconds", "efficiency", "error_message"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for record in mode_records:
                writer.writerow({field: getattr(record, field) for field in fields})
    for mode in modes:
      mode_records = [record for record in records if record.execution_mode == mode]
      for difficulty in PRESETS:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        for axis, case in zip(axes[0], default_cases()):
            for solver in ("rcwa", "neviere"):
                selected = [r for r in mode_records if r.case == case.name and r.difficulty == difficulty and r.solver == solver]
                axis.plot([r.energy_ev for r in selected], [r.median_seconds for r in selected], label=solver)
            axis.set_title(case.name); axis.set_xlabel("Energy (eV)"); axis.set_ylabel("Median seconds")
            axis.grid(True, alpha=0.3); axis.legend()
        figure.suptitle(f"{mode} solver benchmark — {difficulty}")
        figure.tight_layout(); figure.savefig(path / f"solver_runtime_{mode}_{difficulty}.png", dpi=150); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RCWA and Neviere solver runtime")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/solver_benchmark"))
    parser.add_argument("--points", type=int, default=None,
                        help="Energy points (default: 10 serial, 100 multiprocessing)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--multiprocessing", action="store_true",
                        help="Use multiprocessing and the default 100-point sweep")
    args = parser.parse_args()
    energies = None if args.points is None else benchmark_energies(count=args.points)
    run_solver_benchmark(energies_ev=energies, repeats=args.repeats,
                         warmups=args.warmups, output_dir=args.output_dir,
                         multiprocessing=args.multiprocessing)


if __name__ == "__main__":
    main()
