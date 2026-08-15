"""Laminar fixed-angle sweep using the centralized batch runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solver_comparison import (  # noqa: E402
    add_solver_arguments,
    apply_stride,
    solver_output_path,
)

parser = add_solver_arguments(
    argparse.ArgumentParser(description="Laminar 400 l/mm fixed-angle sweep")
)
parser.add_argument("--quick", action="store_true", help="Run a few coarse energy points")
args = parser.parse_args()

quick_mode = args.quick
LIVE_PLOT = False if quick_mode else True
if LIVE_PLOT:
    os.environ.setdefault("MPLBACKEND", "TkAgg")
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import grax  # noqa: E402

repo_root = Path(__file__).resolve().parents[2]
example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = solver_output_path(results_dir / "laminar_fixed_angle_comparison.png", args.solver, args.tag)
grating_plot_path = results_dir / "laminar_fixed_angle_profile.png"
csv_path = solver_output_path(results_dir / "laminar_fixed_angle_all_orders.csv", args.solver, args.tag)
roughness_sigma_nm = 0.5
silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
platinum = pd.read_csv(
    optical_constants_dir / "OC_Pt_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt"

carbon = pd.read_csv(
    optical_constants_dir / "OC_C_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

grating = grax.LaminarGrating(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    depth_nm=14.9,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    top_cap_material=carbon,
    top_cap_thickness_nm=0.7,
    z_resolution_nm=2.0 if quick_mode else 0.1,
    x_resolution_nm=10.0 if quick_mode else 0.1,
)
energies = (
    np.asarray([100.0, 300.0, 600.0], dtype=float)
    if quick_mode
    else np.arange(50.0, 650.1, 1.0)
)
energies = apply_stride(energies, args.stride)
cases = grax.fixed_angle_cases(
    grating=grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
    polarization="p",
)
cases = (
    dict(case, label="fixed-angle", roughness_sigma_nm=roughness_sigma_nm)
    for case in cases
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=5 if quick_mode else 30,
    show_progress=True,
    live_plot=LIVE_PLOT,
    live_plot_x_key="energy_ev",
    live_plot_order_count=1,
    live_plot_reference_data=grax.load_experimental_csv(
        Path(__file__).resolve().parent / "measured_alpha4deg_order1.csv"
    ),
    on_error="fail_fast",
    resume=False,
    max_workers="auto",
    default_solver=args.solver,
)

# Guard the executable part: on macOS the batch runner spawns workers, and a
# spawned worker re-imports this file by path. Without the guard the worker
# re-runs the whole sweep and recursively spawns more workers, which fails with
# BrokenProcessPool before any case completes.
if __name__ == "__main__":
    grating.plot_profile(grating_plot_path)
    batch_result = list(runner.run_cases(cases))

    experimental = grax.load_experimental_csv(
        Path(__file__).resolve().parent / "measured_alpha4deg_order1.csv"
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    successful_cases = [case for case in batch_result if case.status == "ok"]
    axis.plot(
        [case.energy_ev for case in successful_cases],
        [case.selected_efficiency for case in successful_cases],
        "b-o",
        linewidth=0.8,
        markersize=2.0,
        label="simulation",
    )
    axis.plot(
        experimental[:, 0],
        experimental[:, 1],
        "r-s",
        linewidth=0.8,
        markersize=2.0,
        label="measurement",
    )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title(
        "RCWA Simulation vs Experimental Data"
        if args.solver == "rcwa"
        else "Nevière Differential-Method Simulation vs Experimental Data"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    grax.write_all_orders_csv(batch_result, csv_path)

    print(f"Computed {sum(case.status == 'ok' for case in batch_result)} energy points.")
    print(f"Comparison plot saved to: {plot_path}")
    print(f"Grating-period plot saved to: {grating_plot_path}")
    print(f"Simulation CSV saved to: {csv_path}")
