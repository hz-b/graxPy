import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _solver_comparison import (  # noqa: E402
    add_solver_arguments,
    apply_stride,
    solver_output_path,
)

import grax  # noqa: E402

args = add_solver_arguments(
    argparse.ArgumentParser(description="Blazed 600 l/mm monochromator sweep")
).parse_args()

grax.setup_logging(level='INFO', run_id=f'blazed_sweep_{args.solver}')


example_root = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
optical_constants_dir = example_root / "optical_constants"
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = solver_output_path(
    results_dir / "blazed_comparison_monochromator_orders_1_3.csv", args.solver, args.tag
)
plot_path = solver_output_path(
    results_dir / "blazed_comparison_monochromator_orders_1_3.png", args.solver, args.tag
)
profile_plot_path = results_dir / "blazed_comparison_profile.png"

# Import Optical Constants
silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    # skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
gold = pd.read_csv(
    optical_constants_dir / "OC_Au_SSTR.dat",
    # skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

# Define Grating

period_lpermm = 600
blaze_angle_deg = 0.729
anti_blaze_angle_deg = 5.597


grating = grax.BlazedGrating(
    period_lpermm=600,
    blaze_angle_deg=0.729,
    anti_blaze_angle_deg=5.597,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    z_resolution_nm=0.1,
    x_resolution_nm=0.1,
)

# Energy range
energies_ev = apply_stride(np.arange(50.0, 2000.0, 10.0), args.stride)

# Use monochromator_cases helper
cases = grax.monochromator_cases(
    grating=grating,
    energies_ev=energies_ev,
    period_lpermm=period_lpermm,
    diffraction_order=1,
    cff=2.25,
    polarization="p",
)

runner = grax.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    live_plot_order_count=3,
    checkpoint_dir=solver_output_path(results_dir / "checkpoints", args.solver, args.tag),
    resume=False,  # Set to False to force restart from beginning
    backend="numba",
    default_solver=args.solver,
)

# Guard the executable part: on macOS the batch runner spawns workers, and a
# spawned worker re-imports this file by path. Without the guard the worker
# re-runs the whole sweep and recursively spawns more workers, which fails with
# BrokenProcessPool before any case completes.
if __name__ == "__main__":
    grating.plot_profile(profile_plot_path)
    batch_result = list(runner.run_cases(
        cases,
        metadata={
            "grating_type": "BlazedGrating",
            "period_lpermm": period_lpermm,
            "blaze_angle_deg": blaze_angle_deg,
            "anti_blaze_angle_deg": anti_blaze_angle_deg,
            "substrate_material": "Si",
            "layer_material": "Au",
            "layer_thickness_nm": 30.0,
            "fourier_orders": 20,
            "backend": "numba",
            "solver": args.solver,
            "description": "Blazed grating monochromator sweep",
        }
    ))
    grax.write_all_orders_csv(batch_result, csv_path)
    grax.plot_order_subset(
        batch_result,
        plot_path,
        diffraction_orders=[1, 2, 3],
        title="Blazed Grating Monochromator Sweep (600 l/mm, BA=0.729 deg): Orders 1-3",
    )

    print(f"Computed {sum(case.status == 'ok' for case in batch_result)} monochromator points.")
    print(f"Profile plot saved to: {profile_plot_path}")
    print(f"Monochromator sweep CSV saved to: {csv_path}")
    print(f"Monochromator orders plot saved to: {plot_path}")
