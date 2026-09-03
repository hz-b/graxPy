"""Fast multilayer theta-search tutorial example."""

from __future__ import annotations

import argparse

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import grax


parser = argparse.ArgumentParser(description="Multilayer theta-search sweep")
parser.add_argument(
    "--solver",
    choices=("rcwa", "neviere"),
    default="rcwa",
    help="Electromagnetic solver to run. Both compute every diffraction order; "
    "they differ only in how each layer is crossed in z.",
)
parser.add_argument(
    "--polarization",
    choices=("s", "p", "TE", "TM"),
    default="s",
    help="Incident polarization used for every stage of the search "
    "(s/TE or p/TM, canonicalized to s/p).",
)
args = parser.parse_args()

# Keep each solver/polarization run in its own directory so an s and a p run,
# or an rcwa and a neviere run, sit side by side instead of overwriting.
run_tag = f"{args.solver}_{args.polarization.lower()}"

example_root = Path(__file__).resolve().parent
output_dir = example_root / "results" / run_tag
optical_constants_dir = example_root / "optical_constants"
log_dir = output_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

grax.setup_logging(
    level="INFO", log_dir=str(log_dir), run_id=f"multilayer_theta_search_{run_tag}"
)
logger = logging.getLogger("grax.examples.multilayer_theta_search")

energies_ev = np.arange(500.0, 6000.1, 10, dtype=float)
# energies_ev = np.arange(500.0, 6000.1, 200, dtype=float)
logger.info("Running multilayer theta-search example for %d energies.", energies_ev.size)
logger.info("Energy grid (eV): %s", energies_ev.tolist())

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

multilayer_stack = grax.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material=carbon,
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=1.0,
    z_resolution_nm=1.0,
)

# Guard the executable part: the batch runner spawns worker processes on macOS,
# and a spawned worker re-imports this file by path. Without the guard each
# worker re-runs the whole example and recursively spawns more, which fails with
# BrokenProcessPool before any case completes.
if __name__ == "__main__":
    sweep = grax.run_multilayer_theta_search_sweep(
        solver=args.solver,
        polarization=args.polarization,
        grating=grating,
        energies_ev=energies_ev,
        output_dir=output_dir,
        diffraction_order=2,
        multilayer_bragg_order=1,
        rough_scan_half_width_deg=5,
        rough_scan_points=31,
        rough_fourier_orders=2,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        fine_scan_half_width_deg=2,
        fine_scan_points=41,
        fine_fourier_orders=3,
        fine_x_resolution_nm=0.5,
        fine_z_resolution_nm=0.5,
        final_fourier_orders=15,
        final_x_resolution_nm=0.05,
        final_z_resolution_nm=0.05,
        precise_peak_selection_mode="voigt",
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=0.001,
        max_zero_efficiency_retries=3,
        max_workers="auto",
        show_progress=True,
        live_plot=True,
        on_error="fail_fast",
        checkpoint_dir=output_dir / "checkpoints",
        resume=True,
        theta_tracking_mode="auto",
        max_tracking_energy_step_ev=None,
        backend="numba",
    )

    logger.info("Sweep completed successfully.")
    print(f"Summary saved to: {sweep.summary_csv_path}")
    print(f"All-order CSV saved to: {sweep.all_orders_csv_path}")
    print(f"Final efficiency plot saved to: {sweep.energy_efficiency_plot_path}")
    print(f"Workflow plot saved to: {sweep.workflow_plot_path}")
    print(f"Total accumulated scan time: {sweep.total_elapsed_seconds:.1f} s")
    if sweep.profile_plot_path is not None:
        print(f"Profile plot saved to: {sweep.profile_plot_path}")
    if sweep.stack_plot_path is not None:
        print(f"Stack schematic saved to: {sweep.stack_plot_path}")
    print(f"Per-energy theta scans saved to: {sweep.theta_scan_directory}")
    logger.info("Outputs written under: %s", output_dir)
