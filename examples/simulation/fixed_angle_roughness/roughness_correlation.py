"""Roughness correlation-length comparison using live batch plotting.

Compares zero roughness, 1 nm Debye-Waller roughness, and 1 nm random-interface
roughness swept across several lateral correlation lengths -- once with a
single grating period simulated (today's default) and once with the
random-interface roughness spanning several grating periods as one
continuous correlated field (a "supercell").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import grax
from helpers_roughness_kind_comparison import build_grating
from helpers_roughness_correlation import RoughnessRun, csv_path, run_label, run_title, save_grating_plot

matplotlib.use("TkAgg")

# ---------------------------------------------------------------------------
# Variables you may want to change manually
# ---------------------------------------------------------------------------

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "results_roughness_correlation"

# Roughness runs. Zero roughness and Debye-Waller are correlation-independent,
# so they are each run once; random-interface is swept over correlation
# lengths (in nm) at a fixed sigma.
DEBYE_SIGMA_NM = 1.0
RANDOM_INTERFACE_SIGMA_NM = 1.0
RANDOM_INTERFACE_CORRELATION_LENGTHS_NM = [0.0, 1.0, 10.0, 50.0, 100.0]
ROUGHNESS_SEED = 0

# In addition to the single-period random-interface sweep above, also run the
# same correlation-length sweep with the roughness spanning several grating
# periods as one continuous correlated field (a "supercell"). This is what
# lets a correlation length approach or exceed one grating period actually
# show up in the simulated diffraction pattern.
SUPERCELL_NUM_SUPERCELLS = 5

# Simulation settings
GRAZING_ANGLE_DEG = 1.0
ENERGIES_EV = np.arange(50.0, 2200.0, 50.0)
POLARIZATION = "p"
DIFFRACTION_ORDER = 1
FOURIER_ORDERS = 20
# Fourier orders used for the supercell runs. Solver cost scales with
# fourier_orders * num_supercells, so this is set low enough that the
# effective order count (here 4 * 5 = 20) matches the single-period runs
# above instead of multiplying the solve cost by num_supercells.
SUPERCELL_FOURIER_ORDERS = 4
MAX_WORKERS = "auto"
BACKEND = "numba"

# Grating geometry. ``COATING_LAYERS_NM`` lists the coating layers above the
# substrate, bottom-up; each layer gets the same per-layer roughness so the
# example exercises the per-layer roughness system.
SUBSTRATE_MATERIAL = "Si"
COATING_LAYERS_NM = [("Pt", 28.77)]
PERIOD_LPERMM = 400
WIDTH_TO_PERIOD_RATIO = 0.67
DEPTH_NM = 14.9
WALL_ANGLE_DEG = 15.0
X_RESOLUTION_NM = 0.1
Z_RESOLUTION_NM = 0.1

# ---------------------------------------------------------------------------

# Grating geometry passed to ``build_grating`` for every run. ``num_supercells``
# is set per run (see ``_simulation_runs``), not here.
GRATING_CONFIG = dict(
    substrate_material=SUBSTRATE_MATERIAL,
    coating_layers_nm=COATING_LAYERS_NM,
    period_lpermm=PERIOD_LPERMM,
    width_to_period_ratio=WIDTH_TO_PERIOD_RATIO,
    depth_nm=DEPTH_NM,
    wall_angle_deg=WALL_ANGLE_DEG,
    x_resolution_nm=X_RESOLUTION_NM,
    z_resolution_nm=Z_RESOLUTION_NM,
    roughness_seed=ROUGHNESS_SEED,
)


def _simulation_runs() -> list[RoughnessRun]:
    """Return the roughness runs used by the example."""
    return (
        [("baseline", 0.0, None, 1), ("debye-waller", DEBYE_SIGMA_NM, None, 1)]
        + [
            ("random-interface", RANDOM_INTERFACE_SIGMA_NM, correlation_length_nm, 1)
            for correlation_length_nm in RANDOM_INTERFACE_CORRELATION_LENGTHS_NM
        ]
        + [
            ("random-interface", RANDOM_INTERFACE_SIGMA_NM, correlation_length_nm, SUPERCELL_NUM_SUPERCELLS)
            for correlation_length_nm in RANDOM_INTERFACE_CORRELATION_LENGTHS_NM
        ]
    )


def _build_run_grating(roughness_kind: str, roughness_sigma_nm: float, correlation_length_nm: float | None, num_supercells: int):
    """Build the grating for one run in the correlation-length sweep."""
    return build_grating(
        roughness_kind,
        roughness_sigma_nm,
        roughness_correlation_length_nm=correlation_length_nm,
        roughness_num_supercells=num_supercells,
        **GRATING_CONFIG,
    )


def _save_all_grating_plots(runs: list[RoughnessRun]) -> None:
    """Save a whole-grating PDF for every run in the correlation-length sweep."""
    for roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells in runs:
        grating = _build_run_grating(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)
        output_path = save_grating_plot(
            grating,
            OUTPUT_DIR,
            roughness_kind=roughness_kind,
            roughness_sigma_nm=roughness_sigma_nm,
            correlation_length_nm=correlation_length_nm,
            num_supercells=num_supercells,
        )
        print(
            f"Saved grating plot for {run_label(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)} "
            f"to: {output_path}"
        )


def main() -> None:
    """Run the roughness correlation-length comparison example."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the roughness correlation-length comparison example.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only build the gratings and save whole-grating geometry PDFs; do not run simulations.",
    )
    args = parser.parse_args()

    runs = _simulation_runs()
    _save_all_grating_plots(runs)
    if args.geometry_only:
        return

    for roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells in runs:
        grating = _build_run_grating(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)
        cases = grax.fixed_angle_cases(
            grating=grating,
            energies_ev=ENERGIES_EV,
            grazing_angle_deg=GRAZING_ANGLE_DEG,
            polarization=POLARIZATION,
        )
        labeled_cases = (
            dict(
                case,
                label=run_label(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells),
            )
            for case in cases
        )

        fourier_orders = FOURIER_ORDERS if num_supercells == 1 else SUPERCELL_FOURIER_ORDERS
        runner = grax.BatchSimulationRunner(
            default_diffraction_order=DIFFRACTION_ORDER,
            default_fourier_orders=fourier_orders,
            show_progress=True,
            live_plot=True,
            live_plot_x_key="energy_ev",
            live_plot_order_count=1,
            on_error="continue",
            max_workers=MAX_WORKERS,
            backend=BACKEND,
        )
        print(f"\n{run_title(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)}")
        results = list(runner.run_cases(labeled_cases))

        run_csv_path = csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)
        grax.write_all_orders_csv(results, run_csv_path)
        print(
            f"Saved {run_label(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)} results to: "
            f"{run_csv_path} (max_workers={MAX_WORKERS})"
        )

    from comparison_roughness_correlation import plot_roughness_correlation_comparison

    comparison_plot_path = OUTPUT_DIR / "roughness_correlation_order1_comparison.png"
    plot_roughness_correlation_comparison(
        csv_paths=[
            csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)
            for roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells in runs
        ],
        output_path=comparison_plot_path,
    )
    print(f"Comparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    main()
