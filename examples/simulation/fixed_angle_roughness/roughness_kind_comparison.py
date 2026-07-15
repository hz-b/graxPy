"""Roughness-kind comparison (Debye-Waller vs random-interface) using live batch plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import grax
from helpers_roughness_kind_comparison import (
    build_grating,
    case_label,
    csv_path,
    run_title,
    save_grating_plot,
    simulation_runs,
)

matplotlib.use("TkAgg")

# ---------------------------------------------------------------------------
# Variables you may want to change manually
# ---------------------------------------------------------------------------

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "results_roughness_kind_comparison"

# Roughness sweep
ROUGHNESS_LEVELS_NM = [0.0, 0.5, 1.0, 2.0]
ROUGHNESS_KINDS = ["debye-waller", "random-interface"]
BASELINE_KIND = "baseline"
ROUGHNESS_SEED = 0
# Lateral autocorrelation length of the "random-interface" roughness, in nm.
# ``None`` defaults to one tenth of the grating period; ``0.0`` gives an
# uncorrelated (white-noise) interface.
ROUGHNESS_CORRELATION_LENGTH_NM: float | None = 10
# Number of grating periods the "random-interface" roughness field spans as
# one continuous correlated field. 1 keeps today's single-period behavior;
# only meaningful for "random-interface" (ignored for "debye-waller").
ROUGHNESS_NUM_SUPERCELLS = 1

# Simulation settings
GRAZING_ANGLE_DEG = 1.0
ENERGIES_EV = np.arange(50.0, 2200.0, 50.0)
POLARIZATION = "p"
DIFFRACTION_ORDER = 1
FOURIER_ORDERS = 20
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

# Grating geometry passed to ``build_grating`` for every run.
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
    roughness_correlation_length_nm=ROUGHNESS_CORRELATION_LENGTH_NM,
    roughness_num_supercells=ROUGHNESS_NUM_SUPERCELLS,
)


def _save_all_grating_plots(runs: list[tuple[str, float]]) -> None:
    """Save whole-grating PDFs of the example geometries.

    Debye-Waller roughness does not distort the geometry, so a single Debye-Waller
    grating is exported once. ``random-interface`` roughness changes the geometry,
    so one grating is exported per interface-roughness level.
    """
    debye_done = False
    for roughness_kind, roughness_sigma_nm in runs:
        if roughness_kind == "debye-waller":
            if debye_done:
                continue
            debye_done = True
        grating = build_grating(roughness_kind, roughness_sigma_nm, **GRATING_CONFIG)
        output_path = save_grating_plot(
            grating,
            OUTPUT_DIR,
            roughness_kind=roughness_kind,
            roughness_sigma_nm=roughness_sigma_nm,
        )
        print(f"Saved grating plot for {case_label(roughness_kind, roughness_sigma_nm)} to: {output_path}")


def main() -> None:
    """Run the roughness example or only write grating geometry previews."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the roughness-kind comparison example.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only build the gratings and save whole-grating geometry PDFs; do not run simulations.",
    )
    args = parser.parse_args()

    runs = simulation_runs(
        roughness_kinds=ROUGHNESS_KINDS,
        roughness_levels_nm=ROUGHNESS_LEVELS_NM,
        baseline_kind=BASELINE_KIND,
    )
    _save_all_grating_plots(runs)
    if args.geometry_only:
        return

    for roughness_kind, roughness_sigma_nm in runs:
        grating = build_grating(roughness_kind, roughness_sigma_nm, **GRATING_CONFIG)
        cases = grax.fixed_angle_cases(
            grating=grating,
            energies_ev=ENERGIES_EV,
            grazing_angle_deg=GRAZING_ANGLE_DEG,
            polarization=POLARIZATION,
        )
        labeled_cases = (
            dict(
                case,
                label=case_label(roughness_kind, roughness_sigma_nm),
            )
            for case in cases
        )

        runner = grax.BatchSimulationRunner(
            default_diffraction_order=DIFFRACTION_ORDER,
            default_fourier_orders=FOURIER_ORDERS,
            show_progress=True,
            live_plot=True,
            live_plot_x_key="energy_ev",
            live_plot_order_count=1,
            on_error="continue",
            max_workers=MAX_WORKERS,
            backend=BACKEND,
        )
        print(f"\n{run_title(roughness_kind, roughness_sigma_nm)}")
        results = list(runner.run_cases(labeled_cases))

        run_csv_path = csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm)
        grax.write_all_orders_csv(results, run_csv_path)
        print(
            f"Saved {case_label(roughness_kind, roughness_sigma_nm)} results to: {run_csv_path} "
            f"(max_workers={MAX_WORKERS})"
        )

    from comparison_roughness_kind_comparison import plot_roughness_comparison

    comparison_plot_path = OUTPUT_DIR / "roughness_kind_comparison_order1_comparison.png"
    plot_roughness_comparison(
        csv_paths=[csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm) for roughness_kind, roughness_sigma_nm in runs],
        output_path=comparison_plot_path,
    )
    print(f"Comparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    main()
