"""Roughness-kind comparison (Debye-Waller vs random-interface) using live batch plotting.

Compares zero roughness, 1 nm Debye-Waller roughness, and 1 nm random-interface
roughness at a few supercell counts (the random-interface field spanning 1, 5,
or 10 grating periods as one continuous correlated field). Each random-interface
point is itself an average over several independent roughness realizations
(``ROUGHNESS_NUM_REALIZATIONS``), so a full run is noticeably more expensive
than a single solve per point -- supercells and realizations both multiply cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import grax
from helpers_order_spectrum import closest_case, save_order_spectrum_plot
from helpers_roughness_kind_comparison import (
    RoughnessRun,
    build_grating,
    case_label,
    csv_path,
    order_spectrum_plot_path,
    run_title,
    save_grating_plot,
)

matplotlib.use("TkAgg")

# ---------------------------------------------------------------------------
# Variables you may want to change manually
# ---------------------------------------------------------------------------

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "results_roughness_kind_comparison"
PLOTS_DIR = Path(__file__).resolve().parent / "plots_roughness_kind_comparison"

# Photon energy at which the per-run diffraction-order spectrum plots are
# made. ``None`` picks the energy closest to the middle of ``ENERGIES_EV``.
CENTRAL_ENERGY_EV: float | None = None

# Roughness runs. Baseline and Debye-Waller are supercell-independent, so
# they are each run once; random-interface is swept over supercell counts at
# a fixed sigma.
DEBYE_SIGMA_NM = 1.0
RANDOM_INTERFACE_SIGMA_NM = 1.0
RANDOM_INTERFACE_NUM_SUPERCELLS = [10] # [1, 5, 10]
# Base seed for the random-interface roughness ensemble. ``None`` draws real
# entropy (a genuinely random surface each run, so results are not
# reproducible run to run); set an explicit int to reproduce one specific
# ensemble for debugging.
ROUGHNESS_SEED: int | None = None
# Number of independent roughness realizations averaged per simulated point
# (see ``RoughnessSpec.num_realizations``). The library default (8) is
# usually fine as-is; exposed here for visibility/override.
ROUGHNESS_NUM_REALIZATIONS = 8
# Lateral autocorrelation length of the "random-interface" roughness, in nm.
# ``None`` defaults to one tenth of the grating period; ``0.0`` gives an
# uncorrelated (white-noise) interface.
ROUGHNESS_CORRELATION_LENGTH_NM: float | None = 10

# Simulation settings
GRAZING_ANGLE_DEG = 1.0
ENERGIES_EV = np.arange(50.0, 2200.0, 50.0)
POLARIZATION = "p"
DIFFRACTION_ORDER = 1
FOURIER_ORDERS = 20
# Fourier orders used for the supercell runs (num_supercells > 1). Solver
# cost scales with fourier_orders * num_supercells (and the solver hard-caps
# around 100 effective orders), so this is set low enough that even the
# largest supercell count here stays well within that limit.
SUPERCELL_FOURIER_ORDERS = 20
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
    roughness_correlation_length_nm=ROUGHNESS_CORRELATION_LENGTH_NM,
    roughness_num_realizations=ROUGHNESS_NUM_REALIZATIONS,
)


FAMILIES = ["baseline", "debye-waller", "random-interface"]


def _simulation_runs(*, families: list[str]) -> list[RoughnessRun]:
    """Return the roughness runs used by the example, filtered to ``families``."""
    all_runs: list[RoughnessRun] = (
        [("baseline", 0.0, 1), ("debye-waller", DEBYE_SIGMA_NM, 1)]
        + [
            ("random-interface", RANDOM_INTERFACE_SIGMA_NM, num_supercells)
            for num_supercells in RANDOM_INTERFACE_NUM_SUPERCELLS
        ]
    )
    return [run for run in all_runs if run[0] in families]


def _build_run_grating(roughness_kind: str, roughness_sigma_nm: float, num_supercells: int):
    """Build the grating for one run in the roughness-kind sweep."""
    return build_grating(
        roughness_kind,
        roughness_sigma_nm,
        roughness_num_supercells=num_supercells,
        **GRATING_CONFIG,
    )


def _save_all_grating_plots(runs: list[RoughnessRun]) -> None:
    """Save whole-grating PDFs of the example geometries.

    Debye-Waller roughness does not distort the geometry, so a single Debye-Waller
    grating is exported once. ``random-interface`` roughness changes the geometry,
    so one grating is exported per supercell count.
    """
    for roughness_kind, roughness_sigma_nm, num_supercells in runs:
        grating = _build_run_grating(roughness_kind, roughness_sigma_nm, num_supercells)
        output_path = save_grating_plot(
            grating,
            PLOTS_DIR,
            roughness_kind=roughness_kind,
            roughness_sigma_nm=roughness_sigma_nm,
            num_supercells=num_supercells,
        )
        print(
            f"Saved grating plot for {case_label(roughness_kind, roughness_sigma_nm, num_supercells)} "
            f"to: {output_path}"
        )


def main() -> None:
    """Run the roughness example or only write grating geometry previews."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the roughness-kind comparison example.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only build the gratings and save whole-grating geometry PDFs; do not run simulations.",
    )
    parser.add_argument(
        "--family",
        choices=["all", *FAMILIES],
        default="all",
        help="Only run this roughness family (baseline, debye-waller, or random-interface). "
        "Default: run all families.",
    )
    args = parser.parse_args()
    families = FAMILIES if args.family == "all" else [args.family]

    runs = _simulation_runs(families=families)
    _save_all_grating_plots(runs)
    if args.geometry_only:
        return

    for roughness_kind, roughness_sigma_nm, num_supercells in runs:
        grating = _build_run_grating(roughness_kind, roughness_sigma_nm, num_supercells)
        cases = grax.fixed_angle_cases(
            grating=grating,
            energies_ev=ENERGIES_EV,
            grazing_angle_deg=GRAZING_ANGLE_DEG,
            polarization=POLARIZATION,
        )
        labeled_cases = (
            dict(
                case,
                label=case_label(roughness_kind, roughness_sigma_nm, num_supercells),
            )
            for case in cases
        )

        fourier_orders = FOURIER_ORDERS if num_supercells == 1 else SUPERCELL_FOURIER_ORDERS
        runner = grax.BatchSimulationRunner(
            diffraction_order=DIFFRACTION_ORDER,
            fourier_orders=fourier_orders,
            show_progress=True,
            live_plot=False,
            on_error="continue",
            max_workers=MAX_WORKERS,
            backend=BACKEND,
        )
        print(f"\n{run_title(roughness_kind, roughness_sigma_nm, num_supercells)}")
        results = list(runner.run_cases(labeled_cases))

        run_csv_path = csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm, num_supercells)
        grax.write_all_orders_csv(results, run_csv_path)
        print(
            f"Saved {case_label(roughness_kind, roughness_sigma_nm, num_supercells)} results to: "
            f"{run_csv_path} (max_workers={MAX_WORKERS})"
        )

        central_energy_ev = (
            CENTRAL_ENERGY_EV if CENTRAL_ENERGY_EV is not None else float(ENERGIES_EV[len(ENERGIES_EV) // 2])
        )
        central_case = closest_case(results, central_energy_ev)
        spectrum_plot_path = order_spectrum_plot_path(PLOTS_DIR, roughness_kind, roughness_sigma_nm, num_supercells)
        save_order_spectrum_plot(
            central_case.orders,
            central_case.efficiency_all,
            energy_ev=central_case.energy_ev,
            title=case_label(roughness_kind, roughness_sigma_nm, num_supercells),
            output_path=spectrum_plot_path,
        )
        print(f"Saved order-spectrum plot at {central_case.energy_ev:.0f} eV to: {spectrum_plot_path}")

    from comparison_roughness_kind_comparison import plot_roughness_comparison

    # Only plot CSVs that correspond to a run this script currently defines
    # (all families, so running one family at a time still builds up a
    # complete comparison once all of them have been run) -- not any other
    # CSV that happens to sit in the results directory from an older config
    # (different sigma, a supercell count no longer swept, etc.).
    expected_csv_paths = [
        csv_path(OUTPUT_DIR, roughness_kind, roughness_sigma_nm, num_supercells)
        for roughness_kind, roughness_sigma_nm, num_supercells in _simulation_runs(families=FAMILIES)
    ]
    comparison_plot_path = PLOTS_DIR / "roughness_kind_comparison_order1_comparison.png"
    plot_roughness_comparison(
        csv_paths=[path for path in expected_csv_paths if path.exists()],
        output_path=comparison_plot_path,
    )
    print(f"Comparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    main()
