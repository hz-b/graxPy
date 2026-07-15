"""Fixed-angle roughness comparison using live batch plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import grax
from grax.stacks import LayerSpec, assemble_custom_stack

matplotlib.use("TkAgg")

ROUGHNESS_LEVELS_NM = [0.0, 0.5, 1.0, 2.0]
ROUGHNESS_KINDS = ["debye-waller", "random-interface"]
MAX_WORKERS = "auto"
BASELINE_KIND = "baseline"

# Coating layers above the substrate, bottom-up. Each layer is assigned the same
# per-layer roughness so the example exercises the per-layer roughness system.
COATING_LAYERS_NM = [("Pt", 28.77)]


def _roughness_slug(roughness_sigma_nm: float) -> str:
    """Return a filename-safe roughness identifier."""
    return str(roughness_sigma_nm).replace(".", "p")


def _case_label(roughness_kind: str, roughness_sigma_nm: float) -> str:
    """Return a readable label for one roughness level."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    return f"{roughness_kind} sigma={roughness_sigma_nm:.1f} nm"


def _run_title(roughness_kind: str, roughness_sigma_nm: float) -> str:
    """Return a terminal title for one simulation run."""
    if roughness_sigma_nm == 0.0:
        return "Running sigma zero baseline with no roughness"
    if roughness_kind == "debye-waller":
        return f"Running Debye-Waller roughness with sigma={roughness_sigma_nm:.1f} nm"
    return f"Running random-interface roughness with sigma={roughness_sigma_nm:.1f} nm"


def _csv_path(roughness_kind: str, roughness_sigma_nm: float) -> Path:
    """Return the CSV path for one roughness run."""
    slug = _roughness_slug(roughness_sigma_nm)
    return output_dir / f"fixed_angle_roughness_{roughness_kind}_sigma_{slug}_all_orders.csv"


def _grating_plot_path(roughness_kind: str, roughness_sigma_nm: float) -> Path:
    """Return the PDF path for one grating geometry close-up."""
    slug = _roughness_slug(roughness_sigma_nm)
    return output_dir / f"fixed_angle_roughness_{roughness_kind}_sigma_{slug}_grating_closeup.pdf"


def _build_grating(roughness_kind: str, roughness_sigma_nm: float) -> grax.LaminarGrating:
    """Build the example grating with the selected per-layer roughness model."""
    # Assign the same roughness to the substrate boundary and to every coating
    # layer through the per-layer roughness system. The grating-level
    # ``RoughnessSpec`` only carries the kind (and seed); its ``sigma_nm`` stays
    # 0.0 so the per-interface values drive the roughness. ``substrate_roughness_sigma_nm``
    # roughens interface 0 (the substrate/coating boundary), which otherwise
    # falls back to the default and stays flat.
    stack = assemble_custom_stack(
        substrate_material="Si",
        substrate_roughness_sigma_nm=roughness_sigma_nm,
        layers_bottom_up=[
            LayerSpec(
                material=material,
                thickness_nm=thickness_nm,
                roughness_sigma_nm=roughness_sigma_nm,
            )
            for material, thickness_nm in COATING_LAYERS_NM
        ],
    )
    roughness = None
    if roughness_sigma_nm > 0.0:
        roughness = grax.RoughnessSpec(
            kind=roughness_kind,
            sigma_nm=0.0,
            seed=0,
        )
    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material="Si",
        coating_stack=stack,
        x_resolution_nm=0.1,
        z_resolution_nm=0.1,
        roughness=roughness,
    )


def _simulation_runs() -> list[tuple[str, float]]:
    """Return the roughness runs used by the example."""
    return [(BASELINE_KIND, 0.0)] + [
        (roughness_kind, roughness_sigma_nm)
        for roughness_kind in ROUGHNESS_KINDS
        for roughness_sigma_nm in ROUGHNESS_LEVELS_NM
        if roughness_sigma_nm > 0.0
    ]


def _save_grating_closeup_plot(
    grating: grax.LaminarGrating,
    *,
    roughness_kind: str,
    roughness_sigma_nm: float,
) -> Path:
    """Save a close-up PDF of the generated grating material geometry."""
    x_grid, z_grid, material_map, material_labels = grating._material_plot_data(num_periods=1)
    material_colors = grating._plot_material_colors(
        coating_stack=grating.resolved_stack(),
        material_labels=material_labels,
    )
    color_map = plt.matplotlib.colors.ListedColormap(material_colors)
    color_map.set_bad(color="#ffffff", alpha=1.0)
    masked_material_map = np.ma.masked_less(material_map, 0)

    period_nm = grating.period_nm
    ridge_edge_nm = period_nm * grating.width_to_period_ratio
    x_half_window_nm = 45.0
    x_min_nm = max(0.0, ridge_edge_nm - x_half_window_nm)
    x_max_nm = min(period_nm, ridge_edge_nm + x_half_window_nm)
    z_min_nm = max(0.0, float(np.min(z_grid)))
    z_max_nm = min(float(np.max(z_grid)), grating.profile_depth_nm() + grating.resolved_stack().total_thickness_nm + 3.0)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.imshow(
        masked_material_map,
        origin="lower",
        aspect="auto",
        extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
        cmap=color_map,
        interpolation="nearest",
        vmin=0,
        vmax=max(len(material_labels) - 1, 1),
    )
    legend_handles = [
        plt.matplotlib.patches.Patch(color=color_map.colors[index], label=label)
        for index, label in enumerate(material_labels)
    ]
    axis.set_xlim(x_min_nm, x_max_nm)
    axis.set_ylim(z_min_nm, z_max_nm)
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("z (nm)")
    axis.set_title(f"{_case_label(roughness_kind, roughness_sigma_nm)} grating close-up")
    axis.legend(handles=legend_handles, loc="upper right")
    axis.grid(False)

    output_path = _grating_plot_path(roughness_kind, roughness_sigma_nm)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_all_grating_closeup_plots(simulation_runs: list[tuple[str, float]]) -> None:
    """Save grating close-up PDFs for the geometry-distorting roughness runs.

    Only ``random-interface`` roughness changes the geometry, so Debye-Waller
    runs (whose close-ups would look identical to the baseline) are skipped.
    """
    for roughness_kind, roughness_sigma_nm in simulation_runs:
        if roughness_kind == "debye-waller":
            continue
        grating = _build_grating(roughness_kind, roughness_sigma_nm)
        output_path = _save_grating_closeup_plot(
            grating,
            roughness_kind=roughness_kind,
            roughness_sigma_nm=roughness_sigma_nm,
        )
        print(f"Saved grating close-up for {_case_label(roughness_kind, roughness_sigma_nm)} to: {output_path}")


output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)
grazing_angle_deg = 4.0
energies_ev = np.arange(50.0, 650.0, 10.0)


def main() -> None:
    """Run the roughness example or only write grating geometry previews."""
    parser = argparse.ArgumentParser(description="Run the fixed-angle roughness comparison example.")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only build the gratings and save close-up geometry PDFs; do not run simulations.",
    )
    args = parser.parse_args()

    simulation_runs = _simulation_runs()
    _save_all_grating_closeup_plots(simulation_runs)
    if args.geometry_only:
        return

    for roughness_kind, roughness_sigma_nm in simulation_runs:
        grating = _build_grating(roughness_kind, roughness_sigma_nm)
        cases = grax.fixed_angle_cases(
            grating=grating,
            energies_ev=energies_ev,
            grazing_angle_deg=grazing_angle_deg,
            polarization="p",
        )
        labeled_cases = (
            dict(
                case,
                label=_case_label(roughness_kind, roughness_sigma_nm),
            )
            for case in cases
        )

        runner = grax.BatchSimulationRunner(
            default_diffraction_order=1,
            default_fourier_orders=20,
            show_progress=True,
            live_plot=True,
            live_plot_x_key="energy_ev",
            live_plot_order_count=1,
            on_error="continue",
            max_workers=MAX_WORKERS,
            backend="numba",
        )
        print(f"\n{_run_title(roughness_kind, roughness_sigma_nm)}")
        results = list(runner.run_cases(labeled_cases))

        csv_path = _csv_path(roughness_kind, roughness_sigma_nm)
        grax.write_all_orders_csv(results, csv_path)
        print(
            f"Saved {_case_label(roughness_kind, roughness_sigma_nm)} results to: {csv_path} "
            f"(max_workers={MAX_WORKERS})"
        )

    from comparison_fixed_angle_roughness import plot_roughness_comparison

    comparison_plot_path = output_dir / "fixed_angle_roughness_order1_comparison.png"
    plot_roughness_comparison(
        csv_paths=[
            _csv_path(roughness_kind, roughness_sigma_nm) for roughness_kind, roughness_sigma_nm in simulation_runs
        ],
        output_path=comparison_plot_path,
    )
    print(f"Comparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    main()
