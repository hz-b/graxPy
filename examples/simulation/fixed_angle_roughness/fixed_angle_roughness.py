"""Fixed-angle roughness comparison using live batch plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

import grax

matplotlib.use("TkAgg")

ROUGHNESS_LEVELS_NM = [0.0, 0.5, 1.0, 2.0]
ROUGHNESS_KINDS = ["debye-waller", "random-interface"]
MAX_WORKERS = "auto"
BASELINE_KIND = "baseline"


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


def _build_grating(roughness_kind: str, roughness_sigma_nm: float) -> grax.LaminarGrating:
    """Build the example grating with the selected roughness model."""
    roughness = None
    if roughness_sigma_nm > 0.0:
        roughness = grax.RoughnessSpec(
            kind=roughness_kind,
            sigma_nm=roughness_sigma_nm,
            seed=0,
        )
    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=28.77,
        x_resolution_nm=0.1,
        z_resolution_nm=0.1,
        roughness=roughness,
    )


output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)
grazing_angle_deg = 4.0
energies_ev = np.arange(50.0, 650.0, 2.0)

simulation_runs = [(BASELINE_KIND, 0.0)] + [
    (roughness_kind, roughness_sigma_nm)
    for roughness_kind in ROUGHNESS_KINDS
    for roughness_sigma_nm in ROUGHNESS_LEVELS_NM
    if roughness_sigma_nm > 0.0
]

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

from comparison_fixed_angle_roughness import plot_roughness_comparison  # noqa: E402

comparison_plot_path = output_dir / "fixed_angle_roughness_order1_comparison.png"
plot_roughness_comparison(
    csv_paths=[
        _csv_path(roughness_kind, roughness_sigma_nm) for roughness_kind, roughness_sigma_nm in simulation_runs
    ],
    output_path=comparison_plot_path,
)
print(f"Comparison plot saved to: {comparison_plot_path}")
