"""Optimize a blazed grating against measured monochromator data."""

from __future__ import annotations

import json

import pandas as pd
from grax import BlazedGrating

from grax_opt import (
    optimize_to_measurements,
)
from example_config import (
    anti_blaze_angle_deg,
    optimizer_backend,
    optimizer_max_workers,
    batch_size,
    blaze_angle_deg,
    cff,
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    layer_thickness_nm,
    measurement_path,
    optical_constants_dir,
    period_lpermm,
    random_seed,
    results_dir,
    top_cap_thickness_nm,
    top_cap_material_name,
    total_trials,
    use_top_cap,
    x_resolution_nm,
    z_resolution_nm,
)

silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

gold = pd.read_csv(
    optical_constants_dir / "OC_Au_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"
results_dir.mkdir(parents=True, exist_ok=True)
top_cap_material = carbon if use_top_cap else None

def build_grating(parameters: dict[str, float]) -> BlazedGrating:
    """Build blazed grating from measurement-fit parameters."""

    return BlazedGrating(
        period_lpermm=period_lpermm,
        blaze_angle_deg=float(parameters["blaze_angle_deg"]),
        anti_blaze_angle_deg=float(parameters["anti_blaze_angle_deg"]),
        substrate_material=silicon,
        layer_material=gold,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=top_cap_material,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


spec = {
    "build_grating": build_grating,
    "parameter_bounds": {
        "blaze_angle_deg": (0.5, 1.2),
        "anti_blaze_angle_deg": (2.0, 10.0),
        "top_cap_thickness_nm": (0.3, 2.0),
    },
    "measurement_path": measurement_path,
    "output_dir": results_dir,
    "angle_mode": "cff",
    "cff": cff,
    "diffraction_order": diffraction_order,
    "fourier_orders": fourier_orders,
    "validate_physical_results": True,
    "total_trials": total_trials,
    "batch_size": batch_size,
    "random_seed": random_seed,
    "backend": optimizer_backend,
    "max_workers": optimizer_max_workers,
    "experiment_name": "blazed_fit",
    "save_best_fit_plot": True,
    "evaluation_energies_ev": list(evaluation_energies_ev),
}


def main() -> None:
    try:
        result = optimize_to_measurements(spec)
    except ImportError as error:
        print(error)
        print("Install the optional optimizer dependency first: `pip install .[opt]`.")
        raise SystemExit(1) from error

    fitted_parameters_path = results_dir / "fitted_parameters.json"
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    payload["result_json_path"] = str(result.result_json_path)
    payload["trial_history_csv_path"] = str(result.trial_history_csv_path)
    payload["best_fit_plot_path"] = (
        None if result.best_fit_plot_path is None else str(result.best_fit_plot_path)
    )
    payload["loss_history_plot_path"] = (
        None if result.loss_history_plot_path is None else str(result.loss_history_plot_path)
    )
    fitted_parameters_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Measurement: {result.measurement_path}")
    print(f"Optimizer backend request: {optimizer_backend}")
    print(f"Optimizer max_workers request: {optimizer_max_workers}")
    print(
        f"Baseline top-cap setting: use_top_cap={use_top_cap}, "
        f"material={top_cap_material_name if use_top_cap else 'None'}, "
        f"thickness_nm={top_cap_thickness_nm}"
    )
    print(f"Best loss: {result.best_loss:.6g}")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Completed trials: {result.completed_trials}")
    print(f"Stopped early: {result.stopped_early}")
    if result.early_stop_reason is not None:
        print(f"Early-stop reason: {result.early_stop_reason}")
    print(f"Fitted parameters JSON: {fitted_parameters_path}")
    print(f"Best result JSON: {result.result_json_path}")
    print(f"Trial history CSV: {result.trial_history_csv_path}")
    if result.best_fit_plot_path is not None:
        print(f"Best-fit plot: {result.best_fit_plot_path}")
    if result.loss_history_plot_path is not None:
        print(f"Loss-history plot: {result.loss_history_plot_path}")


if __name__ == "__main__":
    main()
