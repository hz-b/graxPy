"""Run simulation using fitted parameters for the blazed optimizer example."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import grax
from example_config import (
    anti_blaze_angle_deg as baseline_anti_blaze_angle_deg,
    blaze_angle_deg as baseline_blaze_angle_deg,
    cff as baseline_cff,
    diffraction_order as baseline_diffraction_order,
    fourier_orders as baseline_fourier_orders,
    layer_thickness_nm as baseline_layer_thickness_nm,
    measurement_path,
    optical_constants_dir,
    period_lpermm as baseline_period_lpermm,
    results_dir,
    simulation_backend as baseline_simulation_backend,
    top_cap_thickness_nm as baseline_top_cap_thickness_nm,
    use_top_cap,
    x_resolution_nm as baseline_x_resolution_nm,
    z_resolution_nm as baseline_z_resolution_nm,
)


def main() -> None:

    results_dir.mkdir(parents=True, exist_ok=True)

    fitted_parameters_path = results_dir / "fitted_parameters.json"
    if not fitted_parameters_path.exists():
        raise FileNotFoundError(
            f"Missing fitted parameters file: {fitted_parameters_path}. Run 0_fit_blazed_grating.py first."
        )

    silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    silicon.attrs["name"] = "Si"
    gold = pd.read_csv(optical_constants_dir / "OC_Au_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
    gold.attrs["name"] = "Au"
    carbon = pd.read_csv(
        optical_constants_dir / "n_C_cxro.txt",
        skiprows=1,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    carbon.attrs["name"] = "C"

    measurement = pd.read_csv(
        measurement_path,
        sep=r"\s+",
        header=None,
        names=["energy_ev", "efficiency"],
    ).apply(pd.to_numeric, errors="coerce").dropna()
    energies = np.asarray(measurement["energy_ev"], dtype=float)

    payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
    fitted_grating_parameters = {
        "period_lpermm": float(baseline_period_lpermm),
        "blaze_angle_deg": float(baseline_blaze_angle_deg),
        "anti_blaze_angle_deg": float(baseline_anti_blaze_angle_deg),
        "layer_thickness_nm": float(baseline_layer_thickness_nm),
        "top_cap_thickness_nm": float(baseline_top_cap_thickness_nm),
        "x_resolution_nm": float(baseline_x_resolution_nm),
        "z_resolution_nm": float(baseline_z_resolution_nm),
    }
    fitted_grating_parameters.update(dict(payload.get("best_grating_parameters", {})))
    fitted_grating_parameters["substrate_material"] = silicon
    fitted_grating_parameters["layer_material"] = gold
    fitted_grating_parameters["top_cap_material"] = carbon if use_top_cap else None
    fitted_grating = grax.BlazedGrating(**fitted_grating_parameters)

    cases = grax.monochromator_cases(
        grating=fitted_grating,
        energies_ev=energies,
        period_lpermm=float(fitted_grating_parameters["period_lpermm"]),
        diffraction_order=int(payload.get("diffraction_order", baseline_diffraction_order)),
        cff=float(payload.get("cff", baseline_cff)),
        polarization="p",
    )

    runner = grax.BatchSimulationRunner(
        diffraction_order=int(payload.get("diffraction_order", baseline_diffraction_order)),
        fourier_orders=int(payload.get("fourier_orders", baseline_fourier_orders)),
        max_workers="auto",
        show_progress=True,
        live_plot=False,
        on_error="fail_fast",
        backend="numba",
    )
    results = list(runner.run_cases(cases))

    output_csv_path = results_dir / "simulated_curve_fitted.csv"
    grax.write_all_orders_csv(results, output_csv_path)
    print(f"Fitted-parameter simulation CSV: {output_csv_path}")
    print(
        "Fitted simulation settings: "
        f"cff={float(payload.get('cff', baseline_cff))}, "
        f"fourier_orders={int(payload.get('fourier_orders', baseline_fourier_orders))}, "
        "simulation_backend=numba, "
        f"x_resolution_nm={fitted_grating.x_resolution_nm}, "
        f"z_resolution_nm={fitted_grating.z_resolution_nm}, "
        f"top_cap_material={'C' if use_top_cap else 'None'}, "
        "top_cap_thickness_source=best_grating_parameters"
    )
    print(
        "Baseline defaults (used only if payload fields are missing): "
        f"x_resolution_nm={baseline_x_resolution_nm}, z_resolution_nm={baseline_z_resolution_nm}"
    )


if __name__ == "__main__":
    main()
