"""Run fixed-angle simulation using the measurement-fit example fitted parameters."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import grax as rp
from grax_opt import load_measurement_data
from example_config import (
    diffraction_order as baseline_diffraction_order,
    fourier_orders as baseline_fourier_orders,
    grazing_angle_deg,
    evaluation_energies_ev,
    measurement_path,
    optical_constants_dir,
    results_dir,
    simulation_backend as baseline_simulation_backend,
)

results_dir.mkdir(parents=True, exist_ok=True)

fitted_parameters_path = results_dir / "fitted_parameters.json"
if not fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted parameters file: {fitted_parameters_path}. "
        "Run 0_fit_dynamic_laminar_grating.py first."
    )

silicon = pd.read_csv(
    optical_constants_dir / "n_Si_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

platinum = pd.read_csv(
    optical_constants_dir / "n_Pt_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

measurement = load_measurement_data(measurement_path)
energies_ev = np.asarray(evaluation_energies_ev, dtype=float)
if energies_ev.size == 0:
    energies_ev = np.asarray(measurement.energy_ev, dtype=float)

payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
best_grating_parameters = dict(payload["best_grating_parameters"])
best_grating_parameters["substrate_material"] = silicon
best_grating_parameters["layer_material"] = platinum
best_grating_parameters["top_cap_material"] = carbon
fitted_grating = rp.LaminarGrating(**best_grating_parameters)

best_result_path = results_dir / "best_result.json"
best_result_payload = {}
if best_result_path.exists():
    best_result_payload = json.loads(best_result_path.read_text(encoding="utf-8"))

cases = rp.fixed_angle_cases(
    grating=fitted_grating,
    energies_ev=energies_ev,
    grazing_angle_deg=grazing_angle_deg,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=int(
        best_result_payload.get("diffraction_order", baseline_diffraction_order)
    ),
    default_fourier_orders=int(
        best_result_payload.get("fourier_orders", baseline_fourier_orders)
    ),
    max_workers="auto",
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend=str(best_result_payload.get("backend_effective", baseline_simulation_backend)),
)
results = list(runner.run_cases(cases))

output_csv_path = results_dir / "simulated_curve_fitted.csv"
rp.write_all_orders_csv(results, output_csv_path)

print(f"Fitted-parameter simulation CSV: {output_csv_path}")
print(
    "Fitted simulation settings: "
    f"grazing_angle_deg={grazing_angle_deg}, "
    f"fourier_orders={int(best_result_payload.get('fourier_orders', baseline_fourier_orders))}, "
    f"simulation_backend={str(best_result_payload.get('backend_effective', baseline_simulation_backend))}"
)
