"""Run simulation using fitted parameters for the blazed optimizer example."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import grax as rp

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
results_dir = example_root / "results" / "blazed_fit"
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

measurement = pd.read_csv(
    example_root / "GR600-BEIChem_energy-Cff2.5.dat",
    sep=r"\s+",
    header=None,
    names=["energy_ev", "efficiency"],
).apply(pd.to_numeric, errors="coerce").dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
fitted_grating_parameters = dict(payload["best_grating_parameters"])
fitted_grating_parameters["substrate_material"] = silicon
fitted_grating_parameters["layer_material"] = gold
fitted_grating = rp.BlazedGrating(**fitted_grating_parameters)

cases = rp.monochromator_cases(
    grating=fitted_grating,
    energies_ev=energies,
    period_lpermm=float(fitted_grating_parameters["period_lpermm"]),
    diffraction_order=int(payload.get("diffraction_order", 1)),
    cff=float(payload.get("cff", 2.5)),
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=int(payload.get("diffraction_order", 1)),
    default_fourier_orders=int(payload.get("fourier_orders", 20)),
    max_workers="auto",
    show_progress=True,
    live_plot=False,
    on_error="fail_fast",
    backend=str(payload.get("backend", "numba")),
)
results = list(runner.run_cases(cases))

output_csv_path = results_dir / "simulated_curve_fitted.csv"
rp.write_all_orders_csv(results, output_csv_path)
print(f"Fitted-parameter simulation CSV: {output_csv_path}")
