"""Run the fixed-angle simulation using fitted grating parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import grax as rp

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants" / "old"
results_dir = example_root / "results" / "laminar_fit"
results_dir.mkdir(parents=True, exist_ok=True)

fitted_parameters_path = results_dir / "fitted_parameters.json"
if not fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted parameters file: {fitted_parameters_path}. Run 0_fit_laminar_grating.py first."
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

measurement = pd.read_csv(
    example_root / "measured_alpha4deg_order1.csv",
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
energies = np.asarray(measurement["energy_ev"], dtype=float)

payload = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
fitted_grating_parameters = dict(payload["best_grating_parameters"])
fitted_grating_parameters["substrate_material"] = silicon
fitted_grating_parameters["layer_material"] = platinum
fitted_grating_parameters["top_cap_material"] = carbon
fitted_grating = rp.LaminarGrating(**fitted_grating_parameters)

cases = rp.fixed_angle_cases(
    grating=fitted_grating,
    energies_ev=energies,
    grazing_angle_deg=4.0,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=int(payload.get("fourier_orders", 15)),
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
