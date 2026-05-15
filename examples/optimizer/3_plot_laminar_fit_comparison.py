"""Plot measurement, design-parameter simulation, and fitted simulation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

example_root = Path(__file__).resolve().parent
results_dir = example_root / "results" / "laminar_fit"
results_dir.mkdir(parents=True, exist_ok=True)

measurement_path = example_root / "measured_alpha4deg_order1.csv"
fitted_parameters_path = results_dir / "fitted_parameters.json"
initial_csv_path = results_dir / "simulated_curve_initial.csv"
fitted_csv_path = results_dir / "simulated_curve_fitted.csv"
comparison_plot_path = results_dir / "laminar_fit_measurement_comparison.png"

if not fitted_parameters_path.exists():
    raise FileNotFoundError(
        f"Missing fitted parameters file: {fitted_parameters_path}. "
        "Run 0_fit_laminar_grating.py first."
    )
if not initial_csv_path.exists():
    raise FileNotFoundError(
        f"Missing initial simulation CSV: {initial_csv_path}. "
        "Run 1_run_simulation_design_parameters.py first."
    )
if not fitted_csv_path.exists():
    raise FileNotFoundError(
        f"Missing fitted simulation CSV: {fitted_csv_path}. "
        "Run 2_run_simulation_fitted_parameters.py first."
    )

measurement = pd.read_csv(
    measurement_path,
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
initial_frame = pd.read_csv(initial_csv_path)
fitted_frame = pd.read_csv(fitted_csv_path)
fitted_parameters = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
evaluation_energies_ev = [
    float(energy_ev) for energy_ev in fitted_parameters.get("evaluation_energies_ev", [])
]

initial_m1 = initial_frame[initial_frame["order"] == -1].sort_values("energy_ev")
fitted_m1 = fitted_frame[fitted_frame["order"] == -1].sort_values("energy_ev")

figure, axis = plt.subplots(figsize=(11, 7))
axis.plot(measurement["energy_ev"], measurement["efficiency"], "o", label="Measurement", markersize=3.0)
axis.plot(initial_m1["energy_ev"], initial_m1["efficiency"], "-", label="Initial design")
axis.plot(fitted_m1["energy_ev"], fitted_m1["efficiency"], "-", label="Fitted")
if evaluation_energies_ev:
    y_min, y_max = axis.get_ylim()
    axis.vlines(
        evaluation_energies_ev,
        y_min,
        y_max,
        colors="0.7",
        linestyles=":",
        linewidth=0.8,
        label="Optimization energies",
    )
    axis.set_ylim(y_min, y_max)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency")
axis.set_title("Laminar fit: measurement vs initial vs fitted")
axis.grid(alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Comparison plot: {comparison_plot_path}")
