"""Plot measurement, design simulation, and fitted simulation for blazed fitting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

example_root = Path(__file__).resolve().parent
results_dir = example_root / "results" / "blazed_fit"

measurement_path = example_root / "GR600-BEIChem_energy-Cff2.5.dat"
fitted_parameters_path = results_dir / "fitted_parameters.json"
initial_csv_path = results_dir / "simulated_curve_initial.csv"
fitted_csv_path = results_dir / "simulated_curve_fitted.csv"
comparison_plot_path = results_dir / "blazed_fit_measurement_comparison.png"

measurement = pd.read_csv(
    measurement_path,
    sep=r"\s+",
    header=None,
    names=["energy_ev", "efficiency"],
).apply(pd.to_numeric, errors="coerce").dropna()
initial_frame = pd.read_csv(initial_csv_path)
fitted_frame = pd.read_csv(fitted_csv_path)
fitted_parameters = json.loads(fitted_parameters_path.read_text(encoding="utf-8"))
evaluation_energies_ev = [float(v) for v in fitted_parameters.get("evaluation_energies_ev", [])]

initial_m1 = initial_frame[initial_frame["order"] == -1].sort_values("energy_ev")
fitted_m1 = fitted_frame[fitted_frame["order"] == -1].sort_values("energy_ev")

figure, axis = plt.subplots(figsize=(11, 7))
axis.plot(measurement["energy_ev"], measurement["efficiency"], "o", label="Measurement", markersize=2.5)
axis.plot(initial_m1["energy_ev"], initial_m1["efficiency"], "-", label="Initial design")
axis.plot(fitted_m1["energy_ev"], fitted_m1["efficiency"], "-", label="Fitted")
if evaluation_energies_ev:
    y_min, y_max = axis.get_ylim()
    axis.vlines(evaluation_energies_ev, y_min, y_max, colors="red", linestyles=":", linewidth=0.8, label="Optimization energies")
    axis.set_ylim(y_min, y_max)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency")
axis.set_title("Blazed fit: measurement vs initial vs fitted")
axis.grid(alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Comparison plot: {comparison_plot_path}")
