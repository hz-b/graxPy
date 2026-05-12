"""Plot measured, design-parameter, and fitted laminar simulations together."""

from __future__ import annotations

import os
from pathlib import Path

# os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "grax-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

LINE_WIDTH = 2.0
MARKER_SIZE = 1
FONTSIZE_LABELS = 16
FONTSIZE_TITLE = 20
FONTSIZE_LEGEND = 14
FONTSIZE_TICKS = 14

example_root = Path(__file__).resolve().parent
results_dir = example_root / "results"
fit_output_dir = results_dir / "laminar_discrete_fit_with_top_layer"
roughness_fit_output_dir = results_dir / "laminar_discrete_fit_with_roughness"
roughness_only_fit_output_dir = results_dir / "laminar_discrete_fit_roughness_only"
measured_path = example_root / "measured_alpha4deg_order1.csv"
design_simulation_path = results_dir / "laminar_fixed_angle_all_orders.csv"
fit_simulation_path = fit_output_dir / "simulated_curve.csv"
roughness_fit_simulation_path = roughness_fit_output_dir / "simulated_curve.csv"
roughness_only_fit_simulation_path = roughness_only_fit_output_dir / "simulated_curve.csv"
plot_path = results_dir / "laminar_design_fit_measurement_comparison.png"

if not design_simulation_path.exists():
    raise FileNotFoundError(
        f"Missing design simulation CSV: {design_simulation_path}. "
        "Run fixed_angle_sweep_design_param.py first."
    )
if not fit_simulation_path.exists():
    raise FileNotFoundError(
        f"Missing fitted simulation CSV: {fit_simulation_path}. "
        "Run run_simulation_fit_laminar_grating.py first."
    )

measured = pd.read_csv(
    measured_path,
    sep=";",
    skiprows=3,
    decimal=",",
    names=["energy_ev", "efficiency"],
).dropna()
design_simulation = pd.read_csv(design_simulation_path)
fit_simulation = pd.read_csv(fit_simulation_path)
roughness_fit_simulation = (
    pd.read_csv(roughness_fit_simulation_path)
    if roughness_fit_simulation_path.exists()
    else None
)
roughness_only_fit_simulation = (
    pd.read_csv(roughness_only_fit_simulation_path)
    if roughness_only_fit_simulation_path.exists()
    else None
)
def extract_m1_curve(frame: pd.DataFrame, path: Path) -> tuple[pd.Series, pd.Series]:
    if "order" not in frame.columns or "efficiency" not in frame.columns:
        raise KeyError(f"Missing long-format order/efficiency columns in {path}.")
    m1 = frame[frame["order"] == -1].sort_values("energy_ev")
    return m1["energy_ev"], m1["efficiency"]

design_energy, design_eff = extract_m1_curve(design_simulation, design_simulation_path)
fit_energy, fit_eff = extract_m1_curve(fit_simulation, fit_simulation_path)
roughness_fit_curve = (
    extract_m1_curve(roughness_fit_simulation, roughness_fit_simulation_path)
    if roughness_fit_simulation is not None
    else None
)
roughness_only_curve = (
    extract_m1_curve(roughness_only_fit_simulation, roughness_only_fit_simulation_path)
    if roughness_only_fit_simulation is not None
    else None
)

plt.figure(figsize=(19.2, 14.4))
plt.plot(
    measured["energy_ev"],
    measured["efficiency"],
    "o",
    label="Measured alpha=4 deg order 1",
    markersize=MARKER_SIZE,
)
plt.plot(
    design_energy,
    design_eff,
    label="Reticolopy design parameters (order -1)",
    linewidth=LINE_WIDTH,
)
plt.plot(
    fit_energy,
    fit_eff,
    label="Reticolopy fitted parameters (selected order 1)",
    linewidth=LINE_WIDTH,
)
if roughness_fit_curve is not None:
    plt.plot(
        roughness_fit_curve[0],
        roughness_fit_curve[1],
        label="Reticolopy fitted parameters + roughness",
        linewidth=LINE_WIDTH,
    )
if roughness_only_curve is not None:
    plt.plot(
        roughness_only_curve[0],
        roughness_only_curve[1],
        label="Reticolopy fitted geometry + roughness-only fit",
        linewidth=LINE_WIDTH,
    )

plt.xlabel("Energy (eV)", fontsize=FONTSIZE_LABELS)
plt.ylabel("Efficiency / Intensity", fontsize=FONTSIZE_LABELS)
plt.title("Laminar comparison: measured, design, and fitted", fontsize=FONTSIZE_TITLE)
plt.grid(alpha=0.3)
plt.legend(fontsize=FONTSIZE_LEGEND)
plt.xticks(fontsize=FONTSIZE_TICKS)
plt.yticks(fontsize=FONTSIZE_TICKS)
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Comparison plot saved to: {plot_path}")
