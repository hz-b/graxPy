"""Compare blazed multilayer efficiencies from grax and DiffraMod."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



base_path = Path(__file__).resolve().parent
project_root = base_path.parent.parent
theta_search_results_path = project_root / "examples" / "simulation" / "multilayer_theta_search" / "results"

grax_results = pd.read_csv(base_path / "results" / "blazed_multilayer_all_orders.csv")
grax_order = grax_results[grax_results["order"] == -2].copy()
grax_order = grax_order.sort_values("energy_ev")

theta_search_results = pd.read_csv(
    theta_search_results_path / "multilayer_theta_search_all_orders.csv"
)
theta_search_order = theta_search_results[theta_search_results["order"] == -2].copy()
theta_search_order = theta_search_order.sort_values("energy_ev")

diffmod_results = pd.read_csv(
    base_path / "simulation" / "DiffractMod_CrC_d4.8_N60_new.dat",
    sep=r"\s+",
    engine="python",
)
diffmod_results = diffmod_results[["Energy", "Efficiency(GR)"]].copy()
diffmod_results = diffmod_results.apply(pd.to_numeric, errors="coerce").dropna()

plt.figure(figsize=(10, 6))
plt.plot(
    grax_order["energy_ev"],
    grax_order["efficiency"],
    label="grax energy-angle",
    linewidth=1.0,
    linestyle=":",
)
plt.plot(
    theta_search_order["energy_ev"],
    theta_search_order["efficiency"],
    label="grax theta-search",
    linewidth=1.0,
    # marker="o",
    linestyle="--",
)
plt.plot(
    diffmod_results["Energy"],
    diffmod_results["Efficiency(GR)"],
    label="DiffraMod",
    linewidth=1.0,
    linestyle="-",
)

plt.xlabel("Energy (eV)")
plt.ylabel("Efficiency (2nd order)")
plt.title("Blazed Multilayer Comparison: 2nd Order")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = base_path / "comparison_blazed_multilayer_sweep.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}")

plt.figure(figsize=(10, 6))
plt.plot(
    grax_order["energy_ev"],
    grax_order["efficiency"],
    label="grax energy-angle",
    linewidth=1.0,
    linestyle="-",
)
plt.plot(
    theta_search_order["energy_ev"],
    theta_search_order["efficiency"],
    label="grax theta-search",
    linewidth=1.0,
    linestyle="--",
)
plt.plot(
    diffmod_results["Energy"],
    diffmod_results["Efficiency(GR)"],
    label="DiffraMod",
    linewidth=1.0,
    linestyle=":",
)

plt.xlim(550, 600)
plt.ylim(0.0, 0.4)
plt.xlabel("Energy (eV)")
plt.ylabel("Efficiency (2nd order)")
plt.title("Blazed Multilayer Comparison: 2nd Order (550-600 eV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

zoom_output_path = base_path / "comparison_blazed_multilayer_sweep_550_600eV.png"
plt.savefig(zoom_output_path, dpi=150, bbox_inches="tight")
print(f"Zoom plot saved to: {zoom_output_path}")
