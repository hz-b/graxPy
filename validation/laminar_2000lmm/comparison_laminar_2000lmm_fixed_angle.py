"""Compare fixed-angle 2000 l/mm simulations against reference data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


example_root = Path(__file__).resolve().parent
results_dir = example_root / "results"
simulation_dir = example_root / "simulation"
output_file = results_dir / "comparison_laminar_2000lmm_fixed_angle.png"
comparison_order = -1

CASE_CONFIG = {
    1: {
        "results_file": results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_all_orders.csv",
        "simulation_file": simulation_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-1deg.dat",
    },
    2: {
        "results_file": results_dir / "laminar_2000lmm_fixed_angle_alpha2deg_all_orders.csv",
        "simulation_file": simulation_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-2deg.dat",
    },
    4: {
        "results_file": results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_all_orders.csv",
        "simulation_file": simulation_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-4deg.dat",
    },
}

missing_results = [str(config["results_file"]) for config in CASE_CONFIG.values() if not config["results_file"].exists()]
if missing_results:
    raise FileNotFoundError(
        "Missing grax results file(s). Run the fixed-angle simulations first:\n"
        + "\n".join(missing_results)
    )

figure, axes = plt.subplots(3, 1, figsize=(11, 14), sharex=False)

for axis, (angle_deg, config) in zip(axes, CASE_CONFIG.items(), strict=True):
    df_grax = pd.read_csv(config["results_file"])
    df_grax = df_grax[df_grax["order"] == comparison_order].copy().sort_values("energy_ev")
    df_reference = pd.read_csv(
        config["simulation_file"],
        sep=r"\s+",
        engine="python",
        header=None,
        names=["energy_ev", "efficiency"],
    )

    axis.plot(
        df_grax["energy_ev"],
        df_grax["efficiency"],
        linewidth=1.8,
        label=f"grax order {comparison_order}",
    )
    axis.plot(
        df_reference["energy_ev"],
        df_reference["efficiency"],
        linestyle="--",
        linewidth=1.4,
        label=config["simulation_file"].name,
    )
    axis.set_title(f"Laminar 2000 l/mm Fixed-Angle Comparison at Alpha = {angle_deg} deg")
    axis.set_xlabel("Energy (eV)")
    axis.set_ylabel("Efficiency")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

figure.tight_layout()
figure.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Combined comparison plot saved to: {output_file}")
