"""Compare grax and DiffractMod grating-angle trajectories versus energy."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base_path = Path(__file__).resolve().parent
project_root = base_path.parent.parent
simulation_path = base_path / "simulation"
theta_search_results_path = project_root / "examples" / "simulation" / "multilayer_theta_search" / "results"

grax_summary = pd.read_csv(theta_search_results_path / "multilayer_theta_search_summary.csv")
grax_angles = grax_summary[["energy_ev", "selected_grazing_angle_deg"]].copy()
grax_angles = grax_angles.rename(columns={"selected_grazing_angle_deg": "grax_angle_deg"})
grax_angles = grax_angles.apply(pd.to_numeric, errors="coerce").dropna().sort_values("energy_ev")

diffmod = pd.read_csv(
    simulation_path / "DiffractMod_CrC_d4.8_N60.dat",
    sep=r"\s+",
    engine="python",
)
diffmod_angles = diffmod[["Energy", "alpha"]].copy()
diffmod_angles = diffmod_angles.rename(columns={"Energy": "energy_ev", "alpha": "diffmod_angle_deg"})
diffmod_angles = diffmod_angles.apply(pd.to_numeric, errors="coerce").dropna().sort_values("energy_ev")

comparison = pd.merge(grax_angles, diffmod_angles, on="energy_ev", how="inner")
figure, axis = plt.subplots(figsize=(10, 6))

axis.plot(
    comparison["energy_ev"],
    comparison["grax_angle_deg"],
    "-",
    label="grax (theta-search)",
    linewidth=1.0,
)
axis.plot(
    comparison["energy_ev"],
    comparison["diffmod_angle_deg"],
    ":",
    label="DiffractMod (alpha)",
    linewidth=1.5,
)
axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Grating angle (deg)")
axis.set_title("Blazed Multilayer: Energy vs Grating Angle")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")

figure.tight_layout()

output_path = base_path / "comparison_blazed_multilayer_grating_angle.png"
figure.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(figure)
print(f"Plot saved to: {output_path}")
