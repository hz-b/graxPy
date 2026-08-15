"""Compare blazed multilayer efficiencies from grax and DiffraMod."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Both solvers' results are plotted when present. A fresh run writes
# ``*_rcwa.csv`` / ``*_neviere.csv``; the unsuffixed CSV is the older checked-in
# artifact and is only used as a fallback, because it predates several solver
# changes and pairing it against a fresh run would show that drift as though it
# were a difference between the two methods.
SOLVER_LABELS = {"rcwa": "graxpy (RCWA)", "neviere": "graxpy (Nevière DM)"}
# The two solvers agree to ~1e-11, so without a dashed overlay the second curve
# hides the first completely and the plot looks as though one is missing.
SOLVER_STYLES = {"rcwa": {"linestyle": "-"}, "neviere": {"linestyle": (0, (6, 4))}}


def load_solver_curves(base_csv, order):
    """Return (label, energy, efficiency, style) for each solver that has been run.

    Args:
        base_csv: Historical unsuffixed all-orders CSV path for this case.
        order: Signed diffraction order to extract (reflected orders are negative).

    Returns:
        List of plottable curves, skipping solvers with no results yet.
    """

    curves = []
    for solver in ("rcwa", "neviere"):
        candidates = [base_csv.with_name(f"{base_csv.stem}_{solver}{base_csv.suffix}")]
        if solver == "rcwa":
            candidates.append(base_csv)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            print(f"  note: no {solver} results yet, skipping that curve")
            continue
        frame = pd.read_csv(path)
        selected = frame[frame["order"] == order].sort_values("energy_ev")
        if selected.empty:
            print(f"  note: {path.name} has no order {order}, skipping that curve")
            continue
        print(f"  {SOLVER_LABELS[solver]:<22} <- {path.name}  ({len(selected)} points)")
        curves.append(
            (
                SOLVER_LABELS[solver],
                selected["energy_ev"],
                selected["efficiency"],
                dict(SOLVER_STYLES[solver]),
            )
        )
    return curves



base_path = Path(__file__).resolve().parent
project_root = base_path.parent.parent
theta_search_results_path = project_root / "examples" / "simulation" / "multilayer_theta_search" / "results"

print("graxpy curves:")
grax_curves = load_solver_curves(
    base_path / "results" / "blazed_multilayer_all_orders.csv",
    order=-2,
)

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
for curve_label, curve_energy, curve_efficiency, curve_style in grax_curves:
    plt.plot(
        curve_energy,
        curve_efficiency,
        label=f"{curve_label} energy-angle",
        linewidth=1.0,
        marker=".",
        markersize=3,
        **curve_style,
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
for curve_label, curve_energy, curve_efficiency, curve_style in grax_curves:
    plt.plot(
        curve_energy,
        curve_efficiency,
        label=f"{curve_label} energy-angle",
        linewidth=1.0,
        marker=".",
        markersize=4,
        **curve_style,
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
