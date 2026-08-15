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

# =========================
# PLOT CONFIGURATION - Easy to adjust
# =========================
LINE_WIDTH = 2.0        # Thickness of all plot lines
MARKER_SIZE = 6         # Size of markers (if any)
FONTSIZE_LABELS = 16    # Font size for axis labels
FONTSIZE_TITLE = 20     # Font size for plot title
FONTSIZE_LEGEND = 14    # Font size for legend
FONTSIZE_TICKS = 14     # Font size for axis ticks
# =========================
# Paths
# =========================
example_root = Path(__file__).resolve().parent
measured_file = example_root / "measured_alpha4deg_order1.csv"
sim_file = example_root / "results" / "laminar_fixed_angle_all_orders.csv"

sim_dir = example_root / "Simulations"

diffmod_path = sim_dir / "Simulations_REFLEC_DiffMOD.txt"
reflec_path = sim_dir / "Simulations_REFLEC_REFLEC(SPECS-cont).txt"

reticolo_path = sim_dir / "SLAG_simulation_400lmm_alpha4.0deg_order1_sub_Si_layer_Pt_20260423_164155.csv"

# =========================
# Measured data
# =========================
df_meas = pd.read_csv(
    measured_file,
    sep=';',
    skiprows=3,
    decimal=',',
    names=["Energy_eV", "Intensity"]
).dropna()
# =========================
# graxpy simulations (-1 order), one curve per solver that has been run
# =========================
print("graxpy curves:")
grax_curves = load_solver_curves(sim_file, order=-1)

# =========================
# RETICOLO simulation
# =========================
df_slag = pd.read_csv(
    reticolo_path
).dropna()

# =========================
# DiffMod file 3 (coarse)
# =========================
df_f3 = pd.read_csv(
    diffmod_path,
    sep=r"\s+",
    engine="python",
)
# =========================
# REFLEC file 4 (fine)
# =========================
df_f4 = pd.read_csv(
    reflec_path,
    sep=r"\s+",
    skiprows=3,
    engine="python",
    names=[
        "Energy",
        "cff2.25_order1",
        "alpha2_order1",
        "alpha4_order1",
        "cff2.25_order2",
        "cff2.25_order3"
    ]
)
# =========================
# Plot
# =========================
plt.figure(figsize=(19.2, 14.4))

plt.plot(df_meas["Energy_eV"], df_meas["Intensity"],
         label="Measured data", linewidth=LINE_WIDTH)
for curve_label, curve_energy, curve_efficiency, curve_style in grax_curves:
    plt.plot(curve_energy, curve_efficiency,
             label=curve_label, linewidth=LINE_WIDTH, **curve_style)
plt.plot(df_slag["PhotonEnergy_eV"], df_slag["DiffractionEfficiency"],
         label="Reticolo", linewidth=LINE_WIDTH)
plt.plot(df_f3["Energy"], df_f3["efficiency_4deg"],
         label="DiffMod", linewidth=LINE_WIDTH)
plt.plot(df_f4["Energy"], df_f4["alpha4_order1"],
         label="REFLEC", linewidth=LINE_WIDTH)

plt.xlabel("Energy (eV)", fontsize=FONTSIZE_LABELS)
plt.ylabel("Efficiency / Intensity", fontsize=FONTSIZE_LABELS)
plt.title("Comparison To Other Codes", fontsize=FONTSIZE_TITLE, fontweight='bold')
plt.grid()
plt.legend(fontsize=FONTSIZE_LEGEND)

# Set tick label sizes
plt.xticks(fontsize=FONTSIZE_TICKS)
plt.yticks(fontsize=FONTSIZE_TICKS)
plt.savefig(example_root / "comparison_laminar_fixed_angle.png", dpi=300, bbox_inches='tight')
plt.close()
