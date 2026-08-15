"""Compare laminar 150 l/mm monochromator results against external tables."""

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


example_root = Path(__file__).resolve().parent
results_file = example_root / "results" / "laminar_150lmm_monochromator_all_orders.csv"
simulation_dir = example_root / "simulation"
output_file = example_root / "comparison_laminar_150lmm_monochromator.png"

FILE_PLOT_OPTIONS: dict[str, bool] = {
    "RR-test-lGR150-gd60-nonSP-20_dm_de_r_m1_single.dat": True,
    "RR-test-lGR150-gd60-Energy_dm_de_r_m1_single.dat": False,
    "external_monochromator_reference.csv": False,
}

# Optional per-file subset of external columns to plot.
# Empty or missing entry means: plot all non-energy columns.
FILE_COLUMN_SELECTION: dict[str, list[str]] = {
    "RR-test-lGR150-gd60-Energy_dm_de_r_m1_single.dat": ["20_harm"],
}

if not results_file.exists():
    raise FileNotFoundError(
        f"Missing grax results file: {results_file}. "
        "Run laminar_150lmm_monochromator_sweep.py first."
    )
if not simulation_dir.exists():
    raise FileNotFoundError(f"Missing simulation directory: {simulation_dir}")


def _load_external_table(path: Path) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    """Load one external table and return energy series with cleaned efficiency series."""

    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path, on_bad_lines="skip")
    else:
        dataframe = pd.read_csv(path, sep=r"\s+", engine="python", on_bad_lines="skip")

    dataframe = dataframe.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    energy_candidates = [column for column in dataframe.columns if "energy" in column.lower()]
    if not energy_candidates:
        raise ValueError(f"No energy column found in {path.name}")
    energy_column = energy_candidates[0]

    dataframe[energy_column] = pd.to_numeric(dataframe[energy_column], errors="coerce")
    dataframe = dataframe.dropna(subset=[energy_column])
    energy_series = dataframe[energy_column]

    efficiency_columns = [column for column in dataframe.columns if column != energy_column]
    selected_columns = FILE_COLUMN_SELECTION.get(path.name, [])
    if selected_columns:
        efficiency_columns = [column for column in efficiency_columns if column in selected_columns]
    cleaned_series: list[tuple[str, pd.Series]] = []
    for column in efficiency_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        # Remove non-physical entries above 1 as requested.
        values = values.where(values <= 1.0)
        cleaned_series.append((column, values))

    return energy_series, cleaned_series


print("graxpy curves:")
grax_curves = load_solver_curves(results_file, order=-1)

external_files = sorted([path for path in simulation_dir.iterdir() if path.is_file()])
if not external_files:
    raise FileNotFoundError(f"No external files found in {simulation_dir}")

figure, axis = plt.subplots(figsize=(12, 8))
for curve_label, curve_energy, curve_efficiency, curve_style in grax_curves:
    axis.plot(
        curve_energy,
        curve_efficiency,
        label=f"{curve_label}, order -1 -> +1",
        linewidth=2.2,
        **curve_style,
    )

for external_path in external_files:
    if not FILE_PLOT_OPTIONS.get(external_path.name, False):
        continue
    energy, efficiency_series = _load_external_table(external_path)
    for column_name, values in efficiency_series:
        label = f"{external_path.stem}:{column_name}"
        axis.plot(
            energy,
            values,
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
            label=label,
        )

axis.set_xlabel("Energy (eV)")
axis.set_ylabel("Efficiency / Intensity")
axis.set_title("Laminar 150 l/mm Monochromator Comparison")
axis.grid(True, alpha=0.3)
axis.legend(loc="best", fontsize=8)
figure.tight_layout()
figure.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Comparison plot saved to: {output_file}")
