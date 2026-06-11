"""Compare laminar 150 l/mm monochromator results against external tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


df_grax = pd.read_csv(results_file)
df_grax_m1 = df_grax[df_grax["order"] == -1].copy().sort_values("energy_ev")

external_files = sorted([path for path in simulation_dir.iterdir() if path.is_file()])
if not external_files:
    raise FileNotFoundError(f"No external files found in {simulation_dir}")

figure, axis = plt.subplots(figsize=(12, 8))
axis.plot(
    df_grax_m1["energy_ev"],
    df_grax_m1["efficiency"],
    label="grax (order -1 -> +1)",
    linewidth=2.2,
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
