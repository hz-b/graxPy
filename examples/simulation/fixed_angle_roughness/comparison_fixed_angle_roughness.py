"""Plot fixed-angle order-1 roughness comparisons from saved CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _label_from_csv_path(path: Path) -> str:
    """Return the legend label encoded in one roughness CSV filename."""

    stem = path.stem
    prefix = "fixed_angle_roughness_sigma_"
    suffix = "_all_orders"
    if stem.startswith(prefix) and stem.endswith(suffix):
        raw_value = stem[len(prefix) : -len(suffix)].replace("p", ".")
        return f"sigma={float(raw_value):.1f} nm"
    return stem


def _load_first_order_series(csv_path: Path) -> tuple[list[float], list[float]]:
    """Load energy and first-order efficiency from one all-orders CSV file."""

    energies_ev: list[float] = []
    efficiencies: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["order"]) != -1:
                continue
            energies_ev.append(float(row["energy_ev"]))
            efficiencies.append(float(row["efficiency"]))
    return energies_ev, efficiencies


def plot_roughness_comparison(*, csv_paths: list[Path], output_path: Path) -> None:
    """Plot order-1 efficiency curves for all provided roughness CSVs."""

    figure, axis = plt.subplots(figsize=(10, 7))
    for csv_path in csv_paths:
        energies_ev, efficiencies = _load_first_order_series(csv_path)
        axis.plot(
            energies_ev,
            efficiencies,
            linewidth=1.0,
            label=_label_from_csv_path(csv_path),
        )

    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("First-Order Diffraction Efficiency")
    axis.set_title("Fixed-Angle Roughness Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    results_dir = example_dir / "results"
    csv_paths = sorted(results_dir.glob("fixed_angle_roughness_sigma_*_all_orders.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No roughness CSV outputs found in {results_dir}. Run fixed_angle_roughness.py first."
        )
    plot_roughness_comparison(
        csv_paths=csv_paths,
        output_path=results_dir / "fixed_angle_roughness_order1_comparison.png",
    )


if __name__ == "__main__":
    main()
