"""Plot fixed-angle order-1 roughness comparisons from saved CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _roughness_from_csv_path(path: Path) -> tuple[str, float]:
    """Return the roughness kind and sigma encoded in one CSV filename."""
    stem = path.stem
    prefix = "fixed_angle_roughness_"
    suffix = "_all_orders"
    if stem.startswith(prefix) and stem.endswith(suffix):
        roughness_tokens = stem[len(prefix) : -len(suffix)]
        roughness_kind, raw_value = roughness_tokens.split("_sigma_", maxsplit=1)
        return roughness_kind, float(raw_value.replace("p", "."))
    return stem, float("inf")


def _label_for_roughness(*, roughness_kind: str, roughness_sigma_nm: float) -> str:
    """Return the plot label for one roughness curve."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    pretty_kind = "Debye-Waller" if roughness_kind == "debye-waller" else "random-interface"
    return f"{pretty_kind} sigma={roughness_sigma_nm:.1f} nm"


def _color_for_sigma(roughness_sigma_nm: float) -> str:
    """Return the shared color for one sigma value."""
    if roughness_sigma_nm == 0.0:
        return "black"
    colors_by_sigma = {
        0.5: "tab:blue",
        1.0: "tab:orange",
        2.0: "tab:green",
    }
    return colors_by_sigma.get(roughness_sigma_nm, "tab:gray")


def _linestyle_for_kind(roughness_kind: str) -> str:
    """Return the line style for one roughness kind."""
    if roughness_kind == "baseline":
        return "-"
    return "--" if roughness_kind == "random-interface" else "-"


def _sorted_csv_paths(csv_paths: list[Path]) -> list[Path]:
    """Return CSV paths ordered by sigma, then Debye-Waller before random-interface."""
    kind_order = {"baseline": 0, "debye-waller": 1, "random-interface": 2}
    return sorted(
        csv_paths,
        key=lambda path: (
            _roughness_from_csv_path(path)[1],
            kind_order.get(_roughness_from_csv_path(path)[0], 99),
            path.name,
        ),
    )


def _load_first_order_series(csv_path: Path) -> tuple[list[float], list[float]]:
    """Load energy and first-order efficiency from one all-orders CSV file."""
    samples: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["order"]) != -1:
                continue
            samples.append((float(row["energy_ev"]), float(row["efficiency"])))
    samples.sort(key=lambda item: item[0])
    energies_ev = [energy_ev for energy_ev, _efficiency in samples]
    efficiencies = [efficiency for _energy_ev, efficiency in samples]
    return energies_ev, efficiencies


def plot_roughness_comparison(*, csv_paths: list[Path], output_path: Path) -> None:
    """Plot order-1 efficiency curves for all provided roughness CSVs."""
    figure, axis = plt.subplots(figsize=(10, 7))
    plotted_zero = False
    for csv_path in _sorted_csv_paths(csv_paths):
        roughness_kind, roughness_sigma_nm = _roughness_from_csv_path(csv_path)
        if roughness_sigma_nm == 0.0 and plotted_zero:
            continue
        energies_ev, efficiencies = _load_first_order_series(csv_path)
        axis.plot(
            energies_ev,
            efficiencies,
            linewidth=1.0,
            color=_color_for_sigma(roughness_sigma_nm),
            linestyle=_linestyle_for_kind(roughness_kind),
            label=_label_for_roughness(
                roughness_kind=roughness_kind,
                roughness_sigma_nm=roughness_sigma_nm,
            ),
        )
        if roughness_sigma_nm == 0.0:
            plotted_zero = True

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
    """Plot all roughness CSV outputs from the example results directory."""
    example_dir = Path(__file__).resolve().parent
    results_dir = example_dir / "results"
    csv_paths = sorted(results_dir.glob("fixed_angle_roughness_*_sigma_*_all_orders.csv"))
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
