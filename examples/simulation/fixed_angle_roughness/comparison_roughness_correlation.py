"""Plot roughness correlation-length comparisons from saved CSV files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROUGHNESS_KINDS = {"baseline", "debye-waller", "random-interface"}
_FILENAME_RE = re.compile(
    r"^roughness_correlation_(?P<kind>.+)_sigma_(?P<sigma>[^_]+)_corr_(?P<corr>[^_]+)"
    r"(?:_supercells_(?P<supercells>\d+))?_all_orders$"
)

_CORRELATION_COLORS = {
    0.0: "tab:blue",
    1.0: "tab:orange",
    10.0: "tab:green",
    50.0: "tab:red",
    100.0: "tab:purple",
}


def _parse_csv_path(path: Path) -> tuple[str, float, float | None, int]:
    """Return the roughness kind, sigma, correlation length, and supercell count in one CSV filename."""
    match = _FILENAME_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Unable to parse roughness correlation CSV filename: {path.name!r}")
    roughness_kind = match.group("kind")
    if roughness_kind not in ROUGHNESS_KINDS:
        raise ValueError(f"Unknown roughness kind in {path.name!r}: {roughness_kind!r}")
    roughness_sigma_nm = float(match.group("sigma").replace("p", "."))
    corr_token = match.group("corr")
    correlation_length_nm = None if corr_token == "na" else float(corr_token.replace("p", "."))
    num_supercells = int(match.group("supercells")) if match.group("supercells") is not None else 1
    return roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells


def _label_for_run(
    *,
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int,
) -> str:
    """Return the plot label for one run."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    if roughness_kind == "debye-waller":
        return f"Debye-Waller sigma={roughness_sigma_nm:.1f} nm"
    label = f"random-interface correlation={correlation_length_nm:.0f} nm"
    if num_supercells != 1:
        label += f", supercells={num_supercells}"
    return label


def _color_for_run(*, roughness_kind: str, correlation_length_nm: float | None) -> str:
    """Return the plot color for one run."""
    if roughness_kind == "baseline":
        return "black"
    if roughness_kind == "debye-waller":
        return "gray"
    return _CORRELATION_COLORS.get(correlation_length_nm, "tab:brown")


def _linestyle_for_run(*, roughness_kind: str, num_supercells: int) -> str:
    """Return the line style for one run."""
    if roughness_kind != "random-interface":
        return "-"
    return ":" if num_supercells != 1 else "--"


def _sort_key(path: Path) -> tuple[int, float, int]:
    """Return a sort key ordering baseline, then Debye-Waller, then correlation lengths ascending, then supercells."""
    roughness_kind, _roughness_sigma_nm, correlation_length_nm, num_supercells = _parse_csv_path(path)
    kind_order = {"baseline": 0, "debye-waller": 1, "random-interface": 2}
    return kind_order.get(roughness_kind, 99), correlation_length_nm or -1.0, num_supercells


def _load_first_order_series(csv_path: Path) -> tuple[list[float], list[float]]:
    """Load energy and first-order efficiency from one all-orders CSV file."""
    samples: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # order may be a fractional "physical order" string (e.g. "-0.333333")
            # when the run used a supercell, so compare as float, not int.
            if abs(float(row["order"]) - (-1.0)) > 1e-6:
                continue
            samples.append((float(row["energy_ev"]), float(row["efficiency"])))
    samples.sort(key=lambda item: item[0])
    energies_ev = [energy_ev for energy_ev, _efficiency in samples]
    efficiencies = [efficiency for _energy_ev, efficiency in samples]
    return energies_ev, efficiencies


def plot_roughness_correlation_comparison(*, csv_paths: list[Path], output_path: Path) -> None:
    """Plot order-1 efficiency curves for all provided roughness-correlation CSVs."""
    figure, axis = plt.subplots(figsize=(10, 7))
    for csv_path in sorted(csv_paths, key=_sort_key):
        roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells = _parse_csv_path(csv_path)
        energies_ev, efficiencies = _load_first_order_series(csv_path)
        axis.plot(
            energies_ev,
            efficiencies,
            linewidth=1.0,
            color=_color_for_run(roughness_kind=roughness_kind, correlation_length_nm=correlation_length_nm),
            linestyle=_linestyle_for_run(roughness_kind=roughness_kind, num_supercells=num_supercells),
            label=_label_for_run(
                roughness_kind=roughness_kind,
                roughness_sigma_nm=roughness_sigma_nm,
                correlation_length_nm=correlation_length_nm,
                num_supercells=num_supercells,
            ),
        )

    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("First-Order Diffraction Efficiency")
    axis.set_title("Roughness Correlation-Length Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _current_csv_paths(results_dir: Path) -> list[Path]:
    """Return only CSV outputs for the roughness correlation-length comparison."""
    csv_paths: list[Path] = []
    for csv_path in results_dir.glob("roughness_correlation_*_all_orders.csv"):
        try:
            _parse_csv_path(csv_path)
        except ValueError:
            continue
        csv_paths.append(csv_path)
    return sorted(csv_paths, key=_sort_key)


def main() -> None:
    """Plot all roughness correlation-length CSV outputs from the example results directory."""
    example_dir = Path(__file__).resolve().parent
    results_dir = example_dir / "results_roughness_correlation"
    plots_dir = example_dir / "plots_roughness_correlation"
    csv_paths = _current_csv_paths(results_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No roughness correlation CSV outputs found in {results_dir}. Run roughness_correlation.py first."
        )
    plot_roughness_correlation_comparison(
        csv_paths=csv_paths,
        output_path=plots_dir / "roughness_correlation_order1_comparison.png",
    )


if __name__ == "__main__":
    main()
