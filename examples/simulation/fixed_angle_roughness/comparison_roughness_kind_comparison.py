"""Plot roughness-kind order-1 comparisons from saved CSV files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROUGHNESS_KINDS = {"baseline", "debye-waller", "random-interface"}
_FILENAME_RE = re.compile(
    r"^roughness_kind_comparison_(?P<kind>.+)_sigma_(?P<sigma>[^_]+)(?:_supercells_(?P<supercells>\d+))?_all_orders$"
)

_SUPERCELL_COLORS = {
    1: "tab:orange",
    3: "tab:blue",
    5: "tab:green",
    10: "tab:red",
}

# Supercell counts the comparison plot always attempts for random-interface
# roughness, independent of whatever ``RANDOM_INTERFACE_NUM_SUPERCELLS`` is
# currently set to in ``roughness_kind_comparison.py`` (that config only
# controls what a given run computes, not what this plot should show). Counts
# with no saved CSV yet are silently skipped.
RANDOM_INTERFACE_SUPERCELLS_TO_PLOT = [1, 3, 5, 10]


def _parse_csv_path(path: Path) -> tuple[str, float, int]:
    """Return the roughness kind, sigma, and supercell count encoded in one CSV filename."""
    match = _FILENAME_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Unable to parse roughness CSV filename: {path.name!r}")
    roughness_kind = match.group("kind")
    if roughness_kind not in ROUGHNESS_KINDS:
        raise ValueError(f"Unknown roughness kind in {path.name!r}: {roughness_kind!r}")
    roughness_sigma_nm = float(match.group("sigma").replace("p", "."))
    num_supercells = int(match.group("supercells")) if match.group("supercells") is not None else 1
    return roughness_kind, roughness_sigma_nm, num_supercells


def _label_for_roughness(*, roughness_kind: str, roughness_sigma_nm: float, num_supercells: int) -> str:
    """Return the plot label for one roughness curve."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    if roughness_kind == "debye-waller":
        return f"Debye-Waller sigma={roughness_sigma_nm:.1f} nm"
    return f"random-interface sigma={roughness_sigma_nm:.1f} nm, supercells={num_supercells}"


def _color_for_run(*, roughness_kind: str, num_supercells: int) -> str:
    """Return the plot color for one run."""
    if roughness_kind == "baseline":
        return "black"
    if roughness_kind == "debye-waller":
        return "gray"
    return _SUPERCELL_COLORS.get(num_supercells, "tab:brown")


def _linestyle_for_kind(roughness_kind: str) -> str:
    """Return the line style for one roughness kind."""
    if roughness_kind == "baseline":
        return "-"
    return "--" if roughness_kind == "random-interface" else "-"


def _sort_key(path: Path) -> tuple[int, int]:
    """Return a sort key ordering baseline, then Debye-Waller, then supercell counts ascending."""
    roughness_kind, _roughness_sigma_nm, num_supercells = _parse_csv_path(path)
    kind_order = {"baseline": 0, "debye-waller": 1, "random-interface": 2}
    return kind_order.get(roughness_kind, 99), num_supercells


def _current_roughness_csv_paths(results_dir: Path) -> list[Path]:
    """Return CSV outputs for the runs this comparison plot expects.

    Deferred import avoids a circular import (that script imports
    ``plot_roughness_comparison`` from this module). Baseline and Debye-Waller
    are supercell-independent, so those come from the script's current config
    (their sigma there is stable). Random-interface always tries every count
    in ``RANDOM_INTERFACE_SUPERCELLS_TO_PLOT`` (1, 3, 5, 10) regardless of what
    ``RANDOM_INTERFACE_NUM_SUPERCELLS`` currently computes -- a run that only
    swept one or two of those counts still gets its curves plotted, and counts
    with no saved CSV are silently skipped rather than raising.
    """
    import roughness_kind_comparison as _script

    expected_runs = {
        (roughness_kind, roughness_sigma_nm, num_supercells)
        for roughness_kind, roughness_sigma_nm, num_supercells in _script._simulation_runs(
            families=_script.FAMILIES
        )
        if roughness_kind != "random-interface"
    }
    if "random-interface" in _script.FAMILIES:
        expected_runs.update(
            ("random-interface", _script.RANDOM_INTERFACE_SIGMA_NM, num_supercells)
            for num_supercells in RANDOM_INTERFACE_SUPERCELLS_TO_PLOT
        )

    expected_paths = {
        _script.csv_path(results_dir, roughness_kind, roughness_sigma_nm, num_supercells)
        for roughness_kind, roughness_sigma_nm, num_supercells in expected_runs
    }
    return sorted((path for path in expected_paths if path.exists()), key=_sort_key)


def _load_first_order_series(csv_path: Path) -> tuple[list[float], list[float]]:
    """Load energy and first-order efficiency from one all-orders CSV file."""
    samples: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # order may be a fractional "physical order" string (e.g. "-0.9") when
            # the run used a supercell, so compare as float, not int.
            if abs(float(row["order"]) - (-1.0)) > 1e-6:
                continue
            samples.append((float(row["energy_ev"]), float(row["efficiency"])))
    samples.sort(key=lambda item: item[0])
    energies_ev = [energy_ev for energy_ev, _efficiency in samples]
    efficiencies = [efficiency for _energy_ev, efficiency in samples]
    return energies_ev, efficiencies


def plot_roughness_comparison(*, csv_paths: list[Path], output_path: Path) -> None:
    """Plot order-1 efficiency curves for all provided roughness CSVs."""
    figure, axis = plt.subplots(figsize=(10, 7))
    for csv_path in sorted(csv_paths, key=_sort_key):
        roughness_kind, roughness_sigma_nm, num_supercells = _parse_csv_path(csv_path)
        energies_ev, efficiencies = _load_first_order_series(csv_path)
        axis.plot(
            energies_ev,
            efficiencies,
            linewidth=1.0,
            color=_color_for_run(roughness_kind=roughness_kind, num_supercells=num_supercells),
            linestyle=_linestyle_for_kind(roughness_kind),
            label=_label_for_roughness(
                roughness_kind=roughness_kind,
                roughness_sigma_nm=roughness_sigma_nm,
                num_supercells=num_supercells,
            ),
        )

    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("First-Order Diffraction Efficiency")
    axis.set_title("Roughness-Kind Comparison")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Plot all roughness CSV outputs from the example results directory."""
    example_dir = Path(__file__).resolve().parent
    results_dir = example_dir / "results_roughness_kind_comparison"
    plots_dir = example_dir / "plots_roughness_kind_comparison"
    csv_paths = _current_roughness_csv_paths(results_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No roughness CSV outputs found in {results_dir}. Run roughness_kind_comparison.py first."
        )
    plot_roughness_comparison(
        csv_paths=csv_paths,
        output_path=plots_dir / "roughness_kind_comparison_order1_comparison.png",
    )


if __name__ == "__main__":
    main()
