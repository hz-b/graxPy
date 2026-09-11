"""Step 2: sweep chosen (d, blaze) designs over energy.

Runs :meth:`grax.MultilayerGratingDesigner.run_energy_scan` for one or more
``(d_spacing_nm, blaze_angle_deg)`` designs read from ``results/survey/survey.csv``:

* default -- one scan per d-spacing, at that d's optimal blaze angle;
* ``--best`` -- only the single ``(d, blaze)`` with the highest survey efficiency;
* ``--pairs`` -- exactly the designs you list.

Besides the standard sweep artifacts under ``results/energy_scan/``, each
design gets a plot in the shared ``results/plots/`` folder (alongside
``0_run_survey.py``'s headline plots): selected efficiency versus energy,
titled with the coating (``CONFIG.coating_label``, "Ru/B4C" here), the
diffraction order and this design's d-spacing and blaze angle, named
``efficiency_vs_energy_<materials>_order<n>_d<d>nm_blaze<b>deg.png`` -- that
is the plot to open.

``--eval`` runs no new solves: it re-reads the existing
``results/energy_scan/*/multilayer_theta_search_summary.csv`` files
(:meth:`grax.MultilayerGratingDesigner.evaluate_energy_scan`) for the selected
designs, or for every design folder found when no selector is given; it also
regenerates the titled plot in ``results/plots/``.

Examples::

    python 1_run_energy_scan.py
    python 1_run_energy_scan.py --best
    python 1_run_energy_scan.py --pairs "3.0,1.1; 4.5,0.9"
    python 1_run_energy_scan.py --eval

The executable body is guarded because the theta-search sweep spawns worker
processes that re-import this file by path.
"""

from __future__ import annotations

import argparse

import pandas as pd
from rub4c_design_parameters import CONFIG

from grax import MultilayerGratingDesigner


def _parse_pairs(text: str) -> list[tuple[float, float]]:
    """Parse ``"d,blaze; d,blaze"`` into ``(d, blaze)`` float pairs."""

    pairs: list[tuple[float, float]] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        d_text, blaze_text = chunk.split(",")
        pairs.append((float(d_text), float(blaze_text)))
    if not pairs:
        raise ValueError(f"no (d, blaze) pairs parsed from {text!r}")
    return pairs


def _survey_table() -> pd.DataFrame:
    """Return the usable rows of ``results/survey/survey.csv``."""

    csv_path = CONFIG.survey_dir / "survey.csv"
    if not csv_path.is_file():
        raise SystemExit(
            f"{csv_path} not found -- run 0_run_survey.py first or pass --pairs."
        )
    table = pd.read_csv(csv_path).dropna(subset=["peak_efficiency"])
    if table.empty:
        raise SystemExit("survey.csv has no usable rows; pass --pairs explicitly.")
    return table


def _optimal_pairs_from_survey() -> list[tuple[float, float]]:
    """Return one ``(d, optimal_blaze)`` pair per d-spacing from the survey CSV."""

    table = _survey_table()
    best = table.loc[table.groupby("d_nm")["peak_efficiency"].idxmax()]
    return [(float(row.d_nm), float(row.blaze_deg)) for row in best.itertuples()]


def _best_pair_from_survey() -> list[tuple[float, float]]:
    """Return only the single ``(d, blaze)`` with the highest survey efficiency."""

    row = _survey_table().loc[lambda frame: frame["peak_efficiency"].idxmax()]
    print(
        f"Best survey design: d = {float(row.d_nm):.3f} nm, "
        f"blaze = {float(row.blaze_deg):.3f} deg (efficiency {float(row.peak_efficiency):.4g})"
    )
    return [(float(row.d_nm), float(row.blaze_deg))]


def main() -> None:
    """Parse the design selector and run the energy scan."""

    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--pairs",
        default=None,
        help='Semicolon-separated "d_nm,blaze_deg" designs to scan.',
    )
    selector.add_argument(
        "--best",
        action="store_true",
        help="Scan only the single (d, blaze) with the highest survey efficiency.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Re-read existing energy_scan/ results (no new solves). With no "
        "selector, evaluates every design folder found.",
    )
    args = parser.parse_args()

    designer = MultilayerGratingDesigner(CONFIG)
    if args.eval and not (args.pairs or args.best):
        results = designer.evaluate_energy_scan()
    else:
        if args.pairs:
            pairs = _parse_pairs(args.pairs)
        elif args.best:
            pairs = _best_pair_from_survey()
        else:
            pairs = _optimal_pairs_from_survey()
        results = (
            designer.evaluate_energy_scan(pairs)
            if args.eval
            else designer.run_energy_scan(pairs)
        )

    for scan in results:
        print(
            f"d = {scan.d_spacing_nm:.3f} nm, blaze = {scan.blaze_angle_deg:.3f} deg "
            f"-> {scan.summary_csv_path}"
        )
        print(f"  plot: {scan.titled_plot_path}")


if __name__ == "__main__":
    main()
