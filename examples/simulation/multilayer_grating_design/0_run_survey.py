"""Step 1: run the Ru/B4C d-spacing x blaze-angle survey.

Runs :meth:`grax.MultilayerGratingDesigner.run_survey` with the shared
``rub4c_design_parameters.CONFIG``. Writes ``results/survey/survey.csv``, three
headline plots in ``results/plot/`` (``optimal_blaze_vs_d_spacing.png``,
``max_efficiency_vs_d_spacing.png`` and ``efficiency_heatmap_d_vs_blaze.png`` --
the full ``(d, blaze) -> efficiency`` map with the optimal-blaze ridge overlaid),
and a per-run folder tree under
``results/survey/runs/``: one ``d<d>nm/`` folder per period (with an
``overlay.png`` of its blaze angles, chosen one highlighted) and inside it one
``blaze<b>deg/`` folder per run holding that theta search's full output
(``multilayer_theta_search_summary.csv``, ``*_all_orders.csv``, ``theta_scans/``,
profile and stack plots, ``checkpoints/``) plus a ``search_parameters.json``.

``--eval`` runs no new solves: it re-reads the per-run summary CSVs already on
disk (:meth:`grax.MultilayerGratingDesigner.evaluate_survey`), rebuilds
``survey.csv`` and regenerates every plot. Use it after a completed run to
iterate on the analysis.

The executable body is guarded because a later step spawns worker processes that
re-import the example package by path.
"""

from __future__ import annotations

import argparse

from rub4c_design_parameters import CONFIG

from grax import MultilayerGratingDesigner


def main() -> None:
    """Run (or, with ``--eval``, re-evaluate) the survey."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Re-evaluate from existing results/ (no new solves): "
        "rebuild survey.csv and every plot.",
    )
    args = parser.parse_args()

    designer = MultilayerGratingDesigner(CONFIG)
    result = designer.evaluate_survey() if args.eval else designer.run_survey()

    print(f"Survey CSV:            {result.combined_csv_path}")
    print(f"Optimal-blaze curve:  {result.plot_path}")
    print(f"Max-efficiency curve: {result.efficiency_plot_path}")
    print(f"Efficiency heatmap:   {result.heatmap_plot_path}")
    print(
        f"Runs: {len(result.run_dirs)} theta-search folders in "
        f"{len(result.period_dirs)} period folders under {result.runs_dir}"
    )
    for d_nm, blaze_opt, eff in zip(
        result.d_values_nm, result.optimal_blaze_deg, result.optimal_blaze_efficiency
    ):
        print(f"  d {d_nm:.3f} nm -> optimal blaze {blaze_opt:.3f} deg (efficiency {eff:.4g})")


if __name__ == "__main__":
    main()
