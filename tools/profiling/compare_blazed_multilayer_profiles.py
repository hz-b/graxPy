"""Compare baseline and candidate blazed multilayer profiling summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the comparison tool."""

    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate blazed multilayer profiling CSVs."
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline matrix summary CSV.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate matrix summary CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "comparisons",
        help="Directory for comparison text and CSV outputs.",
    )
    return parser


def _load_rows(path: Path) -> list[dict[str, str]]:
    """Load profiling comparison rows from one CSV file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    """Return the merge key shared by baseline and candidate rows."""

    return (
        row["energy_ev"],
        row["grazing_angle_deg"],
        row["fourier_orders"],
        row["x_resolution_nm"],
        row["z_resolution_nm"],
    )


def build_comparison_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    """Return merged before/after rows with absolute and relative deltas."""

    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    candidate_by_key = {_row_key(row): row for row in candidate_rows}
    shared_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    comparison_rows: list[dict[str, float | str]] = []
    for key in shared_keys:
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        baseline_total = float(baseline["total_wall_seconds"])
        candidate_total = float(candidate["total_wall_seconds"])
        baseline_texture = float(baseline["texture_generation_seconds"])
        candidate_texture = float(candidate["texture_generation_seconds"])
        baseline_efficiency = float(baseline["selected_efficiency"])
        candidate_efficiency = float(candidate["selected_efficiency"])
        comparison_rows.append(
            {
                "energy_ev": key[0],
                "grazing_angle_deg": key[1],
                "fourier_orders": key[2],
                "x_resolution_nm": key[3],
                "z_resolution_nm": key[4],
                "baseline_total_wall_seconds": baseline_total,
                "candidate_total_wall_seconds": candidate_total,
                "total_wall_seconds_delta": candidate_total - baseline_total,
                "total_wall_speedup": (
                    baseline_total / candidate_total if candidate_total > 0.0 else 0.0
                ),
                "baseline_texture_generation_seconds": baseline_texture,
                "candidate_texture_generation_seconds": candidate_texture,
                "texture_generation_seconds_delta": candidate_texture - baseline_texture,
                "texture_generation_speedup": (
                    baseline_texture / candidate_texture if candidate_texture > 0.0 else 0.0
                ),
                "baseline_texture_count": int(float(baseline["texture_count"])),
                "candidate_texture_count": int(float(candidate["texture_count"])),
                "baseline_unique_texture_count": int(float(baseline["unique_texture_count"])),
                "candidate_unique_texture_count": int(float(candidate["unique_texture_count"])),
                "baseline_selected_efficiency": baseline_efficiency,
                "candidate_selected_efficiency": candidate_efficiency,
                "selected_efficiency_delta": candidate_efficiency - baseline_efficiency,
            }
        )
    return comparison_rows


def write_comparison_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write comparison rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_comparison_report(rows: list[dict[str, float | str]]) -> str:
    """Return a concise human-readable comparison report."""

    if not rows:
        return "No shared profiling configurations found.\n"
    mean_total_speedup = sum(float(row["total_wall_speedup"]) for row in rows) / float(len(rows))
    mean_texture_speedup = sum(float(row["texture_generation_speedup"]) for row in rows) / float(len(rows))
    max_efficiency_delta = max(abs(float(row["selected_efficiency_delta"])) for row in rows)
    lines = [
        "Blazed Multilayer Profiling Comparison",
        "",
        f"shared_configurations={len(rows)}",
        f"mean_total_wall_speedup={mean_total_speedup:.6f}",
        f"mean_texture_generation_speedup={mean_texture_speedup:.6f}",
        f"max_selected_efficiency_delta={max_efficiency_delta:.12e}",
        "",
        "configuration summary",
    ]
    for row in rows:
        lines.append(
            " ".join(
                [
                    f"fo={row['fourier_orders']}",
                    f"x={row['x_resolution_nm']}",
                    f"z={row['z_resolution_nm']}",
                    f"total_speedup={float(row['total_wall_speedup']):.6f}",
                    f"texture_speedup={float(row['texture_generation_speedup']):.6f}",
                    f"eff_delta={float(row['selected_efficiency_delta']):.12e}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the profiling comparison tool."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    baseline_rows = _load_rows(args.baseline)
    candidate_rows = _load_rows(args.candidate)
    comparison_rows = build_comparison_rows(baseline_rows, candidate_rows)
    baseline_label = args.baseline.stem.replace("profile_matrix_summary_", "")
    candidate_label = args.candidate.stem.replace("profile_matrix_summary_", "")
    comparison_csv = args.output_dir / f"comparison_{baseline_label}_vs_{candidate_label}.csv"
    comparison_txt = args.output_dir / f"comparison_{baseline_label}_vs_{candidate_label}.txt"
    write_comparison_csv(comparison_csv, comparison_rows)
    report = format_comparison_report(comparison_rows)
    comparison_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved CSV: {comparison_csv}")
    print(f"Saved report: {comparison_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
