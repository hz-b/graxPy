"""Plot multi-energy baseline vs Numba RCWA comparison results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Plot multi-energy numba vs legacy comparison")
parser.add_argument(
    "--csv-path",
    type=Path,
    default=Path(__file__).resolve().parent / "results" / "multi_energy_numba_vs_legacy.csv",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "results",
)
args = parser.parse_args()

if not args.csv_path.exists():
    raise FileNotFoundError(
        f"Comparison CSV not found: {args.csv_path}. "
        "Run profile_multi_energy_numba_vs_legacy.py first."
    )

args.output_dir.mkdir(parents=True, exist_ok=True)

data = np.genfromtxt(args.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
energy_ev = np.asarray(data["energy_ev"], dtype=float)
baseline_total = np.asarray(data["baseline_total_s"], dtype=float)
numba_total = np.asarray(data["numba_total_s"], dtype=float)
baseline_peak_mb = np.asarray(data["baseline_peak_mb"], dtype=float)
numba_peak_mb = np.asarray(data["numba_peak_mb"], dtype=float)
speedup = np.asarray(data["speedup_baseline_over_numba"], dtype=float)
baseline_eff_m1 = np.asarray(data["baseline_eff_order_m1"], dtype=float)
numba_eff_m1 = np.asarray(data["numba_eff_order_m1"], dtype=float)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax = axes[0, 0]
ax.plot(energy_ev, baseline_total, marker="o", linestyle="-", label="baseline total")
ax.plot(energy_ev, numba_total, marker="o", linestyle=":", label="numba total")
ax.set_title("Total Runtime vs Energy")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Seconds")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(energy_ev, speedup, marker="o", color="tab:green")
ax.axhline(float(np.mean(speedup)), color="tab:green", linestyle="--", alpha=0.6, label="mean speedup")
ax.set_title("Baseline/Numba Speedup")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Speedup (x)")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 0]
ax.plot(energy_ev, baseline_peak_mb, marker="o", linestyle="-", label="baseline peak MB")
ax.plot(energy_ev, numba_peak_mb, marker="o", linestyle=":", label="numba peak MB")
ax.set_title("Peak Memory vs Energy")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("MB")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 1]
ax.plot(energy_ev, baseline_eff_m1, marker="o", linestyle="-", label="baseline (-1 order)")
ax.plot(energy_ev, numba_eff_m1, marker="o", linestyle=":", label="numba (-1 order)")
ax.set_title("Diffracted Efficiency Comparison (-1 Order)")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Efficiency")
ax.grid(True, alpha=0.3)
ax.legend()

for axis in axes.flat:
    axis.ticklabel_format(style="plain", useOffset=False, axis="both")

figure_path = args.output_dir / "multi_energy_numba_vs_legacy_plots.png"
fig.savefig(figure_path, dpi=180)
plt.close(fig)

print(f"Saved plot figure: {figure_path}")
