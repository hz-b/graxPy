"""Plot multilayer multi-energy baseline vs Numba RCWA comparison results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "multi_energy_multilayer_numba_vs_numpy.csv"
PLOT_PATH = RESULTS_DIR / "multi_energy_multilayer_numba_vs_legacy_plots.png"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Comparison CSV not found: {CSV_PATH}. "
        "Run profile_multi_energy_multilayer_numba_vs_legacy.py first."
    )

data = np.genfromtxt(CSV_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
energy_ev = np.asarray(data["energy_ev"], dtype=float)
numpy_total = np.asarray(data["numpy_total_s"], dtype=float)
numba_total = np.asarray(data["numba_total_s"], dtype=float)
numpy_peak_mb = np.asarray(data["numpy_peak_mb"], dtype=float)
numba_peak_mb = np.asarray(data["numba_peak_mb"], dtype=float)
speedup = np.asarray(data["speedup_numpy_over_numba"], dtype=float)
numpy_eff_m1 = np.asarray(data["numpy_eff_order_m1"], dtype=float)
numba_eff_m1 = np.asarray(data["numba_eff_order_m1"], dtype=float)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax = axes[0, 0]
ax.plot(energy_ev, numpy_total, marker="o", linestyle="-", label="numpy total")
ax.plot(energy_ev, numba_total, marker="o", linestyle=":", label="numba total")
ax.set_title("Total Runtime vs Energy")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Seconds")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(energy_ev, speedup, marker="o", color="tab:green")
ax.axhline(float(np.mean(speedup)), color="tab:green", linestyle="--", alpha=0.6, label="mean speedup")
ax.set_title("Numpy/Numba Speedup")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Speedup (x)")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 0]
ax.plot(energy_ev, numpy_peak_mb, marker="o", linestyle="-", label="numpy peak MB")
ax.plot(energy_ev, numba_peak_mb, marker="o", linestyle=":", label="numba peak MB")
ax.set_title("Peak Memory vs Energy")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("MB")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 1]
ax.plot(energy_ev, numpy_eff_m1, marker="o", linestyle="-", label="numpy (-1 order)")
ax.plot(energy_ev, numba_eff_m1, marker="o", linestyle=":", label="numba (-1 order)")
ax.set_title("Diffracted Efficiency Comparison (-1 Order)")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Efficiency")
ax.grid(True, alpha=0.3)
ax.legend()

for axis in axes.flat:
    axis.ticklabel_format(style="plain", useOffset=False, axis="both")

fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Saved plot figure: {PLOT_PATH}")
