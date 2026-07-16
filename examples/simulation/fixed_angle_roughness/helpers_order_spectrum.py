"""Shared diffraction-order-spectrum plotting for the roughness examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def closest_case(cases, target_energy_ev: float):
    """Return the case whose ``energy_ev`` is closest to ``target_energy_ev``."""
    ok_cases = [case for case in cases if case.status == "ok"]
    if not ok_cases:
        raise ValueError("No successful cases to select an energy from.")
    return min(ok_cases, key=lambda case: abs(case.energy_ev - target_energy_ev))


def save_order_spectrum_plot(
    orders: np.ndarray,
    efficiency_all: np.ndarray,
    *,
    energy_ev: float,
    title: str,
    output_path: Path,
) -> Path:
    """Save a diffraction-order-vs-efficiency plot for one energy point."""
    order_values = np.asarray(orders, dtype=float)
    efficiency_values = np.asarray(efficiency_all, dtype=float)
    sort_index = np.argsort(order_values)
    order_values = order_values[sort_index]
    efficiency_values = efficiency_values[sort_index]

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    markerline, stemlines, baseline = axis.stem(order_values, efficiency_values, basefmt=" ")
    plt.setp(markerline, markersize=4)
    plt.setp(stemlines, linewidth=1.0)
    axis.set_xlabel("Diffraction order")
    axis.set_ylabel("Intensity (efficiency)")
    axis.set_title(f"{title} at {energy_ev:.0f} eV")
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
