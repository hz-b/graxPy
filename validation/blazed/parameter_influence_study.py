"""Study the influence of simulation parameters on grating efficiency.

This script analyzes how three parameters affect the first-order efficiency
at three different energies (100, 600, 2000 eV):
1. x_resolution: horizontal discretization
2. z_resolution: vertical discretization
3. fourier_orders: number of Fourier orders

Results are plotted in a 3xN grid: rows = energies, columns = parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import grax

# =============================================================================
# Configuration
# =============================================================================



ENERGIES_EEV = [100.0, 600.0, 2000.0]

PARAMETERS = {
    "x_resolution_nm": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], dtype=float),
    "z_resolution_nm": np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], dtype=float),
    "fourier_orders": np.arange(5, 26, 5, dtype=int),
}

# Enable logging
grax.setup_logging(level='INFO', run_id='parameter_study')

# =============================================================================
# Setup
# =============================================================================
repo_root = Path(__file__).resolve().parents[0]
optical_constants_dir = repo_root / "optical_constants"
results_dir = repo_root / "results" / "parameter_influence_study"
results_dir.mkdir(parents=True, exist_ok=True)
silicon = pd.read_csv(
    optical_constants_dir / "n_Si_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
gold = pd.read_csv(
    optical_constants_dir / "n_Au_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

PERIOD_LPERMM = 600
BLAZE_ANGLE_DEG = 0.729
base_grating_params = {
    "period_lpermm": PERIOD_LPERMM,
    "blaze_angle_deg": BLAZE_ANGLE_DEG,
    "substrate_material": silicon,
    "layer_material": gold,
    "layer_thickness_nm": 30.0,
    "top_cap_material": None,
    "top_cap_thickness_nm": 0.0,
}

grating = grax.BlazedGrating(**base_grating_params)

# =============================================================================
# Run parameter sweeps
# =============================================================================

sweeps_dir = results_dir / "sweeps"
sweeps_dir.mkdir(parents=True, exist_ok=True)
all_rows = []
runner = grax.BatchSimulationRunner(
    diffraction_order=1,
    fourier_orders=15,
    show_progress=True,
    on_error="continue",
    resume=False,
    backend="numba",
)

for energy_ev in ENERGIES_EEV:
    grazing_angle_deg = float(
        grax.monochromator_grazing_angles_deg(
            np.asarray([energy_ev], dtype=float),
            period_lpermm=grating.period_lpermm,
            diffraction_order=1,
            cff=2.25,
        )[0]
    )
    for parameter, values in PARAMETERS.items():
        cases = []
        for index, value in enumerate(values):
            case = {
                "case_id": f"{energy_ev:.1f}-{parameter}-{index:08d}",
                "label": f"{parameter}={value} at {energy_ev:.1f} eV",
                "grating": grating,
                "energy_ev": float(energy_ev),
                "grazing_angle_deg": grazing_angle_deg,
                "polarization": "p",
                "backend": "numba",
            }
            if parameter == "fourier_orders":
                case["fourier_orders"] = int(value)
            else:
                case[parameter] = float(value)
            cases.append(case)

        results = list(runner.run_cases(cases))
        csv_path = sweeps_dir / f"{parameter}_{energy_ev:.1f}eV.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("parameter,value,energy_ev,grazing_angle_deg,status,efficiency,error_message\n")
            for value, result in zip(values, results):
                efficiency = result.selected_efficiency if result.status == "ok" else np.nan
                error_message = "" if result.error_message is None else result.error_message.replace(",", ";")
                handle.write(
                    f"{parameter},{value},{energy_ev},{grazing_angle_deg},"
                    f"{result.status},{efficiency},{error_message}\n"
                )
                all_rows.append((energy_ev, parameter, float(value), result.status, float(efficiency)))

# =============================================================================
# Plot results
# =============================================================================

print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

figure, axes = plt.subplots(len(ENERGIES_EEV), len(PARAMETERS), figsize=(14, 9), squeeze=False)
for row_index, energy_ev in enumerate(ENERGIES_EEV):
    for column_index, parameter in enumerate(PARAMETERS):
        axis = axes[row_index, column_index]
        rows = [row for row in all_rows if row[0] == energy_ev and row[1] == parameter]
        axis.plot([row[2] for row in rows], [row[4] for row in rows], "o-")
        axis.set_title(f"{energy_ev:.0f} eV - {parameter}")
        axis.set_xlabel(parameter)
        axis.set_ylabel("Efficiency")
        axis.grid(True, alpha=0.3)
        if parameter != "fourier_orders":
            axis.set_xscale("log")
figure.tight_layout()
plot_path = results_dir / "parameter_influence_study.png"
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

print(f"Sweep CSVs saved to: {sweeps_dir}")
print(f"Plot saved to: {plot_path}")
print("\nDone!")
