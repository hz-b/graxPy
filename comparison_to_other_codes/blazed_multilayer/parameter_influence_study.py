"""Study parameter influence for the blazed multilayer comparison grating."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax as rp

ENERGIES_EV = [5000.0, 6000.0]

PARAMETERS = {
    "x_resolution_nm": np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], dtype=float),
    "z_resolution_nm": np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], dtype=float),
    "fourier_orders": np.arange(5, 26, 5, dtype=int),
}

rp.setup_logging(level="INFO", run_id="blazed_multilayer_parameter_study")

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root / "optical_constants"
simulation_dir = example_root / "simulation"
results_dir = example_root / "results" / "parameter_influence_study"
results_dir.mkdir(parents=True, exist_ok=True)
sweeps_dir = results_dir / "sweeps"
sweeps_dir.mkdir(parents=True, exist_ok=True)

silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
chromium = pd.read_csv(
    optical_constants_dir / "OC_Cr_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
chromium.attrs["name"] = "Cr"
carbon = pd.read_csv(
    optical_constants_dir / "OC_C_SSTR.dat",
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"

reference_data = pd.read_csv(
    simulation_dir / "DiffractMod_CrC_d4.8_N60.dat",
    sep=r"\s+",
    engine="python",
)
reference_data = reference_data[["Energy", "alpha"]].copy()
reference_data = reference_data.apply(pd.to_numeric, errors="coerce").dropna()
reference_data = reference_data.sort_values("Energy").reset_index(drop=True)

multilayer_stack = rp.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material=carbon,
)

grating = rp.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=0.1,
    z_resolution_nm=0.1,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=2,
    default_fourier_orders=15,
    show_progress=True,
    on_error="continue",
    resume=False,
)

all_rows = []

for energy_ev in ENERGIES_EV:
    grazing_angle_deg = float(
        np.interp(
            energy_ev,
            reference_data["Energy"].to_numpy(dtype=float),
            reference_data["alpha"].to_numpy(dtype=float),
        )
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
                "solver_backend": "s_matrix",
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

figure, axes = plt.subplots(len(ENERGIES_EV), len(PARAMETERS), figsize=(14, 9), squeeze=False)
for row_index, energy_ev in enumerate(ENERGIES_EV):
    for column_index, parameter in enumerate(PARAMETERS):
        axis = axes[row_index, column_index]
        rows = [row for row in all_rows if row[0] == energy_ev and row[1] == parameter]
        axis.plot([row[2] for row in rows], [row[4] for row in rows], "o-")
        axis.set_title(f"{energy_ev:.0f} eV - {parameter}")
        axis.set_xlabel(parameter)
        axis.set_ylabel("2nd-order efficiency")
        axis.grid(True, alpha=0.3)
        if parameter != "fourier_orders":
            axis.set_xscale("log")
figure.tight_layout()

plot_path = results_dir / "parameter_influence_study.png"
profile_plot_path = results_dir / "blazed_multilayer_profile.png"
stack_plot_path = results_dir / "multilayer_stack_schematic.png"

figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)
grating.plot_profile(profile_plot_path)
multilayer_stack.plot_schematic(stack_plot_path)

print(f"Sweep CSVs saved to: {sweeps_dir}")
print(f"Plot saved to: {plot_path}")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Stack schematic saved to: {stack_plot_path}")
