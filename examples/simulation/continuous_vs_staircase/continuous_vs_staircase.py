"""Why the differential method does not need a fine z grid.

Both solvers normally see the same geometry: a grating profile is sliced into
z-layers of ``z_resolution_nm``, each layer treated as z-invariant. That
staircase is an approximation of the real profile, and its error shrinks as the
slices get thinner, which is why a converged run needs a fine z grid and pays
for it in runtime.

``NeviereOptions(z_sampling="continuous")`` drops the staircase entirely. The
differential method re-expands the permittivity from the true profile as it
integrates, so its answer does not depend on ``z_resolution_nm`` at all.

This example sweeps ``z_resolution_nm`` over a sinusoidal profile and plots
three curves:

- RCWA, staircase: converges as z refines
- Nevière with ``z_sampling="textures"``: the same staircase, so it converges
  identically -- this is what makes the two solvers comparable elsewhere
- Nevière with ``z_sampling="continuous"``: flat, because it never sees a
  staircase to converge from

The flat line is the converged answer. The distance from a staircase curve to it
is the discretization error a staircase run is carrying at that resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import grax

parser = argparse.ArgumentParser(description="Staircase vs continuous z sampling")
parser.add_argument(
    "--polarization",
    choices=("s", "p"),
    default="p",
    help="Incident polarization.",
)
args = parser.parse_args()

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

PERIOD_LPERMM = 800
DEPTH_NM = 12.0
ENERGY_EV = 300.0
GRAZING_ANGLE_DEG = 5.0
FOURIER_ORDERS = 10
DIFFRACTION_ORDER = 1

period_nm = 1e6 / PERIOD_LPERMM
x_points_nm = np.linspace(0.0, period_nm, 129)
z_points_nm = DEPTH_NM * 0.5 * (1.0 - np.cos(2.0 * np.pi * x_points_nm / period_nm))


def build_grating(z_resolution_nm: float) -> grax.ProfileGrating:
    """Return the sinusoidal grating sliced at one z resolution.

    Args:
        z_resolution_nm: Vertical slice thickness in nanometers.

    Returns:
        Configured sinusoidal profile grating.
    """

    return grax.ProfileGrating(
        period_lpermm=PERIOD_LPERMM,
        x_points_nm=x_points_nm,
        z_points_nm=z_points_nm,
        substrate_material="Si",
        layer_material="Au",
        layer_thickness_nm=20.0,
        x_resolution_nm=1.0,
        z_resolution_nm=z_resolution_nm,
    )


def efficiency(z_resolution_nm: float, solver: str, solver_options=None) -> float:
    """Return the selected-order efficiency for one configuration.

    Args:
        z_resolution_nm: Vertical slice thickness in nanometers.
        solver: ``"rcwa"`` or ``"neviere"``.
        solver_options: Optional solver settings.

    Returns:
        Efficiency of the selected reflected order.
    """

    result = grax.run_simulation(
        grating=build_grating(z_resolution_nm),
        energy_ev=ENERGY_EV,
        grazing_angle_deg=GRAZING_ANGLE_DEG,
        diffraction_order=DIFFRACTION_ORDER,
        fourier_orders=FOURIER_ORDERS,
        polarization=args.polarization,
        solver=solver,
        solver_options=solver_options,
        validate_physical_results=False,
    )
    return float(result.selected_efficiency)


z_resolutions_nm = np.asarray([2.0, 1.0, 0.5, 0.25, 0.1, 0.05], dtype=float)
continuous_options = grax.NeviereOptions(z_sampling="continuous", sample_phase=0.005)

curves = {
    "RCWA (staircase)": [efficiency(z, "rcwa") for z in z_resolutions_nm],
    "Nevière (staircase)": [efficiency(z, "neviere") for z in z_resolutions_nm],
    "Nevière (continuous)": [
        efficiency(z, "neviere", continuous_options) for z in z_resolutions_nm
    ],
}
curves = {label: np.asarray(values, dtype=float) for label, values in curves.items()}

converged = curves["Nevière (continuous)"].mean()
continuous_spread = float(np.ptp(curves["Nevière (continuous)"]))

figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
value_axis, error_axis = axes

styles = {
    "RCWA (staircase)": {"linestyle": "-", "marker": "o"},
    "Nevière (staircase)": {"linestyle": (0, (6, 4)), "marker": "s"},
    "Nevière (continuous)": {"linestyle": "-.", "marker": "^"},
}
for label, values in curves.items():
    value_axis.plot(z_resolutions_nm, values, linewidth=1.6, markersize=5,
                    label=label, **styles[label])
value_axis.set_xscale("log")
value_axis.set_ylabel(f"Order {DIFFRACTION_ORDER} efficiency")
value_axis.set_title(
    f"Sinusoidal grating at {ENERGY_EV:.0f} eV, {args.polarization} polarization: "
    "staircase convergence vs continuous sampling"
)
value_axis.grid(True, alpha=0.3)
value_axis.legend(loc="best")

staircase_errors = np.concatenate([
    np.abs(curves[label] - converged)
    for label in ("RCWA (staircase)", "Nevière (staircase)")
])
error_floor = staircase_errors[staircase_errors > 0].min() / 50.0
for label, values in curves.items():
    error = np.abs(values - converged)
    if label == "Nevière (continuous)":
        # Exactly zero at every resolution, which a log axis cannot draw. Pin it
        # to a floor and label it honestly rather than letting it vanish.
        error = np.full_like(error, error_floor)
        label = f"{label} - exactly 0, shown at the axis floor"
    error_axis.plot(z_resolutions_nm, error, linewidth=1.4, markersize=5,
                    label=label, **styles[label.split(" - ")[0]])
error_axis.set_xscale("log")
error_axis.set_yscale("log")
error_axis.set_ylim(error_floor / 3.0, staircase_errors.max() * 3.0)
error_axis.invert_xaxis()
error_axis.set_xlabel("z_resolution_nm  (finer to the right)")
error_axis.set_ylabel("|efficiency - converged|")
error_axis.grid(True, alpha=0.3, which="both")
error_axis.legend(loc="best", fontsize=8)

figure.tight_layout()
plot_path = output_dir / f"continuous_vs_staircase_{args.polarization}.png"
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

csv_path = output_dir / f"continuous_vs_staircase_{args.polarization}.csv"
np.savetxt(
    csv_path,
    np.column_stack([z_resolutions_nm, *curves.values()]),
    delimiter=",",
    header="z_resolution_nm,rcwa_staircase,neviere_staircase,neviere_continuous",
    comments="",
)

print(f"Swept z_resolution_nm from {z_resolutions_nm.max()} down to {z_resolutions_nm.min()} nm.")
print(f"  continuous sampling spread over the whole sweep : {continuous_spread:.3e}")
print(f"  coarsest staircase error (z = {z_resolutions_nm.max()} nm)          : "
      f"{abs(curves['RCWA (staircase)'][0] - converged):.3e}")
print(f"  finest staircase error   (z = {z_resolutions_nm.min()} nm)         : "
      f"{abs(curves['RCWA (staircase)'][-1] - converged):.3e}")
print("  the two staircase curves agree with each other to "
      f"{np.max(np.abs(curves['RCWA (staircase)'] - curves['Nevière (staircase)'])):.3e}")
print(f"Plot saved to: {plot_path}")
print(f"CSV saved to: {csv_path}")
