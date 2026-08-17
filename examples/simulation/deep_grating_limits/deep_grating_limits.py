"""Where the modal solver runs out of range and the differential method does not.

The two solvers normally agree to ~1e-11, so most comparisons between them are
uninteresting. This is the case where they genuinely differ.

The modal (RCWA) solver treats each layer as z-invariant and evaluates
``q / sinh(q d)`` across the whole layer at once. For a deep, high-contrast
grating the evanescent orders make ``q d`` large, ``sinh`` overflows to infinity,
and the solve raises "Modal layer function produced NaN/Inf values".

The differential method never forms that quantity. It caps the optical thickness
of any transfer matrix it builds (``NeviereOptions.block_phase``) and combines
the pieces with an interface-response cascade, which is an R-matrix propagation
and cannot accumulate a growing exponential.

Geometry is the published RETICOLO ``exemple1_1D`` lamellar grating: wavelength
6 um, period 10 um, refractive index 1.5 ridges over vacuum grooves, at normal
incidence. Groove depth is swept from a fraction of a wavelength to well past
the point where the modal solver gives up.

The transmission oscillates with depth because a deep grating acts as a
Fabry-Perot cavity. The depths are log-spaced, so those fringes are deliberately
undersampled at the deep end: the jaggedness there is aliasing of real
interference, not solver noise. The energy balance in the lower panel is the
check that the deep points are trustworthy.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grax.solvers import res0, res1, res2, res2_dm
from grax.solvers.common import prepare_layer_stack, propagating_energy_balance

parser = argparse.ArgumentParser(
    description="Compare solver depth limits on a deep lamellar grating"
)
parser.add_argument(
    "--polarization",
    choices=("s", "p"),
    default="p",
    help="Incident polarization. TM (p) is the harder case for the factorization rules.",
)
args = parser.parse_args()

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

WAVELENGTH_NM = 6000.0
PERIOD_NM = 10000.0
FOURIER_ORDERS = 25
# n = 1.5 ridges over [0, 1] and [9, 10] um, vacuum grooves between.
GRATING_TEXTURE = [
    np.array([1000.0, 9000.0]),
    np.array([1.5, 1.0], dtype=complex),
]
STACK = [1.0, 1.5, GRATING_TEXTURE]

depths_nm = np.geomspace(2.0e3, 1.0e6, 28)


def solve(depth_nm: float, solver: str) -> tuple[float, float]:
    """Return zeroth-order transmission and total propagating energy at one depth.

    Args:
        depth_nm: Groove depth in nanometers.
        solver: ``"rcwa"`` or ``"neviere"``.

    Returns:
        Zeroth-order transmission and the summed propagating efficiency, both
        ``nan`` when the solver cannot reach this depth.
    """

    parm = res0(1 if args.polarization == "s" else -1)
    aa = res1(
        WAVELENGTH_NM,
        PERIOD_NM,
        STACK,
        FOURIER_ORDERS,
        0.0,
        parm,
        _fourier_backend="numba",
    )
    profile = ([0.0, float(depth_nm), 0.0], [0, 2, 1])
    try:
        # The overflow that ends the modal solver's range is the subject of this
        # example, not a fault to report, so silence the numpy warning it emits
        # on the way to raising.
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            result = res2(aa, profile, parm) if solver == "rcwa" else res2_dm(aa, profile, parm)
    except ValueError:
        # The modal solver raises once sinh(q d) overflows. That is the point of
        # this example, so record it rather than aborting the sweep.
        return float("nan"), float("nan")

    zeroth = int(np.where(result.inc_top_reflected.order == 0)[0][0])
    n_top, n_bottom, _ = prepare_layer_stack(aa, profile)
    balance = propagating_energy_balance(
        result.inc_top_reflected,
        result.inc_top_transmitted,
        wavelength=aa.wavelength,
        period=aa.period,
        beta0=aa.beta0,
        n_top=n_top,
        n_bottom=n_bottom,
    )
    return float(result.inc_top_transmitted.efficiency[zeroth]), float(balance["total"])


transmission = {solver: [] for solver in ("rcwa", "neviere")}
energy_balance = {solver: [] for solver in ("rcwa", "neviere")}
for depth_nm in depths_nm:
    for solver in ("rcwa", "neviere"):
        zeroth, total = solve(float(depth_nm), solver)
        transmission[solver].append(zeroth)
        energy_balance[solver].append(total)

for solver in ("rcwa", "neviere"):
    transmission[solver] = np.asarray(transmission[solver], dtype=float)
    energy_balance[solver] = np.asarray(energy_balance[solver], dtype=float)

depths_in_wavelengths = depths_nm / WAVELENGTH_NM
rcwa_ok = np.isfinite(transmission["rcwa"])
last_rcwa_depth = depths_in_wavelengths[rcwa_ok].max() if rcwa_ok.any() else float("nan")
deepest = depths_in_wavelengths.max()

figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
efficiency_axis, balance_axis = axes

for solver, style in (("rcwa", "-"), ("neviere", "--")):
    efficiency_axis.plot(
        depths_in_wavelengths,
        transmission[solver],
        style,
        linewidth=1.6,
        marker="o",
        markersize=3,
        label=f"{solver} zeroth-order transmission",
    )
efficiency_axis.axvline(last_rcwa_depth, color="0.4", linestyle=":", linewidth=1.2)
efficiency_axis.annotate(
    f"modal solver stops here\n({last_rcwa_depth:.1f} wavelengths)",
    xy=(last_rcwa_depth, 0.44),
    xytext=(last_rcwa_depth * 1.6, 0.30),
    fontsize=9,
    arrowprops={"arrowstyle": "->", "color": "0.4"},
)
efficiency_axis.set_xscale("log")
efficiency_axis.set_ylabel("Zeroth-order transmission")
efficiency_axis.set_title(
    f"Deep lamellar grating, {args.polarization} polarization: solver depth limits"
)
efficiency_axis.grid(True, alpha=0.3)
efficiency_axis.legend(loc="lower left", fontsize=9)

for solver, style in (("rcwa", "-"), ("neviere", "--")):
    balance_axis.plot(
        depths_in_wavelengths,
        np.abs(1.0 - energy_balance[solver]),
        style,
        linewidth=1.4,
        marker="o",
        markersize=3,
        label=f"{solver}",
    )
balance_axis.set_yscale("log")
balance_axis.set_xscale("log")
balance_axis.set_xlabel("Groove depth (wavelengths)")
balance_axis.set_ylabel("|1 - energy balance|")
balance_axis.grid(True, alpha=0.3, which="both")
balance_axis.legend(loc="best")

figure.tight_layout()
plot_path = output_dir / f"deep_grating_limits_{args.polarization}.png"
figure.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

csv_path = output_dir / f"deep_grating_limits_{args.polarization}.csv"
np.savetxt(
    csv_path,
    np.column_stack(
        [
            depths_nm,
            depths_in_wavelengths,
            transmission["rcwa"],
            transmission["neviere"],
            energy_balance["rcwa"],
            energy_balance["neviere"],
        ]
    ),
    delimiter=",",
    header="depth_nm,depth_wavelengths,t0_rcwa,t0_neviere,energy_rcwa,energy_neviere",
    comments="",
)

matched = rcwa_ok & np.isfinite(transmission["neviere"])
print(f"Swept {len(depths_nm)} depths from "
      f"{depths_in_wavelengths.min():.2f} to {deepest:.0f} wavelengths.")
print(f"  modal solver succeeded up to     : {last_rcwa_depth:.1f} wavelengths "
      f"({int(rcwa_ok.sum())}/{len(depths_nm)} depths)")
print(f"  differential method succeeded up to: {deepest:.0f} wavelengths "
      f"({int(np.isfinite(transmission['neviere']).sum())}/{len(depths_nm)} depths)")
if matched.any():
    deviation = np.abs(transmission["rcwa"][matched] - transmission["neviere"][matched]).max()
    print(f"  where both work, max |dT0|       : {deviation:.3e}")
worst_balance = np.nanmax(np.abs(1.0 - energy_balance["neviere"]))
print(f"  differential-method energy balance stays within {worst_balance:.1e} of 1")
print(f"Plot saved to: {plot_path}")
print(f"CSV saved to: {csv_path}")
