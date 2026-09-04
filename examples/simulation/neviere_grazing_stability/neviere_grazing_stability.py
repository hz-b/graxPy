"""Nevière p-polarised stability at very small grazing angles.

Reproduces, in miniature, the regime a Mo/B4C second-order theta-search sweep
wandered into on 2026-09-04: its tracked-theta logic followed a near-zero
p-polarised feature down to ~0.29 deg -- far below the Bragg estimate -- and a
serial (``MAX_WORKERS=1``) run then crashed natively on Linux/OpenBLAS.

The differential method issues thousands of tiny dense solves per point; a
threaded BLAS both wastes time on dispatch and, on some OpenBLAS builds, crashes
under that pattern. ``grax.run_simulation`` now pins the Nevière solve to a
single BLAS thread, and the interface-response cascade raises a clean error
instead of feeding non-finite values to LAPACK. This script sweeps the grazing
angle from a normal Bragg angle down to 0.01 deg and prints the p-polarised
order-2 efficiency at each point; every value should be finite and the script
should exit cleanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import grax

EXAMPLE_ROOT = Path(__file__).resolve().parent


def build_grating() -> grax.BlazedGrating:
    """Return a coarse Mo/B4C multilayer-coated blazed grating."""

    substrate = grax.MaterialSpec("Si", 2.33)
    return grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=0.5,
        coating_stack=grax.MultilayerStack(
            substrate_material=substrate,
            material_a=grax.MaterialSpec("Mo", 10.22),
            material_b=grax.MaterialSpec("C", 2.52),
            d_period_nm=4.5,
            gamma=0.4,
            n_bilayers=40,
            top_material=grax.MaterialSpec("C", 2.52),
        ),
        substrate_material=substrate,
        x_resolution_nm=0.5,
        z_resolution_nm=0.5,
    )


def main() -> None:
    """Sweep the grazing angle down into the crash regime and report results."""

    grating = build_grating()
    energy_ev = 9428.571428571428  # the blaze-scan point right after 9000 eV
    angles_deg = [0.88, 0.50, 0.29, 0.10, 0.05, 0.01]

    print(f"Nevière / p / order 2 / E = {energy_ev:.1f} eV")
    for grazing_angle_deg in angles_deg:
        result = grax.run_simulation(
            grating=grating,
            energy_ev=energy_ev,
            grazing_angle_deg=grazing_angle_deg,
            diffraction_order=2,
            fourier_orders=25,
            polarization="p",
            solver="neviere",
            backend="numba",
            validate_physical_results=False,
        )
        finite = bool(np.all(np.isfinite(result.efficiency_all)))
        print(
            f"  grazing = {grazing_angle_deg:5.2f} deg  "
            f"eff(order 2) = {result.selected_efficiency:.3e}  all_finite = {finite}"
        )
    print("Completed without a native crash.")


if __name__ == "__main__":
    main()
