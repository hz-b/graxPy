"""Test 2: shallow sinusoid, where the envelope is no longer trivial.

The flat interface is the clean diagnostic but a degenerate one -- its envelope
is a single harmonic by symmetry. A corrugated surface couples orders, so the
envelope acquires real structure, and the question becomes how *many* harmonics
that structure needs and whether that number tracks the geometry or the
wavelength.

Sweeps the envelope mode count at a fixed quadrature, against RCWA on identical
geometry, and reports the envelope's own spectrum so the smoothness claim is
measured rather than asserted.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import grax
from grax.solvers import res0, res1, res2
from grax.solvers._carrier import res2_hf
from grax.solvers.integral import IntegralOptions, res2_im

PERIOD_NM = 2500.0
GRAZING_DEG = 2.0
POLARIZATION = -1
REPORTED = 3


def sinusoid_grating(depth_nm: float):
    """Return a shallow sinusoid on Si with the benchmark period."""

    x = np.linspace(0.0, PERIOD_NM, 257)
    z = 0.5 * depth_nm * (1.0 - np.cos(2.0 * np.pi * x / PERIOD_NM))
    return grax.ProfileGrating(
        period_lpermm=1e6 / PERIOD_NM,
        x_points_nm=x,
        z_points_nm=z,
        substrate_material="Si",
        layer_material="Si",
        layer_thickness_nm=0.0,
        z_resolution_nm=0.2,
        x_resolution_nm=2.0,
    )


def main() -> None:
    """Compare classical BIE, carrier-extracted BIE and RCWA."""

    parser = argparse.ArgumentParser(description="Shallow-sinusoid carrier-extraction test")
    parser.add_argument("--energy", type=float, default=100.0)
    parser.add_argument("--depth-nm", type=float, default=20.0)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--fourier-orders", type=int, default=30)
    args = parser.parse_args()

    grating = sinusoid_grating(args.depth_nm)
    wavelength = 1239.8 / args.energy
    beta0 = float(np.sin(np.deg2rad(90.0 - GRAZING_DEG)))
    orders = np.arange(-REPORTED, REPORTED + 1, dtype=float)
    common = dict(
        grating=grating,
        wavelength_nm=wavelength,
        period_nm=PERIOD_NM,
        orders=orders,
        beta0=beta0,
        polarization=POLARIZATION,
        photon_energy_ev=args.energy,
    )

    print(
        f"sinusoid on Si, d = {PERIOD_NM:.0f} nm, depth {args.depth_nm:g} nm "
        f"(h/d = {args.depth_nm / PERIOD_NM:.4f}), {args.energy:.0f} eV, "
        f"d/lambda = {PERIOD_NM / wavelength:.0f}, {GRAZING_DEG} deg grazing, TM"
    )

    textures, profile = grating.build_textures(args.energy, n_inc=1.0 + 0.0j)
    parm = res0(POLARIZATION)
    started = time.perf_counter()
    aa = res1(wavelength, PERIOD_NM, textures, args.fourier_orders, beta0, parm)
    reference = res2(aa, profile, parm)
    rcwa_seconds = time.perf_counter() - started
    reference_orders = np.asarray(reference.inc_top_reflected.order, dtype=float)
    keep = np.isin(reference_orders, orders)
    reference_efficiency = np.real(reference.inc_top_reflected.efficiency)[keep]

    labels = "".join(f"{'m=' + str(int(-o)):>12}" for o in orders)
    print(f"\n{'method':>24}{labels}{'max dev':>10}{'secs':>8}")
    print("-" * (24 + 12 * orders.size + 18))
    print(
        f"{'RCWA (reference)':>24}"
        + "".join(f"{v:>12.7f}" for v in reference_efficiency)
        + f"{'':>10}{rcwa_seconds:>8.1f}"
    )

    started = time.perf_counter()
    classical = res2_im(
        **common,
        options=IntegralOptions(
            boundary_points=args.nodes, discretization="nystrom", corner_grading=1.0
        ),
    )
    classical_seconds = time.perf_counter() - started
    classical_efficiency = np.real(np.asarray(classical.inc_top_reflected.efficiency))
    print(
        f"{'classical BIE N=' + str(args.nodes):>24}"
        + "".join(f"{v:>12.7f}" for v in classical_efficiency)
        + f"{np.max(np.abs(classical_efficiency - reference_efficiency)):>10.1e}"
        + f"{classical_seconds:>8.1f}"
    )

    envelopes = None
    for modes in (0, 1, 2, 4, 8, 16):
        started = time.perf_counter()
        result, envelopes = res2_hf(
            **common,
            envelope_modes=modes,
            quadrature_nodes=args.nodes,
            return_envelopes=True,
        )
        seconds = time.perf_counter() - started
        efficiency = np.real(np.asarray(result.inc_top_reflected.efficiency))
        print(
            f"{'hf BIE M=' + str(modes) + ' (' + str(2 * modes + 1) + ' dof)':>24}"
            + "".join(f"{v:>12.7f}" for v in efficiency)
            + f"{np.max(np.abs(efficiency - reference_efficiency)):>10.1e}"
            + f"{seconds:>8.1f}",
            flush=True,
        )

    print("\nenvelope spectrum, |phi~_m| relative to the m = 0 coefficient:")
    field = np.abs(envelopes[-1, 0])
    centre = field.size // 2
    for offset in range(0, min(9, centre + 1)):
        print(f"  |m| = {offset:2d}   {field[centre + offset] / field[centre]:.3e}")
    print(
        "\nA decaying spectrum here is the claim under test: the envelope is smooth,\n"
        "so a handful of harmonics carries it however many wavelengths fit in a period."
    )


if __name__ == "__main__":
    main()
