"""Stage 1 diagnostic: is N driven by the unknowns or by the quadrature?

``scaling.py`` measures how many panels are needed. It cannot say *why*. Those
are different questions with different consequences:

- If the surface densities themselves need that many degrees of freedom, the
  formulation is expensive here and no change of discretization rescues it.
- If the densities are band-limited to a handful of harmonics and the panels are
  being spent resolving the *oscillatory kernel* instead, then the panel method
  is conflating two grids that a higher-order scheme keeps separate -- and
  trigonometric collocation, which represents the density in the Fourier basis
  while quadrature is handled independently, removes the constraint. That is the
  premise of Stage 2 and this script tests it directly.

The test: solve at a converged panel count, strip the pseudo-periodic carrier
``exp(i alpha_0 x)`` from the densities, take the Fourier spectrum of what
remains, and count how many harmonics carry meaningful amplitude. Sweep
``d / lambda`` and see whether that count saturates while the required panel
count keeps climbing.

Usage::

    python tools/integral_study/spectrum.py --ratios 5 10 25 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cases import build_sinusoid_case  # noqa: E402

from grax.solvers._green import PeriodicGreen, default_ewald_splitting  # noqa: E402
from grax.solvers._operators import layer_operators  # noqa: E402
from grax.solvers.integral import IntegralOptions, build_stack  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Return the parsed command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 25.0, 50.0],
        help="d/lambda values to test (default: 5 10 25 50).",
    )
    parser.add_argument(
        "--panels",
        type=int,
        default=384,
        help="Panel count used for the diagnostic solve (default: 384).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Relative amplitude counted as significant (default: 1e-4).",
    )
    parser.add_argument(
        "--depth-nm", type=float, default=5.0, help="Sinusoid depth (default: 5)."
    )
    return parser.parse_args()


def main() -> int:
    """Run the diagnostic and return a process exit code."""

    args = parse_arguments()
    energy_ev = 100.0
    wavelength_nm = 1239.8 / energy_ev

    print(f"panels={args.panels}, significance threshold={args.threshold:.0e}")
    print(f"{'d/lam':>7} {'h/d':>8} {'phi harm':>9} {'psi harm':>9} {'prop ord':>9}")
    print("-" * 48)

    for ratio in args.ratios:
        period_nm = ratio * wavelength_nm
        case = build_sinusoid_case(
            period_nm=period_nm, depth_nm=args.depth_nm, energy_ev=energy_ev
        )
        phi, psi = _solve_densities(case, panels=args.panels)
        phi_harmonics = _significant_harmonics(phi, case, args.threshold)
        psi_harmonics = _significant_harmonics(psi, case, args.threshold)
        propagating = _propagating_order_count(case)
        print(
            f"{ratio:>7.0f} {args.depth_nm / period_nm:>8.4f} "
            f"{phi_harmonics:>9} {psi_harmonics:>9} {propagating:>9}"
        )

    print()
    print("If the harmonic counts stay flat while scaling.py's N_required climbs,")
    print("the panels are being spent on the kernel, not the unknowns -> Stage 2 pays off.")
    return 0


def _solve_densities(case, *, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the boundary densities for a single-interface case.

    Rebuilds the two-equation system directly rather than going through
    ``res2_im``, because the densities are internal to that function and this
    diagnostic needs them.

    Args:
        case: A single-interface case.
        panels: Panel count.

    Returns:
        ``(phi, psi)`` sampled at the collocation points.

    Raises:
        ValueError: If the case has more than one interface.
    """

    options = IntegralOptions(boundary_points=panels)
    k0 = 2.0 * np.pi / case.wavelength_nm
    alpha0 = k0 * case.beta0

    stack = build_stack(
        case.grating,
        photon_energy_ev=case.energy_ev,
        wavelength_nm=case.wavelength_nm,
        n_inc=1.0 + 0.0j,
        orders=case.reported_orders,
        options=options,
    )
    if stack.interface_count != 1:
        raise ValueError("This diagnostic handles single-interface cases only.")

    interface = stack.interfaces[0]
    n_below, n_above = stack.indices[0], stack.indices[1]

    green_above = PeriodicGreen(
        period=case.period_nm,
        wavenumber=k0 * n_above,
        alpha0=alpha0,
        splitting=default_ewald_splitting(case.period_nm, k0 * n_above),
    )
    green_below = PeriodicGreen(
        period=case.period_nm,
        wavenumber=k0 * n_below,
        alpha0=alpha0,
        splitting=default_ewald_splitting(case.period_nm, k0 * n_below),
    )
    single_above, double_above = layer_operators(
        green_above, target=interface, source=interface, same_interface=True
    )
    single_below, double_below = layer_operators(
        green_below, target=interface, source=interface, same_interface=True
    )

    count = interface.count
    identity = np.eye(count, dtype=complex)
    tau = 1.0 + 0.0j if case.polarization == 1 else (n_below**2) / (n_above**2)

    matrix = np.zeros((2 * count, 2 * count), dtype=complex)
    matrix[:count, :count] = 0.5 * identity - double_above
    matrix[:count, count:] = single_above
    matrix[count:, :count] = 0.5 * identity + double_below
    matrix[count:, count:] = -tau * single_below

    beta0 = complex(green_above.beta(np.asarray([0.0]))[0])
    rhs = np.zeros(2 * count, dtype=complex)
    rhs[:count] = np.exp(
        1j * alpha0 * interface.midpoint[:, 0] - 1j * beta0 * interface.midpoint[:, 1]
    )

    solution = np.linalg.solve(matrix, rhs)
    return solution[:count], solution[count:]


def _significant_harmonics(density: np.ndarray, case, threshold: float) -> int:
    """Return how many Fourier harmonics of the density exceed a threshold.

    The pseudo-periodic carrier is removed first, so what is counted is the
    genuine harmonic content of the density envelope rather than the incident
    phase, which every solver carries and none has to resolve as unknowns.

    Args:
        density: Density sampled at the collocation points.
        case: The case, for the carrier phase and geometry.
        threshold: Relative amplitude counted as significant.

    Returns:
        Number of harmonics above the threshold.
    """

    k0 = 2.0 * np.pi / case.wavelength_nm
    alpha0 = k0 * case.beta0
    # Resample onto a uniform x grid so the FFT means what it should. Panels are
    # graded, so the collocation points are not uniformly spaced in x.
    options = IntegralOptions(boundary_points=density.size)
    stack = build_stack(
        case.grating,
        photon_energy_ev=case.energy_ev,
        wavelength_nm=case.wavelength_nm,
        n_inc=1.0 + 0.0j,
        orders=case.reported_orders,
        options=options,
    )
    x = stack.interfaces[0].midpoint[:, 0]
    envelope = density * np.exp(-1j * alpha0 * x)

    uniform_x = np.linspace(0.0, case.period_nm, density.size, endpoint=False)
    real = np.interp(uniform_x, x, envelope.real, period=case.period_nm)
    imaginary = np.interp(uniform_x, x, envelope.imag, period=case.period_nm)
    spectrum = np.abs(np.fft.fft(real + 1j * imaginary)) / density.size

    peak = float(np.max(spectrum))
    if peak <= 0.0:
        return 0
    return int(np.count_nonzero(spectrum >= threshold * peak))


def _propagating_order_count(case) -> int:
    """Return how many orders propagate in the incident medium.

    Args:
        case: The case.

    Returns:
        Number of propagating orders.
    """

    k0 = 2.0 * np.pi / case.wavelength_nm
    alpha0 = k0 * case.beta0
    reach = int(2.0 * case.period_nm / case.wavelength_nm) + 2
    orders = np.arange(-reach, reach + 1, dtype=float)
    alpha = alpha0 + 2.0 * np.pi * orders / case.period_nm
    return int(np.count_nonzero(k0**2 - alpha**2 > 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
