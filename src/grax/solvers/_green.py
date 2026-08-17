"""Quasi-periodic Green function for the 2D Helmholtz equation.

The boundary-integral solver in :mod:`grax.solvers.integral` needs the Green
function of a *periodic array* of line sources, so that every integral runs over
one grating period instead of the whole infinite surface::

    G(r, r') = (i/4) sum_m H_0^(1)(k |r - r' - m d x_hat|) exp(i alpha_0 m d)

with ``(nabla^2 + k^2) G = -delta`` and the ``exp(-i omega t)`` time convention
that the rest of the package already uses (see ``_kz_branch`` in
:mod:`grax.solvers.common`, which selects ``Im k_z >= 0``).

Two evaluation paths are provided and they must agree:

``spectral``
    The plane-wave (Rayleigh) series

        G = (i / 2d) sum_n (1 / beta_n) exp(i alpha_n X + i beta_n |Y|)

    It converges geometrically in ``|Y|`` and is therefore exact and cheap when
    the two points are vertically separated, but it stalls as ``|Y| -> 0``,
    which is precisely the on-boundary case the solver needs. Kept as the
    reference implementation and as the test oracle.

``ewald``
    Ewald's splitting of the same sum into a spectral part and a spatial part,
    each converging exponentially, at the cost of an ``erfc`` per term. This is
    the production path. The splitting parameter cancels analytically, so
    independence of ``ewald_splitting`` is a strong self-check that holds even
    at ``Y = 0``, where the spectral path cannot serve as an oracle.

Derivation
----------
Both halves come from

    (i/4) H_0^(1)(k R) = (1 / 4 pi) int_0^inf (1/t) exp(-R^2 t + k^2 / 4t) dt

split at ``t = E^2``.

The upper range keeps a Gaussian decay in ``R`` and stays a lattice sum. With
``z = R^2 E^2`` and the generalised exponential integral ``E_p``::

    G_lattice = (1/4pi) sum_m e^(i alpha_0 m d)
                sum_p (1/p!) (k^2 / 4 E^2)^p E_{p+1}(z_m)

Note the ``E^(-2p)``: the ``R`` dependence sits entirely inside ``E_{p+1}``.

The lower range is Poisson summed into

    G_spectral = (i/4d) sum_n (1/beta_n) e^(i alpha_n X)
                 [ e^(-i beta_n |Y|) erfc(u_n + E|Y|)
                 + e^(+i beta_n |Y|) erfc(u_n - E|Y|) ]

with ``u_n = -i beta_n / 2E``. The sign of ``u_n`` matters: the closed form for
``int_0^T t^(-3/2) exp(-a^2 t - b^2/t) dt`` requires ``Re b > 0``, and with
``b^2 = -beta^2/4`` on the ``Im beta >= 0`` branch that forces ``b = -i beta/2``.
The opposite choice makes both terms grow without bound for evanescent orders.
Letting ``E -> inf`` collapses the bracket to ``2 e^(i beta_n |Y|)`` and recovers
the plane-wave series exactly, which is the check that fixes every constant.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx, exp1, hankel1

__all__ = [
    "PeriodicGreen",
    "default_ewald_splitting",
    "free_space_green",
]


# Guards, not expected counts: both Ewald sums decay like a Gaussian, so the
# adaptive reaches below normally land far under these.
_MAX_SPECTRAL_ORDERS = 20000
_MAX_LATTICE_CELLS = 64
# Past this the lattice series has stopped being a practical accelerator and the
# splitting parameter is the thing to fix, so it raises rather than grinding.
_MAX_LATTICE_TERMS = 200
_TINY = 1e-300


def default_ewald_splitting(period: float, wavenumber: complex) -> float:
    """Return a splitting parameter that keeps both Ewald sums convergent.

    The classical balanced choice for a 1D lattice of pitch ``d`` is
    ``E = sqrt(pi) / d``. That is only usable while the period is comparable to
    the wavelength: the lattice sum carries ``(k^2 / 4 E^2)^p / p!``, which needs
    ``p`` beyond ``|k|^2 / 4 E^2`` terms before it starts to decay, so a soft
    X-ray period of several hundred wavelengths would need millions of terms.

    Enforcing ``E >= |k| / 2`` caps that ratio at one, making the lattice series
    converge immediately. The price is paid in the spectral sum, whose reach
    grows like ``E d``, and that cost is linear rather than catastrophic.

    Args:
        period: Lattice pitch in nanometers.
        wavenumber: Medium wavenumber ``k0 * n`` in inverse nanometers.

    Returns:
        Positive splitting parameter in inverse nanometers.
    """

    balanced = np.sqrt(np.pi) / float(period)
    convergent = abs(complex(wavenumber)) / 2.0
    return float(max(balanced, convergent))


class PeriodicGreen:
    """Quasi-periodic Green function and its gradient for one medium.

    One instance is bound to a medium (through ``wavenumber``), a period, and an
    incidence phase, because those fix ``alpha_n`` and ``beta_n``. Evaluation is
    vectorised over arrays of separations.

    Attributes:
        period: Grating period in nanometers.
        wavenumber: ``k0 * n`` for this medium, in inverse nanometers.
        alpha0: In-plane wavenumber of the incident field, inverse nanometers.
        method: ``"ewald"`` or ``"spectral"``.
        splitting: Ewald splitting parameter, inverse nanometers.
    """

    def __init__(
        self,
        *,
        period: float,
        wavenumber: complex,
        alpha0: float,
        method: str = "ewald",
        splitting: float | None = None,
        spectral_orders: int | None = None,
    ) -> None:
        """Bind the Green function to one medium and incidence.

        Args:
            period: Grating period in nanometers.
            wavenumber: ``k0 * n`` for this medium.
            alpha0: In-plane wavenumber of the incident field.
            method: ``"ewald"`` (production) or ``"spectral"`` (reference).
            splitting: Ewald splitting parameter. ``None`` picks a default.
            spectral_orders: Half-width of the ``"spectral"`` sum. ``None``
                picks a value from the period-to-wavelength ratio.

        Raises:
            ValueError: If ``method`` is not one of the two supported names.
        """

        if method not in ("ewald", "spectral"):
            raise ValueError(f"method must be 'ewald' or 'spectral', got {method!r}.")
        self.period = float(period)
        self.wavenumber = complex(wavenumber)
        self.alpha0 = float(alpha0)
        self.method = method
        self.splitting = (
            float(splitting)
            if splitting is not None
            else default_ewald_splitting(self.period, self.wavenumber)
        )
        if spectral_orders is None:
            # Enough to cover every propagating order plus an evanescent margin.
            spectral_orders = int(4 * self.period * abs(self.wavenumber) / (2 * np.pi)) + 40
        self.spectral_orders = int(spectral_orders)
        self._lattice_terms = _lattice_expansion_terms(self.wavenumber, self.splitting)

    # -- shared spectral bookkeeping ---------------------------------------

    def alpha(self, orders: np.ndarray) -> np.ndarray:
        """Return the in-plane wavenumbers of the given orders."""

        return self.alpha0 + 2.0 * np.pi * np.asarray(orders, dtype=float) / self.period

    def beta(self, orders: np.ndarray) -> np.ndarray:
        """Return the out-of-plane wavenumbers, on the ``Im beta >= 0`` branch.

        The branch matches ``grax.solvers.common._kz_branch_array`` so that a
        Rayleigh amplitude produced here means the same thing as one produced by
        the modal solver.
        """

        alpha = self.alpha(orders)
        beta = np.sqrt(self.wavenumber**2 - alpha**2 + 0j)
        flip = (np.imag(beta) < 0) | ((np.abs(np.imag(beta)) < 1e-15) & (np.real(beta) < 0))
        return np.where(flip, -beta, beta)

    # -- public evaluation --------------------------------------------------

    def value(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Return ``G`` for the given separations.

        Args:
            dx: In-plane separation ``x - x'`` in nanometers.
            dy: Out-of-plane separation ``y - y'`` in nanometers.

        Returns:
            Complex array shaped like the broadcast of the inputs.
        """

        dx, dy = _as_pair(dx, dy)
        if self.method == "spectral":
            return self._value_spectral(dx, dy)
        spectral, _, _ = self._ewald_spectral_part(dx, dy)
        lattice, _, _ = self._ewald_lattice_part(dx, dy)
        return spectral + lattice

    def gradient(self, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dG/dx, dG/dy)`` for the given separations.

        The derivatives are taken with respect to the *field* point, so a
        derivative along the source normal picks up an overall minus sign.

        Args:
            dx: In-plane separation ``x - x'`` in nanometers.
            dy: Out-of-plane separation ``y - y'`` in nanometers.

        Returns:
            Pair of complex arrays shaped like the broadcast of the inputs.
        """

        return self.value_and_gradient(dx, dy)[1:]

    def value_and_gradient(
        self, dx: np.ndarray, dy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(G, dG/dx, dG/dy)`` in one pass.

        Both layer operators are needed at the same points, and each Ewald half
        produces its value and its gradient from the same intermediates, so
        asking for them together halves the work compared with separate
        :meth:`value` and :meth:`gradient` calls.

        Args:
            dx: In-plane separation ``x - x'`` in nanometers.
            dy: Out-of-plane separation ``y - y'`` in nanometers.

        Returns:
            Triple of complex arrays shaped like the broadcast of the inputs.
        """

        dx, dy = _as_pair(dx, dy)
        if self.method == "spectral":
            gradient = self._gradient_spectral(dx, dy)
            return self._value_spectral(dx, dy), gradient[0], gradient[1]
        spectral, spectral_dx, spectral_dy = self._ewald_spectral_part(dx, dy)
        lattice, lattice_dx, lattice_dy = self._ewald_lattice_part(dx, dy)
        return (
            spectral + lattice,
            spectral_dx + lattice_dx,
            spectral_dy + lattice_dy,
        )

    # -- spectral (Rayleigh) reference path ---------------------------------

    def _value_spectral(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Return ``G`` from the plane-wave series."""

        orders = np.arange(-self.spectral_orders, self.spectral_orders + 1, dtype=float)
        alpha = self.alpha(orders)
        beta = self.beta(orders)
        phase = np.exp(1j * alpha * dx[..., None] + 1j * beta * np.abs(dy)[..., None])
        return (0.5j / self.period) * np.sum(phase / beta, axis=-1)

    def _gradient_spectral(
        self, dx: np.ndarray, dy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dG/dx, dG/dy)`` from the plane-wave series."""

        orders = np.arange(-self.spectral_orders, self.spectral_orders + 1, dtype=float)
        alpha = self.alpha(orders)
        beta = self.beta(orders)
        phase = np.exp(1j * alpha * dx[..., None] + 1j * beta * np.abs(dy)[..., None])
        prefactor = 0.5j / self.period
        d_dx = prefactor * np.sum(1j * alpha * phase / beta, axis=-1)
        d_dy = prefactor * np.sum(1j * phase, axis=-1) * np.sign(dy)
        return d_dx, d_dy

    # -- Ewald production path ----------------------------------------------

    def _ewald_spectral_part(
        self, dx: np.ndarray, dy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the spectral half of the Ewald sum and its gradient."""

        orders = self._ewald_spectral_orders()
        alpha = self.alpha(orders)
        beta = self.beta(orders)
        abs_dy = np.abs(dy)[..., None]

        lower, upper = _erfc_pair(beta=beta, abs_dy=abs_dy, splitting=self.splitting)
        bracket = lower + upper

        common = np.exp(1j * alpha * dx[..., None]) / beta * (0.25j / self.period)
        value = np.sum(common * bracket, axis=-1)
        d_dx = np.sum(1j * alpha * common * bracket, axis=-1)
        # The two erfc-derivative contributions cancel identically; only the
        # differentiated exponentials survive.
        d_abs_dy = np.sum(common * 1j * beta * (upper - lower), axis=-1)
        return value, d_dx, d_abs_dy * np.sign(dy)

    def _ewald_spectral_orders(self) -> np.ndarray:
        """Return the orders retained in the Ewald spectral sum.

        For large ``|n|`` the term decays like ``exp(-(alpha_n / 2E)^2)``, so the
        reach follows from the splitting parameter and the period alone.
        """

        reach = 3.5 * 2.0 * self.splitting * self.period / (2.0 * np.pi)
        count = min(int(np.ceil(reach)) + 8, _MAX_SPECTRAL_ORDERS)
        return np.arange(-count, count + 1, dtype=float)

    def _ewald_lattice_part(
        self, dx: np.ndarray, dy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the spatial half of the Ewald sum and its gradient.

        The ``m = 0`` image carries the logarithmic singularity of the
        free-space Green function, through ``E_1(z) ~ -gamma - ln z``.
        """

        shift = self._ewald_lattice_cells() * self.period
        rx = dx[..., None] - shift
        ry = np.broadcast_to(dy[..., None], rx.shape)
        r_squared = rx * rx + ry * ry

        series, series_derivative = self._lattice_series(r_squared)
        prefactor = np.exp(1j * self.alpha0 * shift) / (4.0 * np.pi)

        value = np.sum(prefactor * series, axis=-1)
        d_dx = np.sum(prefactor * series_derivative * 2.0 * rx, axis=-1)
        d_dy = np.sum(prefactor * series_derivative * 2.0 * ry, axis=-1)
        return value, d_dx, d_dy

    def _ewald_lattice_cells(self) -> np.ndarray:
        """Return the lattice cell offsets retained in the spatial sum."""

        reach = 4.0 / max(self.splitting * self.period, 1e-12)
        count = min(int(np.ceil(reach)) + 2, _MAX_LATTICE_CELLS)
        return np.arange(-count, count + 1, dtype=float)

    def _lattice_series(self, r_squared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the lattice radial series and its ``d/d(R^2)`` derivative.

        The series is ``sum_p c_p E_{p+1}(z)`` with
        ``c_p = (k^2 / 4 E^2)^p / p!`` and ``z = R^2 E^2``, evaluated through the
        upward recurrence ``E_{p+1}(z) = (exp(-z) - z E_p(z)) / p`` started from
        ``E_1``. Because ``dE_{p+1}/dz = -E_p`` and ``dz/d(R^2) = E^2``, the
        derivative reuses the same ladder one rung down.

        Args:
            r_squared: Squared distance to each lattice image, in nm^2.

        Returns:
            The series and its derivative with respect to ``R^2``.
        """

        splitting_squared = self.splitting**2
        z = np.maximum(np.asarray(r_squared, dtype=float) * splitting_squared, _TINY)
        exp_neg_z = np.exp(-z)

        e_lower = exp_neg_z / z  # E_0
        e_current = exp1(z).astype(complex)  # E_1

        ratio = (self.wavenumber**2) / (4.0 * splitting_squared)
        coefficient = 1.0 + 0.0j

        total = e_current.astype(complex)
        total_derivative = -splitting_squared * e_lower.astype(complex)

        for p in range(1, self._lattice_terms):
            e_next = (exp_neg_z - z * e_current) / p
            coefficient = coefficient * ratio / p
            total = total + coefficient * e_next
            total_derivative = total_derivative - coefficient * splitting_squared * e_current
            e_lower = e_current
            e_current = e_next

        return total, total_derivative


def _as_pair(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return both separations as float arrays broadcast to a common shape."""

    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    return np.broadcast_arrays(dx, dy)


def _lattice_expansion_terms(wavenumber: complex, splitting: float) -> int:
    """Return how many terms the lattice radial series needs.

    The coefficients are ``(k^2 / 4E^2)^p / p!``, so the series turns over once
    ``p`` passes the ratio and then falls factorially.

    Args:
        wavenumber: Medium wavenumber.
        splitting: Ewald splitting parameter.

    Returns:
        Term count, at least a handful and at most a few dozen.
    """

    ratio = abs(complex(wavenumber) ** 2) / (4.0 * splitting**2)
    terms = int(max(8, 4 * ratio + 12))
    if terms > _MAX_LATTICE_TERMS:
        raise ValueError(
            "The Ewald splitting parameter is too small for this wavelength: "
            f"k^2 / 4E^2 = {ratio:.4g} would need about {terms} terms in the lattice "
            f"series, above the {_MAX_LATTICE_TERMS} term limit. The series carries "
            "(k^2 / 4E^2)^p / p!, which only starts to decay past p = k^2 / 4E^2, so a "
            "small splitting parameter overflows long before it converges. Use at least "
            f"ewald_splitting = {abs(complex(wavenumber)) / 2.0:.6g} for this medium, or "
            "leave it unset to take the default."
        )
    return terms


def _erfc_pair(
    *,
    beta: np.ndarray,
    abs_dy: np.ndarray,
    splitting: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two ``exp * erfc`` products of the Ewald spectral term.

    Returns ``(A, B)`` with

        A = exp(-i beta |Y|) erfc(u + E|Y|)
        B = exp(+i beta |Y|) erfc(u - E|Y|),  u = -i beta / 2E

    Neither factor is formed on its own: ``exp(-i beta |Y|)`` grows like
    ``exp(Im beta |Y|)`` for evanescent orders while the ``erfc`` decays just as
    fast. Both products share the same closed-form combined exponent

        -i beta |Y| - (u + E|Y|)^2 = +i beta |Y| - (u - E|Y|)^2
                                   = beta^2 / 4E^2 - E^2 Y^2

    so each is evaluated as that bounded exponential times ``erfcx``, which is
    the scaled complementary error function ``exp(z^2) erfc(z)``.

    Args:
        beta: Out-of-plane wavenumbers on the ``Im beta >= 0`` branch.
        abs_dy: Absolute out-of-plane separation.
        splitting: Ewald splitting parameter.

    Returns:
        The two products, each shaped like the broadcast of the inputs.
    """

    u = -1j * beta / (2.0 * splitting)
    shift = splitting * abs_dy
    combined = np.exp(beta**2 / (4.0 * splitting**2) - shift**2)

    lower = combined * erfcx(u + shift)

    upper_argument = u - shift
    # erfcx is only accurate for Re(z) >= 0; mirror the rest through
    # erfc(z) = 2 - erfc(-z).
    mirrored = np.real(upper_argument) < 0.0
    safe = np.where(mirrored, -upper_argument, upper_argument)
    direct = combined * erfcx(safe)
    upper = np.where(mirrored, 2.0 * np.exp(1j * beta * abs_dy) - direct, direct)
    return lower, upper


def free_space_green(wavenumber: complex, distance: np.ndarray) -> np.ndarray:
    """Return the free-space 2D Green function ``(i/4) H_0^(1)(k R)``.

    Used to peel the logarithmic singularity off the periodic kernel before
    quadrature, and as an independent check of the periodic evaluation.

    Args:
        wavenumber: Medium wavenumber.
        distance: Separation ``R`` in nanometers.

    Returns:
        Complex array shaped like ``distance``.
    """

    return 0.25j * hankel1(0, complex(wavenumber) * np.asarray(distance, dtype=float))
