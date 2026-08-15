"""Core one-point simulation helpers."""

from __future__ import annotations

import csv
import dataclasses
import importlib
import inspect
import logging
import warnings
import weakref
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext as _nullcontext
from copy import copy
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ..gratings import BaseGrating
from ..solvers import res0, res1, res2, res2_dm
from ..solvers.neviere import NeviereOptions, build_grating_epsilon_sampler, coerce_neviere_options
from ._memory import PeakMemorySampler, format_memory_mb
from ._profiling import SolverProfiler
from .models import (
    BatchSimulationResult,
    CaseExecutionResult,
    SimulationResult,
    SingleSimulationResult,
)

logger = logging.getLogger(__name__)

# Above this many effective Fourier orders (fourier_orders * num_supercells),
# warn the caller: RCWA solve cost grows worse than linearly with order
# count, and the solver hard-caps at 100 orders (201 modes) regardless.
_SUPERCELL_FOURIER_ORDER_WARNING_THRESHOLD = 40

#: Electromagnetic solvers selectable through ``solver=``.
SOLVER_NAMES = ("rcwa", "neviere")


def _validate_solver(solver: str) -> str:
    """Return the validated solver name."""

    if solver not in SOLVER_NAMES:
        raise ValueError(f"solver must be one of {SOLVER_NAMES}, got {solver!r}.")
    return solver


def _warn_if_numpy_backend_requested(backend: str, *, stacklevel: int = 3) -> None:
    """Warn when callers explicitly request the deprecated NumPy backend."""

    if str(backend).lower() != "numpy":
        return
    warnings.warn(
        "backend='numpy' is deprecated and will be removed in a future version. "
        "Use backend='numba' or rely on the default numba backend instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _simulation_api():
    """Return the public simulation package for monkeypatch-compatible dispatch."""

    return importlib.import_module("grax.simulation")


def _supports_interactive_pause() -> bool:
    """Return whether the active Matplotlib backend supports interactive pause."""

    return "agg" not in plt.get_backend().lower()


_shown_figures: "weakref.WeakSet[plt.Figure]" = weakref.WeakSet()


def _refresh_interactive_figure(figure: plt.Figure, *, pause_seconds: float = 0.05) -> None:
    """Show and refresh an interactive Matplotlib figure.

    ``plt.show()`` is only invoked the first time a given figure is shown.
    On macOS's native "macosx" backend, re-invoking it on every refresh
    forces the window to the front (orderFront) each time, which prevents
    the user from minimizing it or sending it behind another window.
    """

    if not _supports_interactive_pause():
        figure.canvas.draw_idle()
        return

    if figure not in _shown_figures:
        plt.figure(figure.number)
        plt.show(block=False)
        _shown_figures.add(figure)

    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    plt.pause(pause_seconds)


def _validate_reflected_efficiencies(
    *,
    photon_energy_ev: float,
    grazing_angle_deg: float,
    period_nm: float,
    orders: np.ndarray,
    efficiency_all: np.ndarray,
    min_efficiency: float,
    max_reflected_efficiency: float,
    max_total_reflected_efficiency: float,
) -> None:
    """Validate reflected efficiencies and raise on non-physical results.

    Args:
        photon_energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        period_nm: Grating period in nanometers.
        orders: Calculated diffraction orders.
        efficiency_all: Reflected efficiency for all orders.
        min_efficiency: Minimum accepted efficiency value.
        max_reflected_efficiency: Maximum accepted single-order efficiency.
        max_total_reflected_efficiency: Maximum accepted propagating reflected sum.
    """

    minimum_efficiency = float(np.min(efficiency_all))
    maximum_efficiency = float(np.max(efficiency_all))
    if minimum_efficiency < min_efficiency:
        raise ValueError(
            "Non-physical negative diffraction efficiency detected at "
            f"{photon_energy_ev:.6g} eV: min={minimum_efficiency:.6g}"
        )
    if maximum_efficiency > max_reflected_efficiency:
        raise ValueError(
            "Non-physical reflected diffraction efficiency detected at "
            f"{photon_energy_ev:.6g} eV: max={maximum_efficiency:.6g}"
        )

    wavelength_nm = 1239.8 / photon_energy_ev
    k0 = 2.0 * np.pi / wavelength_nm
    k_parallel = np.sin(np.deg2rad(90.0 - grazing_angle_deg))
    kx = k0 * k_parallel + (2.0 * np.pi * orders / period_nm)
    propagating_mask = np.abs(kx) <= k0 * (1.0 + 1e-9)
    total_reflected_efficiency = float(np.sum(efficiency_all[propagating_mask]))
    if total_reflected_efficiency > max_total_reflected_efficiency:
        raise ValueError(
            "Non-physical total reflected efficiency detected at "
            f"{photon_energy_ev:.6g} eV: sum={total_reflected_efficiency:.6g}"
        )


def _log_simulation_memory_usage(
    *,
    energy_ev: float,
    grazing_angle_deg: float,
    sampler: PeakMemorySampler,
) -> None:
    """Log peak process RSS for a completed simulation when available.

    Args:
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        sampler: Completed peak memory sampler.
    """
    if sampler.peak_memory_bytes is None or sampler.memory_delta_bytes is None:
        return
    logger.info(
        "Simulation completed at %.2f eV, grazing=%.3f deg, peak_ram=%s, ram_delta=%s",
        energy_ev,
        grazing_angle_deg,
        format_memory_mb(sampler.peak_memory_bytes),
        format_memory_mb(sampler.memory_delta_bytes),
    )


def _run_single_realization(
    grating: BaseGrating,
    *,
    energy_ev: float,
    grazing_angle_deg: float,
    polarization: Literal["s", "p"],
    effective_roughness_sigma_nm: float | None,
    num_supercells: int,
    effective_period_nm: float,
    effective_fourier_orders: int,
    _memory_mode: Literal["legacy_dense", "low_memory"],
    _profiler: SolverProfiler | None,
    backend: str,
    solver: str,
    neviere_options: NeviereOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one full solve and return raw (orders, efficiency_all, diffraction_angle_all).

    One "realization" is one concrete roughness draw (or the single
    deterministic solve when there's no roughness/Debye-Waller roughness).
    Averaging several realizations together is the caller's responsibility.
    """

    wavelength_nm = 1239.8 / float(energy_ev)
    k_parallel = np.sin(np.deg2rad(90.0 - float(grazing_angle_deg)))
    with _profiler.record("texture_generation") if _profiler is not None else _nullcontext():
        textures, profile = grating.build_textures(
            float(energy_ev),
            n_inc=1.0 + 0.0j,
            _memory_mode=_memory_mode,
        )

    parm = res0(1 if polarization == "s" else -1)
    aa = res1(
        wavelength_nm,
        effective_period_nm,
        textures,
        effective_fourier_orders,
        k_parallel,
        parm,
        _profiler=_profiler,
        _fourier_backend=backend,
    )
    if solver == "neviere":
        epsilon_sampler = None
        if neviere_options.z_sampling == "continuous":
            epsilon_sampler = build_grating_epsilon_sampler(
                grating,
                photon_energy_ev=float(energy_ev),
                period_nm=effective_period_nm,
                orders=aa.orders,
                fourier_backend=backend,
            )
        ef = res2_dm(
            aa,
            profile,
            parm,
            roughness_sigma_nm=effective_roughness_sigma_nm,
            options=neviere_options,
            epsilon_sampler=epsilon_sampler,
            _profiler=_profiler,
        )
    else:
        ef = res2(
            aa,
            profile,
            parm,
            roughness_sigma_nm=effective_roughness_sigma_nm,
            _profiler=_profiler,
        )

    with _profiler.record("postprocessing") if _profiler is not None else _nullcontext():
        orders = np.asarray(ef.inc_top_reflected.order, dtype=float) / float(num_supercells)
        all_efficiency = np.asarray(
            np.real_if_close(ef.inc_top_reflected.efficiency),
            dtype=float,
        )
        all_diffraction_angle_deg = np.asarray(90.0 - ef.inc_top_reflected.theta, dtype=float)
    return orders, all_efficiency, all_diffraction_angle_deg


def run_simulation(
    *,
    grating: BaseGrating,
    energy_ev: float,
    grazing_angle_deg: float,
    diffraction_order: int = 1,
    fourier_orders: int = 25,
    roughness_sigma_nm: float | None = None,
    polarization: Literal["s", "p"] = "s",
    validate_physical_results: bool = True,
    max_reflected_efficiency: float = 1.05,
    min_efficiency: float = -1e-8,
    max_total_reflected_efficiency: float = 1.05,
    _memory_mode: Literal["legacy_dense", "low_memory"] = "low_memory",
    _profiler: SolverProfiler | None = None,
    backend: str = "numba",
    solver: Literal["rcwa", "neviere"] = "rcwa",
    neviere_options: NeviereOptions | dict[str, object] | None = None,
) -> SingleSimulationResult:
    """Run one simulation case and return a typed result.

    Args:
        grating: Grating profile and material stack.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        diffraction_order: Positive diffraction order to select.
        fourier_orders: Number of Fourier orders on one side of zero.
        roughness_sigma_nm: Optional rms roughness in nanometers.
        polarization: Incident polarization. ``"s"`` selects TE (default);
            ``"p"`` selects TM.
        validate_physical_results: Whether to validate reflected efficiencies.
        max_reflected_efficiency: Maximum allowed single-order reflected efficiency.
        min_efficiency: Minimum allowed efficiency.
        max_total_reflected_efficiency: Maximum allowed sum of propagating reflected efficiencies.
        _memory_mode: Internal texture-generation mode. ``"low_memory"`` is the
            public path. ``"legacy_dense"`` keeps the older dense-grid path
            available for internal regression and debugging.
        backend: Fourier coefficient backend selector. ``"numba"`` is the
            default backend. ``"numpy"`` remains available temporarily for
            compatibility but is deprecated and will be removed in a future
            version.
        solver: Electromagnetic solver. ``"rcwa"`` (default) eigen-decomposes
            each layer; ``"neviere"`` integrates the coupled first-order system
            in ``z`` with the Nevière differential method. Both use the same
            Fourier operators and the same efficiency extraction, so they are
            directly comparable.
        neviere_options: Integration settings for ``solver="neviere"``, as a
            :class:`grax.NeviereOptions` or an equivalent mapping. Ignored by
            the RCWA solver.

    Returns:
        Single-case simulation result.
    """

    if not isinstance(grating, BaseGrating):
        raise TypeError("grating must derive from BaseGrating.")
    if roughness_sigma_nm is not None and roughness_sigma_nm < 0.0:
        raise ValueError("roughness_sigma_nm must be >= 0 when provided.")
    if roughness_sigma_nm is not None and grating.roughness is not None:
        raise ValueError("Pass roughness either on the grating or as roughness_sigma_nm, not both.")
    effective_roughness_sigma_nm = roughness_sigma_nm
    if grating.roughness is not None and grating.roughness.kind == "debye-waller":
        stack = grating.resolved_stack()
        if stack.has_per_layer_roughness():
            # Uncorrelated interface roughness adds in quadrature, reducing the
            # per-layer Debye-Waller damping to a single effective sigma.
            interface_sigmas = stack.interface_roughness_sigmas_bottom_up(
                float(grating.roughness.sigma_nm)
            )
            effective_roughness_sigma_nm = float(
                np.sqrt(sum(sigma * sigma for sigma in interface_sigmas))
            )
        else:
            effective_roughness_sigma_nm = float(grating.roughness.sigma_nm)
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'.")
    if _memory_mode not in {"legacy_dense", "low_memory"}:
        raise ValueError("memory_mode must be 'low_memory' or 'legacy_dense'.")
    _validate_solver(solver)
    resolved_neviere_options = coerce_neviere_options(neviere_options)
    _warn_if_numpy_backend_requested(backend, stacklevel=2)

    logger.info(
        "Running %s simulation at %.2f eV, grazing=%.3f deg, fourier_orders=%s, memory_mode=%s",
        solver,
        energy_ev,
        grazing_angle_deg,
        fourier_orders,
        _memory_mode,
    )
    num_supercells = grating._roughness_num_supercells()
    effective_period_nm = grating.period_nm * num_supercells
    effective_fourier_orders = int(fourier_orders) * num_supercells
    num_realizations = (
        int(grating.roughness.num_realizations)
        if grating.roughness is not None and grating.roughness.kind == "random-interface"
        else 1
    )
    if effective_fourier_orders > _SUPERCELL_FOURIER_ORDER_WARNING_THRESHOLD:
        warnings.warn(
            f"fourier_orders={fourier_orders} with num_supercells={num_supercells} requires "
            f"{effective_fourier_orders} effective Fourier orders. RCWA solve cost grows worse "
            "than linearly with this count, so this combination may be very slow; consider "
            "reducing fourier_orders or num_supercells."
            + (
                f" This is on top of num_realizations={num_realizations}, which multiplies the "
                "cost again (each realization is an independent full solve)."
                if num_realizations > 1
                else ""
            ),
            UserWarning,
            stacklevel=2,
        )

    with PeakMemorySampler() as memory_sampler:
        if num_realizations > 1:
            realization_seeds = grating.roughness.realization_seeds()
            orders: np.ndarray | None = None
            all_diffraction_angle_deg: np.ndarray | None = None
            efficiency_sum: np.ndarray | None = None
            for realization_seed in realization_seeds:
                realization_grating = copy(grating)
                realization_grating.roughness = dataclasses.replace(
                    grating.roughness, seed=realization_seed, num_realizations=1
                )
                realization_orders, realization_efficiency, realization_angle = _run_single_realization(
                    realization_grating,
                    energy_ev=energy_ev,
                    grazing_angle_deg=grazing_angle_deg,
                    polarization=polarization,
                    effective_roughness_sigma_nm=effective_roughness_sigma_nm,
                    num_supercells=num_supercells,
                    effective_period_nm=effective_period_nm,
                    effective_fourier_orders=effective_fourier_orders,
                    _memory_mode=_memory_mode,
                    _profiler=_profiler,
                    backend=backend,
                    solver=solver,
                    neviere_options=resolved_neviere_options,
                )
                if orders is None:
                    orders = realization_orders
                    all_diffraction_angle_deg = realization_angle
                    efficiency_sum = realization_efficiency
                else:
                    efficiency_sum = efficiency_sum + realization_efficiency
            all_efficiency = efficiency_sum / float(num_realizations)
        else:
            orders, all_efficiency, all_diffraction_angle_deg = _run_single_realization(
                grating,
                energy_ev=energy_ev,
                grazing_angle_deg=grazing_angle_deg,
                polarization=polarization,
                effective_roughness_sigma_nm=effective_roughness_sigma_nm,
                num_supercells=num_supercells,
                effective_period_nm=effective_period_nm,
                effective_fourier_orders=effective_fourier_orders,
                _memory_mode=_memory_mode,
                _profiler=_profiler,
                backend=backend,
                solver=solver,
                neviere_options=resolved_neviere_options,
            )

        order_index = np.where(np.isclose(orders, -float(diffraction_order)))[0]
        if len(order_index) != 1:
            raise ValueError(f"Unable to locate diffraction order {diffraction_order}")
        idx = int(order_index[0])

        if validate_physical_results:
            _validate_reflected_efficiencies(
                photon_energy_ev=float(energy_ev),
                grazing_angle_deg=float(grazing_angle_deg),
                period_nm=grating.period_nm,
                orders=orders,
                efficiency_all=all_efficiency,
                min_efficiency=min_efficiency,
                max_reflected_efficiency=max_reflected_efficiency,
                max_total_reflected_efficiency=max_total_reflected_efficiency,
            )

        if _profiler is not None:
            _profiler.finalize()

        result = SingleSimulationResult(
            energy_ev=float(energy_ev),
            grazing_angle_deg=float(grazing_angle_deg),
            orders=orders,
            selected_efficiency=float(all_efficiency[idx]),
            selected_diffraction_angle_deg=float(all_diffraction_angle_deg[idx]),
            efficiency_all=all_efficiency,
            diffraction_angle_all=all_diffraction_angle_deg,
            diffraction_order=int(diffraction_order),
            fourier_orders=int(fourier_orders),
            roughness_sigma_nm=effective_roughness_sigma_nm,
            num_supercells=num_supercells,
            num_realizations=num_realizations,
            polarization=polarization,
            solver=solver,
            solver_options=(
                resolved_neviere_options.to_dict() if solver == "neviere" else None
            ),
        )

    _log_simulation_memory_usage(
        energy_ev=float(energy_ev),
        grazing_angle_deg=float(grazing_angle_deg),
        sampler=memory_sampler,
    )
    return result


run_simulation.__signature__ = inspect.signature(run_simulation).replace(
    parameters=[
        parameter
        for parameter in inspect.signature(run_simulation).parameters.values()
        if parameter.name not in {"_profiler", "_memory_mode"}
    ]
)


def _clone_grating_with_overrides(
    grating: BaseGrating,
    *,
    x_resolution_nm: float | None,
    z_resolution_nm: float | None,
) -> BaseGrating:
    """Return a grating copy with optional resolution overrides."""

    cloned_grating = copy(grating)
    if x_resolution_nm is not None:
        cloned_grating.x_resolution_nm = float(x_resolution_nm)
    if z_resolution_nm is not None:
        cloned_grating.z_resolution_nm = float(z_resolution_nm)
    return cloned_grating



def load_experimental_csv(path: str | Path) -> np.ndarray:
    """Load a semicolon-separated experimental CSV used for comparison.

    Args:
        path: Path to the experimental CSV file.

    Returns:
        Two-column array with energy and efficiency.
    """

    csv_path = Path(path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for _ in range(3):
            next(handle)
        for line in handle:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            energy_text, efficiency_text, *_ = line.split(";")
            rows.append(
                [
                    float(energy_text.replace(",", ".")),
                    float(efficiency_text.replace(",", ".")),
                ]
            )
    return np.asarray(rows, dtype=float)



def efficiency_for_order(
    orders: Sequence[int] | np.ndarray,
    efficiency_all: Sequence[float] | np.ndarray,
    *,
    diffraction_order: int,
) -> float:
    """Return the efficiency for one diffraction order.

    Args:
        orders: Array of calculated diffraction orders.
        efficiency_all: Efficiency array aligned with ``orders``.
        diffraction_order: Positive diffraction order to extract.

    Returns:
        Efficiency for the requested order, or ``nan`` if absent.
    """

    orders_array = np.asarray(orders, dtype=float)
    efficiency_array = np.asarray(efficiency_all, dtype=float)
    order_index = np.where(np.isclose(orders_array, -diffraction_order))[0]
    if order_index.size == 0:
        return float("nan")
    return float(efficiency_array[int(order_index[0])])


def _iter_case_results(
    results: SingleSimulationResult | CaseExecutionResult | BatchSimulationResult | Iterable[CaseExecutionResult],
) -> Iterator[CaseExecutionResult]:
    """Yield case-style results from supported result containers."""

    if isinstance(results, SingleSimulationResult):
        yield CaseExecutionResult(
            case_id="single",
            index=0,
            label=None,
            energy_ev=results.energy_ev,
            grazing_angle_deg=results.grazing_angle_deg,
            orders=results.orders,
            selected_efficiency=results.selected_efficiency,
            selected_diffraction_angle_deg=results.selected_diffraction_angle_deg,
            efficiency_all=results.efficiency_all,
            diffraction_angle_all=results.diffraction_angle_all,
            status="ok",
        )
        return
    if isinstance(results, CaseExecutionResult):
        yield results
        return
    if isinstance(results, BatchSimulationResult):
        yield from results.cases
        return
    yield from results


def write_all_orders_csv(
    results: SingleSimulationResult | CaseExecutionResult | Iterable[CaseExecutionResult],
    output_path: str | Path,
) -> None:
    """Write all-order efficiencies and angles to a stream-friendly CSV.

    Args:
        results: Single result, case result, or result iterable.
        output_path: CSV output path.
    """

    output = Path(output_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "energy_ev",
                "grazing_angle_deg",
                "order",
                "efficiency",
                "diffraction_angle_deg",
            ]
        )
        for result in _iter_case_results(results):
            if result.status != "ok":
                continue
            for order, efficiency, angle in zip(
                np.asarray(result.orders, dtype=float),
                np.asarray(result.efficiency_all, dtype=float),
                np.asarray(result.diffraction_angle_all, dtype=float),
            ):
                order_value = float(order)
                order_cell = int(order_value) if order_value.is_integer() else order_value
                writer.writerow(
                    [
                        result.case_id,
                        float(result.energy_ev),
                        float(result.grazing_angle_deg),
                        order_cell,
                        float(efficiency),
                        float(angle),
                    ]
                )


def plot_order_subset(
    results: CaseExecutionResult | Iterable[CaseExecutionResult],
    output_filename: str | Path,
    *,
    diffraction_orders: Sequence[int],
    title: str,
) -> None:
    """Save an efficiency-vs-energy plot for selected diffraction orders.

    Args:
        results: Case result or result iterable.
        output_filename: Output image path.
        diffraction_orders: Positive diffraction orders to plot.
        title: Plot title.
    """

    collected = sorted(
        [result for result in _iter_case_results(results) if result.status == "ok"],
        key=lambda result: float(result.energy_ev),
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "d", "v", "x"]
    energies = np.asarray([case.energy_ev for case in collected], dtype=float)
    for index, order in enumerate(diffraction_orders):
        order_efficiency = np.asarray(
            [
                efficiency_for_order(
                    case.orders,
                    case.efficiency_all,
                    diffraction_order=order,
                )
                for case in collected
            ],
            dtype=float,
        )
        axis.plot(
            energies,
            order_efficiency,
            f"{markers[index % len(markers)]}-",
            linewidth=1.0,
            markersize=3.0,
            label=f"Order {order}",
        )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(figure)


class GratingSimulation:
    """Object-style wrapper around :func:`run_simulation`.

    Drives whichever solver ``solver=`` selects. Named ``RCWASimulation``
    until a second solver existed, at which point the old name described only
    one of the two things it can do.
    """

    def __init__(
        self,
        *,
        grating: BaseGrating,
        diffraction_order: int = 1,
        fourier_orders: int = 25,
        grazing_angle_deg: float = 4.0,
        polarization: Literal["s", "p"] = "s",
        live_plot: bool = False,
        validate_physical_results: bool = True,
        max_reflected_efficiency: float = 1.05,
        min_efficiency: float = -1e-8,
        max_total_reflected_efficiency: float = 1.05,
        roughness_sigma_nm: float | None = None,
        backend: str = "numba",
        solver: Literal["rcwa", "neviere"] = "rcwa",
        neviere_options: NeviereOptions | dict[str, object] | None = None,
    ) -> None:
        """Initialize a compatibility simulation object."""

        _validate_solver(solver)
        _warn_if_numpy_backend_requested(backend, stacklevel=2)
        self.grating = grating
        self.diffraction_order = diffraction_order
        self.fourier_orders = fourier_orders
        self.grazing_angle_deg = grazing_angle_deg
        self.polarization = polarization
        self.live_plot = live_plot
        self.validate_physical_results = validate_physical_results
        self.max_reflected_efficiency = max_reflected_efficiency
        self.min_efficiency = min_efficiency
        self.max_total_reflected_efficiency = max_total_reflected_efficiency
        self.roughness_sigma_nm = roughness_sigma_nm
        self.backend = backend
        self.solver = solver
        self.neviere_options = neviere_options
        self._live_comparison_figure = None
        self._live_comparison_axis = None

    def run_single(self, photon_energy_ev: float) -> dict[str, float | np.ndarray]:
        """Run one energy and return the legacy dictionary shape."""

        result = _simulation_api().run_simulation(
            grating=self.grating,
            energy_ev=photon_energy_ev,
            grazing_angle_deg=self.grazing_angle_deg,
            diffraction_order=self.diffraction_order,
            fourier_orders=self.fourier_orders,
            polarization=self.polarization,
            roughness_sigma_nm=self.roughness_sigma_nm,
            validate_physical_results=self.validate_physical_results,
            max_reflected_efficiency=self.max_reflected_efficiency,
            min_efficiency=self.min_efficiency,
            max_total_reflected_efficiency=self.max_total_reflected_efficiency,
            backend=self.backend,
            solver=self.solver,
            neviere_options=self.neviere_options,
        )
        return {
            "orders": result.orders,
            "efficiency": result.selected_efficiency,
            "diffraction_angle_deg": result.selected_diffraction_angle_deg,
            "efficiency_all": result.efficiency_all,
            "diffraction_angle_all": result.diffraction_angle_all,
        }

    def run(self, photon_energy_ev: float | list[float] | np.ndarray) -> SimulationResult:
        """Run one or more energies and return a legacy collected sweep."""

        energies = np.atleast_1d(np.asarray(photon_energy_ev, dtype=float))
        simulation_api = _simulation_api()
        single_results = [
            simulation_api.run_simulation(
                grating=self.grating,
                energy_ev=float(energy),
                grazing_angle_deg=self.grazing_angle_deg,
                diffraction_order=self.diffraction_order,
                fourier_orders=self.fourier_orders,
                polarization=self.polarization,
                roughness_sigma_nm=self.roughness_sigma_nm,
                validate_physical_results=self.validate_physical_results,
                max_reflected_efficiency=self.max_reflected_efficiency,
                min_efficiency=self.min_efficiency,
                max_total_reflected_efficiency=self.max_total_reflected_efficiency,
                backend=self.backend,
                solver=self.solver,
                neviere_options=self.neviere_options,
            )
            for energy in energies
        ]
        orders = single_results[0].orders if single_results else np.asarray([], dtype=float)
        for result in single_results[1:]:
            if not np.array_equal(orders, result.orders):
                raise ValueError("Diffraction orders changed between energy points.")
        return SimulationResult(
            energy_ev=energies,
            orders=orders,
            efficiency=np.asarray([result.selected_efficiency for result in single_results], dtype=float),
            diffraction_angle_deg=np.asarray(
                [result.selected_diffraction_angle_deg for result in single_results], dtype=float
            ),
            efficiency_all=np.asarray([result.efficiency_all for result in single_results], dtype=float),
            diffraction_angle_all=np.asarray(
                [result.diffraction_angle_all for result in single_results], dtype=float
            ),
        )

    def _validate_reflected_efficiencies(
        self,
        *,
        photon_energy_ev: float,
        orders: np.ndarray,
        efficiency_all: np.ndarray,
    ) -> None:
        """Validate reflected efficiencies using legacy object settings."""

        _validate_reflected_efficiencies(
            photon_energy_ev=photon_energy_ev,
            grazing_angle_deg=self.grazing_angle_deg,
            period_nm=self.grating.period_nm,
            orders=orders,
            efficiency_all=efficiency_all,
            min_efficiency=self.min_efficiency,
            max_reflected_efficiency=self.max_reflected_efficiency,
            max_total_reflected_efficiency=self.max_total_reflected_efficiency,
        )

    def load_experimental_csv(self, path: str | Path) -> np.ndarray:
        """Load experimental data from CSV."""

        return load_experimental_csv(path)

    def plot_against_experiment(
        self,
        result: SimulationResult | BatchSimulationResult,
        experimental_data: np.ndarray,
        output_filename: str | Path,
        *,
        live_plot: bool | None = None,
    ) -> None:
        """Plot simulation and experimental efficiency curves."""

        simulation_result = result if isinstance(result, SimulationResult) else result.to_simulation_result()
        live_plot_enabled = self.live_plot if live_plot is None else live_plot
        if live_plot_enabled:
            plt.ion()
            figure = self._live_comparison_figure
            axis = self._live_comparison_axis
            if figure is None or axis is None or not plt.fignum_exists(figure.number):
                figure, axis = plt.subplots(figsize=(10, 7))
                self._live_comparison_figure = figure
                self._live_comparison_axis = axis
            axis.clear()
        else:
            figure, axis = plt.subplots(figsize=(10, 7))
        axis.plot(
            simulation_result.energy_ev,
            simulation_result.efficiency,
            "b-o",
            linewidth=0.5,
            markersize=2.0,
            label="Simulation",
        )
        axis.plot(
            experimental_data[:, 0],
            experimental_data[:, 1],
            "r-s",
            linewidth=0.5,
            markersize=2.0,
            label="Experimental Data",
        )
        axis.set_xlabel("Photon Energy (eV)")
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title("RCWA Simulation vs Experimental Data")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(output_filename, dpi=150, bbox_inches="tight")
        if live_plot_enabled:
            _refresh_interactive_figure(figure)
        else:
            plt.close(figure)
