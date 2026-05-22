"""Numerical-convergence optimization helpers for grating simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field

import numpy as np

from grax import BaseGrating, get_default_parameter_study_ranges, run_simulation
from grax.parameter_sweep import ParameterSweepSeries

from .optimize import _resolve_optimizer_backend

__all__ = [
    "SimulationConvergenceConfig",
    "SimulationConvergenceEnergyResult",
    "SimulationConvergenceResult",
    "optimize_simulation_convergence",
]


def _normalize_fourier_values(values: Sequence[int] | None) -> np.ndarray:
    """Normalize Fourier-order sweep values.

    Args:
        values: Candidate Fourier-order values, or ``None`` to use defaults.

    Returns:
        Sorted unique Fourier-order values in ascending order.
    """

    default_fourier, _, _ = get_default_parameter_study_ranges()
    selected_values = default_fourier if values is None else np.asarray(values, dtype=int)
    normalized = np.unique(np.asarray(selected_values, dtype=int))
    if normalized.size == 0:
        raise ValueError("fourier_orders_values must be provided and non-empty.")
    if np.any(normalized <= 0):
        raise ValueError("fourier_orders_values values must be > 0.")
    return np.sort(normalized)


def _normalize_resolution_values(values: Sequence[float] | None, *, default: np.ndarray) -> np.ndarray:
    """Normalize resolution sweep values.

    Args:
        values: Candidate resolution values, or ``None`` to use defaults.
        default: Default candidate values to use when ``values`` is ``None``.

    Returns:
        Sorted unique resolution values in coarse-to-fine order.
    """

    selected_values = default if values is None else np.asarray(values, dtype=float)
    normalized = np.unique(np.asarray(selected_values, dtype=float))
    if normalized.size == 0:
        raise ValueError("Resolution sweep values must be provided and non-empty.")
    if np.any(normalized <= 0.0):
        raise ValueError("Resolution sweep values must be > 0.")
    return np.sort(normalized)[::-1]


def _select_stable_index(series: ParameterSweepSeries, tolerance: float) -> tuple[int, bool]:
    """Select the coarsest index that is stable against all finer values.

    Args:
        series: One sweep series ordered from coarse to fine.
        tolerance: Maximum allowed relative change between adjacent points.

    Returns:
        A tuple of ``(selected_index, converged)``.
    """

    successful_indices = np.where(~series.errors)[0]
    if successful_indices.size == 0:
        raise RuntimeError(f"Sweep {series.parameter!r} produced no successful samples.")
    if np.any(series.errors):
        return int(successful_indices[-1]), False

    efficiencies = np.asarray(series.efficiencies, dtype=float)
    if efficiencies.size == 1:
        return 0, True

    denominator = np.maximum(
        np.maximum(np.abs(efficiencies[:-1]), np.abs(efficiencies[1:])),
        1.0e-12,
    )
    relative_changes = np.abs(np.diff(efficiencies)) / denominator
    suffix_max = np.maximum.accumulate(relative_changes[::-1])[::-1]
    stable_indices = np.where(suffix_max <= tolerance)[0]
    if stable_indices.size == 0:
        return int(efficiencies.size - 1), False
    return int(stable_indices[0]), True


def _run_convergence_sweep(
    *,
    grating: BaseGrating,
    parameter: str,
    values: np.ndarray,
    energy_ev: float,
    grazing_angle_deg: float,
    diffraction_order: int,
    fixed_fourier_orders: int,
    backend: str,
    validate_physical_results: bool,
) -> ParameterSweepSeries:
    """Run one convergence sweep for one energy and one parameter.

    Args:
        grating: Baseline grating used for the sweep.
        parameter: Parameter name being swept.
        values: Ordered sweep values from coarse to fine.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Fixed grazing angle in degrees.
        diffraction_order: Selected diffraction order.
        fixed_fourier_orders: Fourier truncation used when sweeping x/z resolution.
        backend: RCWA backend to use.
        validate_physical_results: Whether to validate physical output ranges.

    Returns:
        Sweep series with efficiencies and failure flags aligned to ``values``.
    """

    efficiencies = np.full(values.shape, np.nan, dtype=float)
    errors = np.zeros(values.shape, dtype=bool)

    for index, value in enumerate(values):
        try:
            if parameter == "fourier_orders":
                sweep_grating = grating
                fourier_orders = int(value)
            else:
                sweep_grating = copy(grating)
                setattr(sweep_grating, parameter, float(value))
                fourier_orders = int(fixed_fourier_orders)

            result = run_simulation(
                grating=sweep_grating,
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(grazing_angle_deg),
                diffraction_order=int(diffraction_order),
                fourier_orders=fourier_orders,
                validate_physical_results=validate_physical_results,
                backend=backend,
            )
            efficiencies[index] = float(result.selected_efficiency)
        except Exception:
            errors[index] = True

    return ParameterSweepSeries(
        parameter=parameter,
        values=np.asarray(values),
        efficiencies=efficiencies,
        errors=errors,
    )


@dataclass(frozen=True)
class SimulationConvergenceConfig:
    """Run configuration for numerical convergence optimization.

    Attributes:
        grating: Baseline grating used for the convergence study.
        energies_ev: Photon energies used to assess convergence.
        grazing_angle_deg: Fixed grazing angle used for all simulations.
        diffraction_order: Positive diffraction order evaluated during the study.
        fourier_orders_values: Candidate Fourier-order values, ordered coarse to fine.
        x_resolution_values: Candidate x-resolution values in nanometers, coarse to fine.
        z_resolution_values: Candidate z-resolution values in nanometers, coarse to fine.
        relative_tolerance: Maximum relative change allowed between adjacent points.
        backend: Requested compute backend.
        validate_physical_results: Whether to validate physical output ranges.
    """

    grating: BaseGrating
    energies_ev: Sequence[float]
    grazing_angle_deg: float
    diffraction_order: int = 1
    fourier_orders_values: Sequence[int] | None = None
    x_resolution_values: Sequence[float] | None = None
    z_resolution_values: Sequence[float] | None = None
    relative_tolerance: float = 5.0e-3
    backend: str = "auto"
    validate_physical_results: bool = True

    def __post_init__(self) -> None:
        """Normalize candidate grids and validate the configuration."""

        object.__setattr__(self, "energies_ev", np.asarray(self.energies_ev, dtype=float))
        object.__setattr__(
            self,
            "fourier_orders_values",
            _normalize_fourier_values(self.fourier_orders_values),
        )
        _, default_x, default_z = get_default_parameter_study_ranges()
        object.__setattr__(
            self,
            "x_resolution_values",
            _normalize_resolution_values(self.x_resolution_values, default=default_x),
        )
        object.__setattr__(
            self,
            "z_resolution_values",
            _normalize_resolution_values(self.z_resolution_values, default=default_z),
        )

        if self.energies_ev.size == 0:
            raise ValueError("energies_ev must be provided and non-empty.")
        if np.any(self.energies_ev <= 0.0):
            raise ValueError("energies_ev values must be > 0.")
        if self.grazing_angle_deg <= 0.0:
            raise ValueError("grazing_angle_deg must be > 0.")
        if self.diffraction_order <= 0:
            raise ValueError("diffraction_order must be > 0.")
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance < 0.0:
            raise ValueError("relative_tolerance must be finite and >= 0.")
        if self.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be one of: auto, numba, numpy.")


@dataclass(frozen=True)
class SimulationConvergenceEnergyResult:
    """Convergence diagnostics for one photon energy.

    Attributes:
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Fixed grazing angle used for the sweeps.
        parameter_sweeps: Sweep series keyed by parameter name.
        selected_indices: Selected stable index for each parameter.
        selected_values: Selected stable value for each parameter.
        parameter_converged: Whether each parameter reached the tolerance target.
    """

    energy_ev: float
    grazing_angle_deg: float
    parameter_sweeps: dict[str, ParameterSweepSeries]
    selected_indices: dict[str, int]
    selected_values: dict[str, float]
    parameter_converged: dict[str, bool]


@dataclass(frozen=True)
class SimulationConvergenceResult:
    """Result bundle returned by the simulation convergence optimizer.

    Attributes:
        energies_ev: Energies included in the study.
        grazing_angle_deg: Fixed grazing angle used for the study.
        diffraction_order: Selected diffraction order.
        relative_tolerance: Relative-change tolerance used for selection.
        backend_requested: Backend requested by the user.
        backend_effective: Backend actually used for the evaluation.
        fourier_orders_values: Fourier-order candidate values.
        x_resolution_values: x-resolution candidate values.
        z_resolution_values: z-resolution candidate values.
        energy_results: Per-energy sweep diagnostics.
        selected_fourier_orders: Coarsest converged Fourier order.
        selected_x_resolution_nm: Coarsest converged x resolution.
        selected_z_resolution_nm: Coarsest converged z resolution.
        converged: Whether all parameters converged for all energies.
    """

    energies_ev: np.ndarray
    grazing_angle_deg: float
    diffraction_order: int
    relative_tolerance: float
    backend_requested: str
    backend_effective: str
    fourier_orders_values: np.ndarray
    x_resolution_values: np.ndarray
    z_resolution_values: np.ndarray
    energy_results: list[SimulationConvergenceEnergyResult] = field(default_factory=list)
    selected_fourier_orders: int = 0
    selected_x_resolution_nm: float = 0.0
    selected_z_resolution_nm: float = 0.0
    converged: bool = False


def optimize_simulation_convergence(
    config: SimulationConvergenceConfig | Mapping[str, object],
) -> SimulationConvergenceResult:
    """Find the coarsest simulation settings that remain numerically stable.

    Args:
        config: Convergence configuration or a plain mapping with the same keys.

    Returns:
        Convergence result bundle with per-energy sweep diagnostics and the
        selected Fourier/order discretization values.
    """

    if not isinstance(config, SimulationConvergenceConfig):
        config = SimulationConvergenceConfig(**dict(config))

    backend_effective = _resolve_optimizer_backend(config.backend)
    fixed_fourier_orders = int(np.max(config.fourier_orders_values))
    energy_results: list[SimulationConvergenceEnergyResult] = []

    for energy_ev in config.energies_ev:
        parameter_sweeps = {
            "fourier_orders": _run_convergence_sweep(
                grating=config.grating,
                parameter="fourier_orders",
                values=np.asarray(config.fourier_orders_values),
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(config.grazing_angle_deg),
                diffraction_order=int(config.diffraction_order),
                fixed_fourier_orders=fixed_fourier_orders,
                backend=backend_effective,
                validate_physical_results=config.validate_physical_results,
            ),
            "x_resolution_nm": _run_convergence_sweep(
                grating=config.grating,
                parameter="x_resolution_nm",
                values=np.asarray(config.x_resolution_values),
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(config.grazing_angle_deg),
                diffraction_order=int(config.diffraction_order),
                fixed_fourier_orders=fixed_fourier_orders,
                backend=backend_effective,
                validate_physical_results=config.validate_physical_results,
            ),
            "z_resolution_nm": _run_convergence_sweep(
                grating=config.grating,
                parameter="z_resolution_nm",
                values=np.asarray(config.z_resolution_values),
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(config.grazing_angle_deg),
                diffraction_order=int(config.diffraction_order),
                fixed_fourier_orders=fixed_fourier_orders,
                backend=backend_effective,
                validate_physical_results=config.validate_physical_results,
            ),
        }

        selected_indices: dict[str, int] = {}
        selected_values: dict[str, float] = {}
        parameter_converged: dict[str, bool] = {}
        for parameter, sweep in parameter_sweeps.items():
            selected_index, converged = _select_stable_index(
                sweep,
                float(config.relative_tolerance),
            )
            selected_indices[parameter] = int(selected_index)
            selected_values[parameter] = float(sweep.values[selected_index])
            parameter_converged[parameter] = bool(converged)

        energy_results.append(
            SimulationConvergenceEnergyResult(
                energy_ev=float(energy_ev),
                grazing_angle_deg=float(config.grazing_angle_deg),
                parameter_sweeps=parameter_sweeps,
                selected_indices=selected_indices,
                selected_values=selected_values,
                parameter_converged=parameter_converged,
            )
        )

    final_selected_indices = {
        parameter: max(
            energy_result.selected_indices[parameter] for energy_result in energy_results
        )
        for parameter in ("fourier_orders", "x_resolution_nm", "z_resolution_nm")
    }
    final_selected_values = {
        "fourier_orders": float(config.fourier_orders_values[final_selected_indices["fourier_orders"]]),
        "x_resolution_nm": float(config.x_resolution_values[final_selected_indices["x_resolution_nm"]]),
        "z_resolution_nm": float(config.z_resolution_values[final_selected_indices["z_resolution_nm"]]),
    }

    return SimulationConvergenceResult(
        energies_ev=np.asarray(config.energies_ev, dtype=float),
        grazing_angle_deg=float(config.grazing_angle_deg),
        diffraction_order=int(config.diffraction_order),
        relative_tolerance=float(config.relative_tolerance),
        backend_requested=str(config.backend),
        backend_effective=backend_effective,
        fourier_orders_values=np.asarray(config.fourier_orders_values),
        x_resolution_values=np.asarray(config.x_resolution_values),
        z_resolution_values=np.asarray(config.z_resolution_values),
        energy_results=energy_results,
        selected_fourier_orders=int(final_selected_values["fourier_orders"]),
        selected_x_resolution_nm=float(final_selected_values["x_resolution_nm"]),
        selected_z_resolution_nm=float(final_selected_values["z_resolution_nm"]),
        converged=all(
            all(energy_result.parameter_converged.values())
            for energy_result in energy_results
        ),
    )
