"""Three-stage multilayer-grating design workflow.

The workflow sizes a periodic multilayer coating for a blazed grating
monochromator working in a chosen diffraction order at fixed CFF, in three
stages driven by a single :class:`MultilayerOptimizationConfig`:

1. :func:`run_d_spacing_study` -- derive a bilayer d-spacing from the grating
   geometry (grazing angle at the configured CFF, then the first-order Bragg
   law) and scan practical d-spacing candidates with XRT planar-multilayer
   reflectivity.
2. :func:`run_gamma_study` -- at the selected d-spacing, scan the bilayer
   thickness ratio ``gamma`` and pick the value with the highest peak
   reflectivity at the target energy.
3. :func:`run_blaze_study` -- build the multilayer-coated blazed grating and
   scan the blaze angle, using graxPy's internal theta search per energy, then
   pick the blaze angle with the highest selected-order efficiency at the target
   energy.

Stages hand values forward only through ``optimization_state.json`` and only when
a config value is the string ``"auto"``. A numeric config value always wins over
anything in the state file, and no stage ever rewrites the config. Stage 2 reads
``config.gamma`` directly -- the gamma suggestion from stage 1 is recorded for
traceability but is not auto-applied.

Two numerical conventions are inherited from the original workflow and kept
deliberately: the geometry d-spacing derivation uses ``HC_EV_NM = 1239.841984``
while :func:`grax.monochromator_grazing_angles_deg` uses ``1239.8`` internally
(immaterial at the 0.1 nm rounding used here), and the XRT reflectivity path puts
``material_a`` on top of a ``material_a`` substrate whereas the graxPy
:class:`grax.MultilayerStack` puts ``material_b`` on top of the configured
substrate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .gratings import BlazedGrating
from .materials import MaterialSpec
from .multilayer_reflectivity import MultilayerReflectivity
from .simulation import monochromator_grazing_angles_deg, run_multilayer_theta_search_sweep
from .simulation.core import normalize_polarization
from .stacks import MultilayerStack

__all__ = [
    "BlazeStudyResult",
    "DSpacingStudyResult",
    "GammaStudyResult",
    "MultilayerOptimizationConfig",
    "d_spacing_bounds_from_bragg_angles",
    "energy_to_wavelength_nm",
    "ensure_target_energy",
    "intersect_search_bounds",
    "resolve_configured_value",
    "run_blaze_study",
    "run_d_spacing_study",
    "run_gamma_study",
    "select_target_energy_optimum",
    "update_optimization_state",
]

HC_EV_NM = 1239.841984


@dataclass(frozen=True)
class MultilayerOptimizationConfig:
    """Every knob for the three multilayer-optimization stages.

    The defaults reproduce the Ru/B4C second-order study the workflow was ported
    from. ``material_a`` / ``material_b`` / ``substrate_material`` are
    ``(name, density_g_cm3)`` pairs (a :class:`grax.MaterialSpec` is also
    accepted). ``d_spacing_nm`` and, downstream, ``gamma`` may be the string
    ``"auto"`` to consume the previous stage's suggestion from the state file.

    Attributes:
        output_dir: Root directory for all generated artifacts.
        d_spacing_nm: Bilayer period in nm, or ``"auto"``.
        gamma: Bilayer thickness ratio (``material_a`` fraction), 0 < gamma < 1.
        blaze_angle_deg: Center blaze angle for the stage-2 scan.
        material_a: Incident-side / top bilayer material.
        material_b: Second bilayer material.
        substrate_material: Grating substrate material.
        n_bilayers: Number of bilayer periods.
        target_energy_ev: Photon energy the stages optimize at.
        grating_density_lpermm: Groove density in lines/mm.
        diffraction_order: Grating diffraction order to optimize.
        cff: Fixed-focus constant used for the geometry grazing angle.
        multilayer_bragg_order: Multilayer Bragg order (distinct from the
            grating diffraction order).
    """

    output_dir: Path

    # Selected values (numeric, or "auto" where noted).
    d_spacing_nm: float | Literal["auto"] = "auto"
    gamma: float = 0.5
    blaze_angle_deg: float = 1.1

    # Materials.
    material_a: Any = ("Ru", 12.1)
    material_b: Any = ("C", 2.52)
    substrate_material: Any = ("Si", 2.33)
    n_bilayers: int = 40

    # Target and grating geometry.
    target_energy_ev: float = 9000.0
    grating_density_lpermm: float = 2400.0
    diffraction_order: int = 2
    cff: float = 2.25
    multilayer_bragg_order: int = 1

    # Per-stage energy grids.
    d_spacing_energy_min_ev: float = 500.0
    d_spacing_energy_max_ev: float = 12000.0
    d_spacing_energy_step_ev: float = 100.0
    d_spacing_energy_quick_step_ev: float = 250.0
    gamma_energy_min_ev: float = 500.0
    gamma_energy_max_ev: float = 12000.0
    gamma_energy_step_ev: float = 100.0
    gamma_energy_quick_step_ev: float = 250.0
    blaze_energy_min_ev: float = 3000.0
    blaze_energy_max_ev: float = 12000.0
    blaze_energy_points: int = 15
    blaze_energy_quick_points: int = 5

    # D-spacing geometry and scan settings.
    bragg_angle_min_deg: float = 0.5
    bragg_angle_max_deg: float = 2.0
    d_spacing_relative_range: float = 0.25
    d_spacing_min_practical_nm: float = 2.0
    d_spacing_max_practical_nm: float = 8.0
    d_spacing_points: int = 21

    # Gamma scan settings.
    gamma_min: float = 0.3
    gamma_max: float = 0.8
    gamma_step: float = 0.1

    # Blaze scan settings.
    blaze_angle_half_range_deg: float = 0.3
    blaze_angle_points: int = 4
    anti_blaze_angle_deg: float = 0.0

    # XRT reflectivity settings.
    xrt_window_deg: float = 0.2
    xrt_angle_points: int = 2001
    xrt_min_angle_deg: float = 0.0
    xrt_individuals: bool = False

    # graxPy theta-search settings.
    grax_x_resolution_nm: float = 0.5
    grax_z_resolution_nm: float = 0.5
    rough_scan_half_width_deg: float = 0.5
    rough_scan_points: int = 61
    rough_fourier_orders: int = 5
    rough_x_resolution_nm: float = 1.0
    rough_z_resolution_nm: float = 1.0
    fine_scan_half_width_deg: float = 0.2
    fine_scan_points: int = 81
    fine_fourier_orders: int = 15
    fine_x_resolution_nm: float = 0.5
    fine_z_resolution_nm: float = 0.5
    final_fourier_orders: int = 25
    final_x_resolution_nm: float = 0.2
    final_z_resolution_nm: float = 0.2
    roughness_sigma_nm: float | None = None
    precise_peak_selection_mode: str = "max"
    retry_on_selected_efficiency_zero: bool = True
    retry_selected_efficiency_threshold: float = 1.0e-4
    max_zero_efficiency_retries: int = 3
    backend: str = "numba"
    solver: str = "neviere"
    polarization: str = "p"

    # Runtime controls.
    quick: bool = False
    max_workers: int | str | None = "auto"
    show_progress: bool = True
    live_plot: bool = False
    on_error: str = "fail_fast"
    checkpoint_interval: int = 1
    resume: bool = True
    theta_tracking_mode: str = "auto"
    max_tracking_energy_step_ev: float | None = None
    save_profile_plot: bool = True
    save_stack_plot: bool = True

    def __post_init__(self) -> None:
        """Validate cross-field constraints."""

        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not (0.0 < float(self.gamma) < 1.0):
            raise ValueError(f"gamma must satisfy 0 < gamma < 1, got {self.gamma!r}")
        if isinstance(self.d_spacing_nm, str):
            if self.d_spacing_nm.strip().lower() != "auto":
                raise ValueError(
                    f"d_spacing_nm must be numeric or 'auto', got {self.d_spacing_nm!r}"
                )
        elif not np.isfinite(float(self.d_spacing_nm)) or float(self.d_spacing_nm) <= 0.0:
            raise ValueError(f"d_spacing_nm must be finite and positive, got {self.d_spacing_nm!r}")
        if self.target_energy_ev <= 0.0:
            raise ValueError("target_energy_ev must be positive")
        if self.n_bilayers < 1:
            raise ValueError("n_bilayers must be at least 1")
        if self.d_spacing_points < 2:
            raise ValueError("d_spacing_points must be at least 2")
        if self.blaze_angle_points < 1:
            raise ValueError("blaze_angle_points must be at least 1")
        if self.blaze_angle_half_range_deg < 0.0:
            raise ValueError("blaze_angle_half_range_deg must be non-negative")
        if self.blaze_energy_points < 2:
            raise ValueError("blaze_energy_points must be at least 2")
        if self.solver not in {"rcwa", "neviere"}:
            raise ValueError(f"solver must be 'rcwa' or 'neviere', got {self.solver!r}")

    @property
    def plot_dir(self) -> Path:
        """Directory for the per-stage summary plots."""

        return self.output_dir / "plot"

    @property
    def state_path(self) -> Path:
        """Path to the cross-stage ``optimization_state.json`` file."""

        return self.output_dir / "optimization_state.json"

    @property
    def d_spacing_results_dir(self) -> Path:
        """Directory for stage-0 (d-spacing) artifacts."""

        return self.output_dir / "0_d_spacing"

    @property
    def gamma_results_dir(self) -> Path:
        """Directory for stage-1 (gamma) artifacts."""

        return self.output_dir / "1_gamma"

    @property
    def blaze_results_dir(self) -> Path:
        """Directory for stage-2 (blaze) artifacts."""

        return self.output_dir / "2_blaze"


@dataclass
class DSpacingStudyResult:
    """Outcome of :func:`run_d_spacing_study`.

    Attributes:
        geometry_grazing_angle_deg: Grazing angle at the target energy and CFF.
        geometry_d_nm: d-spacing from the first-order Bragg law at that angle.
        d_suggested_nm: ``geometry_d_nm`` rounded to 0.1 nm; handed to stages 1-2.
        d_suggested_peak_rp: Peak reflectivity at ``d_suggested_nm`` and target.
        d_reflectivity_best_nm: Numerically best d at the target (diagnostic).
        d_reflectivity_best_peak_rp: Peak reflectivity at ``d_reflectivity_best_nm``.
        search_min_nm: Lower edge of the resolved d-spacing search interval.
        search_max_nm: Upper edge of the resolved d-spacing search interval.
        combined_csv_path: Combined per-d reflectivity table.
        plot_path: Reflectivity-versus-energy summary plot.
        state_path: The updated state file.
        results: The combined reflectivity table.
    """

    geometry_grazing_angle_deg: float
    geometry_d_nm: float
    d_suggested_nm: float
    d_suggested_peak_rp: float
    d_reflectivity_best_nm: float
    d_reflectivity_best_peak_rp: float
    search_min_nm: float
    search_max_nm: float
    combined_csv_path: Path
    plot_path: Path
    state_path: Path
    results: pd.DataFrame = field(repr=False)


@dataclass
class GammaStudyResult:
    """Outcome of :func:`run_gamma_study`.

    Attributes:
        d_spacing_nm: Resolved d-spacing the scan ran at.
        gamma_suggested: Gamma with the highest peak reflectivity at the target.
        gamma_suggested_peak_rp: Peak reflectivity at ``gamma_suggested``.
        combined_csv_path: Combined per-gamma reflectivity table.
        plot_path: Reflectivity-versus-energy summary plot.
        state_path: The updated state file.
        results: The combined reflectivity table.
    """

    d_spacing_nm: float
    gamma_suggested: float
    gamma_suggested_peak_rp: float
    combined_csv_path: Path
    plot_path: Path
    state_path: Path
    results: pd.DataFrame = field(repr=False)


@dataclass
class BlazeStudyResult:
    """Outcome of :func:`run_blaze_study`.

    Attributes:
        d_spacing_nm: Resolved d-spacing the gratings were built with.
        gamma: Gamma the gratings were built with (``config.gamma``).
        blaze_suggested_deg: Blaze angle with the highest selected-order
            efficiency at the target energy.
        blaze_suggested_efficiency: Selected-order efficiency at that blaze.
        combined_csv_path: Combined per-blaze theta-search summary table.
        plot_path: Efficiency-versus-energy summary plot.
        state_path: The updated state file.
        results: The combined theta-search summary table.
    """

    d_spacing_nm: float
    gamma: float
    blaze_suggested_deg: float
    blaze_suggested_efficiency: float
    combined_csv_path: Path
    plot_path: Path
    state_path: Path
    results: pd.DataFrame = field(repr=False)


def energy_to_wavelength_nm(energy_ev: float | Iterable[float]) -> float | np.ndarray:
    """Convert photon energy in eV to wavelength in nm.

    Args:
        energy_ev: One energy or an iterable of energies in eV.

    Returns:
        The wavelength in nm, scalar for scalar input.

    Raises:
        ValueError: If any energy is non-finite or non-positive.
    """

    energy = np.asarray(energy_ev, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError(f"Photon energy must be finite and positive, got {energy_ev!r}")
    wavelength = HC_EV_NM / energy
    return float(wavelength) if wavelength.ndim == 0 else wavelength


def d_spacing_bounds_from_bragg_angles(
    energy_ev: float,
    angle_min_deg: float,
    angle_max_deg: float,
    *,
    bragg_order: int = 1,
) -> tuple[float, float]:
    """Return the d-spacing range whose Bragg grazing angle lies in an interval.

    Args:
        energy_ev: Photon energy in eV.
        angle_min_deg: Minimum grazing Bragg angle in degrees.
        angle_max_deg: Maximum grazing Bragg angle in degrees.
        bragg_order: Positive Bragg order.

    Returns:
        ``(d_min_nm, d_max_nm)``; the larger angle maps to the smaller d.

    Raises:
        ValueError: If the energy, order or angle interval is invalid.
    """

    energy = float(energy_ev)
    angle_min = float(angle_min_deg)
    angle_max = float(angle_max_deg)
    order = int(bragg_order)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"target energy must be finite and positive, got {energy_ev!r}")
    if order <= 0:
        raise ValueError(f"Bragg order must be positive, got {bragg_order!r}")
    if not (0.0 < angle_min < angle_max < 90.0):
        raise ValueError(
            "Bragg grazing-angle limits must satisfy 0 < min < max < 90 deg; "
            f"got {angle_min_deg!r}, {angle_max_deg!r}"
        )
    wavelength_nm = float(energy_to_wavelength_nm(energy))
    d_min = order * wavelength_nm / (2.0 * np.sin(np.deg2rad(angle_max)))
    d_max = order * wavelength_nm / (2.0 * np.sin(np.deg2rad(angle_min)))
    return float(d_min), float(d_max)


def intersect_search_bounds(
    derived_min_nm: float,
    derived_max_nm: float,
    practical_min_nm: float | None = None,
    practical_max_nm: float | None = None,
) -> tuple[float, float]:
    """Intersect a derived d-spacing interval with optional practical limits.

    Args:
        derived_min_nm: Lower derived bound in nm.
        derived_max_nm: Upper derived bound in nm.
        practical_min_nm: Optional practical lower clamp in nm.
        practical_max_nm: Optional practical upper clamp in nm.

    Returns:
        The intersected ``(lower_nm, upper_nm)`` interval.

    Raises:
        ValueError: If any interval is empty or non-positive, or the
            intersection is empty.
    """

    derived_min = float(derived_min_nm)
    derived_max = float(derived_max_nm)
    practical_min = None if practical_min_nm is None else float(practical_min_nm)
    practical_max = None if practical_max_nm is None else float(practical_max_nm)
    if not (np.isfinite(derived_min) and np.isfinite(derived_max)) or derived_min <= 0.0:
        raise ValueError(
            f"Derived d-spacing bounds must be finite and positive: "
            f"{derived_min_nm!r}, {derived_max_nm!r}"
        )
    if derived_min >= derived_max:
        raise ValueError(
            f"Derived d-spacing interval is empty: ({derived_min:.8g}, {derived_max:.8g}) nm"
        )
    if practical_min is not None and (not np.isfinite(practical_min) or practical_min <= 0.0):
        raise ValueError(f"Practical d-spacing minimum must be positive, got {practical_min_nm!r}")
    if practical_max is not None and (not np.isfinite(practical_max) or practical_max <= 0.0):
        raise ValueError(f"Practical d-spacing maximum must be positive, got {practical_max_nm!r}")
    if practical_min is not None and practical_max is not None and practical_min >= practical_max:
        raise ValueError(
            "Practical d-spacing interval is empty: "
            f"({practical_min_nm!r}, {practical_max_nm!r}) nm"
        )
    lower = max(derived_min, practical_min) if practical_min is not None else derived_min
    upper = min(derived_max, practical_max) if practical_max is not None else derived_max
    if lower >= upper:
        raise ValueError(
            "No usable d-spacing interval after intersecting bounds: "
            f"derived=({derived_min:.8g}, {derived_max:.8g}) nm, "
            f"practical=({practical_min_nm!r}, {practical_max_nm!r}) nm"
        )
    return float(lower), float(upper)


def ensure_target_energy(energies: Iterable[float], target_energy_ev: float) -> np.ndarray:
    """Return a sorted energy grid that contains the target energy exactly once.

    Args:
        energies: Candidate energy grid in eV.
        target_energy_ev: Energy that must appear in the returned grid.

    Returns:
        The sorted, de-duplicated grid including ``target_energy_ev``.

    Raises:
        ValueError: If the grid or the target is not finite and positive.
    """

    values = np.asarray(list(energies), dtype=float).reshape(-1)
    target = float(target_energy_ev)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Energy grid must contain finite, positive values")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"Target energy must be finite and positive, got {target_energy_ev!r}")
    close = np.isclose(values, target, rtol=0.0, atol=1.0e-12)
    if np.any(close):
        values[close] = target
    else:
        values = np.append(values, target)
    return np.unique(np.sort(values))


def select_target_energy_optimum(
    results: pd.DataFrame,
    *,
    parameter_column: str,
    metric_column: str,
    target_energy_ev: float,
    energy_column: str,
) -> tuple[float, float]:
    """Return the tested parameter with the largest metric at the target energy.

    Args:
        results: Long-format table with parameter, metric and energy columns.
        parameter_column: Name of the swept-parameter column.
        metric_column: Name of the metric column to maximize.
        target_energy_ev: Energy at which to compare.
        energy_column: Name of the energy column.

    Returns:
        ``(parameter_value, metric_value)`` for the best row at the target.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no usable target-energy row exists.
    """

    missing = {energy_column, parameter_column, metric_column} - set(results.columns)
    if missing:
        raise KeyError(f"Results are missing required columns: {sorted(missing)}")
    target = float(target_energy_ev)
    target_rows = results[
        np.isclose(results[energy_column].astype(float), target, rtol=0.0, atol=1.0e-9)
    ]
    target_rows = target_rows.dropna(subset=[parameter_column, metric_column])
    if target_rows.empty:
        raise ValueError(f"Results contain no usable target-energy row for {target:g} eV")
    best = target_rows.loc[target_rows[metric_column].astype(float).idxmax()]
    return float(best[parameter_column]), float(best[metric_column])


def update_optimization_state(state_path: str | Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the JSON state file and return the full state.

    Args:
        state_path: Path to ``optimization_state.json``. Created if absent.
        updates: Keys to merge in. NumPy scalars and nested mappings/sequences
            are converted to plain JSON types.

    Returns:
        The merged state dictionary.

    Raises:
        ValueError: If the existing file does not hold a JSON object.
    """

    path = Path(state_path)
    state: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise ValueError(f"Optimization state must contain a JSON object: {path}")
        state.update(loaded)

    def json_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value

    state.update({str(key): json_value(value) for key, value in updates.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return state


def resolve_configured_value(
    configured_value: float | str,
    *,
    state_path: str | Path,
    suggestion_key: str,
    parameter_name: str,
) -> float:
    """Resolve a numeric configuration value or the ``"auto"`` sentinel.

    Args:
        configured_value: A number, or ``"auto"`` to read the state file.
        state_path: Path to ``optimization_state.json``.
        suggestion_key: State key holding the upstream suggestion.
        parameter_name: Name used in error messages.

    Returns:
        The resolved positive, finite value.

    Raises:
        ValueError: If a string other than ``"auto"`` is given, the state file
            or key is missing, or the resolved value is not finite and positive.
    """

    if isinstance(configured_value, str):
        if configured_value.strip().lower() != "auto":
            raise ValueError(
                f"{parameter_name} must be numeric or 'auto', got {configured_value!r}"
            )
        path = Path(state_path)
        if not path.exists():
            raise ValueError(
                f"{parameter_name} is 'auto', but optimization state does not exist: {path}"
            )
        with path.open(encoding="utf-8") as stream:
            state = json.load(stream)
        if suggestion_key not in state:
            raise ValueError(
                f"{parameter_name} is 'auto', but state has no {suggestion_key!r}; "
                "run the preceding optimization stage first"
            )
        value: Any = state[suggestion_key]
    else:
        value = configured_value
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Resolved {parameter_name} must be numeric, got {value!r}") from error
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"Resolved {parameter_name} must be finite and positive, got {resolved!r}")
    return resolved


def _material_pair(material: Any) -> tuple[str, float]:
    """Return a ``(name, density_g_cm3)`` pair for a material specification."""

    if isinstance(material, MaterialSpec):
        if material.density_g_cm3 is None:
            raise ValueError(f"MaterialSpec {material.name!r} needs a density for this workflow")
        return str(material.name), float(material.density_g_cm3)
    if isinstance(material, (tuple, list)) and len(material) == 2:
        return str(material[0]), float(material[1])
    raise ValueError(f"material must be a (name, density) pair or MaterialSpec, got {material!r}")


def _stage_energy_grid(config: MultilayerOptimizationConfig, stage: str) -> np.ndarray:
    """Build the energy grid for one stage, honoring ``config.quick``."""

    minimum = float(getattr(config, f"{stage}_energy_min_ev"))
    maximum = float(getattr(config, f"{stage}_energy_max_ev"))
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError(f"Invalid {stage} energy range: {minimum}, {maximum} eV")
    if stage == "blaze":
        points = int(config.blaze_energy_points)
        energies = np.linspace(minimum, maximum, points)
        if config.quick:
            energies = np.linspace(minimum, maximum, max(2, int(config.blaze_energy_quick_points)))
    else:
        step = float(getattr(config, f"{stage}_energy_step_ev"))
        if step <= 0.0:
            raise ValueError(f"{stage}_energy_step_ev must be positive")
        if config.quick:
            step = float(getattr(config, f"{stage}_energy_quick_step_ev"))
            if step <= 0.0:
                raise ValueError(f"{stage}_energy_quick_step_ev must be positive")
        energies = np.arange(minimum, maximum + 0.5 * step, step)
    return ensure_target_energy(energies, config.target_energy_ev)


def _rounded_d_grid(
    lower_nm: float, upper_nm: float, points: int, required_nm: float
) -> np.ndarray:
    """Return a 0.1 nm-rounded d-spacing grid that includes the geometry value.

    Args:
        lower_nm: Lower edge of the search interval in nm.
        upper_nm: Upper edge of the search interval in nm.
        points: Number of ``linspace`` samples before rounding and de-duplication.
        required_nm: Geometry d-spacing that must appear in the grid.

    Returns:
        The sorted, de-duplicated candidate grid rounded to 0.1 nm.

    Raises:
        ValueError: If the rounded interval cannot contain ``required_nm`` or
            fewer than two unique candidates survive.
    """

    if points < 2:
        raise ValueError(f"d_spacing_points must be at least 2, got {points!r}")
    rounded_min = np.ceil(lower_nm * 10.0) / 10.0
    rounded_max = np.floor(upper_nm * 10.0) / 10.0
    required = round(float(required_nm), 1)
    if rounded_min > rounded_max or not (rounded_min <= required <= rounded_max):
        raise ValueError(
            "The automatic d-spacing interval cannot contain the one-decimal "
            f"geometry value {required:.1f} nm: interval=({lower_nm:.8g}, {upper_nm:.8g}) nm"
        )
    grid = np.unique(np.round(np.linspace(rounded_min, rounded_max, int(points)), decimals=1))
    if not np.any(np.isclose(grid, required, rtol=0.0, atol=1.0e-9)):
        grid[np.argmin(np.abs(grid - required))] = required
        grid = np.unique(np.sort(grid))
    if grid.size < 2:
        raise ValueError(
            "The automatic d-spacing interval provides fewer than two unique 0.1 nm candidates"
        )
    return grid


def _reflectivity_metric(config: MultilayerOptimizationConfig) -> str:
    """Return the reflectivity column matching the configured polarization."""

    return "peak_rp" if normalize_polarization(config.polarization) == "p" else "peak_rs"


def _reflectivity_curve(
    config: MultilayerOptimizationConfig,
    d_spacing_nm: float,
    gamma: float,
    output_dir: Path,
    energies_ev: np.ndarray,
) -> pd.DataFrame:
    """Run the XRT reflectivity engine for one ``(d, gamma)`` over the grid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = MultilayerReflectivity(
        config.material_a,
        d_spacing_nm * gamma,
        config.material_b,
        d_spacing_nm * (1.0 - gamma),
        config.n_bilayers,
        save_recap=output_dir,
        individuals=config.xrt_individuals,
    )
    return engine.reflectivity_vs_energy(
        energies_ev,
        bragg_order=config.multilayer_bragg_order,
        window_deg=config.xrt_window_deg,
        angle_range=None,
        angle_points=config.xrt_angle_points,
        min_angle_deg=config.xrt_min_angle_deg,
    )


def _target_metric_for_parameter(
    results: pd.DataFrame,
    *,
    parameter_column: str,
    parameter_value: float,
    metric_column: str,
    target_energy_ev: float,
    energy_column: str,
) -> float:
    """Return the metric for one parameter value at the target energy."""

    energy_match = np.isclose(
        results[energy_column].astype(float), float(target_energy_ev), rtol=0.0, atol=1.0e-9
    )
    parameter_match = np.isclose(
        results[parameter_column].astype(float), float(parameter_value), rtol=0.0, atol=1.0e-9
    )
    rows = results[energy_match & parameter_match]
    rows = rows.dropna(subset=[metric_column])
    if rows.empty:
        raise ValueError(
            f"No {metric_column} result for {parameter_value:g} at {target_energy_ev:g} eV"
        )
    return float(rows.iloc[0][metric_column])


def _plot_reflectivity(
    config: MultilayerOptimizationConfig,
    results: pd.DataFrame,
    *,
    parameter_column: str,
    metric_column: str,
    suggested_value: float,
    suggested_metric: float,
    energy_column: str,
    label_format: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot the per-parameter metric versus energy, highlighting the suggestion."""

    import matplotlib.pyplot as plt

    target_energy = float(config.target_energy_ev)
    figure, axis = plt.subplots(figsize=(10, 6))
    for value, curve in results.groupby(parameter_column, sort=True):
        selected = bool(np.isclose(float(value), suggested_value, rtol=0.0, atol=1.0e-9))
        axis.plot(
            curve[energy_column],
            curve[metric_column],
            marker="o",
            markersize=2,
            linewidth=2.6 if selected else 0.9,
            alpha=1.0 if selected else 0.7,
            label=label_format.format(value=float(value)) + (" (suggested)" if selected else ""),
            zorder=3 if selected else 2,
        )
    axis.axvline(target_energy, color="black", linestyle="--", linewidth=1)
    axis.plot(
        [target_energy], [suggested_metric], marker="o", markersize=7, color="black", zorder=5
    )
    axis.annotate(
        f"{target_energy:g} eV\n{label_format.format(value=suggested_value)}\n"
        f"{metric_column} = {suggested_metric:.4g}",
        xy=(target_energy, suggested_metric),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85},
    )
    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel(f"Peak reflectivity ({normalize_polarization(config.polarization)}-polarized)")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _build_grating(
    config: MultilayerOptimizationConfig,
    d_spacing_nm: float,
    gamma: float,
    blaze_angle_deg: float,
) -> BlazedGrating:
    """Build the multilayer-coated blazed grating for one stage-2 case."""

    material_a_spec = MaterialSpec(*_material_pair(config.material_a))
    material_b_spec = MaterialSpec(*_material_pair(config.material_b))
    substrate_spec = MaterialSpec(*_material_pair(config.substrate_material))
    stack = MultilayerStack(
        substrate_material=substrate_spec,
        material_a=material_a_spec,
        material_b=material_b_spec,
        d_period_nm=d_spacing_nm,
        gamma=gamma,
        n_bilayers=int(config.n_bilayers),
        top_material=material_b_spec,
    )
    anti_blaze_kwargs = (
        {}
        if config.anti_blaze_angle_deg == 0.0
        else {"anti_blaze_angle_deg": float(config.anti_blaze_angle_deg)}
    )
    return BlazedGrating(
        period_lpermm=float(config.grating_density_lpermm),
        blaze_angle_deg=float(blaze_angle_deg),
        coating_stack=stack,
        substrate_material=substrate_spec,
        x_resolution_nm=float(config.grax_x_resolution_nm),
        z_resolution_nm=float(config.grax_z_resolution_nm),
        **anti_blaze_kwargs,
    )


def _run_blaze_case(
    config: MultilayerOptimizationConfig,
    d_spacing_nm: float,
    gamma: float,
    blaze_angle_deg: float,
    energies_ev: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Run graxPy's theta-search sweep for one blaze angle and return its summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = run_multilayer_theta_search_sweep(
        grating=_build_grating(config, d_spacing_nm, gamma, blaze_angle_deg),
        energies_ev=energies_ev,
        output_dir=output_dir,
        diffraction_order=int(config.diffraction_order),
        multilayer_bragg_order=int(config.multilayer_bragg_order),
        rough_scan_half_width_deg=float(config.rough_scan_half_width_deg),
        rough_scan_points=int(config.rough_scan_points),
        rough_fourier_orders=int(config.rough_fourier_orders),
        rough_x_resolution_nm=float(config.rough_x_resolution_nm),
        rough_z_resolution_nm=float(config.rough_z_resolution_nm),
        fine_scan_half_width_deg=float(config.fine_scan_half_width_deg),
        fine_scan_points=int(config.fine_scan_points),
        fine_fourier_orders=int(config.fine_fourier_orders),
        fine_x_resolution_nm=float(config.fine_x_resolution_nm),
        fine_z_resolution_nm=float(config.fine_z_resolution_nm),
        final_fourier_orders=int(config.final_fourier_orders),
        final_x_resolution_nm=float(config.final_x_resolution_nm),
        final_z_resolution_nm=float(config.final_z_resolution_nm),
        roughness_sigma_nm=config.roughness_sigma_nm,
        precise_peak_selection_mode=str(config.precise_peak_selection_mode),
        retry_on_selected_efficiency_zero=bool(config.retry_on_selected_efficiency_zero),
        retry_selected_efficiency_threshold=float(config.retry_selected_efficiency_threshold),
        max_zero_efficiency_retries=int(config.max_zero_efficiency_retries),
        max_workers=config.max_workers,
        show_progress=bool(config.show_progress),
        live_plot=bool(config.live_plot),
        on_error=str(config.on_error),
        checkpoint_dir=output_dir / "checkpoints",
        checkpoint_interval=int(config.checkpoint_interval),
        resume=bool(config.resume),
        theta_tracking_mode=str(config.theta_tracking_mode),
        max_tracking_energy_step_ev=config.max_tracking_energy_step_ev,
        save_profile_plot=bool(config.save_profile_plot),
        save_stack_plot=bool(config.save_stack_plot),
        backend=str(config.backend),
        solver=str(config.solver),
        polarization=str(config.polarization),
    )
    return pd.read_csv(sweep.summary_csv_path)


def run_d_spacing_study(config: MultilayerOptimizationConfig) -> DSpacingStudyResult:
    """Run stage 0: derive and scan the bilayer d-spacing.

    Derives the grazing angle at the target energy and CFF, converts it to a
    d-spacing with the first-order Bragg law, builds a practical 0.1 nm-rounded
    scan grid that includes the geometry value, runs XRT reflectivity over the
    energy grid for every candidate, and writes ``d_suggested_nm`` (the geometry
    value) plus the numerically best d (diagnostic) to the state file.

    Args:
        config: The workflow configuration.

    Returns:
        A :class:`DSpacingStudyResult` with the suggestion, diagnostics and
        artifact paths.
    """

    target_energy = float(config.target_energy_ev)
    wavelength_nm = float(energy_to_wavelength_nm(target_energy))
    geometry_angle = float(
        monochromator_grazing_angles_deg(
            [target_energy],
            period_lpermm=float(config.grating_density_lpermm),
            diffraction_order=int(config.diffraction_order),
            cff=float(config.cff),
        )[0]
    )
    d_geometry = wavelength_nm / (2.0 * np.sin(np.deg2rad(geometry_angle)))
    geometry_min = d_geometry * (1.0 - float(config.d_spacing_relative_range))
    geometry_max = d_geometry * (1.0 + float(config.d_spacing_relative_range))
    angle_derived_min, angle_derived_max = d_spacing_bounds_from_bragg_angles(
        target_energy,
        float(config.bragg_angle_min_deg),
        float(config.bragg_angle_max_deg),
        bragg_order=int(config.multilayer_bragg_order),
    )
    lower, upper = intersect_search_bounds(
        geometry_min, geometry_max, angle_derived_min, angle_derived_max
    )
    lower, upper = intersect_search_bounds(
        lower,
        upper,
        float(config.d_spacing_min_practical_nm),
        float(config.d_spacing_max_practical_nm),
    )
    d_suggested = round(d_geometry, 1)
    d_values = _rounded_d_grid(lower, upper, int(config.d_spacing_points), d_suggested)
    energies = _stage_energy_grid(config, "d_spacing")

    results_dir = config.d_spacing_results_dir
    curves = []
    for d_spacing in d_values:
        print(f"Calculating multilayer reflectivity, d = {d_spacing:.1f} nm")
        curve = _reflectivity_curve(
            config,
            float(d_spacing),
            float(config.gamma),
            results_dir / f"d_{d_spacing:.1f}nm",
            energies,
        )
        curve.insert(0, "d_spacing_nm", float(d_spacing))
        curves.append(curve)
    combined = pd.concat(curves, ignore_index=True)

    metric = _reflectivity_metric(config)
    geometry_metric = _target_metric_for_parameter(
        combined,
        parameter_column="d_spacing_nm",
        parameter_value=d_suggested,
        metric_column=metric,
        target_energy_ev=target_energy,
        energy_column="energy_ev",
    )
    numerical_best_d, numerical_best_metric = select_target_energy_optimum(
        combined,
        parameter_column="d_spacing_nm",
        metric_column=metric,
        target_energy_ev=target_energy,
        energy_column="energy_ev",
    )
    update_optimization_state(
        config.state_path,
        {
            "target_energy_eV": target_energy,
            "wavelength_nm": wavelength_nm,
            "grating_grazing_angle_deg": geometry_angle,
            "d_geometry_estimate_nm": d_geometry,
            "d_geometry_search_min_nm": geometry_min,
            "d_geometry_search_max_nm": geometry_max,
            "d_search_min_nm": lower,
            "d_search_max_nm": upper,
            "d_suggested_nm": d_suggested,
            "d_suggested_peak_rp": geometry_metric,
            "d_reflectivity_best_nm": numerical_best_d,
            "d_reflectivity_best_peak_rp": numerical_best_metric,
        },
    )
    csv_path = results_dir / "d_spacing_study.csv"
    combined.to_csv(csv_path, index=False)
    plot_path = config.plot_dir / "0_d_spacing_study.png"
    _plot_reflectivity(
        config,
        combined,
        parameter_column="d_spacing_nm",
        metric_column=metric,
        suggested_value=d_suggested,
        suggested_metric=geometry_metric,
        energy_column="energy_ev",
        label_format="d = {value:.1f} nm",
        title=f"Multilayer d-spacing study (geometry d = {d_geometry:.3f} nm)",
        output_path=plot_path,
    )
    print(
        f"Target {target_energy:g} eV: grazing angle = {geometry_angle:.6f} deg, "
        f"geometry d = {d_geometry:.6f} nm, suggested d = {d_suggested:.1f} nm"
    )
    print(f"Reflectivity at suggested d: {geometry_metric:.6g}")
    print(f"Numerical best d (diagnostic): {numerical_best_d:.1f} nm ({numerical_best_metric:.6g})")
    print(f"Results CSV: {csv_path}")
    print(f"Optimization state: {config.state_path}")
    return DSpacingStudyResult(
        geometry_grazing_angle_deg=geometry_angle,
        geometry_d_nm=float(d_geometry),
        d_suggested_nm=float(d_suggested),
        d_suggested_peak_rp=float(geometry_metric),
        d_reflectivity_best_nm=float(numerical_best_d),
        d_reflectivity_best_peak_rp=float(numerical_best_metric),
        search_min_nm=float(lower),
        search_max_nm=float(upper),
        combined_csv_path=csv_path,
        plot_path=plot_path,
        state_path=config.state_path,
        results=combined,
    )


def run_gamma_study(config: MultilayerOptimizationConfig) -> GammaStudyResult:
    """Run stage 1: scan the bilayer thickness ratio at the selected d-spacing.

    Resolves ``config.d_spacing_nm`` (numeric, or ``"auto"`` from the state
    file), scans ``gamma`` over ``[gamma_min, gamma_max]`` in ``gamma_step``
    increments, and records the gamma with the highest peak reflectivity at the
    target energy. The config is never modified.

    Args:
        config: The workflow configuration.

    Returns:
        A :class:`GammaStudyResult` with the suggested gamma and artifact paths.
    """

    target_energy = float(config.target_energy_ev)
    d_spacing = resolve_configured_value(
        config.d_spacing_nm,
        state_path=config.state_path,
        suggestion_key="d_suggested_nm",
        parameter_name="d_spacing_nm",
    )
    gamma_values = np.round(
        np.arange(
            float(config.gamma_min),
            float(config.gamma_max) + 0.5 * float(config.gamma_step),
            float(config.gamma_step),
        ),
        3,
    )
    energies = _stage_energy_grid(config, "gamma")

    results_dir = config.gamma_results_dir
    curves = []
    for gamma in gamma_values:
        print(f"Calculating multilayer reflectivity, gamma = {gamma:.3f}")
        curve = _reflectivity_curve(
            config, d_spacing, float(gamma), results_dir / f"gamma_{gamma:.3f}", energies
        )
        curve.insert(0, "gamma", float(gamma))
        curves.append(curve)
    combined = pd.concat(curves, ignore_index=True)

    metric = _reflectivity_metric(config)
    suggested_gamma, suggested_metric = select_target_energy_optimum(
        combined,
        parameter_column="gamma",
        metric_column=metric,
        target_energy_ev=target_energy,
        energy_column="energy_ev",
    )
    update_optimization_state(
        config.state_path,
        {"gamma_suggested": suggested_gamma, "gamma_suggested_peak_rp": suggested_metric},
    )
    csv_path = results_dir / "gamma_study.csv"
    combined.to_csv(csv_path, index=False)
    plot_path = config.plot_dir / "1_gamma_study.png"
    _plot_reflectivity(
        config,
        combined,
        parameter_column="gamma",
        metric_column=metric,
        suggested_value=suggested_gamma,
        suggested_metric=suggested_metric,
        energy_column="energy_ev",
        label_format="gamma = {value:.3f}",
        title=f"Multilayer gamma study (d = {d_spacing:.3f} nm)",
        output_path=plot_path,
    )
    print(
        f"Target {target_energy:g} eV, d = {d_spacing:.6f} nm: suggested gamma = "
        f"{suggested_gamma:.3f} ({metric} = {suggested_metric:.6g})"
    )
    print(f"Results CSV: {csv_path}")
    print(f"Optimization state: {config.state_path}")
    return GammaStudyResult(
        d_spacing_nm=float(d_spacing),
        gamma_suggested=float(suggested_gamma),
        gamma_suggested_peak_rp=float(suggested_metric),
        combined_csv_path=csv_path,
        plot_path=plot_path,
        state_path=config.state_path,
        results=combined,
    )


def run_blaze_study(config: MultilayerOptimizationConfig) -> BlazeStudyResult:
    """Run stage 2: scan the blaze angle with graxPy's theta search.

    Resolves ``config.d_spacing_nm`` (numeric, or ``"auto"`` from the state
    file), uses ``config.gamma`` directly, builds the multilayer-coated blazed
    grating for each blaze angle in the scan, runs
    :func:`grax.run_multilayer_theta_search_sweep` per blaze angle, and records
    the blaze angle with the highest selected-order efficiency at the target
    energy.

    Args:
        config: The workflow configuration.

    Returns:
        A :class:`BlazeStudyResult` with the suggested blaze angle and artifact
        paths.
    """

    target_energy = float(config.target_energy_ev)
    d_spacing = resolve_configured_value(
        config.d_spacing_nm,
        state_path=config.state_path,
        suggestion_key="d_suggested_nm",
        parameter_name="d_spacing_nm",
    )
    gamma = float(config.gamma)
    blaze_center = float(config.blaze_angle_deg)
    blaze_half_range = float(config.blaze_angle_half_range_deg)
    blaze_points = int(config.blaze_angle_points)
    blaze_values = np.linspace(
        blaze_center - blaze_half_range, blaze_center + blaze_half_range, blaze_points
    )
    if np.any(blaze_values <= 0.0):
        raise ValueError(f"Blaze-angle scan must be positive: {blaze_values.tolist()}")
    energies = _stage_energy_grid(config, "blaze")

    results_dir = config.blaze_results_dir
    curves = []
    for blaze in blaze_values:
        print(f"Running theta search, blaze = {blaze:.4f} deg")
        curve = _run_blaze_case(
            config, d_spacing, gamma, float(blaze), energies, results_dir / f"blaze_{blaze:.4f}deg"
        )
        curve.insert(0, "blaze_angle_deg", float(blaze))
        curves.append(curve)
    combined = pd.concat(curves, ignore_index=True)

    suggested_blaze, suggested_efficiency = select_target_energy_optimum(
        combined,
        parameter_column="blaze_angle_deg",
        metric_column="selected_efficiency",
        target_energy_ev=target_energy,
        energy_column="energy_ev",
    )
    update_optimization_state(
        config.state_path,
        {
            "blaze_suggested_deg": suggested_blaze,
            "blaze_suggested_efficiency": suggested_efficiency,
        },
    )
    csv_path = results_dir / "blaze_study.csv"
    combined.to_csv(csv_path, index=False)
    plot_path = config.plot_dir / "2_blaze_study.png"
    _plot_reflectivity(
        config,
        combined,
        parameter_column="blaze_angle_deg",
        metric_column="selected_efficiency",
        suggested_value=suggested_blaze,
        suggested_metric=suggested_efficiency,
        energy_column="energy_ev",
        label_format="blaze = {value:.4f} deg",
        title=f"Blaze study (d = {d_spacing:.3f} nm, gamma = {gamma:.3f})",
        output_path=plot_path,
    )
    print(
        f"Target {target_energy:g} eV, d = {d_spacing:.6f} nm, gamma = {gamma:.3f}: "
        f"suggested blaze = {suggested_blaze:.4f} deg (efficiency = {suggested_efficiency:.6g})"
    )
    print(f"Results CSV: {csv_path}")
    print(f"Optimization state: {config.state_path}")
    return BlazeStudyResult(
        d_spacing_nm=float(d_spacing),
        gamma=gamma,
        blaze_suggested_deg=float(suggested_blaze),
        blaze_suggested_efficiency=float(suggested_efficiency),
        combined_csv_path=csv_path,
        plot_path=plot_path,
        state_path=config.state_path,
        results=combined,
    )
