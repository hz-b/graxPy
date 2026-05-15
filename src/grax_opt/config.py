"""Configuration objects for Ax-based grating optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParameterBounds:
    """Bounds for one optimization parameter."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate bounds ordering."""

        if self.upper <= self.lower:
            raise ValueError("Parameter bounds must satisfy upper > lower.")

    def as_list(self) -> list[float]:
        """Return bounds in Ax-compatible list form."""

        return [float(self.lower), float(self.upper)]


@dataclass(frozen=True)
class InitialBlazedGrating:
    """Fixed and initial parameters for a blazed grating."""

    period_lpermm: float
    blaze_angle_deg: float
    anti_blaze_angle_deg: float | None = None
    substrate_material: Any = "Si"
    layer_material: Any = "Au"
    layer_thickness_nm: float = 30.0
    top_cap_material: Any | None = None
    top_cap_thickness_nm: float = 0.0
    z_resolution_nm: float = 0.1
    x_resolution_nm: float = 0.1

    def __post_init__(self) -> None:
        """Validate required initial values."""

        if self.period_lpermm <= 0.0:
            raise ValueError("period_lpermm must be > 0.")
        if self.blaze_angle_deg <= 0.0:
            raise ValueError("blaze_angle_deg must be > 0.")
        if self.layer_thickness_nm < 0.0:
            raise ValueError("layer_thickness_nm must be >= 0.")
        if self.top_cap_thickness_nm < 0.0:
            raise ValueError("top_cap_thickness_nm must be >= 0.")
        if self.x_resolution_nm <= 0.0 or self.z_resolution_nm <= 0.0:
            raise ValueError("x_resolution_nm and z_resolution_nm must be > 0.")
        if self.anti_blaze_angle_deg is not None and self.anti_blaze_angle_deg <= 0.0:
            raise ValueError("anti_blaze_angle_deg must be > 0 when provided.")


@dataclass(frozen=True)
class InitialLaminarGrating:
    """Fixed and initial parameters for a laminar grating."""

    period_lpermm: float
    width_to_period_ratio: float
    depth_nm: float
    left_wall_angle_deg: float
    right_wall_angle_deg: float
    substrate_material: Any = "Si"
    layer_material: Any = "Pt"
    layer_thickness_nm: float = 30.0
    top_cap_material: Any | None = None
    top_cap_thickness_nm: float = 0.0
    z_resolution_nm: float = 0.1
    x_resolution_nm: float = 0.1

    def __post_init__(self) -> None:
        """Validate required initial values."""

        if self.period_lpermm <= 0.0:
            raise ValueError("period_lpermm must be > 0.")
        if self.width_to_period_ratio < 0.0 or self.width_to_period_ratio > 1.0:
            raise ValueError("width_to_period_ratio must be in [0, 1].")
        if self.depth_nm <= 0.0:
            raise ValueError("depth_nm must be > 0.")
        if self.left_wall_angle_deg < 0.0 or self.right_wall_angle_deg < 0.0:
            raise ValueError("wall angles must be >= 0.")
        if self.left_wall_angle_deg == 0.0 and self.right_wall_angle_deg != 0.0:
            raise ValueError("both wall angles must be zero for rectangular walls.")
        if self.right_wall_angle_deg == 0.0 and self.left_wall_angle_deg != 0.0:
            raise ValueError("both wall angles must be zero for rectangular walls.")
        if self.layer_thickness_nm < 0.0:
            raise ValueError("layer_thickness_nm must be >= 0.")
        if self.top_cap_thickness_nm < 0.0:
            raise ValueError("top_cap_thickness_nm must be >= 0.")
        if self.x_resolution_nm <= 0.0 or self.z_resolution_nm <= 0.0:
            raise ValueError("x_resolution_nm and z_resolution_nm must be > 0.")


@dataclass(frozen=True)
class BlazedAxConfig:
    """Run configuration for Ax-based blazed-grating fitting.

    Attributes:
        initial_grating: Initial/fixed blazed grating definition.
        measurement_path: Path to two-column measurement data (energy, efficiency).
        output_dir: Directory where optimizer artifacts are written.
        cff: Constant-focus factor used by monochromator geometry.
        diffraction_order: Positive diffraction order to optimize.
        fourier_orders: Fourier truncation order for RCWA evaluations.
        roughness_sigma_nm: Optional fixed roughness sigma in nanometers.
        validate_physical_results: Whether to validate physical output ranges.
        total_trials: Total Ax trials to run.
        batch_size: Number of candidates proposed/evaluated per optimizer batch.
        random_seed: Optional random seed for deterministic Ax behavior.
        optimize_period_lpermm: Whether period is an optimization variable.
        optimize_blaze_angle_deg: Whether blaze angle is optimized.
        optimize_anti_blaze_angle_deg: Whether anti-blaze angle is optimized.
        optimize_top_cap_thickness_nm: Whether top-cap thickness is optimized.
        period_lpermm_bounds: Optional bounds for period optimization.
        blaze_angle_deg_bounds: Optional bounds for blaze-angle optimization.
        anti_blaze_angle_deg_bounds: Optional bounds for anti-blaze-angle optimization.
        top_cap_thickness_nm_bounds: Optional bounds for top-cap-thickness optimization.
        objective_name: Objective metric name reported to Ax.
        experiment_name: Experiment label persisted in optimizer artifacts.
        loss_name: Loss metric used for trial scoring.
        failure_penalty: Finite penalty assigned to failed simulations.
        objective_sem: Reported objective standard error for Ax.
        enable_early_stopping: Whether early stopping logic is enabled.
        early_stopping_patience: Allowed non-improving trials before stopping.
        early_stopping_min_relative_improvement: Relative improvement threshold.
        early_stopping_warmup_trials: Trials to complete before early-stop checks.
        save_best_fit_plot: Whether to write ``best_fit.png``.
        save_loss_plot: Whether to write ``optimization_loss_history.png``.
        backend: Requested compute backend (for example ``"auto"``, ``"numba"``, ``"numpy"``).
        evaluation_energies_ev: Discrete energies (eV) used for objective evaluation.
    """

    initial_grating: InitialBlazedGrating
    measurement_path: Path
    output_dir: Path
    cff: float = 2.5
    diffraction_order: int = 1
    fourier_orders: int = 20
    roughness_sigma_nm: float | None = None
    validate_physical_results: bool = True
    total_trials: int = 20
    batch_size: int = 1
    random_seed: int | None = None
    optimize_period_lpermm: bool = True
    optimize_blaze_angle_deg: bool = False
    optimize_anti_blaze_angle_deg: bool = False
    optimize_top_cap_thickness_nm: bool = False
    period_lpermm_bounds: ParameterBounds | None = None
    blaze_angle_deg_bounds: ParameterBounds | None = None
    anti_blaze_angle_deg_bounds: ParameterBounds | None = None
    top_cap_thickness_nm_bounds: ParameterBounds | None = None
    objective_name: str = "loss"
    experiment_name: str = "blazed_grating_fit"
    loss_name: str = "mse"
    failure_penalty: float = 1.0e6
    objective_sem: float = 1.0e-6
    enable_early_stopping: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_relative_improvement: float = 5.0e-3
    early_stopping_warmup_trials: int = 8
    save_best_fit_plot: bool = True
    save_loss_plot: bool = True
    backend: str = "auto"
    evaluation_energies_ev: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize paths and validate configuration."""

        object.__setattr__(self, "measurement_path", Path(self.measurement_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not any(
            (
                self.optimize_period_lpermm,
                self.optimize_blaze_angle_deg,
                self.optimize_anti_blaze_angle_deg,
                self.optimize_top_cap_thickness_nm,
            )
        ):
            raise ValueError("At least one blazed optimization parameter must be enabled.")
        if self.cff <= 0.0:
            raise ValueError("cff must be > 0.")
        if self.diffraction_order <= 0:
            raise ValueError("diffraction_order must be > 0.")
        if self.fourier_orders <= 0:
            raise ValueError("fourier_orders must be > 0.")
        if self.roughness_sigma_nm is not None and self.roughness_sigma_nm < 0.0:
            raise ValueError("roughness_sigma_nm must be >= 0 when provided.")
        if self.total_trials <= 0:
            raise ValueError("total_trials must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.failure_penalty <= 0.0:
            raise ValueError("failure_penalty must be > 0.")
        if not math.isfinite(self.objective_sem) or self.objective_sem <= 0.0:
            raise ValueError("objective_sem must be finite and > 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if (
            not math.isfinite(self.early_stopping_min_relative_improvement)
            or self.early_stopping_min_relative_improvement < 0.0
        ):
            raise ValueError(
                "early_stopping_min_relative_improvement must be finite and >= 0."
            )
        if self.early_stopping_warmup_trials < 0:
            raise ValueError("early_stopping_warmup_trials must be >= 0.")
        if self.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be one of: auto, numba, numpy.")
        if len(self.evaluation_energies_ev) == 0:
            raise ValueError("evaluation_energies_ev must be provided and non-empty.")
        normalized_energies = sorted(
            {float(energy_ev) for energy_ev in self.evaluation_energies_ev}
        )
        if any(energy_ev <= 0.0 for energy_ev in normalized_energies):
            raise ValueError("evaluation_energies_ev values must be > 0.")
        object.__setattr__(self, "evaluation_energies_ev", normalized_energies)
        if self.optimize_anti_blaze_angle_deg:
            if (
                self.initial_grating.anti_blaze_angle_deg is None
                and self.anti_blaze_angle_deg_bounds is None
            ):
                raise ValueError(
                    "anti_blaze_angle_deg requires an initial value or explicit bounds "
                    "when optimized."
                )
        if self.optimize_top_cap_thickness_nm and self.initial_grating.top_cap_material is None:
            raise ValueError(
                "top_cap_material must be set when optimizing top_cap_thickness_nm."
            )


@dataclass(frozen=True)
class LaminarAxConfig:
    """Run configuration for Ax-based laminar-grating fitting.

    Attributes:
        initial_grating: Initial/fixed laminar grating definition.
        measurement_path: Path to two-column measurement data (energy, efficiency).
        output_dir: Directory where optimizer artifacts are written.
        angle_mode: Evaluation geometry mode (``"fixed"`` or ``"cff"``).
        grazing_angle_deg: Fixed grazing angle in degrees when ``angle_mode="fixed"``.
        cff: Constant-focus factor used when ``angle_mode="cff"``.
        diffraction_order: Positive diffraction order to optimize.
        fourier_orders: Fourier truncation order for RCWA evaluations.
        roughness_sigma_nm: Optional fixed roughness sigma in nanometers.
        validate_physical_results: Whether to validate physical output ranges.
        total_trials: Total Ax trials to run.
        batch_size: Number of candidates proposed/evaluated per optimizer batch.
        random_seed: Optional random seed for deterministic Ax behavior.
        optimize_period_lpermm: Whether period is an optimization variable.
        optimize_width_to_period_ratio: Whether width/period ratio is optimized.
        optimize_depth_nm: Whether groove depth is optimized.
        optimize_left_wall_angle_deg: Whether left wall angle is optimized.
        optimize_right_wall_angle_deg: Whether right wall angle is optimized.
        optimize_top_cap_thickness_nm: Whether top-cap thickness is optimized.
        optimize_roughness_sigma_nm: Whether roughness sigma is optimized.
        period_lpermm_bounds: Bounds for period optimization when enabled.
        width_to_period_ratio_bounds: Bounds for width/period-ratio optimization.
        depth_nm_bounds: Bounds for depth optimization.
        left_wall_angle_deg_bounds: Bounds for left-wall-angle optimization.
        right_wall_angle_deg_bounds: Bounds for right-wall-angle optimization.
        top_cap_thickness_nm_bounds: Bounds for top-cap-thickness optimization.
        roughness_sigma_nm_bounds: Bounds for roughness optimization.
        objective_name: Objective metric name reported to Ax.
        experiment_name: Experiment label persisted in optimizer artifacts.
        loss_name: Loss metric used for trial scoring.
        failure_penalty: Finite penalty assigned to failed simulations.
        objective_sem: Reported objective standard error for Ax.
        enable_early_stopping: Whether early stopping logic is enabled.
        early_stopping_patience: Allowed non-improving trials before stopping.
        early_stopping_min_relative_improvement: Relative improvement threshold.
        early_stopping_warmup_trials: Trials to complete before early-stop checks.
        save_best_fit_plot: Whether to write ``best_fit.png``.
        save_loss_plot: Whether to write ``optimization_loss_history.png``.
        backend: Requested compute backend (for example ``"auto"``, ``"numba"``, ``"numpy"``).
        evaluation_energies_ev: Discrete energies (eV) used for objective evaluation.
    """

    initial_grating: InitialLaminarGrating
    measurement_path: Path
    output_dir: Path
    angle_mode: str = "fixed"
    grazing_angle_deg: float = 4.0
    cff: float = 2.5
    diffraction_order: int = 1
    fourier_orders: int = 20
    roughness_sigma_nm: float | None = None
    validate_physical_results: bool = True
    total_trials: int = 20
    batch_size: int = 1
    random_seed: int | None = None
    optimize_period_lpermm: bool = True
    optimize_width_to_period_ratio: bool = True
    optimize_depth_nm: bool = True
    optimize_left_wall_angle_deg: bool = True
    optimize_right_wall_angle_deg: bool = True
    optimize_top_cap_thickness_nm: bool = True
    optimize_roughness_sigma_nm: bool = False
    period_lpermm_bounds: ParameterBounds | None = None
    width_to_period_ratio_bounds: ParameterBounds | None = None
    depth_nm_bounds: ParameterBounds | None = None
    left_wall_angle_deg_bounds: ParameterBounds | None = None
    right_wall_angle_deg_bounds: ParameterBounds | None = None
    top_cap_thickness_nm_bounds: ParameterBounds | None = None
    roughness_sigma_nm_bounds: ParameterBounds | None = None
    objective_name: str = "loss"
    experiment_name: str = "laminar_grating_fit"
    loss_name: str = "mse"
    failure_penalty: float = 1.0e6
    objective_sem: float = 1.0e-6
    enable_early_stopping: bool = False
    early_stopping_patience: int = 8
    early_stopping_min_relative_improvement: float = 5.0e-3
    early_stopping_warmup_trials: int = 8
    save_best_fit_plot: bool = True
    save_loss_plot: bool = True
    backend: str = "auto"
    evaluation_energies_ev: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize paths and validate configuration."""

        object.__setattr__(self, "measurement_path", Path(self.measurement_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.angle_mode not in {"fixed", "cff"}:
            raise ValueError("angle_mode must be either 'fixed' or 'cff'.")
        if self.angle_mode == "fixed" and self.grazing_angle_deg <= 0.0:
            raise ValueError("grazing_angle_deg must be > 0 for fixed angle mode.")
        if self.cff <= 0.0:
            raise ValueError("cff must be > 0.")
        if self.diffraction_order <= 0:
            raise ValueError("diffraction_order must be > 0.")
        if self.fourier_orders <= 0:
            raise ValueError("fourier_orders must be > 0.")
        if self.roughness_sigma_nm is not None and self.roughness_sigma_nm < 0.0:
            raise ValueError("roughness_sigma_nm must be >= 0 when provided.")
        if self.total_trials <= 0:
            raise ValueError("total_trials must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.failure_penalty <= 0.0:
            raise ValueError("failure_penalty must be > 0.")
        if not math.isfinite(self.objective_sem) or self.objective_sem <= 0.0:
            raise ValueError("objective_sem must be finite and > 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if (
            not math.isfinite(self.early_stopping_min_relative_improvement)
            or self.early_stopping_min_relative_improvement < 0.0
        ):
            raise ValueError(
                "early_stopping_min_relative_improvement must be finite and >= 0."
            )
        if self.early_stopping_warmup_trials < 0:
            raise ValueError("early_stopping_warmup_trials must be >= 0.")
        if self.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be one of: auto, numba, numpy.")
        if len(self.evaluation_energies_ev) == 0:
            raise ValueError("evaluation_energies_ev must be provided and non-empty.")
        normalized_energies = sorted(
            {float(energy_ev) for energy_ev in self.evaluation_energies_ev}
        )
        if any(energy_ev <= 0.0 for energy_ev in normalized_energies):
            raise ValueError("evaluation_energies_ev values must be > 0.")
        object.__setattr__(self, "evaluation_energies_ev", normalized_energies)
        if self.optimize_top_cap_thickness_nm and self.initial_grating.top_cap_material is None:
            raise ValueError(
                "top_cap_material must be set when optimizing top_cap_thickness_nm."
            )

        required_bounds = {
            "period_lpermm_bounds": self.optimize_period_lpermm,
            "width_to_period_ratio_bounds": self.optimize_width_to_period_ratio,
            "depth_nm_bounds": self.optimize_depth_nm,
            "left_wall_angle_deg_bounds": self.optimize_left_wall_angle_deg,
            "right_wall_angle_deg_bounds": self.optimize_right_wall_angle_deg,
            "top_cap_thickness_nm_bounds": self.optimize_top_cap_thickness_nm,
            "roughness_sigma_nm_bounds": self.optimize_roughness_sigma_nm,
        }
        for field_name, is_optimized in required_bounds.items():
            if is_optimized and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be provided when optimized.")
