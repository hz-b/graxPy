"""Typed simulation result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import numpy as np

ExecutionMode = Literal["inline", "subprocess"]
ErrorPolicy = Literal["continue", "fail_fast"]
CaseStatus = Literal["ok", "error"]
MaxWorkers = Union[int, Literal["all", "auto"], None]
AUTO_WORKER_MEMORY_SAFETY_FACTOR = 1.35
AUTO_WORKER_MEMORY_RESERVE_BYTES = 4 * 1024**3

@dataclass
class ThetaSearchDiagnostics:
    """Diagnostic data produced by one multilayer theta-search solve.

    Attributes:
        estimated_grazing_angle_deg: Analytical initial grazing-angle estimate.
        rough_grazing_angles_deg: Grazing angles sampled during the rough scan.
        rough_efficiencies: Selected-order efficiencies from the rough scan.
        precise_grazing_angles_deg: Grazing angles sampled during the precise scan.
        precise_efficiencies: Selected-order efficiencies from the precise scan.
        selected_grazing_angle_deg: Final selected grazing angle in degrees.
        selected_efficiency: Final selected efficiency from the precise scan.
        precise_fwhm_deg: Full width at half maximum from the precise scan.
        theta_tracking_center_mode: Center source used for this point.
        theta_tracking_auto_classification: Auto-classification for the energy step.
        theta_tracking_previous_energy_ev: Previous successful energy used for continuity tracking.
        theta_tracking_previous_grazing_angle_deg: Previous successful grazing angle used for continuity tracking.
        theta_tracking_used_previous_theta: Whether the previous theta was used as the primary center.
        theta_tracking_bragg_fallback_triggered: Whether a Bragg-centered fallback solve was evaluated.
        theta_tracking_continuity_rejected: Whether a candidate was rejected by the continuity guard.
        precise_peak_selection_mode_requested: Requested precise-scan peak-selection mode.
        precise_peak_selection_mode_used: Mode that actually supplied the final selected theta.
        precise_peak_fit_fallback_used: Whether fit-mode fallback was required.
        precise_peak_fitted_center_deg: Fitted peak center in degrees, if available.
        precise_peak_fitted_fwhm_deg: Fitted FWHM-like width in degrees, if available.
        precise_peak_fitted_theta_deg: Theta samples used to render the fitted profile.
        precise_peak_fitted_efficiencies: Fitted profile values at ``precise_peak_fitted_theta_deg``.
    """

    estimated_grazing_angle_deg: float
    rough_grazing_angles_deg: np.ndarray
    rough_efficiencies: np.ndarray
    precise_grazing_angles_deg: np.ndarray
    precise_efficiencies: np.ndarray
    selected_grazing_angle_deg: float
    selected_efficiency: float
    precise_fwhm_deg: float | None = None
    theta_tracking_center_mode: str = "bragg"
    theta_tracking_auto_classification: str = "initial"
    theta_tracking_previous_energy_ev: float | None = None
    theta_tracking_previous_grazing_angle_deg: float | None = None
    theta_tracking_used_previous_theta: bool = False
    theta_tracking_bragg_fallback_triggered: bool = False
    theta_tracking_continuity_rejected: bool = False
    precise_peak_selection_mode_requested: str = "max"
    precise_peak_selection_mode_used: str = "max"
    precise_peak_fit_fallback_used: bool = False
    precise_peak_fitted_center_deg: float | None = None
    precise_peak_fitted_fwhm_deg: float | None = None
    precise_peak_fitted_theta_deg: np.ndarray | None = None
    precise_peak_fitted_efficiencies: np.ndarray | None = None


@dataclass
class SingleSimulationResult:
    """Result for one grating simulation case.

    Attributes:
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        orders: Calculated diffraction orders. These are physical orders and may
            be fractional (spaced at ``1/num_supercells``) when supercell
            roughness (``RoughnessSpec.num_supercells > 1``) is active.
        selected_efficiency: Efficiency for the selected diffraction order.
        selected_diffraction_angle_deg: Diffraction angle for the selected order.
        efficiency_all: Efficiency for all calculated diffraction orders.
        diffraction_angle_all: Diffraction angle for all calculated diffraction orders.
        diffraction_order: Positive selected diffraction order.
        fourier_orders: Fourier truncation order used for the solve.
        roughness_sigma_nm: Optional rms roughness in nanometers.
        num_supercells: Number of grating periods spanned by the roughness
            field at solve time (mirrors ``RoughnessSpec.num_supercells``);
            ``1`` when no supercell roughness was in effect.
        num_realizations: Number of independent random-interface roughness
            realizations averaged into this result (mirrors
            ``RoughnessSpec.num_realizations``); ``1`` when no averaging was
            in effect (including all Debye-Waller and no-roughness runs).
        solver: Electromagnetic solver that produced this result, ``"rcwa"`` or
            ``"neviere"``.
        solver_options: Solver settings that produced this result, or ``None``
            when the solver takes none. Recorded so a checkpointed differential-
            method run is reproducible: the solver name alone does not pin the
            integration settings.
        theta_search_diagnostics: Optional transient theta-search diagnostics.
        retry_triggered: Whether zero-efficiency retry logic was triggered.
        retry_attempts: Number of additional retry attempts after the first run.
        retry_status: Retry outcome status.
        selected_efficiency_is_exact_zero: Whether final selected efficiency is exactly 0.0.
        selected_efficiency_below_retry_threshold: Whether final selected efficiency
            is below or equal to the configured retry threshold.
        theta_tracking_center_mode: Center source used for the theta search.
        theta_tracking_auto_classification: Auto-classification for the energy step.
        theta_tracking_previous_energy_ev: Previous successful energy used for continuity tracking.
        theta_tracking_previous_grazing_angle_deg: Previous successful grazing angle used
            for continuity tracking.
        theta_tracking_used_previous_theta: Whether the previous theta was used as the primary center.
        theta_tracking_bragg_fallback_triggered: Whether a Bragg-centered fallback solve was evaluated.
        theta_tracking_continuity_rejected: Whether a candidate was rejected by the continuity guard.
    """

    energy_ev: float
    grazing_angle_deg: float
    orders: np.ndarray
    selected_efficiency: float
    selected_diffraction_angle_deg: float
    efficiency_all: np.ndarray
    diffraction_angle_all: np.ndarray
    diffraction_order: int
    fourier_orders: int
    roughness_sigma_nm: float | None = None
    num_supercells: int = 1
    num_realizations: int = 1
    polarization: str = "s"
    solver: str = "rcwa"
    solver_options: dict[str, object] | None = None
    theta_search_diagnostics: ThetaSearchDiagnostics | None = None
    retry_triggered: bool = False
    retry_attempts: int = 0
    retry_status: str = "not_needed"
    selected_efficiency_is_exact_zero: bool = False
    selected_efficiency_below_retry_threshold: bool = False
    theta_tracking_center_mode: str = "bragg"
    theta_tracking_auto_classification: str = "initial"
    theta_tracking_previous_energy_ev: float | None = None
    theta_tracking_previous_grazing_angle_deg: float | None = None
    theta_tracking_used_previous_theta: bool = False
    theta_tracking_bragg_fallback_triggered: bool = False
    theta_tracking_continuity_rejected: bool = False

    @property
    def efficiency(self) -> float:
        """Return the selected diffraction-order efficiency."""

        return self.selected_efficiency

    @property
    def diffraction_angle_deg(self) -> float:
        """Return the selected diffraction-order angle in degrees."""

        return self.selected_diffraction_angle_deg


@dataclass
class CaseExecutionResult:
    """Result for one batch-executed simulation case.

    Attributes:
        case_id: Stable user-provided case identifier.
        index: Zero-based execution index among the provided cases.
        label: Optional case label.
        energy_ev: Photon energy in electronvolts.
        grazing_angle_deg: Grazing incidence angle in degrees.
        orders: Calculated diffraction orders. These are physical orders and may
            be fractional (spaced at ``1/num_supercells``) when supercell
            roughness (``RoughnessSpec.num_supercells > 1``) is active.
        selected_efficiency: Efficiency for the selected diffraction order.
        selected_diffraction_angle_deg: Diffraction angle for the selected order.
        efficiency_all: Efficiency for all calculated diffraction orders.
        diffraction_angle_all: Diffraction angle for all calculated diffraction orders.
        status: Case execution status.
        error_message: Error message if the case failed.
        peak_memory_bytes: Peak tracked memory during execution, when profiling is enabled.
        wall_seconds: Total wall time during execution, when profiling is enabled.
        case_data: Original serializable case metadata without the grating object.
        solver: Electromagnetic solver that produced this result, ``"rcwa"`` or
            ``"neviere"``.
        solver_options: Solver settings that produced this result, or ``None``
            when the solver takes none.
        theta_search_diagnostics: Optional transient theta-search diagnostics.
        retry_triggered: Whether zero-efficiency retry logic was triggered.
        retry_attempts: Number of additional retry attempts after the first run.
        retry_status: Retry outcome status.
        selected_efficiency_is_exact_zero: Whether final selected efficiency is exactly 0.0.
        selected_efficiency_below_retry_threshold: Whether final selected efficiency
            is below or equal to the configured retry threshold.
        theta_tracking_center_mode: Center source used for the theta search.
        theta_tracking_auto_classification: Auto-classification for the energy step.
        theta_tracking_previous_energy_ev: Previous successful energy used for continuity tracking.
        theta_tracking_previous_grazing_angle_deg: Previous successful grazing angle used
            for continuity tracking.
        theta_tracking_used_previous_theta: Whether the previous theta was used as the primary center.
        theta_tracking_bragg_fallback_triggered: Whether a Bragg-centered fallback solve was evaluated.
        theta_tracking_continuity_rejected: Whether a candidate was rejected by the continuity guard.
    """

    case_id: str
    index: int
    label: str | None
    energy_ev: float
    grazing_angle_deg: float
    orders: np.ndarray
    selected_efficiency: float
    selected_diffraction_angle_deg: float
    efficiency_all: np.ndarray
    diffraction_angle_all: np.ndarray
    status: CaseStatus
    error_message: str | None = None
    peak_memory_bytes: int | None = None
    wall_seconds: float | None = None
    case_data: dict[str, object] = field(default_factory=dict)
    polarization: str = "s"
    solver: str = "rcwa"
    solver_options: dict[str, object] | None = None
    theta_search_diagnostics: ThetaSearchDiagnostics | None = None
    retry_triggered: bool = False
    retry_attempts: int = 0
    retry_status: str = "not_needed"
    selected_efficiency_is_exact_zero: bool = False
    selected_efficiency_below_retry_threshold: bool = False
    theta_tracking_center_mode: str = "bragg"
    theta_tracking_auto_classification: str = "initial"
    theta_tracking_previous_energy_ev: float | None = None
    theta_tracking_previous_grazing_angle_deg: float | None = None
    theta_tracking_used_previous_theta: bool = False
    theta_tracking_bragg_fallback_triggered: bool = False
    theta_tracking_continuity_rejected: bool = False


@dataclass
class SimulationResult:
    """Compatibility container for small collected simulation sweeps."""

    energy_ev: np.ndarray
    orders: np.ndarray
    efficiency: np.ndarray
    diffraction_angle_deg: np.ndarray
    efficiency_all: np.ndarray
    diffraction_angle_all: np.ndarray


@dataclass
class BatchSimulationResult:
    """Compatibility container for explicitly collected batch results."""

    cases: list[CaseExecutionResult]

    @property
    def successful_cases(self) -> list[CaseExecutionResult]:
        """Return only successful cases."""

        return [case for case in self.cases if case.status == "ok"]

    @property
    def energy_ev(self) -> np.ndarray:
        """Return energies for successful cases."""

        return np.asarray([case.energy_ev for case in self.successful_cases], dtype=float)

    @property
    def grazing_angle_deg(self) -> np.ndarray:
        """Return grazing angles for successful cases."""

        return np.asarray([case.grazing_angle_deg for case in self.successful_cases], dtype=float)

    @property
    def orders(self) -> np.ndarray:
        """Return diffraction orders for successful cases."""

        successful_cases = self.successful_cases
        if not successful_cases:
            return np.asarray([], dtype=float)
        return np.asarray(successful_cases[0].orders, dtype=float)

    @property
    def efficiency(self) -> np.ndarray:
        """Return selected-order efficiencies for successful cases."""

        return np.asarray([case.selected_efficiency for case in self.successful_cases], dtype=float)

    @property
    def diffraction_angle_deg(self) -> np.ndarray:
        """Return selected-order diffraction angles for successful cases."""

        return np.asarray(
            [case.selected_diffraction_angle_deg for case in self.successful_cases],
            dtype=float,
        )

    @property
    def efficiency_all(self) -> np.ndarray:
        """Return all-order efficiencies for successful cases."""

        successful_cases = self.successful_cases
        if not successful_cases:
            return np.empty((0, 0), dtype=float)
        return np.asarray([case.efficiency_all for case in successful_cases], dtype=float)

    @property
    def diffraction_angle_all(self) -> np.ndarray:
        """Return all-order diffraction angles for successful cases."""

        successful_cases = self.successful_cases
        if not successful_cases:
            return np.empty((0, 0), dtype=float)
        return np.asarray([case.diffraction_angle_all for case in successful_cases], dtype=float)

    def to_simulation_result(self) -> SimulationResult:
        """Convert successful batch cases to a small collected sweep result."""

        return SimulationResult(
            energy_ev=self.energy_ev,
            orders=self.orders,
            efficiency=self.efficiency,
            diffraction_angle_deg=self.diffraction_angle_deg,
            efficiency_all=self.efficiency_all,
            diffraction_angle_all=self.diffraction_angle_all,
        )


@dataclass
class MultilayerThetaSearchSweepResult:
    """Collected result and output paths for a multilayer theta-search sweep.

    Attributes:
        batch_result: Collected batch result for the sweep.
        summary_csv_path: CSV containing one selected theta/efficiency per energy.
        all_orders_csv_path: CSV containing all reflected orders for every energy.
        energy_efficiency_plot_path: Plot of final selected efficiency versus energy.
        workflow_plot_path: Combined plot of final sweep results and first-energy diagnostics.
        theta_scan_directory: Directory containing one precise theta-scan CSV/PNG pair per energy.
        profile_plot_path: Optional saved grating profile plot.
        stack_plot_path: Optional saved multilayer stack schematic.
        total_elapsed_seconds: Total accumulated sweep runtime in seconds, including
            prior resumed runs when checkpoint metadata is available.
        current_run_elapsed_seconds: Runtime spent in the current process invocation.
    """

    batch_result: BatchSimulationResult
    summary_csv_path: Path
    all_orders_csv_path: Path
    energy_efficiency_plot_path: Path
    workflow_plot_path: Path
    theta_scan_directory: Path
    profile_plot_path: Path | None = None
    stack_plot_path: Path | None = None
    total_elapsed_seconds: float = 0.0
    current_run_elapsed_seconds: float = 0.0
