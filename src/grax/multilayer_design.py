"""Multilayer-grating design workflow: a d-spacing / blaze-angle survey plus energy scans.

The workflow sizes a periodic multilayer coating for a blazed grating working in a
chosen diffraction order, in two steps driven by a single
:class:`MultilayerDesignConfig`:

1. :meth:`MultilayerGratingDesigner.run_survey` -- for every ``(d_spacing, blaze
   angle)`` pair on a 2-D grid, build the multilayer-coated blazed grating and run
   the multilayer theta search at the optimization energy through
   :func:`grax.run_multilayer_theta_search_sweep`, with its ``output_dir`` set to
   that pair's own folder. Each run therefore saves the full standard artifact
   bundle under ``survey/runs/d<d>nm/blaze<b>deg/`` -- the selected-result summary
   CSV, the all-orders CSV, the incidence-angle scan CSV/PNG, the profile and
   stack plots, a checkpoint, and a ``search_parameters.json`` recording what was
   fed in so the search can be reviewed and tuned. From every pair's selected
   result the survey reports, per d-spacing, the blaze angle with the highest
   efficiency (``argmax_blaze``); the headline artifacts are two curves -- optimal
   blaze angle versus d-spacing, and max efficiency versus d-spacing. Each
   ``d<d>nm/`` folder also gets an ``overlay.png`` of its blaze angles' scans with
   the chosen one highlighted.

2. :meth:`MultilayerGratingDesigner.run_energy_scan` -- for one or more
   ``(d_spacing, blaze angle)`` pairs picked off the survey, sweep a fixed coated
   grating over an explicit energy range with
   :func:`grax.run_multilayer_theta_search_sweep`, yielding efficiency versus
   energy and ``theta*(E)`` for each design.

There is no fixed-focus (CFF) input: the incident angle is a result of the theta
search for every pair, not an assumption. ``gamma`` is held fixed and the
anti-blaze angle defaults to ``0`` (a plain sawtooth); both are later
optimizations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gratings import BlazedGrating
from .materials import MaterialSpec
from .simulation import (
    estimate_multilayer_bragg_angle_deg,
    run_multilayer_theta_search_sweep,
)
from .stacks import MultilayerStack

__all__ = [
    "EnergyScanResult",
    "MultilayerDesignConfig",
    "MultilayerGratingDesigner",
    "StageProgress",
    "SurveyResult",
    "ThetaSearchScanSettings",
]

HC_EV_NM = 1239.841984


@dataclass(frozen=True)
class StageProgress:
    """Progress report emitted before each scanned item by a workflow step.

    Attributes:
        stage: ``"survey"`` or ``"energy_scan"``.
        completed: Items finished so far.
        total: Total items in the scan.
        current_label: Human-readable label of the item about to run, or
            ``"done"`` on the final call once the scan has finished.
    """

    stage: str
    completed: int
    total: int
    current_label: str


@dataclass(frozen=True)
class ThetaSearchScanSettings:
    """Rough/fine/final incidence-angle scan settings for one theta search.

    :class:`MultilayerDesignConfig` carries two independent instances --
    ``survey_scan_settings`` and ``energy_scan_settings`` -- so the (d, blaze)
    survey (many single-energy searches, one per grid cell) and the per-design
    energy scan (few designs, many energies each) can be tuned for speed or
    accuracy independently.

    Attributes:
        rough_scan_half_width_deg: Half-width of the coarse first pass around
            the seed angle.
        rough_scan_points: Number of angles sampled in the rough pass.
        rough_fourier_orders: Fourier truncation order used during the rough pass.
        rough_x_resolution_nm / rough_z_resolution_nm: Grating mesh override
            during the rough pass.
        fine_scan_half_width_deg: Half-width of the precise pass around the
            rough-pass maximum.
        fine_scan_points: Number of angles sampled in the precise pass.
        fine_fourier_orders: Fourier truncation order used during the precise pass.
        fine_x_resolution_nm / fine_z_resolution_nm: Grating mesh override
            during the precise pass.
        final_fourier_orders: Fourier truncation order for the final solve at
            the selected angle.
        final_x_resolution_nm / final_z_resolution_nm: Grating mesh override
            for the final solve.
        precise_peak_selection_mode: ``"max"``, ``"gauss"`` or ``"voigt"``.
        roughness_sigma_nm: Optional rms interface roughness in nm.
    """

    rough_scan_half_width_deg: float = 1.5
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
    precise_peak_selection_mode: str = "max"
    roughness_sigma_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate the peak-selection mode."""

        if self.precise_peak_selection_mode not in {"max", "gauss", "voigt"}:
            raise ValueError(
                "precise_peak_selection_mode must be 'max', 'gauss' or 'voigt', got "
                f"{self.precise_peak_selection_mode!r}"
            )


@dataclass(frozen=True)
class MultilayerDesignConfig:
    """Every knob for the multilayer-grating design survey and energy scans.

    Fields fall into three groups, matching the section comments below and in
    the example ``rub4c_design_parameters.py``:

    * **Shared** -- grating geometry, materials, solver/runtime. Read by both
      :meth:`MultilayerGratingDesigner.run_survey` and
      :meth:`MultilayerGratingDesigner.run_energy_scan`.
    * **Survey only** -- read only by ``run_survey`` /
      :meth:`MultilayerGratingDesigner.evaluate_survey`.
    * **Energy scan only** -- read only by ``run_energy_scan`` /
      :meth:`MultilayerGratingDesigner.evaluate_energy_scan`.

    ``material_a`` / ``material_b`` / ``substrate_material`` are
    ``(name, density_g_cm3)`` pairs (a :class:`grax.MaterialSpec` is also
    accepted).

    Attributes:
        output_dir: Root directory for all generated artifacts.
        grating_density_lpermm: Groove density in lines/mm.
        diffraction_order: Grating diffraction order to optimize.
        multilayer_bragg_order: Multilayer Bragg order (distinct from the grating
            diffraction order).
        material_a: Incident-side / top bilayer material.
        material_b: Second bilayer material.
        substrate_material: Grating substrate material.
        n_bilayers: Number of bilayer periods.
        gamma: Bilayer thickness ratio (``material_a`` fraction), 0 < gamma < 1;
            held fixed for both steps.
        anti_blaze_angle_deg: Anti-blaze facet angle; ``0`` (default) builds a
            plain sawtooth.
        coating_label: Display label for the coating in plot titles, e.g.
            ``"Ru/B4C"``. Defaults to ``"<material_a name>/<material_b name>"``
            when ``None`` -- set this explicitly whenever a material is modelled
            with a stand-in optical-constants table (for example B4C modelled
            with the carbon table), so titles show the real compound.
        x_resolution_nm / z_resolution_nm: Grating profile discretization.
        backend / solver / polarization: Solver selection.
        checkpoint / resume: Checkpointing for both steps.
        save_profile_plot / save_stack_plot: Per-run diagnostic plots.
        target_energy_ev: Photon energy the survey optimizes at.
        d_min_nm / d_max_nm / d_points: Inclusive d-spacing grid in nm.
        blaze_min_deg / blaze_max_deg / blaze_points: Inclusive blaze-angle grid.
        survey_scan_settings: :class:`ThetaSearchScanSettings` for the survey --
            one single-energy search per ``(d, blaze)`` grid cell.
        on_error: ``"continue"`` records a NaN row for a failing survey cell and
            keeps going; ``"fail_fast"`` re-raises.
        energy_scan_min_ev / energy_scan_max_ev / energy_scan_points: Explicit
            energy grid.
        energy_scan_settings: :class:`ThetaSearchScanSettings` for the energy
            scan -- one search per energy, for each chosen design.
        max_workers: Batch worker count (``"auto"`` calibrates).
        show_progress: Whether a progress bar is shown.
        theta_tracking_mode / max_tracking_energy_step_ev: How the seed angle
            for each energy is chosen (the survey always seeds from the Bragg
            estimate instead).
    """

    output_dir: Path

    # ================================================================== #
    # Shared -- grating geometry, materials, solver/runtime.              #
    # Read by both run_survey and run_energy_scan.                        #
    # ================================================================== #
    grating_density_lpermm: float = 2400.0
    diffraction_order: int = 2
    multilayer_bragg_order: int = 1

    material_a: Any = ("Ru", 12.1)
    material_b: Any = ("C", 2.52)
    substrate_material: Any = ("Si", 2.33)
    n_bilayers: int = 40
    gamma: float = 0.5
    anti_blaze_angle_deg: float = 0.0
    coating_label: str | None = None

    x_resolution_nm: float = 0.5
    z_resolution_nm: float = 0.5

    backend: str = "numba"
    solver: str = "neviere"
    polarization: str = "p"
    checkpoint: bool = True
    resume: bool = True
    save_profile_plot: bool = True
    save_stack_plot: bool = True

    # ================================================================== #
    # Survey only -- read by run_survey / evaluate_survey.                #
    # ================================================================== #
    target_energy_ev: float = 9000.0

    d_min_nm: float = 1.5
    d_max_nm: float = 6.0
    d_points: int = 4
    blaze_min_deg: float = 0.6
    blaze_max_deg: float = 2.0
    blaze_points: int = 6

    # The rough half-width is wide because the scan is seeded from the
    # multilayer Bragg estimate, which overshoots the true grating optimum for
    # shallow inside orders; the window has to reach well below it across the
    # whole d grid.
    survey_scan_settings: ThetaSearchScanSettings = field(
        default_factory=ThetaSearchScanSettings
    )
    on_error: str = "continue"

    # ================================================================== #
    # Energy scan only -- read by run_energy_scan / evaluate_energy_scan. #
    # ================================================================== #
    energy_scan_min_ev: float = 3000.0
    energy_scan_max_ev: float = 12000.0
    energy_scan_points: int = 15
    energy_scan_settings: ThetaSearchScanSettings = field(
        default_factory=ThetaSearchScanSettings
    )
    max_workers: int | str | None = "auto"
    show_progress: bool = True
    theta_tracking_mode: str = "auto"
    max_tracking_energy_step_ev: float | None = None

    def __post_init__(self) -> None:
        """Validate cross-field constraints."""

        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not (0.0 < float(self.gamma) < 1.0):
            raise ValueError(f"gamma must satisfy 0 < gamma < 1, got {self.gamma!r}")
        if self.target_energy_ev <= 0.0:
            raise ValueError("target_energy_ev must be positive")
        if self.n_bilayers < 1:
            raise ValueError("n_bilayers must be at least 1")
        if self.multilayer_bragg_order < 1:
            raise ValueError("multilayer_bragg_order must be at least 1")
        if not (0.0 < float(self.d_min_nm) < float(self.d_max_nm)):
            raise ValueError(
                f"d grid must satisfy 0 < d_min_nm < d_max_nm, got "
                f"{self.d_min_nm!r}, {self.d_max_nm!r}"
            )
        if self.d_points < 2:
            raise ValueError("d_points must be at least 2")
        if not (0.0 < float(self.blaze_min_deg) <= float(self.blaze_max_deg)):
            raise ValueError(
                f"blaze grid must satisfy 0 < blaze_min_deg <= blaze_max_deg, got "
                f"{self.blaze_min_deg!r}, {self.blaze_max_deg!r}"
            )
        if self.blaze_points < 1:
            raise ValueError("blaze_points must be at least 1")
        if self.anti_blaze_angle_deg < 0.0:
            raise ValueError("anti_blaze_angle_deg must be non-negative")
        if not (0.0 < float(self.energy_scan_min_ev) < float(self.energy_scan_max_ev)):
            raise ValueError(
                "energy scan must satisfy 0 < energy_scan_min_ev < energy_scan_max_ev"
            )
        if self.energy_scan_points < 2:
            raise ValueError("energy_scan_points must be at least 2")
        if self.solver not in {"rcwa", "neviere"}:
            raise ValueError(f"solver must be 'rcwa' or 'neviere', got {self.solver!r}")
        if self.on_error not in {"continue", "fail_fast"}:
            raise ValueError(f"on_error must be 'continue' or 'fail_fast', got {self.on_error!r}")

    @property
    def survey_dir(self) -> Path:
        """Directory for the survey CSV."""

        return self.output_dir / "survey"

    @property
    def plot_dir(self) -> Path:
        """Directory for the survey headline plots and the titled energy-scan plots."""

        return self.output_dir / "plots"

    @property
    def runs_dir(self) -> Path:
        """Root of the survey-run folder tree (``d<d>nm/blaze<b>deg/``)."""

        return self.output_dir / "survey" / "runs"

    def period_dir(self, d_spacing_nm: float) -> Path:
        """Return one d-spacing's folder, holding its per-blaze run folders."""

        return self.runs_dir / f"d{float(d_spacing_nm):.3f}nm"

    def run_dir(self, d_spacing_nm: float, blaze_angle_deg: float) -> Path:
        """Return one ``(d, blaze)`` run's folder, nested under its period folder."""

        return self.period_dir(d_spacing_nm) / f"blaze{float(blaze_angle_deg):.3f}deg"

    @property
    def energy_scan_dir(self) -> Path:
        """Root directory for per-design energy scans."""

        return self.output_dir / "energy_scan"

    def d_grid_nm(self) -> np.ndarray:
        """Return the inclusive d-spacing grid in nm."""

        return np.linspace(float(self.d_min_nm), float(self.d_max_nm), int(self.d_points))

    def blaze_grid_deg(self) -> np.ndarray:
        """Return the inclusive blaze-angle grid in degrees."""

        return np.linspace(
            float(self.blaze_min_deg), float(self.blaze_max_deg), int(self.blaze_points)
        )

    def energy_grid_ev(self) -> np.ndarray:
        """Return the inclusive energy grid for the energy scan."""

        return np.linspace(
            float(self.energy_scan_min_ev),
            float(self.energy_scan_max_ev),
            int(self.energy_scan_points),
        )


@dataclass
class SurveyResult:
    """Outcome of :meth:`MultilayerGratingDesigner.run_survey`.

    The ``*_map`` arrays are shaped ``(len(d_values_nm), len(blaze_values_deg))``.
    Cells that failed (or were never reached on an early stop) hold ``NaN``.

    Attributes:
        d_values_nm: The d-spacing grid.
        blaze_values_deg: The blaze-angle grid.
        efficiency_map: Peak selected-order efficiency per ``(d, blaze)``.
        incidence_angle_map_deg: ``theta*`` per ``(d, blaze)``.
        fwhm_map_deg: Precise-scan FWHM per ``(d, blaze)``.
        bragg_estimate_map_deg: Analytical Bragg-angle seed per ``(d, blaze)``.
        optimal_blaze_deg: Per d-spacing, the blaze angle with the largest
            efficiency (``NaN`` where the whole row failed).
        optimal_blaze_efficiency: Efficiency along ``optimal_blaze_deg``.
        optimal_blaze_incidence_deg: ``theta*`` along ``optimal_blaze_deg``.
        combined_csv_path: Long-format per-cell table.
        plot_path: The optimal blaze angle versus d-spacing curve.
        efficiency_plot_path: The max efficiency versus d-spacing curve, each
            point labelled with the blaze angle that achieved it.
        heatmap_plot_path: 2-D heatmap of ``efficiency_map`` -- d-spacing (x)
            against blaze angle (y), colour is peak efficiency (z) -- with the
            ``optimal_blaze_deg`` ridge overlaid.
        runs_dir: Root of the survey-run folder tree.
        period_dirs: One folder per d-spacing that solved at least one cell
            (``runs/d<d>nm/``), each holding an ``overlay.png`` of every blaze
            angle's theta search plus one sub-folder per blaze angle. In d-grid
            order.
        run_dirs: One leaf folder per solved cell
            (``runs/d<d>nm/blaze<b>deg/``), each holding that theta search's full
            :func:`grax.run_multilayer_theta_search_sweep` output
            (``multilayer_theta_search_summary.csv``, ``*_all_orders.csv``,
            ``theta_scans/``, profile/stack plots, ``checkpoints/``) plus a
            ``search_parameters.json``. In scan order.
        aborted: Whether the scan stopped early on a ``should_continue`` signal.
        results: The per-cell table.
    """

    d_values_nm: np.ndarray
    blaze_values_deg: np.ndarray
    efficiency_map: np.ndarray
    incidence_angle_map_deg: np.ndarray
    fwhm_map_deg: np.ndarray
    bragg_estimate_map_deg: np.ndarray
    optimal_blaze_deg: np.ndarray
    optimal_blaze_efficiency: np.ndarray
    optimal_blaze_incidence_deg: np.ndarray
    combined_csv_path: Path
    plot_path: Path
    efficiency_plot_path: Path
    heatmap_plot_path: Path
    runs_dir: Path
    period_dirs: list[Path]
    run_dirs: list[Path]
    aborted: bool
    results: pd.DataFrame = field(repr=False)


@dataclass
class EnergyScanResult:
    """Outcome of one design's energy scan.

    Attributes:
        d_spacing_nm: Bilayer period the grating was built with.
        blaze_angle_deg: Blaze angle the grating was built with.
        output_dir: Directory holding this design's sweep artifacts.
        summary_csv_path: One selected theta/efficiency per energy.
        all_orders_csv_path: All reflected orders per energy.
        energy_efficiency_plot_path: Selected efficiency versus energy, with
            graxPy's generic title (as written by
            :func:`grax.run_multilayer_theta_search_sweep`).
        titled_plot_path: The same curve, titled with the coating
            (:attr:`MultilayerDesignConfig.coating_label`) and this design's
            d-spacing and blaze angle -- the plot to actually look at. Lives in
            :attr:`MultilayerDesignConfig.plot_dir` (shared with the survey's
            headline plots) as
            ``efficiency_vs_energy_<materials>_order<n>_d<d>nm_blaze<b>deg.png``,
            since every design's plot lands in the same folder.
        results: The summary table.
    """

    d_spacing_nm: float
    blaze_angle_deg: float
    output_dir: Path
    summary_csv_path: Path
    all_orders_csv_path: Path
    energy_efficiency_plot_path: Path
    titled_plot_path: Path
    results: pd.DataFrame = field(repr=False)


def _material_pair(material: Any) -> tuple[str, float]:
    """Return a ``(name, density_g_cm3)`` pair for a material specification."""

    if isinstance(material, MaterialSpec):
        if material.density_g_cm3 is None:
            raise ValueError(f"MaterialSpec {material.name!r} needs a density for this workflow")
        return str(material.name), float(material.density_g_cm3)
    if isinstance(material, (tuple, list)) and len(material) == 2:
        return str(material[0]), float(material[1])
    raise ValueError(f"material must be a (name, density) pair or MaterialSpec, got {material!r}")


def _coating_label(config: MultilayerDesignConfig) -> str:
    """Return the coating's display label for plot titles.

    Uses ``config.coating_label`` when set; otherwise derives
    ``"<material_a name>/<material_b name>"`` from the configured materials.
    """

    if config.coating_label:
        return str(config.coating_label)
    name_a, _ = _material_pair(config.material_a)
    name_b, _ = _material_pair(config.material_b)
    return f"{name_a}/{name_b}"


def _energy_scan_title(
    config: MultilayerDesignConfig, d_spacing_nm: float, blaze_angle_deg: float
) -> str:
    """Return the title for one design's efficiency-versus-energy plot."""

    return (
        f"{_coating_label(config)} multilayer grating (order {config.diffraction_order}): "
        f"d = {d_spacing_nm:.3f} nm, blaze = {blaze_angle_deg:.3f} deg"
    )


def _filename_slug(text: str) -> str:
    """Return ``text`` with filesystem-unsafe characters collapsed to ``-``."""

    slug = re.sub(r"[^A-Za-z0-9.+_-]+", "-", text.strip())
    return slug.strip("-") or "coating"


def _energy_scan_plot_filename(
    config: MultilayerDesignConfig, d_spacing_nm: float, blaze_angle_deg: float
) -> str:
    """Return the filename for one design's titled efficiency-versus-energy plot.

    All designs share :attr:`MultilayerDesignConfig.plot_dir`, so the filename
    itself carries the coating, diffraction order, d-spacing and blaze angle:
    ``efficiency_vs_energy_<materials>_order<n>_d<d>nm_blaze<b>deg.png``.
    """

    materials = _filename_slug(_coating_label(config))
    return (
        f"efficiency_vs_energy_{materials}_order{int(config.diffraction_order)}_"
        f"d{d_spacing_nm:.3f}nm_blaze{blaze_angle_deg:.3f}deg.png"
    )


def _emit_progress(
    callback: Callable[[StageProgress], None] | None,
    *,
    stage: str,
    completed: int,
    total: int,
    current_label: str,
) -> None:
    """Report progress through ``callback`` when one was supplied."""

    if callback is not None:
        callback(
            StageProgress(
                stage=stage, completed=completed, total=total, current_label=current_label
            )
        )


class MultilayerGratingDesigner:
    """Run the multilayer-grating design survey and per-design energy scans."""

    def __init__(self, config: MultilayerDesignConfig) -> None:
        """Store the shared configuration."""

        self.config = config

    # ------------------------------------------------------------------ #
    # Grating construction                                               #
    # ------------------------------------------------------------------ #
    def _build_grating(self, d_spacing_nm: float, blaze_angle_deg: float) -> BlazedGrating:
        """Build the multilayer-coated blazed grating for one ``(d, blaze)`` pair."""

        config = self.config
        material_a_spec = MaterialSpec(*_material_pair(config.material_a))
        material_b_spec = MaterialSpec(*_material_pair(config.material_b))
        substrate_spec = MaterialSpec(*_material_pair(config.substrate_material))
        stack = MultilayerStack(
            substrate_material=substrate_spec,
            material_a=material_a_spec,
            material_b=material_b_spec,
            d_period_nm=float(d_spacing_nm),
            gamma=float(config.gamma),
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
            x_resolution_nm=float(config.x_resolution_nm),
            z_resolution_nm=float(config.z_resolution_nm),
            **anti_blaze_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Step 1: (d, blaze) survey                                          #
    # ------------------------------------------------------------------ #
    def _survey_cell(
        self, d_spacing_nm: float, blaze_angle_deg: float
    ) -> tuple[dict[str, float], Path]:
        """Run one ``(d, blaze)`` multilayer theta search and save its full output.

        The search runs through :func:`grax.run_multilayer_theta_search_sweep` at
        the single optimization energy, with ``output_dir`` set to this run's
        folder, so it drops the standard artifact bundle there (selected-result
        summary CSV, all-orders CSV, incidence-angle scan CSV/PNG, profile and
        stack plots, checkpoint). A ``search_parameters.json`` is added so the
        search settings can be reviewed and tuned. Returns the scalar summary and
        the run folder.
        """

        config = self.config
        grating = self._build_grating(d_spacing_nm, blaze_angle_deg)
        bragg_estimate_deg = estimate_multilayer_bragg_angle_deg(
            grating=grating,
            energy_ev=float(config.target_energy_ev),
            multilayer_bragg_order=int(config.multilayer_bragg_order),
        )
        run_dir = config.run_dir(d_spacing_nm, blaze_angle_deg)
        run_dir.mkdir(parents=True, exist_ok=True)
        scan = config.survey_scan_settings

        sweep = run_multilayer_theta_search_sweep(
            grating=grating,
            energies_ev=[float(config.target_energy_ev)],
            output_dir=run_dir,
            diffraction_order=int(config.diffraction_order),
            multilayer_bragg_order=int(config.multilayer_bragg_order),
            rough_scan_half_width_deg=float(scan.rough_scan_half_width_deg),
            rough_scan_points=int(scan.rough_scan_points),
            rough_fourier_orders=int(scan.rough_fourier_orders),
            rough_x_resolution_nm=float(scan.rough_x_resolution_nm),
            rough_z_resolution_nm=float(scan.rough_z_resolution_nm),
            fine_scan_half_width_deg=float(scan.fine_scan_half_width_deg),
            fine_scan_points=int(scan.fine_scan_points),
            fine_fourier_orders=int(scan.fine_fourier_orders),
            fine_x_resolution_nm=float(scan.fine_x_resolution_nm),
            fine_z_resolution_nm=float(scan.fine_z_resolution_nm),
            final_fourier_orders=int(scan.final_fourier_orders),
            final_x_resolution_nm=float(scan.final_x_resolution_nm),
            final_z_resolution_nm=float(scan.final_z_resolution_nm),
            roughness_sigma_nm=scan.roughness_sigma_nm,
            precise_peak_selection_mode=str(scan.precise_peak_selection_mode),
            max_workers=1,
            show_progress=False,
            on_error="fail_fast",
            checkpoint_dir=(run_dir / "checkpoints") if config.checkpoint else None,
            checkpoint_interval=1,
            resume=bool(config.resume and config.checkpoint),
            theta_tracking_mode="bragg",
            save_profile_plot=bool(config.save_profile_plot),
            save_stack_plot=bool(config.save_stack_plot),
            backend=str(config.backend),
            solver=str(config.solver),
            polarization=str(config.polarization),
        )
        (run_dir / "search_parameters.json").write_text(
            json.dumps(
                self._search_parameters_record(
                    d_spacing_nm=float(d_spacing_nm),
                    blaze_angle_deg=float(blaze_angle_deg),
                    bragg_estimate_deg=float(bragg_estimate_deg),
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        row = pd.read_csv(sweep.summary_csv_path).iloc[0]
        selected_angle = float(row["selected_grazing_angle_deg"])
        fwhm_value = row.get("precise_fwhm_deg", float("nan"))
        summary = {
            "bragg_estimate_deg": float(bragg_estimate_deg),
            "incidence_angle_deg": selected_angle,
            "peak_efficiency": float(row["selected_efficiency"]),
            "precise_fwhm_deg": float(fwhm_value) if pd.notna(fwhm_value) else float("nan"),
            "edge_clipped": _edge_clipped_from_scan(sweep.theta_scan_directory, selected_angle),
        }
        return summary, run_dir

    def run_survey(
        self,
        *,
        progress_callback: Callable[[StageProgress], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> SurveyResult:
        """Run the ``(d, blaze)`` survey at the optimization energy.

        Args:
            progress_callback: Optional callable invoked with a
                :class:`StageProgress` before each cell and once more when the
                scan finishes.
            should_continue: Optional callable checked before each cell; when it
                returns ``False`` the scan stops early and the optimal blaze per
                d-spacing is computed from the completed cells (``aborted`` is set
                on the result).

        Returns:
            A :class:`SurveyResult` with the maps, the per-d optimal blaze
            angle, and the per-run folder tree under ``survey/runs/``.

        Raises:
            RuntimeError: If ``should_continue`` stops the scan before any cell
                has been evaluated.
        """

        config = self.config
        d_values = config.d_grid_nm()
        blaze_values = config.blaze_grid_deg()
        n_d = d_values.size
        n_blaze = blaze_values.size
        total = n_d * n_blaze

        efficiency_map = np.full((n_d, n_blaze), np.nan)
        incidence_map = np.full((n_d, n_blaze), np.nan)
        fwhm_map = np.full((n_d, n_blaze), np.nan)
        bragg_map = np.full((n_d, n_blaze), np.nan)
        run_dir_by_cell: dict[tuple[int, int], Path] = {}
        rows: list[dict[str, float]] = []
        completed = 0
        aborted = False

        for i_d, d_spacing in enumerate(d_values):
            if aborted:
                break
            for i_b, blaze in enumerate(blaze_values):
                if should_continue is not None and not should_continue():
                    aborted = True
                    break
                _emit_progress(
                    progress_callback,
                    stage="survey",
                    completed=completed,
                    total=total,
                    current_label=f"d = {d_spacing:.2f} nm, blaze = {blaze:.3f} deg",
                )
                print(
                    f"Multilayer theta search: d = {d_spacing:.2f} nm, blaze = {blaze:.3f} deg"
                )
                try:
                    summary, run_dir = self._survey_cell(float(d_spacing), float(blaze))
                except Exception as error:  # noqa: BLE001 - per-cell isolation
                    if config.on_error == "fail_fast":
                        raise
                    print(
                        f"  cell failed (d = {d_spacing:.2f} nm, blaze = {blaze:.3f} deg): {error}"
                    )
                    summary = {
                        "bragg_estimate_deg": float("nan"),
                        "incidence_angle_deg": float("nan"),
                        "peak_efficiency": float("nan"),
                        "precise_fwhm_deg": float("nan"),
                        "edge_clipped": float("nan"),
                    }
                    run_dir = None
                if run_dir is not None:
                    run_dir_by_cell[(i_d, i_b)] = run_dir
                efficiency_map[i_d, i_b] = summary["peak_efficiency"]
                incidence_map[i_d, i_b] = summary["incidence_angle_deg"]
                fwhm_map[i_d, i_b] = summary["precise_fwhm_deg"]
                bragg_map[i_d, i_b] = summary["bragg_estimate_deg"]
                rows.append(
                    {
                        "d_nm": float(d_spacing),
                        "blaze_deg": float(blaze),
                        "bragg_estimate_deg": summary["bragg_estimate_deg"],
                        "incidence_angle_deg": summary["incidence_angle_deg"],
                        "peak_efficiency": summary["peak_efficiency"],
                        "precise_fwhm_deg": summary["precise_fwhm_deg"],
                        "edge_clipped": summary["edge_clipped"],
                    }
                )
                completed += 1

        if not rows:
            raise RuntimeError("multilayer design survey aborted before any result")
        _emit_progress(
            progress_callback,
            stage="survey",
            completed=completed,
            total=total,
            current_label="done",
        )
        return self._finalize_survey(
            d_values=d_values,
            blaze_values=blaze_values,
            efficiency_map=efficiency_map,
            incidence_map=incidence_map,
            fwhm_map=fwhm_map,
            bragg_map=bragg_map,
            run_dir_by_cell=run_dir_by_cell,
            rows=rows,
            aborted=aborted,
        )

    def evaluate_survey(self) -> SurveyResult:
        """Re-derive the survey outputs from the runs already on disk.

        Runs no new solves: reads every
        ``survey/runs/d<d>nm/blaze<b>deg/multilayer_theta_search_summary.csv``
        that exists, rebuilds ``survey/survey.csv``, the per-period overlays and
        the two headline curves, and returns a fresh :class:`SurveyResult`. Use
        it to regenerate the analysis after a completed :meth:`run_survey`.

        Raises:
            RuntimeError: If no completed run folder is found under
                ``config.runs_dir``.
        """

        config = self.config
        found: dict[tuple[float, float], Path] = {}
        if config.runs_dir.is_dir():
            for run_dir in sorted(config.runs_dir.glob("d*nm/blaze*deg")):
                if not (run_dir / "multilayer_theta_search_summary.csv").is_file():
                    continue
                try:
                    d_nm = float(run_dir.parent.name[1:-2])
                    blaze = float(run_dir.name[5:-3])
                except ValueError:
                    continue
                found[(d_nm, blaze)] = run_dir
        if not found:
            raise RuntimeError(
                f"no completed survey runs found under {config.runs_dir}; run run_survey() first"
            )

        d_values = np.array(sorted({d for d, _ in found}), dtype=float)
        blaze_values = np.array(sorted({b for _, b in found}), dtype=float)
        n_d, n_blaze = d_values.size, blaze_values.size
        efficiency_map = np.full((n_d, n_blaze), np.nan)
        incidence_map = np.full((n_d, n_blaze), np.nan)
        fwhm_map = np.full((n_d, n_blaze), np.nan)
        bragg_map = np.full((n_d, n_blaze), np.nan)
        run_dir_by_cell: dict[tuple[int, int], Path] = {}
        rows: list[dict[str, float]] = []
        for (d_nm, blaze), run_dir in sorted(found.items()):
            i_d = int(np.argmin(np.abs(d_values - d_nm)))
            i_b = int(np.argmin(np.abs(blaze_values - blaze)))
            row = pd.read_csv(run_dir / "multilayer_theta_search_summary.csv").iloc[0]
            selected_angle = float(row["selected_grazing_angle_deg"])
            fwhm_value = row.get("precise_fwhm_deg", float("nan"))
            bragg_estimate = float("nan")
            params_path = run_dir / "search_parameters.json"
            if params_path.is_file():
                bragg_estimate = float(
                    json.loads(params_path.read_text()).get("bragg_estimate_deg", float("nan"))
                )
            summary = {
                "bragg_estimate_deg": bragg_estimate,
                "incidence_angle_deg": selected_angle,
                "peak_efficiency": float(row["selected_efficiency"]),
                "precise_fwhm_deg": float(fwhm_value) if pd.notna(fwhm_value) else float("nan"),
                "edge_clipped": _edge_clipped_from_scan(run_dir / "theta_scans", selected_angle),
            }
            run_dir_by_cell[(i_d, i_b)] = run_dir
            efficiency_map[i_d, i_b] = summary["peak_efficiency"]
            incidence_map[i_d, i_b] = summary["incidence_angle_deg"]
            fwhm_map[i_d, i_b] = summary["precise_fwhm_deg"]
            bragg_map[i_d, i_b] = summary["bragg_estimate_deg"]
            rows.append(
                {
                    "d_nm": float(d_nm),
                    "blaze_deg": float(blaze),
                    "bragg_estimate_deg": summary["bragg_estimate_deg"],
                    "incidence_angle_deg": summary["incidence_angle_deg"],
                    "peak_efficiency": summary["peak_efficiency"],
                    "precise_fwhm_deg": summary["precise_fwhm_deg"],
                    "edge_clipped": summary["edge_clipped"],
                }
            )
        print(f"Re-evaluating {len(found)} completed runs under {config.runs_dir}")
        return self._finalize_survey(
            d_values=d_values,
            blaze_values=blaze_values,
            efficiency_map=efficiency_map,
            incidence_map=incidence_map,
            fwhm_map=fwhm_map,
            bragg_map=bragg_map,
            run_dir_by_cell=run_dir_by_cell,
            rows=rows,
            aborted=False,
        )

    def _finalize_survey(
        self,
        *,
        d_values: np.ndarray,
        blaze_values: np.ndarray,
        efficiency_map: np.ndarray,
        incidence_map: np.ndarray,
        fwhm_map: np.ndarray,
        bragg_map: np.ndarray,
        run_dir_by_cell: dict[tuple[int, int], Path],
        rows: list[dict[str, float]],
        aborted: bool,
    ) -> SurveyResult:
        """Aggregate the per-cell results into ``survey.csv``, the plots and a result."""

        config = self.config
        n_blaze = blaze_values.size
        optimal_blaze, optimal_eff, optimal_inc = _optimal_blaze_per_d(
            blaze_values, efficiency_map, incidence_map
        )
        combined = pd.DataFrame(rows)
        config.survey_dir.mkdir(parents=True, exist_ok=True)
        csv_path = config.survey_dir / "survey.csv"
        combined.to_csv(csv_path, index=False)

        period_dirs: list[Path] = []
        run_dirs: list[Path] = []
        for i_d, d_spacing in enumerate(d_values):
            d_spacing = float(d_spacing)
            cells = [
                (float(blaze_values[i_b]), run_dir_by_cell[(i_d, i_b)])
                for i_b in range(n_blaze)
                if (i_d, i_b) in run_dir_by_cell
            ]
            if not cells:
                continue
            run_dirs.extend(run_dir for _, run_dir in cells)
            period_dir = config.period_dir(d_spacing)
            self._plot_period_overlay(
                d_spacing_nm=d_spacing,
                cells=cells,
                chosen_blaze_deg=(
                    float(optimal_blaze[i_d]) if np.isfinite(optimal_blaze[i_d]) else None
                ),
                output_path=period_dir / "overlay.png",
            )
            period_dirs.append(period_dir)

        config.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = config.plot_dir / "optimal_blaze_vs_d_spacing.png"
        self._plot_optimal_blaze(
            d_values=d_values,
            optimal_blaze=optimal_blaze,
            optimal_eff=optimal_eff,
            output_path=plot_path,
        )
        efficiency_plot_path = config.plot_dir / "max_efficiency_vs_d_spacing.png"
        self._plot_max_efficiency(
            d_values=d_values,
            optimal_blaze=optimal_blaze,
            optimal_eff=optimal_eff,
            output_path=efficiency_plot_path,
        )
        heatmap_plot_path = config.plot_dir / "efficiency_heatmap_d_vs_blaze.png"
        self._plot_efficiency_heatmap(
            d_values=d_values,
            blaze_values=blaze_values,
            efficiency_map=efficiency_map,
            optimal_blaze=optimal_blaze,
            output_path=heatmap_plot_path,
        )

        print(f"Survey CSV: {csv_path}")
        print(f"Per-run results: {len(run_dirs)} folders under {config.runs_dir}")
        for d_spacing, blaze_opt, eff in zip(d_values, optimal_blaze, optimal_eff):
            print(
                f"  d {d_spacing:.3f} nm -> optimal blaze {blaze_opt:.3f} deg "
                f"(efficiency {eff:.4g})"
            )
        return SurveyResult(
            d_values_nm=d_values,
            blaze_values_deg=blaze_values,
            efficiency_map=efficiency_map,
            incidence_angle_map_deg=incidence_map,
            fwhm_map_deg=fwhm_map,
            bragg_estimate_map_deg=bragg_map,
            optimal_blaze_deg=optimal_blaze,
            optimal_blaze_efficiency=optimal_eff,
            optimal_blaze_incidence_deg=optimal_inc,
            combined_csv_path=csv_path,
            plot_path=plot_path,
            efficiency_plot_path=efficiency_plot_path,
            heatmap_plot_path=heatmap_plot_path,
            runs_dir=config.runs_dir,
            period_dirs=period_dirs,
            run_dirs=run_dirs,
            aborted=aborted,
            results=combined,
        )

    # ------------------------------------------------------------------ #
    # Step 2: per-design energy scan                                     #
    # ------------------------------------------------------------------ #
    def run_energy_scan(
        self,
        pairs: Iterable[tuple[float, float]],
        *,
        progress_callback: Callable[[StageProgress], None] | None = None,
    ) -> list[EnergyScanResult]:
        """Sweep chosen ``(d_spacing_nm, blaze_angle_deg)`` designs over energy.

        Args:
            pairs: ``(d_spacing_nm, blaze_angle_deg)`` designs, typically read
                off the survey.
            progress_callback: Optional callable invoked with a
                :class:`StageProgress` before each design and once when finished.

        Returns:
            One :class:`EnergyScanResult` per design, in input order.

        Raises:
            ValueError: If ``pairs`` is empty.
        """

        config = self.config
        scan = config.energy_scan_settings
        designs = [(float(d), float(b)) for d, b in pairs]
        if not designs:
            raise ValueError("run_energy_scan requires at least one (d_spacing, blaze) pair")
        energies = config.energy_grid_ev()
        total = len(designs)
        results: list[EnergyScanResult] = []
        for index, (d_spacing, blaze) in enumerate(designs):
            _emit_progress(
                progress_callback,
                stage="energy_scan",
                completed=index,
                total=total,
                current_label=f"d = {d_spacing:.3f} nm, blaze = {blaze:.3f} deg",
            )
            design_dir = config.energy_scan_dir / f"d{d_spacing:.3f}nm_blaze{blaze:.3f}deg"
            design_dir.mkdir(parents=True, exist_ok=True)
            sweep = run_multilayer_theta_search_sweep(
                grating=self._build_grating(d_spacing, blaze),
                energies_ev=energies,
                output_dir=design_dir,
                diffraction_order=int(config.diffraction_order),
                multilayer_bragg_order=int(config.multilayer_bragg_order),
                rough_scan_half_width_deg=float(scan.rough_scan_half_width_deg),
                rough_scan_points=int(scan.rough_scan_points),
                rough_fourier_orders=int(scan.rough_fourier_orders),
                rough_x_resolution_nm=float(scan.rough_x_resolution_nm),
                rough_z_resolution_nm=float(scan.rough_z_resolution_nm),
                fine_scan_half_width_deg=float(scan.fine_scan_half_width_deg),
                fine_scan_points=int(scan.fine_scan_points),
                fine_fourier_orders=int(scan.fine_fourier_orders),
                fine_x_resolution_nm=float(scan.fine_x_resolution_nm),
                fine_z_resolution_nm=float(scan.fine_z_resolution_nm),
                final_fourier_orders=int(scan.final_fourier_orders),
                final_x_resolution_nm=float(scan.final_x_resolution_nm),
                final_z_resolution_nm=float(scan.final_z_resolution_nm),
                roughness_sigma_nm=scan.roughness_sigma_nm,
                precise_peak_selection_mode=str(scan.precise_peak_selection_mode),
                max_workers=config.max_workers,
                show_progress=bool(config.show_progress),
                on_error="fail_fast",
                checkpoint_dir=(design_dir / "checkpoints") if config.checkpoint else None,
                checkpoint_interval=1,
                resume=bool(config.resume and config.checkpoint),
                theta_tracking_mode=str(config.theta_tracking_mode),
                max_tracking_energy_step_ev=config.max_tracking_energy_step_ev,
                save_profile_plot=bool(config.save_profile_plot),
                save_stack_plot=bool(config.save_stack_plot),
                backend=str(config.backend),
                solver=str(config.solver),
                polarization=str(config.polarization),
            )
            scan_results = pd.read_csv(sweep.summary_csv_path)
            config.plot_dir.mkdir(parents=True, exist_ok=True)
            titled_plot_path = config.plot_dir / _energy_scan_plot_filename(
                config, d_spacing, blaze
            )
            self._plot_energy_scan(
                d_spacing_nm=d_spacing,
                blaze_angle_deg=blaze,
                results=scan_results,
                output_path=titled_plot_path,
            )
            results.append(
                EnergyScanResult(
                    d_spacing_nm=d_spacing,
                    blaze_angle_deg=blaze,
                    output_dir=design_dir,
                    summary_csv_path=sweep.summary_csv_path,
                    all_orders_csv_path=sweep.all_orders_csv_path,
                    energy_efficiency_plot_path=sweep.energy_efficiency_plot_path,
                    titled_plot_path=titled_plot_path,
                    results=scan_results,
                )
            )
        _emit_progress(
            progress_callback,
            stage="energy_scan",
            completed=total,
            total=total,
            current_label="done",
        )
        return results

    def evaluate_energy_scan(
        self, pairs: Iterable[tuple[float, float]] | None = None
    ) -> list[EnergyScanResult]:
        """Collect energy-scan results already on disk, running no new solves.

        Reads
        ``energy_scan/d<d>nm_blaze<b>deg/multilayer_theta_search_summary.csv`` for
        each requested design (or, when ``pairs`` is ``None``, for every design
        folder found) and returns the matching :class:`EnergyScanResult` objects.

        Raises:
            RuntimeError: If no requested/found design folder holds a summary CSV.
        """

        config = self.config
        if pairs is None:
            designs = []
            if config.energy_scan_dir.is_dir():
                for design_dir in sorted(config.energy_scan_dir.glob("d*nm_blaze*deg")):
                    stem = design_dir.name
                    try:
                        d_text, blaze_text = stem.split("_blaze")
                        designs.append((float(d_text[1:-2]), float(blaze_text[:-3])))
                    except ValueError:
                        continue
        else:
            designs = [(float(d), float(b)) for d, b in pairs]

        results: list[EnergyScanResult] = []
        for d_spacing, blaze in designs:
            design_dir = config.energy_scan_dir / f"d{d_spacing:.3f}nm_blaze{blaze:.3f}deg"
            summary_csv = design_dir / "multilayer_theta_search_summary.csv"
            if not summary_csv.is_file():
                print(f"  skipping {design_dir} (no summary CSV)")
                continue
            scan_results = pd.read_csv(summary_csv)
            config.plot_dir.mkdir(parents=True, exist_ok=True)
            titled_plot_path = config.plot_dir / _energy_scan_plot_filename(
                config, d_spacing, blaze
            )
            self._plot_energy_scan(
                d_spacing_nm=d_spacing,
                blaze_angle_deg=blaze,
                results=scan_results,
                output_path=titled_plot_path,
            )
            results.append(
                EnergyScanResult(
                    d_spacing_nm=d_spacing,
                    blaze_angle_deg=blaze,
                    output_dir=design_dir,
                    summary_csv_path=summary_csv,
                    all_orders_csv_path=design_dir
                    / "multilayer_theta_search_all_orders.csv",
                    energy_efficiency_plot_path=design_dir
                    / "multilayer_theta_search_energy_vs_efficiency.png",
                    titled_plot_path=titled_plot_path,
                    results=scan_results,
                )
            )
        if not results:
            raise RuntimeError(
                f"no completed energy scans found under {config.energy_scan_dir}"
            )
        print(f"Re-evaluated {len(results)} energy scans under {config.energy_scan_dir}")
        return results

    # ------------------------------------------------------------------ #
    # Plotting                                                           #
    # ------------------------------------------------------------------ #
    def _plot_optimal_blaze(
        self,
        *,
        d_values: np.ndarray,
        optimal_blaze: np.ndarray,
        optimal_eff: np.ndarray,
        output_path: Path,
    ) -> None:
        """Write the headline curve: optimal blaze angle versus d-spacing.

        The line is coloured by peak efficiency; a sparse set of points is
        labelled with their efficiency so the text stays readable on fine grids.
        """

        import matplotlib.pyplot as plt

        config = self.config
        finite = np.isfinite(optimal_blaze)
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(d_values[finite], optimal_blaze[finite], "-", color="0.6", zorder=1)
        scatter = axis.scatter(
            d_values[finite],
            optimal_blaze[finite],
            c=optimal_eff[finite],
            cmap="viridis",
            zorder=2,
        )
        figure.colorbar(scatter, ax=axis, label="peak efficiency")
        for i in _label_indices(optimal_eff):
            axis.annotate(
                f"{optimal_eff[i]:.3g}",
                xy=(d_values[i], optimal_blaze[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Bilayer d-spacing (nm)")
        axis.set_ylabel("Optimal blaze angle (deg)")
        axis.set_title(
            f"Optimal blaze angle versus d-spacing at {config.target_energy_ev:g} eV "
            f"(order {config.diffraction_order}; colour/labels: peak efficiency)"
        )
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _plot_max_efficiency(
        self,
        *,
        d_values: np.ndarray,
        optimal_blaze: np.ndarray,
        optimal_eff: np.ndarray,
        output_path: Path,
    ) -> None:
        """Write max selected-order efficiency versus d-spacing.

        The line is coloured by the optimal blaze angle; a sparse set of points
        (the peak and where the optimal blaze changes) is labelled with the blaze
        angle so the text stays readable on fine grids.
        """

        import matplotlib.pyplot as plt

        config = self.config
        finite = np.isfinite(optimal_eff)
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(d_values[finite], optimal_eff[finite], "-", color="0.6", zorder=1)
        scatter = axis.scatter(
            d_values[finite],
            optimal_eff[finite],
            c=optimal_blaze[finite],
            cmap="plasma",
            zorder=2,
        )
        figure.colorbar(scatter, ax=axis, label="optimal blaze angle (deg)")
        for i in _label_indices(optimal_eff, change_series=optimal_blaze):
            axis.annotate(
                f"blaze {optimal_blaze[i]:.3f} deg",
                xy=(d_values[i], optimal_eff[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Bilayer d-spacing (nm)")
        axis.set_ylabel(f"Max selected-order efficiency ({config.polarization}-pol)")
        axis.set_title(
            f"Max efficiency versus d-spacing at {config.target_energy_ev:g} eV "
            f"(order {config.diffraction_order}; colour/labels: optimal blaze angle)"
        )
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _plot_efficiency_heatmap(
        self,
        *,
        d_values: np.ndarray,
        blaze_values: np.ndarray,
        efficiency_map: np.ndarray,
        optimal_blaze: np.ndarray,
        output_path: Path,
    ) -> None:
        """Write the full ``(d, blaze) -> efficiency`` heatmap.

        x is d-spacing, y is blaze angle, colour is peak selected-order
        efficiency -- the whole ``efficiency_map``, not just the per-d optimum.
        The ``optimal_blaze_deg`` ridge is overlaid so the two headline curves
        and this map read together.
        """

        import matplotlib.pyplot as plt

        config = self.config
        figure, axis = plt.subplots(figsize=(9, 6))
        mesh = axis.pcolormesh(
            d_values,
            blaze_values,
            efficiency_map.T,
            shading="nearest",
            cmap="viridis",
        )
        figure.colorbar(
            mesh, ax=axis, label=f"peak selected-order efficiency ({config.polarization}-pol)"
        )
        finite = np.isfinite(optimal_blaze)
        axis.plot(
            d_values[finite],
            optimal_blaze[finite],
            "o-",
            color="white",
            markeredgecolor="black",
            linewidth=1.6,
            label="optimal blaze per d",
        )
        axis.set_xlabel("Bilayer d-spacing (nm)")
        axis.set_ylabel("Blaze angle (deg)")
        axis.set_title(
            f"Peak efficiency over (d, blaze) at {config.target_energy_ev:g} eV "
            f"(order {config.diffraction_order})"
        )
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _plot_energy_scan(
        self,
        *,
        d_spacing_nm: float,
        blaze_angle_deg: float,
        results: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Write one design's efficiency-versus-energy curve, titled with the design.

        The title names the coating (:func:`_coating_label`) and this design's
        d-spacing and blaze angle, so the plot is self-describing without
        needing its folder name.
        """

        import matplotlib.pyplot as plt

        config = self.config
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(
            results["energy_ev"],
            results["selected_efficiency"],
            "-",
            color="tab:blue",
        )
        axis.set_xlabel("Photon energy (eV)")
        axis.set_ylabel(f"Selected-order efficiency ({config.polarization}-pol)")
        axis.set_title(_energy_scan_title(config, d_spacing_nm, blaze_angle_deg))
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _search_parameters_record(
        self, *, d_spacing_nm: float, blaze_angle_deg: float, bragg_estimate_deg: float
    ) -> dict[str, object]:
        """Build ``search_parameters.json`` for one run so the search can be tuned.

        The run's numerical *results* are already in the sweep's own artifacts
        (``multilayer_theta_search_summary.csv``, ``*_all_orders.csv``,
        ``theta_scans/``); this records only what was fed into the search.
        """

        config = self.config
        scan = config.survey_scan_settings
        return {
            "d_spacing_nm": float(d_spacing_nm),
            "blaze_angle_deg": float(blaze_angle_deg),
            "target_energy_ev": float(config.target_energy_ev),
            "diffraction_order": int(config.diffraction_order),
            "multilayer_bragg_order": int(config.multilayer_bragg_order),
            "bragg_estimate_deg": float(bragg_estimate_deg),
            "theta_seed": "multilayer Bragg estimate (theta_tracking_mode='bragg')",
            "rough_scan_half_width_deg": float(scan.rough_scan_half_width_deg),
            "rough_scan_points": int(scan.rough_scan_points),
            "rough_fourier_orders": int(scan.rough_fourier_orders),
            "rough_x_resolution_nm": float(scan.rough_x_resolution_nm),
            "rough_z_resolution_nm": float(scan.rough_z_resolution_nm),
            "fine_scan_half_width_deg": float(scan.fine_scan_half_width_deg),
            "fine_scan_points": int(scan.fine_scan_points),
            "fine_fourier_orders": int(scan.fine_fourier_orders),
            "fine_x_resolution_nm": float(scan.fine_x_resolution_nm),
            "fine_z_resolution_nm": float(scan.fine_z_resolution_nm),
            "final_fourier_orders": int(scan.final_fourier_orders),
            "final_x_resolution_nm": float(scan.final_x_resolution_nm),
            "final_z_resolution_nm": float(scan.final_z_resolution_nm),
            "precise_peak_selection_mode": str(scan.precise_peak_selection_mode),
            "roughness_sigma_nm": (
                None if scan.roughness_sigma_nm is None else float(scan.roughness_sigma_nm)
            ),
            "grating_x_resolution_nm": float(config.x_resolution_nm),
            "grating_z_resolution_nm": float(config.z_resolution_nm),
            "solver": str(config.solver),
            "polarization": str(config.polarization),
            "backend": str(config.backend),
        }

    def _plot_period_overlay(
        self,
        *,
        d_spacing_nm: float,
        cells: list[tuple[float, Path]],
        chosen_blaze_deg: float | None,
        output_path: Path,
    ) -> None:
        """Overlay every blaze angle's theta scan at one d-spacing.

        The blaze angle that won (highest peak efficiency, ``chosen_blaze_deg``)
        is drawn bold with a ``(chosen)`` legend entry, a dashed vertical line at
        its selected incidence angle and a large star at its peak; the other
        blaze angles are drawn light with a small star at each selected peak.
        """

        import matplotlib.pyplot as plt

        config = self.config
        figure, axis = plt.subplots(figsize=(9, 6))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for order, (blaze, run_dir) in enumerate(cells):
            frame = _read_theta_scan_frame(run_dir)
            if frame is None:
                continue
            is_chosen = chosen_blaze_deg is not None and bool(
                np.isclose(blaze, chosen_blaze_deg, rtol=0.0, atol=1.0e-9)
            )
            color = color_cycle[order % len(color_cycle)]
            rough = frame[frame["scan_type"] == "rough"]
            precise = frame[frame["scan_type"] == "precise"]
            axis.plot(
                rough["grazing_angle_deg"],
                rough["selected_efficiency"],
                ":",
                color=color,
                linewidth=0.9,
                alpha=0.6 if is_chosen else 0.35,
            )
            axis.plot(
                precise["grazing_angle_deg"],
                precise["selected_efficiency"],
                "-",
                color=color,
                linewidth=2.8 if is_chosen else 1.3,
                alpha=1.0 if is_chosen else 0.75,
                label=f"blaze {blaze:.3f} deg" + (" (chosen)" if is_chosen else ""),
                zorder=5 if is_chosen else 3,
            )
            peak = frame[frame["is_selected_peak"] == 1]
            if not peak.empty:
                peak_angle = float(peak["grazing_angle_deg"].iloc[0])
                peak_eff = float(peak["selected_efficiency"].iloc[0])
                axis.plot(
                    [peak_angle],
                    [peak_eff],
                    "*",
                    color=color,
                    markersize=20 if is_chosen else 9,
                    markeredgecolor="black",
                    zorder=6 if is_chosen else 4,
                )
                if is_chosen:
                    axis.axvline(
                        peak_angle, color=color, linestyle="--", linewidth=1.2, alpha=0.7
                    )
                    axis.text(
                        0.02,
                        0.72,
                        f"chosen: blaze {blaze:.3f} deg\n"
                        f"theta* = {peak_angle:.3f} deg\n"
                        f"efficiency = {peak_eff:.4g}",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round", "fc": "white", "ec": color, "alpha": 0.9},
                    )
        axis.set_xlabel("Incidence (grazing) angle (deg)")
        axis.set_ylabel(f"Selected-order efficiency ({config.polarization}-pol)")
        axis.set_title(
            f"Multilayer theta search at d = {d_spacing_nm:.3f} nm, "
            f"{config.target_energy_ev:g} eV (dotted: rough, solid: precise, *: selected peak)"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def _read_theta_scan_frame(run_dir: Path) -> pd.DataFrame | None:
    """Return the rough/precise scan table a sweep wrote under ``run_dir``."""

    scan_files = sorted((run_dir / "theta_scans").glob("theta_scan_*.csv"))
    if not scan_files:
        return None
    return pd.read_csv(scan_files[0])


def _edge_clipped_from_scan(theta_scan_directory: Path, selected_angle_deg: float) -> float:
    """Flag whether the selected angle sits within 5% of a precise-scan edge."""

    scan_files = sorted(Path(theta_scan_directory).glob("theta_scan_*.csv"))
    if not scan_files:
        return float("nan")
    frame = pd.read_csv(scan_files[0])
    precise = np.asarray(
        frame.loc[frame["scan_type"] == "precise", "grazing_angle_deg"], dtype=float
    )
    if precise.size == 0:
        return float("nan")
    span = float(precise.max() - precise.min())
    margin = 0.05 * span if span > 0.0 else 0.0
    return float(
        selected_angle_deg <= precise.min() + margin
        or selected_angle_deg >= precise.max() - margin
    )


def _label_indices(
    values: np.ndarray,
    *,
    change_series: np.ndarray | None = None,
    max_labels: int = 14,
) -> list[int]:
    """Pick a readable subset of indices to annotate on a per-d headline curve.

    Always keeps the first and last finite points and the efficiency peak; adds
    every point where ``change_series`` steps to a new value, then tops up (or
    thins) to about ``max_labels`` evenly spaced points.
    """

    finite = np.where(np.isfinite(values))[0]
    if finite.size == 0:
        return []
    peak = int(finite[int(np.nanargmax(values[finite]))])
    chosen = {int(finite[0]), int(finite[-1]), peak}
    if change_series is not None:
        previous: float | None = None
        for i in finite:
            current = float(change_series[i])
            if not np.isfinite(current):
                continue
            if previous is None or not np.isclose(current, previous):
                chosen.add(int(i))
            previous = current
    if len(chosen) < max_labels:
        step = max(1, finite.size // max_labels)
        chosen.update(int(i) for i in finite[::step])
    ordered = sorted(chosen)
    if len(ordered) > max_labels:
        keep = {ordered[0], ordered[-1], peak}
        rest = [i for i in ordered if i not in keep]
        step = max(1, len(rest) // max(1, max_labels - len(keep)))
        keep.update(rest[::step])
        ordered = sorted(keep)
    return ordered


def _optimal_blaze_per_d(
    blaze_values: np.ndarray,
    efficiency_map: np.ndarray,
    incidence_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(blaze_opt, efficiency, theta*)`` per d-spacing row of the maps."""

    n_d = efficiency_map.shape[0]
    blaze_opt = np.full(n_d, np.nan)
    eff_opt = np.full(n_d, np.nan)
    inc_opt = np.full(n_d, np.nan)
    for i_d in range(n_d):
        row = efficiency_map[i_d, :]
        if not np.any(np.isfinite(row)):
            continue
        i_b = int(np.nanargmax(row))
        blaze_opt[i_d] = float(blaze_values[i_b])
        eff_opt[i_d] = float(row[i_b])
        inc_opt[i_d] = float(incidence_map[i_d, i_b])
    return blaze_opt, eff_opt, inc_opt
