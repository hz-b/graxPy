"""Top-level package for grax."""

from __future__ import annotations

import logging

# Attach a no-op handler to the package logger so that when an application has
# not configured logging, grax log records are not emitted to stderr by
# ``logging.lastResort``. This matters for the spawned batch workers, which
# re-import grax but never call ``setup_logging``: without this, every
# ``logger.warning`` inside a worker leaks to the terminal.
logging.getLogger("grax").addHandler(logging.NullHandler())

from .afm_grating import AFMGrating
from .afm_preprocessing import AFMPreprocessing
from .materials import (
    MaterialSpec,
    available_material_symbols,
    material_density_catalog,
    material_density_g_cm3,
)
from .gratings import BaseGrating, BlazedGrating, LaminarGrating, ProfileGrating
from .simulation.core import normalize_polarization
from .parameter_sweep import (
    ParameterStudyEnergyResult,
    ParameterStudyResult,
    ParameterSweepSeries,
    get_default_parameter_study_ranges,
    plot_parameter_study,
    run_parameter_study,
)
from .solvers import NeviereOptions, res0, res1, res2, res2_dm
from .roughness import RoughnessSpec
from .stacks import (
    BaseStack,
    CustomStack,
    LayerSpec,
    MultilayerStack,
    SingleLayerStack,
    assemble_custom_stack,
    build_multilayer_stack,
    build_single_layer_stack,
)
from .simulation import (
    BatchSimulationRunner,
    CaseExecutionResult,
    MultilayerThetaSearchSweepResult,
    SingleSimulationResult,
    ThetaSearchDiagnostics,
    efficiency_for_order,
    energy_angle_cases,
    estimate_multilayer_bragg_angle_deg,
    fixed_angle_cases,
    load_experimental_csv,
    multilayer_theta_search_cases,
    monochromator_cases,
    monochromator_grazing_angles_deg,
    plot_order_subset,
    run_multilayer_theta_search,
    run_multilayer_theta_search_sweep,
    run_simulation,
    write_all_orders_csv,
)
from .multilayer_reflectivity import MultilayerReflectivity
from .multilayer_optimization import (
    BlazeStudyResult,
    DSpacingStudyResult,
    GammaStudyResult,
    MultilayerOptimizationConfig,
    run_blaze_study,
    run_d_spacing_study,
    run_gamma_study,
)
from .slag import SlagConfig, default_example_slag_config, run_example_slag, simulate_single_energy

__all__ = [
    "AFMGrating",
    "AFMPreprocessing",
    "BaseGrating",
    "BaseStack",
    "BatchSimulationRunner",
    "BlazeStudyResult",
    "BlazedGrating",
    "CaseExecutionResult",
    "CustomStack",
    "DSpacingStudyResult",
    "GammaStudyResult",
    "LayerSpec",
    "MaterialSpec",
    "LaminarGrating",
    "MultilayerOptimizationConfig",
    "MultilayerReflectivity",
    "MultilayerThetaSearchSweepResult",
    "MultilayerStack",
    "NeviereOptions",
    "ParameterStudyEnergyResult",
    "ParameterStudyResult",
    "ParameterSweepSeries",
    "ProfileGrating",
    "RoughnessSpec",
    "SingleLayerStack",
    "SingleSimulationResult",
    "SlagConfig",
    "ThetaSearchDiagnostics",
    "assemble_custom_stack",
    "available_material_symbols",
    "build_multilayer_stack",
    "build_single_layer_stack",
    "default_example_slag_config",
    "efficiency_for_order",
    "energy_angle_cases",
    "estimate_multilayer_bragg_angle_deg",
    "fixed_angle_cases",
    "get_default_parameter_study_ranges",
    "load_experimental_csv",
    "material_density_catalog",
    "material_density_g_cm3",
    "multilayer_theta_search_cases",
    "monochromator_cases",
    "normalize_polarization",
    "monochromator_grazing_angles_deg",
    "plot_parameter_study",
    "plot_order_subset",
    "res0",
    "res1",
    "res2",
    "res2_dm",
    "run_blaze_study",
    "run_d_spacing_study",
    "run_example_slag",
    "run_gamma_study",
    "run_parameter_study",
    "run_multilayer_theta_search",
    "run_multilayer_theta_search_sweep",
    "run_simulation",
    "setup_logging",
    "simulate_single_energy",
    "write_all_orders_csv",
]


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_dir: str | None = None,
    run_id: str | None = None,
) -> None:
    """Configure logging for grax simulations.

    Args:
        level: Logging level (``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``).
        log_file: Optional explicit file path to write logs. Overrides log_dir/run_id.
        log_dir: Directory to store log files. Defaults to ``results/logs``.
        run_id: Optional unique identifier for this run. If None, uses timestamp.
    """

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if log_file:
        handler = logging.FileHandler(log_file, mode="w")
    else:
        if log_dir is None:
            log_dir = "results/logs"

        from datetime import datetime
        from pathlib import Path

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_filename = f"{run_id}.log" if run_id else datetime.now().strftime("%Y%m%d_%H%M%S.log")
        handler = logging.FileHandler(log_path / log_filename, mode="w")
        print(f"Logging to: {log_path / log_filename}")

    handler.setFormatter(formatter)
    handler.setLevel(level)

    root_logger = logging.getLogger("grax")
    root_logger.setLevel(level)

    # Do not let grax records bubble to the root logger: they are already
    # written to this file handler, and a StreamHandler attached to the root
    # by the host application would otherwise re-print every one of them.
    root_logger.propagate = False

    # Guard against duplicate file handlers if setup_logging is called twice
    # (e.g. an example re-run in the same interpreter): a second identical
    # FileHandler would write every record to the log twice.
    new_target = getattr(handler, "baseFilename", None)
    for existing in list(root_logger.handlers):
        if (
            isinstance(existing, logging.FileHandler)
            and getattr(existing, "baseFilename", None) == new_target
        ):
            root_logger.removeHandler(existing)
            existing.close()
    root_logger.addHandler(handler)

    logging.getLogger("numpy").setLevel(level)
    logging.getLogger("scipy").setLevel(level)
