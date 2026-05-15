"""Compatibility wrapper that runs numbered simulation and plotting scripts."""

from __future__ import annotations

import runpy
from pathlib import Path

script_dir = Path(__file__).resolve().parent

runpy.run_path(str(script_dir / "1_run_simulation_design_parameters.py"), run_name="__main__")
runpy.run_path(str(script_dir / "2_run_simulation_fitted_parameters.py"), run_name="__main__")
runpy.run_path(str(script_dir / "3_plot_laminar_fit_comparison.py"), run_name="__main__")
