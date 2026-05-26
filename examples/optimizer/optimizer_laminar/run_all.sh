#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Running 0_fit_laminar_grating.py"
python "${SCRIPT_DIR}/0_fit_laminar_grating.py"

echo "==> Running 0b_fit_laminar_grating_tied_walls.py"
python "${SCRIPT_DIR}/0b_fit_laminar_grating_tied_walls.py"

echo "==> Running 1_run_simulation_design_parameters.py"
python "${SCRIPT_DIR}/1_run_simulation_design_parameters.py"

echo "==> Running 2_run_simulation_fitted_parameters.py"
python "${SCRIPT_DIR}/2_run_simulation_fitted_parameters.py"

echo "==> Running 2b_run_simulation_tied_wall_fitted_parameters.py"
python "${SCRIPT_DIR}/2b_run_simulation_tied_wall_fitted_parameters.py"

echo "==> Running 3_plot_laminar_fit_comparison.py"
python "${SCRIPT_DIR}/3_plot_laminar_fit_comparison.py"
