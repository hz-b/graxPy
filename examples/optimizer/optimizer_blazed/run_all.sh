#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/0_fit_blazed_grating.py"
# python "${SCRIPT_DIR}/1_run_simulation_design_parameters.py"
python "${SCRIPT_DIR}/2_run_simulation_fitted_parameters.py"
# python "${SCRIPT_DIR}/1b_run_simulation_fitted_parameters_alternative.py"
python "${SCRIPT_DIR}/3_plot_blazed_fit_comparison.py"
