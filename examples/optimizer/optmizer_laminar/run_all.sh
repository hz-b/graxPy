#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run optimizer fit
python "${SCRIPT_DIR}/0_fit_laminar_grating.py"

# Run simulation with design parameters
# python "${SCRIPT_DIR}/1_run_simulation_design_parameters.py"

# Run simulation with fitted parameters
python "${SCRIPT_DIR}/2_run_simulation_fitted_parameters.py"

# Plot measurement vs design vs fitted
python "${SCRIPT_DIR}/3_plot_laminar_fit_comparison.py"
