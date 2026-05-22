#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Fitting measurement-fit laminar grating"
python "${SCRIPT_DIR}/0_fit_dynamic_laminar_grating.py"

echo "==> Simulating measurement-fit design parameters"
python "${SCRIPT_DIR}/1_run_simulation_design_parameters.py"

echo "==> Simulating measurement-fit fitted parameters"
python "${SCRIPT_DIR}/2_run_simulation_fitted_parameters.py"

echo "==> Plotting measurement-fit comparison"
python "${SCRIPT_DIR}/3_plot_dynamic_fit_comparison.py"
