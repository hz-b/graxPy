#!/usr/bin/env bash
# Run the multilayer-grating design survey, then the energy scan of its ridge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "==> Running 0_run_survey.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/0_run_survey.py"

echo "==> Running 1_run_energy_scan.py (survey ridge)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/1_run_energy_scan.py"
