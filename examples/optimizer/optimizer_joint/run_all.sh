#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOLVER="${SOLVER:-rcwa}"

echo "==> Running 0_generate_measurements.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/0_generate_measurements.py"

echo "==> Running 1_fit_joint.py (--solver ${SOLVER})"
"${PYTHON_BIN}" "${SCRIPT_DIR}/1_fit_joint.py" --solver "${SOLVER}"

echo "==> Running 2_resume_and_extend.py (--solver ${SOLVER})"
"${PYTHON_BIN}" "${SCRIPT_DIR}/2_resume_and_extend.py" --solver "${SOLVER}"

echo "==> Running 3_plot_joint_fit_comparison.py (--solver ${SOLVER})"
"${PYTHON_BIN}" "${SCRIPT_DIR}/3_plot_joint_fit_comparison.py" --solver "${SOLVER}"
