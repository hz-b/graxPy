#!/usr/bin/env bash
# Run the three Ru/B4C multilayer optimization stages in order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOLVER="${SOLVER:-neviere}"

echo "==> Running 0_ru_b4c_d_spacing_study.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/0_ru_b4c_d_spacing_study.py"

echo "==> Running 1_ru_b4c_gamma_study.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/1_ru_b4c_gamma_study.py"

echo "==> Running 2_ru_b4c_blaze_study.py (--solver ${SOLVER})"
"${PYTHON_BIN}" "${SCRIPT_DIR}/2_ru_b4c_blaze_study.py" --solver "${SOLVER}"
