#!/usr/bin/env bash
# Run all grating profile generation scripts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "$SCRIPT_DIR/laminar_no_top_cap.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/laminar_with_top_cap.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/blazed_no_top_cap.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/blazed_with_top_cap.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/blazed_multilayer_custom_stack.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/sinusoidal_custom_profile.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/afm_preprocessing_blazed_profile.py" "$@"
"${PYTHON_BIN}" "$SCRIPT_DIR/afm_preprocessing_laminar_profile.py" "$@"
