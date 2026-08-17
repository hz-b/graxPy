#!/usr/bin/env bash
# Curated profile and optimizer workflows. Tutorial-image synchronization is
# handled by tools/build_docs.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Running grating examples..."
PYTHON_BIN="${PYTHON_BIN}" bash "$SCRIPT_DIR/grating/run_all.sh"

echo "Running optimizer examples..."
PYTHON_BIN="${PYTHON_BIN}" bash "$SCRIPT_DIR/optimizer/run_all.sh"

echo "All curated examples completed."
