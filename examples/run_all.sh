#!/bin/bash
# Curated examples runner only.
# This script does not copy assets into docs; docs-sync ownership is:
# docs/tutorials/assets/gratings/scripts/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Running grating examples..."
bash "$SCRIPT_DIR/grating/run_all.sh"

echo "Running simulation examples..."

for script_path in "$SCRIPT_DIR"/simulation/*/*.py; do
    script_name="$(basename "$script_path")"
    echo "  $PYTHON_BIN $script_name"
    "$PYTHON_BIN" "$script_path"
done

echo "All curated examples completed."
