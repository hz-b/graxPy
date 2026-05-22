#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# echo "==> Running dynamic optimizer workflow"
# bash "${SCRIPT_DIR}/dynamic_optimizer/run_all.sh"

echo "==> Running blazed optimizer workflow"
bash "${SCRIPT_DIR}/optimizer_blazed/run_all.sh"

echo "==> Running laminar optimizer workflow"
bash "${SCRIPT_DIR}/optimizer_laminar/run_all.sh"

echo "==> All optimizer workflows completed"
