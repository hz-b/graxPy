#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "==> Running blazed optimizer workflow"
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/optimizer_blazed/run_all.sh"

echo "==> Running laminar optimizer workflow"
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/optimizer_laminar/run_all.sh"

echo "==> Running joint measurement-fit workflow"
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/optimizer_joint/run_all.sh"

echo "==> All optimizer workflows completed"
