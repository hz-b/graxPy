#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed or not in PATH." >&2
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Using existing virtual environment at .venv"
else
  echo "Creating virtual environment at .venv"
  uv venv .venv
fi

echo "Installing graxpy (import namespace: grax) in editable mode (-e)"
uv pip install --python .venv/bin/python -e .

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"
