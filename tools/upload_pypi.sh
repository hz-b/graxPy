#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PUBLISH_VENV="${PROJECT_ROOT}/.venv_publish"
TOKEN_FILE="${PROJECT_ROOT}/.token"

get_named_token() {
    local token_file="$1"
    local header="$2"
    awk -v header="$header" '
        $0 ~ "^# " header " token[[:space:]]*$" {in_block=1; next}
        in_block && $0 ~ "^#" {in_block=0}
        in_block && $0 !~ "^[[:space:]]*$" {print; exit}
    ' "$token_file"
}

cd "${PROJECT_ROOT}"

echo "============================================================"
echo "Preparing publish environment"
echo "============================================================"

# Note: no shell-level deactivation is required.
# We always run using ${PUBLISH_VENV}/bin/python explicitly.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Detected active virtualenv: ${VIRTUAL_ENV} (ignored; using publish venv explicitly)"
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Detected active conda env: ${CONDA_DEFAULT_ENV} (ignored; using publish venv explicitly)"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "Error: 'uv' is not installed."
    echo
    echo "Install it with:"
    echo
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo
    echo "or see:"
    echo
    echo "  https://docs.astral.sh/uv/"
    echo
    exit 1
fi

if [[ ! -d "${PUBLISH_VENV}" ]]; then
    echo "Creating dedicated publish virtual environment..."
    uv venv "${PUBLISH_VENV}"
fi

PYTHON_BIN="${PUBLISH_VENV}/bin/python"

# Some venvs may not include pip; repair in-place if needed.
if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    echo "pip is missing in publish venv; bootstrapping with ensurepip..."
    "${PYTHON_BIN}" -m ensurepip --upgrade
fi

echo "Ensuring publish dependencies are installed..."
"${PYTHON_BIN}" -m pip install --upgrade \
    pip \
    build \
    twine

echo
echo "============================================================"
echo "Cleaning previous builds"
echo "============================================================"

rm -rf build dist *.egg-info

echo
echo "============================================================"
echo "Building package"
echo "============================================================"

"${PYTHON_BIN}" -m build

echo
echo "============================================================"
echo "Checking built distributions"
echo "============================================================"

if ! ls dist/* >/dev/null 2>&1; then
    echo "Error: no distributions were produced in dist/."
    exit 1
fi

echo
echo "============================================================"
echo "Uploading to PyPI"
echo "============================================================"

if [[ -f "${TOKEN_FILE}" ]]; then
    TOKEN_VALUE="$(get_named_token "${TOKEN_FILE}" "PyPI")"
    if [[ -z "${TOKEN_VALUE}" ]]; then
        echo "Error: could not find '# PyPI token' entry in ${TOKEN_FILE}."
        exit 1
    fi
    echo "Using PyPI token from ${TOKEN_FILE} (non-interactive upload)."
    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="${TOKEN_VALUE}"
    "${PYTHON_BIN}" -m twine upload --non-interactive dist/*
else
    "${PYTHON_BIN}" -m twine upload dist/*
fi

echo
echo "============================================================"
echo "Done"
echo "============================================================"
echo
echo "Install command:"
echo
echo "pip install graxpy"
