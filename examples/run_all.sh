#!/usr/bin/env bash
# Run every example group. Tutorial-image synchronization is handled by
# tools/build_docs.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_grating=false
run_optimizer=false
run_simulations=false

if [[ $# -eq 0 ]]; then
    run_grating=true
    run_optimizer=true
    run_simulations=true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --grating)
            run_grating=true
            ;;
        --optimizer)
            run_optimizer=true
            ;;
        --simulations)
            run_simulations=true
            ;;
        --all)
            run_grating=true
            run_optimizer=true
            run_simulations=true
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_all.sh [--grating] [--optimizer] [--simulations] [--all]

With no options, runs all example groups.  Specify one or more group options
to run only those groups.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ "$run_grating" == true ]]; then
    echo "Running grating examples..."
    PYTHON_BIN="${PYTHON_BIN}" bash "$SCRIPT_DIR/grating/run_all.sh"
fi

if [[ "$run_optimizer" == true ]]; then
    echo "Running optimizer examples..."
    PYTHON_BIN="${PYTHON_BIN}" bash "$SCRIPT_DIR/optimizer/run_all.sh"
fi

if [[ "$run_simulations" == true ]]; then
    echo "Running simulation examples..."
    PYTHON_BIN="${PYTHON_BIN}" bash "$SCRIPT_DIR/simulation/run_all.sh"
fi

echo "Selected example groups completed."
