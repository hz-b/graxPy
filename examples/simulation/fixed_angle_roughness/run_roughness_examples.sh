#!/usr/bin/env bash
# Run the roughness example scripts.
#
# Usage:
#   ./run_roughness_examples.sh [--study kind|correlation|both] [--mode sim|eval|all] [--geometry-only]
#
# --study   Which study to run:
#             kind        Debye-Waller vs random-interface roughness comparison.
#             correlation Random-interface correlation-length sweep.
#             both        Run both studies (default).
#
# --mode    What to run for each selected study:
#             sim   Run the simulation script only. It already saves the
#                   comparison plot itself once its runs finish.
#             eval  Re-plot the comparison figure only, from CSVs already on
#                   disk. Does not run any simulation.
#             all   Run the simulation script, then also re-run the standalone
#                   comparison script explicitly (default).
#
# --geometry-only  Passed through to the simulation script(s): only save
#                   whole-grating geometry PDFs, skip running simulations.
#                   Ignored in --mode eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STUDY="both"
MODE="all"
GEOMETRY_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --study)
            STUDY="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --geometry-only)
            GEOMETRY_ONLY=true
            shift
            ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

case "$STUDY" in
    kind|correlation|both) ;;
    *)
        echo "Invalid --study: $STUDY (expected kind, correlation, or both)" >&2
        exit 1
        ;;
esac

case "$MODE" in
    sim|eval|all) ;;
    *)
        echo "Invalid --mode: $MODE (expected sim, eval, or all)" >&2
        exit 1
        ;;
esac

if [[ -x "$SCRIPT_DIR/../../../.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/../../../.venv/bin/python"
else
    PYTHON="python3"
fi

SIM_ARGS=()
if [[ "$GEOMETRY_ONLY" == true ]]; then
    SIM_ARGS+=(--geometry-only)
fi

run_kind_study() {
    if [[ "$MODE" == "sim" || "$MODE" == "all" ]]; then
        echo "=== Running roughness-kind comparison simulation ==="
        "$PYTHON" roughness_kind_comparison.py ${SIM_ARGS[@]+"${SIM_ARGS[@]}"}
    fi
    if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
        echo "=== Plotting roughness-kind comparison ==="
        "$PYTHON" comparison_roughness_kind_comparison.py
    fi
}

run_correlation_study() {
    if [[ "$MODE" == "sim" || "$MODE" == "all" ]]; then
        echo "=== Running roughness correlation-length simulation ==="
        "$PYTHON" roughness_correlation.py ${SIM_ARGS[@]+"${SIM_ARGS[@]}"}
    fi
    if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
        echo "=== Plotting roughness correlation-length comparison ==="
        "$PYTHON" comparison_roughness_correlation.py
    fi
}

if [[ "$STUDY" == "kind" || "$STUDY" == "both" ]]; then
    run_kind_study
fi

if [[ "$STUDY" == "correlation" || "$STUDY" == "both" ]]; then
    run_correlation_study
fi
