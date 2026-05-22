#!/usr/bin/env bash

set -euo pipefail

LAUNCH_HTML=false
SKIP_IMAGE_SYNC=false
BUILD_HTML=false
BUILD_LATEX=false
BUILD_PDF=false
EXPLICIT_BUILD_SELECTION=false

for arg in "$@"; do
    case "$arg" in
        --html)
            BUILD_HTML=true
            EXPLICIT_BUILD_SELECTION=true
            ;;
        --pdf)
            BUILD_LATEX=true
            BUILD_PDF=true
            EXPLICIT_BUILD_SELECTION=true
            ;;
        --open)
            LAUNCH_HTML=true
            ;;
        --skip-image-sync|--skip-example-sync)
            SKIP_IMAGE_SYNC=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--html] [--pdf] [--open] [--skip-image-sync]"
            exit 1
            ;;
    esac
done

if [[ "${EXPLICIT_BUILD_SELECTION}" == false ]]; then
    if [[ "${LAUNCH_HTML}" == true ]]; then
        BUILD_HTML=true
    else
        BUILD_HTML=true
        BUILD_LATEX=true
        BUILD_PDF=true
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCS_DIR="${PROJECT_ROOT}/docs"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Python interpreter not found (tried '${PYTHON_BIN}' and 'python3')."
        exit 1
    fi
fi

HTML_BUILD_DIR="${DOCS_DIR}/_build/html"
LATEX_BUILD_DIR="${DOCS_DIR}/_build/latex"

if [[ "${SKIP_IMAGE_SYNC}" == false ]]; then
    echo "============================================================"
    echo "Syncing tutorial images from examples/"
    echo "============================================================"
    GRATING_IMAGE_DIR="${DOCS_DIR}/tutorials/images/gratings"
    SIM_IMAGE_DIR="${DOCS_DIR}/tutorials/images/simulation"
    OPTIMIZER_IMAGE_DIR="${DOCS_DIR}/tutorials/images/optimizer/laminar_fit"
    OPTIMIZER_BLAZED_IMAGE_DIR="${DOCS_DIR}/tutorials/images/optimizer/blazed_fit"
    NUMBA_SPEED_IMAGE_DIR="${DOCS_DIR}/tutorials/images/numba_speed"
    COMPARISON_IMAGE_DIR="${DOCS_DIR}/comparison-to-other-codes/images"
    HOWTO_IMAGE_DIR="${DOCS_DIR}/how-to/images"
    mkdir -p "${GRATING_IMAGE_DIR}" "${SIM_IMAGE_DIR}" "${OPTIMIZER_IMAGE_DIR}" "${OPTIMIZER_BLAZED_IMAGE_DIR}" "${NUMBA_SPEED_IMAGE_DIR}" "${COMPARISON_IMAGE_DIR}" "${HOWTO_IMAGE_DIR}"

    cp "${PROJECT_ROOT}/examples/grating/results/laminar_no_top_cap.png" \
      "${GRATING_IMAGE_DIR}/laminar_no_top_cap.png"
    cp "${PROJECT_ROOT}/examples/grating/results/laminar_with_top_cap.png" \
      "${GRATING_IMAGE_DIR}/laminar_with_top_cap.png"
    cp "${PROJECT_ROOT}/examples/grating/results/blazed_no_top_cap.png" \
      "${GRATING_IMAGE_DIR}/blazed_no_top_cap.png"
    cp "${PROJECT_ROOT}/examples/grating/results/blazed_with_top_cap.png" \
      "${GRATING_IMAGE_DIR}/blazed_with_top_cap.png"
    cp "${PROJECT_ROOT}/examples/grating/results/blazed_multilayer_custom_stack.png" \
      "${GRATING_IMAGE_DIR}/blazed_multilayer_custom_stack.png"
    cp "${PROJECT_ROOT}/examples/grating/results/blazed_multilayer_custom_stack_schematic.png" \
      "${GRATING_IMAGE_DIR}/blazed_multilayer_custom_stack_schematic.png"
    cp "${PROJECT_ROOT}/examples/grating/results/sinusoidal_custom_profile.png" \
      "${HOWTO_IMAGE_DIR}/sinusoidal_custom_profile.png"

    cp "${PROJECT_ROOT}/examples/simulation/batch_user_cases/results/batch_user_cases_orders_1_3_vs_depth.png" \
      "${SIM_IMAGE_DIR}/batch_user_cases_orders_1_3_vs_depth.png"
    cp "${PROJECT_ROOT}/examples/simulation/fixed_angle_sweep/results/fixed_angle_orders_1_3.png" \
      "${SIM_IMAGE_DIR}/fixed_angle_orders_1_3.png"
    cp "${PROJECT_ROOT}/examples/simulation/monochromator_sweep/results/monochromator_orders_1_3.png" \
      "${SIM_IMAGE_DIR}/monochromator_orders_1_3.png"
    cp "${PROJECT_ROOT}/examples/simulation/energy_angle_sweep/results/energy_angle_multilayer_fast.png" \
      "${SIM_IMAGE_DIR}/energy_angle_multilayer_fast.png"
    cp "${PROJECT_ROOT}/examples/simulation/multilayer_theta_search/results/multilayer_theta_search_workflow.png" \
      "${SIM_IMAGE_DIR}/multilayer_theta_search_workflow.png"
    cp "${PROJECT_ROOT}/examples/simulation/parameter_study/results/parameter_study_grid.png" \
      "${SIM_IMAGE_DIR}/parameter_study_grid.png"
    cp "${PROJECT_ROOT}/examples/optimizer/optimizer_laminar/results/laminar_fit/best_fit.png" \
      "${OPTIMIZER_IMAGE_DIR}/best_fit.png"
    cp "${PROJECT_ROOT}/examples/optimizer/optimizer_laminar/results/laminar_fit/optimization_loss_history.png" \
      "${OPTIMIZER_IMAGE_DIR}/optimization_loss_history.png"
    cp "${PROJECT_ROOT}/examples/optimizer/optimizer_laminar/results/laminar_fit/laminar_fit_measurement_comparison.png" \
      "${OPTIMIZER_IMAGE_DIR}/laminar_fit_measurement_comparison.png"
    if [[ -f "${PROJECT_ROOT}/examples/optimizer/optimizer_blazed/results/blazed_fit/best_fit.png" ]]; then
      cp "${PROJECT_ROOT}/examples/optimizer/optimizer_blazed/results/blazed_fit/best_fit.png" \
        "${OPTIMIZER_BLAZED_IMAGE_DIR}/best_fit.png"
      cp "${PROJECT_ROOT}/examples/optimizer/optimizer_blazed/results/blazed_fit/optimization_loss_history.png" \
        "${OPTIMIZER_BLAZED_IMAGE_DIR}/optimization_loss_history.png"
      cp "${PROJECT_ROOT}/examples/optimizer/optimizer_blazed/results/blazed_fit/blazed_fit_measurement_comparison.png" \
        "${OPTIMIZER_BLAZED_IMAGE_DIR}/blazed_fit_measurement_comparison.png"
    else
      echo "Warning: blazed optimizer images not found; run examples/optimizer/optimizer_blazed workflow first."
    fi
    cp "${PROJECT_ROOT}/tools/numba_speed/results/multi_energy_numba_vs_legacy_plots.png" \
      "${NUMBA_SPEED_IMAGE_DIR}/multi_energy_numba_vs_legacy_plots.png"
    cp "${PROJECT_ROOT}/tools/numba_speed/results/multi_energy_multilayer_numba_vs_legacy_plots.png" \
      "${NUMBA_SPEED_IMAGE_DIR}/multi_energy_multilayer_numba_vs_legacy_plots.png"

    cp "${PROJECT_ROOT}/examples/simulation/blazed_multilayer_sweep/results/blazed_multilayer_all_orders.csv" \
      "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/simulation/grax_multilayer_theta_search_all_orders.csv"
    (
      cd "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer"
      "${PYTHON_BIN}" comparison_blazed_multilayer_sweep.py
      "${PYTHON_BIN}" comparison_blazed_multilayer_grating_angle.py
    )

    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed/comparison_blazed_monochromator_sweep.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_blazed_monochromator_sweep.png"

    # Regenerate blazed-multilayer comparison figures from current example results.
    # The scripts read theta-search CSVs directly from examples/ and only emit PNGs.
    "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/comparison_blazed_multilayer_sweep.py" >/dev/null
    "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/comparison_blazed_multilayer_grating_angle.py" >/dev/null

    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/results/multilayer_stack_schematic.png" \
      "${COMPARISON_IMAGE_DIR}/multilayer_stack_schematic.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/results/blazed_multilayer_profile.png" \
      "${COMPARISON_IMAGE_DIR}/blazed_multilayer_profile.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/comparison_blazed_multilayer_sweep.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_blazed_multilayer_sweep.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/comparison_blazed_multilayer_sweep_550_600eV.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_blazed_multilayer_sweep_550_600eV.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/blazed_multilayer/comparison_blazed_multilayer_grating_angle.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_blazed_multilayer_grating_angle.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/laminar/comparison_laminar_fixed_angle.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_laminar_fixed_angle.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/laminar_150lmm/results/laminar_150lmm_monochromator_profile.png" \
      "${COMPARISON_IMAGE_DIR}/laminar_150lmm_monochromator_profile.png"
    cp "${PROJECT_ROOT}/comparison_to_other_codes/laminar_150lmm/comparison_laminar_150lmm_monochromator.png" \
      "${COMPARISON_IMAGE_DIR}/comparison_laminar_150lmm_monochromator.png"
    echo
fi

rm -rf "${DOCS_DIR}/_build"

if [[ "${BUILD_HTML}" == true ]]; then
    echo "============================================================"
    echo "Building HTML documentation"
    echo "============================================================"
    "${PYTHON_BIN}" -m sphinx -b html "${DOCS_DIR}" "${HTML_BUILD_DIR}"
fi

if [[ "${BUILD_LATEX}" == true ]]; then
    if [[ "${BUILD_HTML}" == true ]]; then
        echo
    fi
    echo "============================================================"
    echo "Building LaTeX documentation"
    echo "============================================================"
    "${PYTHON_BIN}" -m sphinx -b latex "${DOCS_DIR}" "${LATEX_BUILD_DIR}"
fi

if [[ "${BUILD_PDF}" == true ]]; then
    if [[ "${BUILD_HTML}" == true || "${BUILD_LATEX}" == true ]]; then
        echo
    fi
    echo "============================================================"
    echo "Building PDF"
    echo "============================================================"
    make -C "${LATEX_BUILD_DIR}"
fi

PDF_FILE=""
if [[ "${BUILD_PDF}" == true && -d "${LATEX_BUILD_DIR}" ]]; then
    PDF_FILE="$(find "${LATEX_BUILD_DIR}" -maxdepth 1 -name '*.pdf' | head -n 1)"
fi

echo
echo "============================================================"
echo "Build completed"
echo "============================================================"
if [[ "${BUILD_HTML}" == true ]]; then
    echo "HTML:"
    echo "  ${HTML_BUILD_DIR}/index.html"
fi

if [[ "${BUILD_PDF}" == true && -n "${PDF_FILE}" ]]; then
    echo
    echo "PDF:"
    echo "  ${PDF_FILE}"
fi

if [[ "${LAUNCH_HTML}" == true ]]; then
    echo
    echo "Opening HTML documentation..."
    HTML_INDEX="${HTML_BUILD_DIR}/index.html"
    DOCS_BROWSER="${DOCS_BROWSER:-google-chrome}"
    if command -v "${DOCS_BROWSER}" >/dev/null 2>&1; then
        if "${DOCS_BROWSER}" "file://${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif "${DOCS_BROWSER}" "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v xdg-open >/dev/null 2>&1 && xdg-open "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v gio >/dev/null 2>&1 && gio open "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v python3 >/dev/null 2>&1 && python3 -m webbrowser "file://${HTML_INDEX}" >/dev/null 2>&1; then
            :
        else
            echo "Could not auto-open browser."
            echo "Open manually: ${HTML_INDEX}"
        fi
    elif command -v xdg-open >/dev/null 2>&1; then
        if xdg-open "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v gio >/dev/null 2>&1 && gio open "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v python3 >/dev/null 2>&1 && python3 -m webbrowser "file://${HTML_INDEX}" >/dev/null 2>&1; then
            :
        else
            echo "Could not auto-open browser."
            echo "Open manually: ${HTML_INDEX}"
        fi
    elif command -v gio >/dev/null 2>&1; then
        if gio open "${HTML_INDEX}" >/dev/null 2>&1; then
            :
        elif command -v python3 >/dev/null 2>&1 && python3 -m webbrowser "file://${HTML_INDEX}" >/dev/null 2>&1; then
            :
        else
            echo "Could not auto-open browser."
            echo "Open manually: ${HTML_INDEX}"
        fi
    elif command -v python3 >/dev/null 2>&1; then
        if ! python3 -m webbrowser "file://${HTML_INDEX}" >/dev/null 2>&1; then
            echo "Could not auto-open browser."
            echo "Open manually: ${HTML_INDEX}"
        fi
    else
        echo "Could not auto-open browser."
        echo "Open manually: ${HTML_INDEX}"
    fi
fi
