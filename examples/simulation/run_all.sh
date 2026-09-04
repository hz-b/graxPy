#!/usr/bin/env bash
# Run every standalone simulation example and its dedicated comparison plots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_example() {
    local relative_path="$1"
    echo "==> Running ${relative_path}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/${relative_path}"
}

# Core one-dimensional simulations and their polarization comparisons.
run_example "single_simulation/single_simulation.py"
run_example "single_simulation/polarization_comparison.py"
run_example "fixed_angle_sweep/fixed_angle_sweep.py"
run_example "fixed_angle_sweep/polarization_comparison.py"
run_example "monochromator_sweep/monochromator_sweep.py"
run_example "monochromator_sweep/polarization_comparison.py"
run_example "energy_angle_sweep/energy_angle_sweep.py"
run_example "energy_angle_sweep/polarization_comparison.py"
run_example "batch_user_cases/batch_user_cases.py"
run_example "batch_user_cases/polarization_comparison.py"

# Multilayer, convergence, and solver-comparison studies.
run_example "blazed_multilayer_sweep/blazed_multilayer_sweep.py"
run_example "blazed_multilayer_sweep/polarization_comparison.py"
run_example "blazed_multilayer_memory_comparison/blazed_multilayer_memory_comparison.py"
run_example "multilayer_theta_search/multilayer_theta_search.py"
run_example "parameter_study/parameter_study.py"
run_example "neviere_solver/neviere_solver.py"
run_example "neviere_grazing_stability/neviere_grazing_stability.py"
run_example "deep_grating_limits/deep_grating_limits.py"
run_example "continuous_vs_staircase/continuous_vs_staircase.py"
run_example "solver_runtime/solver_runtime.py"

# This wrapper runs both roughness studies and re-generates their comparison
# figures after their simulation outputs are available.
echo "==> Running fixed_angle_roughness/run_roughness_examples.sh"
PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/fixed_angle_roughness/run_roughness_examples.sh"

echo "==> All simulation examples completed"
