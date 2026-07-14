#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
results_dir="${script_dir}/results"
log_dir="${results_dir}/logs"
mkdir -p "${log_dir}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

show_usage() {
  cat <<'EOF'
Usage: run_fixed_angle_validations.sh [mode] [--no-git]

Modes:
  baseline          Run only the baseline fixed-angle simulations.
  layered           Run only the nominal layered fixed-angle simulations.
  fitted            Run the three best-fit fixed-angle simulations.
  fitted-edge-excluded
                    Run the three edge-excluded best-fit fixed-angle simulations.
  fitted-all        Run both fitted simulation families.
  family-full-range Run the full-range optimizations and the matching fitted simulations.
  family-edge-excluded
                    Run the edge-excluded optimizations and the matching fitted simulations.
  family-all        Run both optimization families together with their matching fitted simulations.
  validations       Run baseline and layered fixed-angle simulations, then comparison.
  optimizations     Run the three top-layer optimization scripts.
  optimizations-edge-excluded
                    Run the three edge-excluded top-layer optimization scripts.
  optimizations-all Run both optimization families.
  parameter-study   Run the layered parameter study script.
  comparison        Run only the comparison plot script.
  all               Run baseline, layered, both fitted families, both optimization families,
                    parameter study, and comparison.

Defaults:
  If no mode is provided, the script runs: validations

Options:
  --no-git          Skip git add/commit/push for the results directory.
  -h, --help        Show this help message.
EOF
}

run_python_script() {
  local script_name="$1"
  local log_name="$2"

  echo "Running ${script_name}"
  "${python_bin}" "${script_dir}/${script_name}" 2>&1 | tee "${log_dir}/${log_name}"
}

run_comparison() {
  run_python_script "comparison_laminar_2000lmm_fixed_angle.py" "comparison_laminar_2000lmm_fixed_angle.log"
}

run_validations() {
  run_baseline
  run_layered
  run_comparison
}

run_baseline() {
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg.py" "laminar_2000lmm_fixed_angle_alpha1deg.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg.py" "laminar_2000lmm_fixed_angle_alpha2deg.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg.py" "laminar_2000lmm_fixed_angle_alpha4deg.log"
}

run_layered() {
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg_layered.py" "laminar_2000lmm_fixed_angle_alpha1deg_layered.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg_layered.py" "laminar_2000lmm_fixed_angle_alpha2deg_layered.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg_layered.py" "laminar_2000lmm_fixed_angle_alpha4deg_layered.log"
}

run_fitted() {
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg_fitted.py" "laminar_2000lmm_fixed_angle_alpha1deg_fitted.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg_fitted.py" "laminar_2000lmm_fixed_angle_alpha2deg_fitted.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg_fitted.py" "laminar_2000lmm_fixed_angle_alpha4deg_fitted.log"
}

run_fitted_edge_excluded() {
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg_edge_excluded_fitted.py" "laminar_2000lmm_fixed_angle_alpha1deg_edge_excluded_fitted.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg_edge_excluded_fitted.py" "laminar_2000lmm_fixed_angle_alpha2deg_edge_excluded_fitted.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg_edge_excluded_fitted.py" "laminar_2000lmm_fixed_angle_alpha4deg_edge_excluded_fitted.log"
}

run_fitted_all() {
  run_fitted
  run_fitted_edge_excluded
}

run_family_full_range() {
  run_optimizations
  run_fitted
}

run_family_edge_excluded() {
  run_optimizations_edge_excluded
  run_fitted_edge_excluded
}

run_family_all() {
  run_family_full_range
  run_family_edge_excluded
}

run_optimizations() {
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha1deg.py" "laminar_2000lmm_optimize_top_layers_alpha1deg.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha2deg.py" "laminar_2000lmm_optimize_top_layers_alpha2deg.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha4deg.py" "laminar_2000lmm_optimize_top_layers_alpha4deg.log"
}

run_optimizations_edge_excluded() {
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha1deg_edge_excluded.py" "laminar_2000lmm_optimize_top_layers_alpha1deg_edge_excluded.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha2deg_edge_excluded.py" "laminar_2000lmm_optimize_top_layers_alpha2deg_edge_excluded.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha4deg_edge_excluded.py" "laminar_2000lmm_optimize_top_layers_alpha4deg_edge_excluded.log"
}

run_optimizations_all() {
  run_optimizations
  run_optimizations_edge_excluded
}

run_parameter_study() {
  run_python_script "laminar_2000lmm_layered_parameter_study.py" "laminar_2000lmm_layered_parameter_study.log"
}

mode="validations"
do_git="yes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    baseline|layered|fitted|fitted-edge-excluded|fitted-all|family-full-range|family-edge-excluded|family-all|validations|optimizations|optimizations-edge-excluded|optimizations-all|parameter-study|comparison|all)
      mode="$1"
      shift
      ;;
    --no-git)
      do_git="no"
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_usage >&2
      exit 1
      ;;
  esac
done

case "${mode}" in
  baseline)
    run_baseline
    ;;
  layered)
    run_layered
    ;;
  fitted)
    run_fitted
    ;;
  fitted-edge-excluded)
    run_fitted_edge_excluded
    ;;
  fitted-all)
    run_fitted_all
    ;;
  family-full-range)
    run_family_full_range
    ;;
  family-edge-excluded)
    run_family_edge_excluded
    ;;
  family-all)
    run_family_all
    ;;
  validations)
    run_validations
    ;;
  optimizations)
    run_optimizations
    ;;
  optimizations-edge-excluded)
    run_optimizations_edge_excluded
    ;;
  optimizations-all)
    run_optimizations_all
    ;;
  parameter-study)
    run_parameter_study
    ;;
  comparison)
    run_comparison
    ;;
  all)
    run_baseline
    run_layered
    run_fitted_all
    run_optimizations_all
    run_parameter_study
    run_comparison
    ;;
esac

if [[ "${do_git}" == "no" ]]; then
  echo "Batch complete for mode: ${mode}. Skipping git add/commit/push."
  exit 0
fi

current_branch="$(git -C "${repo_root}" branch --show-current)"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

git -C "${repo_root}" add "${results_dir}"

if git -C "${repo_root}" diff --cached --quiet -- "${results_dir}"; then
  echo "No result changes to commit."
else
  git -C "${repo_root}" commit -m "Add laminar 2000 l/mm results (${timestamp})"
  git -C "${repo_root}" push origin "${current_branch}"
fi

echo "Batch complete for mode: ${mode}"
