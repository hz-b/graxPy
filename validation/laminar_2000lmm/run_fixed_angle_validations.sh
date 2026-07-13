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
  validations       Run baseline and layered fixed-angle simulations, then comparison.
  optimizations     Run the three top-layer optimization scripts.
  parameter-study   Run the layered parameter study script.
  comparison        Run only the comparison plot script.
  all               Run validations, optimizations, and parameter study.

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
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg.py" "laminar_2000lmm_fixed_angle_alpha1deg.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg.py" "laminar_2000lmm_fixed_angle_alpha2deg.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg.py" "laminar_2000lmm_fixed_angle_alpha4deg.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha1deg_layered.py" "laminar_2000lmm_fixed_angle_alpha1deg_layered.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha2deg_layered.py" "laminar_2000lmm_fixed_angle_alpha2deg_layered.log"
  run_python_script "laminar_2000lmm_fixed_angle_alpha4deg_layered.py" "laminar_2000lmm_fixed_angle_alpha4deg_layered.log"
  run_comparison
}

run_optimizations() {
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha1deg.py" "laminar_2000lmm_optimize_top_layers_alpha1deg.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha2deg.py" "laminar_2000lmm_optimize_top_layers_alpha2deg.log"
  run_python_script "laminar_2000lmm_optimize_top_layers_alpha4deg.py" "laminar_2000lmm_optimize_top_layers_alpha4deg.log"
}

run_parameter_study() {
  run_python_script "laminar_2000lmm_layered_parameter_study.py" "laminar_2000lmm_layered_parameter_study.log"
}

mode="validations"
do_git="yes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    validations|optimizations|parameter-study|comparison|all)
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
  validations)
    run_validations
    ;;
  optimizations)
    run_optimizations
    ;;
  parameter-study)
    run_parameter_study
    ;;
  comparison)
    run_comparison
    ;;
  all)
    run_validations
    run_optimizations
    run_parameter_study
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
