#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
results_dir="${script_dir}/results"
log_dir="${results_dir}/logs"
mkdir -p "${log_dir}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_python_script() {
  local script_name="$1"
  local log_name="$2"

  echo "Running ${script_name}"
  python3 "${script_dir}/${script_name}" 2>&1 | tee "${log_dir}/${log_name}"
}

run_python_script "laminar_2000lmm_fixed_angle_alpha1deg.py" "laminar_2000lmm_fixed_angle_alpha1deg.log"
run_python_script "laminar_2000lmm_fixed_angle_alpha2deg.py" "laminar_2000lmm_fixed_angle_alpha2deg.log"
run_python_script "laminar_2000lmm_fixed_angle_alpha4deg.py" "laminar_2000lmm_fixed_angle_alpha4deg.log"
run_python_script "laminar_2000lmm_fixed_angle_alpha1deg_layered.py" "laminar_2000lmm_fixed_angle_alpha1deg_layered.log"
run_python_script "laminar_2000lmm_fixed_angle_alpha2deg_layered.py" "laminar_2000lmm_fixed_angle_alpha2deg_layered.log"
run_python_script "laminar_2000lmm_fixed_angle_alpha4deg_layered.py" "laminar_2000lmm_fixed_angle_alpha4deg_layered.log"
run_python_script "comparison_laminar_2000lmm_fixed_angle.py" "comparison_laminar_2000lmm_fixed_angle.log"

current_branch="$(git -C "${repo_root}" branch --show-current)"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

git -C "${repo_root}" add "${results_dir}"

if git -C "${repo_root}" diff --cached --quiet -- "${results_dir}"; then
  echo "No result changes to commit."
else
  git -C "${repo_root}" commit -m "Add laminar 2000 l/mm fixed-angle validation results (${timestamp})"
  git -C "${repo_root}" push origin "${current_branch}"
fi

echo "Fixed-angle validation batch complete."
