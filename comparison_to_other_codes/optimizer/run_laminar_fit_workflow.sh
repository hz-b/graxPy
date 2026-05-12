#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_laminar_fit_workflow.sh [plain|roughness|roughness-only|both|all] [--quick-sim]

Modes:
  plain           Run fit + simulation without roughness, then update the plot.
  roughness       Run fit + simulation with geometry and roughness optimized together.
  roughness-only  Run plain fit, then fit/simulate only roughness on that geometry.
  both            Run plain and geometry+roughness workflows, then update the plot.
  all             Run plain, geometry+roughness, and roughness-only workflows.

Options:
  --quick-sim  Pass --quick only to simulation scripts. Fits still run normally.
  -h, --help   Show this help.
USAGE
}

mode="${1:-both}"
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$mode" != "plain" && "$mode" != "roughness" && "$mode" != "roughness-only" && "$mode" != "both" && "$mode" != "all" ]]; then
  echo "Unknown mode: $mode" >&2
  usage >&2
  exit 2
fi
shift || true

simulation_args=()
while (($#)); do
  case "$1" in
    --quick-sim)
      simulation_args+=(--quick)
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

run_plain() {
  echo "==> Fitting laminar grating without roughness"
  python fit_laminar_grating.py
  echo "==> Simulating laminar fitted parameters without roughness"
  python run_simulation_fit_laminar_grating.py "${simulation_args[@]}"
}

run_roughness() {
  echo "==> Fitting laminar grating with roughness"
  python fit_laminar_grating_with_roughness.py
  echo "==> Simulating laminar fitted parameters with roughness"
  python run_simulation_fit_laminar_grating_with_roughness.py "${simulation_args[@]}"
}

run_roughness_only() {
  echo "==> Fitting only roughness using the plain fitted geometry"
  python fit_laminar_roughness_only.py
  echo "==> Simulating plain fitted geometry with roughness-only fit"
  python run_simulation_fit_laminar_roughness_only.py "${simulation_args[@]}"
}

case "$mode" in
  plain)
    run_plain
    ;;
  roughness)
    run_roughness
    ;;
  roughness-only)
    run_plain
    run_roughness_only
    ;;
  both)
    run_plain
    run_roughness
    ;;
  all)
    run_plain
    run_roughness
    run_roughness_only
    ;;
esac

echo "==> Updating laminar fit comparison plot"
python plot_laminar_fit_comparison.py
echo "==> Done"
