#!/bin/bash
# Run all grating profile generation scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/laminar_no_top_cap.py" "$@"
python "$SCRIPT_DIR/laminar_with_top_cap.py" "$@"
python "$SCRIPT_DIR/blazed_no_top_cap.py" "$@"
python "$SCRIPT_DIR/blazed_with_top_cap.py" "$@"
python "$SCRIPT_DIR/blazed_multilayer_custom_stack.py" "$@"
python "$SCRIPT_DIR/sinusoidal_custom_profile.py" "$@"
python "$SCRIPT_DIR/afm_preprocessing_blazed_profile.py" "$@"
python "$SCRIPT_DIR/afm_preprocessing_laminar_profile.py" "$@"
