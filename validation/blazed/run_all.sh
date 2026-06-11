#!/bin/bash
set -e

echo "=================================================="
echo "Running parameter influence study scripts"
echo "=================================================="

cd "$(dirname "$0")"

echo ""
echo "1. Running parameter_influence_study.py"
echo "--------------------------------------------------"
python parameter_influence_study.py

echo ""
echo "2. Running parameter_influence_study_stable.py"
echo "--------------------------------------------------"
python parameter_influence_study_stable.py

echo ""
echo "=================================================="
echo "All scripts completed!"
echo "=================================================="
