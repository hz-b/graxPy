#!/bin/bash
#
# run_with_restart.sh
# Wrapper script that automatically restarts energy_sweep.py after crashes
#
# Usage: ./run_with_restart.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results/logs"
LOG_FILE="$LOG_DIR/wrapper_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "========================================" | tee "$LOG_FILE"
echo "Simulation Auto-Restart Wrapper" | tee "$LOG_FILE"
echo "Started: $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee "$LOG_FILE"
echo "========================================" | tee "$LOG_FILE"

run_count=0

while true; do
    run_count=$((run_count + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Run #$run_count - $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Run the simulation
    cd "$SCRIPT_DIR"
    python energy_sweep.py 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    
    # Check result
    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "SUCCESS! Simulation completed." | tee -a "$LOG_FILE"
        echo "Finished: $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        exit 0
    fi
    
    # Log the failure
    echo "" | tee -a "$LOG_FILE"
    echo "!!! Run #$run_count failed with exit code $exit_code !!!" | tee -a "$LOG_FILE"
    
    # Identify crash type
    if [ $exit_code -eq 139 ]; then
        echo "!!! SEGFAULT detected (SIGSEGV) !!!" | tee -a "$LOG_FILE"
    elif [ $exit_code -eq 134 ]; then
        echo "!!! ABORT detected (SIGABRT) !!!" | tee -a "$LOG_FILE"
    elif [ $exit_code -eq 152 ]; then
        echo "!!! STOP detected (SIGUSR1) - possible memory issue !!!" | tee -a "$LOG_FILE"
    else
        echo "!!! Script error or exception !!!" | tee -a "$LOG_FILE"
    fi
    
    # Restart
    echo "Restarting in 5 seconds..." | tee -a "$LOG_FILE"
    sleep 5
done
