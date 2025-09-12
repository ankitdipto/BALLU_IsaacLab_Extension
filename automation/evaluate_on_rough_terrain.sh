#!/bin/bash

# Script to run play_rough.py multiple times with user-provided parameters
# Accepts: load_run, checkpoint, and range of trn_height values

# Default values
DEFAULT_LOAD_RUN="2025-09-04_17-36-05_trnCrclm_envs4096_pltWd2.0_trnSz7.0_ltPen2.0_negRewExp-1"
DEFAULT_CHECKPOINT="model_600.pt"
DEFAULT_TRN_HEIGHT_START=0.025
DEFAULT_TRN_HEIGHT_END=0.035
DEFAULT_TRN_HEIGHT_STEP=0.001

echo "=== BALLU Rough Terrain Evaluation Script ==="
echo

# Read user input with defaults
read -p "Enter load_run [default: $DEFAULT_LOAD_RUN]: " load_run
load_run=${load_run:-$DEFAULT_LOAD_RUN}

read -p "Enter checkpoint [default: $DEFAULT_CHECKPOINT]: " checkpoint
checkpoint=${checkpoint:-$DEFAULT_CHECKPOINT}

read -p "Enter start trn_height [default: $DEFAULT_TRN_HEIGHT_START]: " trn_height_start
trn_height_start=${trn_height_start:-$DEFAULT_TRN_HEIGHT_START}

read -p "Enter end trn_height [default: $DEFAULT_TRN_HEIGHT_END]: " trn_height_end
trn_height_end=${trn_height_end:-$DEFAULT_TRN_HEIGHT_END}

read -p "Enter trn_height step [default: $DEFAULT_TRN_HEIGHT_STEP]: " trn_height_step
trn_height_step=${trn_height_step:-$DEFAULT_TRN_HEIGHT_STEP}

# Validate inputs using basic arithmetic since bc might not be available
if [[ $(echo "$trn_height_start > $trn_height_end" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo "Error: start height must be less than or equal to end height"
    exit 1
fi

if [[ $(echo "$trn_height_step <= 0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo "Error: step must be greater than 0"
    exit 1
fi

echo
echo "Configuration:"
echo "  load_run: $load_run"
echo "  checkpoint: $checkpoint"
echo "  trn_height range: $trn_height_start to $trn_height_end (step: $trn_height_step)"
echo

# Generate height values using seq
height_values=$(seq -f "%.3f" $trn_height_start $trn_height_step $trn_height_end)
total_runs=$(echo "$height_values" | wc -w)

echo "Total runs: $total_runs"
echo "Height values: $height_values"
echo

# Counter for progress
current_run=0

# Loop through each height value
for trn_height in $height_values; do
    current_run=$((current_run + 1))
    echo "[$current_run/$total_runs] Running with trn_height = $trn_height"

    python scripts/rsl_rl/play_rough.py --task Isc-Vel-BALLU-rough-play \
                                        --load_run "$load_run" \
                                        --checkpoint "$checkpoint" \
                                        --headless \
                                        --video \
                                        --num_envs 1 \
                                        --video_length 399 \
                                        --trn_height "$trn_height"

    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $?"
    fi

    echo

    # Add a small delay between runs to avoid overwhelming the system
    sleep 2
done

echo "All runs completed!"