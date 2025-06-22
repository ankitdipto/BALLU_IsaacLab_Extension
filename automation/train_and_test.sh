#!/bin/bash

# Exit on any error
set -e

# First argument is the task name, second argument is the common folder name
TASK="$1"
COMMON_FOLDER="$2"
MAX_ITERATIONS="$3"
shift 3  # Remove the first three arguments, leaving only the seeds

# Get current timestamp
TIMESTAMP=$(date +"%m-%d_%H-%M-%S")
# Add timestamp to common folder
COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_${COMMON_FOLDER}"

# Check if required arguments were provided
if [ -z "$TASK" ] || [ -z "$COMMON_FOLDER" ]; then
    echo "Error: Task name and common folder name must be provided"
    echo "Usage: $0 <task_name> <common_folder_name> <max_iterations> <seed1> <seed2> ..."
    echo "Example: $0 Isaac-Vel-BALLU-priv my_experiment 4000 42 43 44"
    exit 1
fi

echo "Starting training for task: $TASK"
echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

for SEED in "$@"
do
        # Experiment 1
        echo "Training with seed: $SEED"
        python scripts/rsl_rl/train.py --task "$TASK" \
                                       --num_envs 4096 \
                                       --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
                                       --seed "$SEED" \
                                       --headless \
                                       --max_iterations "$MAX_ITERATIONS"

        echo "Testing with seed: $SEED"
        python scripts/rsl_rl/play.py --task "$TASK" \
                                     --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
                                     --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
                                     --headless \
                                     --video \
                                     --num_envs 1 \
                                     --video_length 399
        
        # Check if play script succeeded
        if [ $? -ne 0 ]; then
            echo "Play script failed for experiment 1 with seed $SEED. Terminating all remaining experiments."
            exit 1
        fi
        
done

echo "All training and testing runs completed successfully"