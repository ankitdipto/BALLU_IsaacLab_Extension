#!/bin/bash

# First argument is the task name, second argument is the common folder name
TASK="$1"
COMMON_FOLDER="$2"
shift 2  # Remove the first two arguments, leaving only the seeds

# Get current timestamp
TIMESTAMP=$(date +"%m-%d_%H-%M-%S")
# Add timestamp to common folder
COMMON_FOLDER_WITH_TIMESTAMP="${COMMON_FOLDER}_${TIMESTAMP}"

# Check if required arguments were provided
if [ -z "$TASK" ] || [ -z "$COMMON_FOLDER" ]; then
    echo "Error: Task name and common folder name must be provided"
    echo "Usage: $0 <task_name> <common_folder_name> <seed1> <seed2> ..."
    echo "Example: $0 Isaac-Vel-BALLU-priv my_experiment 42 43 44"
    exit 1
fi

echo "Starting training for task: $TASK"
echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

for SEED in "$@"
do
echo "Training with seed: $SEED"
python scripts/rsl_rl/train.py --task "$TASK" \
                              --num_envs 16 \
                              --run_name "seed-$SEED" \
                              --max_iterations 2000 \
                              --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
                              env.rewards.track_lin_vel_xy_exp.params.std=0.2 \
                              --seed "$SEED" \
                              --headless
done

echo "All training runs completed in common folder: $COMMON_FOLDER_WITH_TIMESTAMP"