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

# Check if required arguments were provided
if [ -z "$TASK" ] || [ -z "$COMMON_FOLDER" ]; then
    echo "Error: Task name and common folder name must be provided"
    echo "Usage: $0 <task_name> <common_folder_name> <max_iterations> <seed1> <seed2> ..."
    echo "Example: $0 Isc-Vel-BALLU-encoder my_experiment 1000 42 43 44"
    exit 1
fi

# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rew_std_0.5_symWt_0"
COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_${COMMON_FOLDER}"


echo "Starting training for task: $TASK"
echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
for fl_ratio in "$@"
do
        echo "Training with fl_ratio: $fl_ratio"
        python scripts/rsl_rl/train.py --task "$TASK" \
                                       --num_envs 4096 \
                                       --run_name "1-obstacle_fl_ratio_seed44_$fl_ratio" \
                                       --fl_ratio "$fl_ratio" \
                                       --headless \
                                       --seed 44 \
                                       --max_iterations "$MAX_ITERATIONS"
                                       # --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \

        # echo "Testing with seed: $SEED on final model"
        # python scripts/rsl_rl/play.py --task "$TASK" \
        #                              --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
        #                              --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
        #                              --headless \
        #                              --video \
        #                              --num_envs 1 \
        #                              --video_length 399 \
        #                              --device cuda:0
        
        # echo "Testing with seed: $SEED on best model"
        # python scripts/rsl_rl/play.py --task "$TASK" \
        #                              --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
        #                              --checkpoint "model_best.pt" \
        #                              --headless \
        #                              --video \
        #                              --num_envs 1 \
        #                              --video_length 399 \
        #                              --device cuda:0
                                     
        # Check if play script succeeded
        # if [ $? -ne 0 ]; then
        #     echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
        #     exit 1
        # fi
        
done

# python scripts/rsl_rl/train.py --task Isc-Vel-BALLU-1-obstacle --num_envs 4096 --run_name "1_obstacle_crclm_frwdVel3.0_jump1.0_eplen500_Hzn40_rewChgEnd600_seed0" --seed 0 --headless --max_iterations 5000 --device cuda:0
# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rewStd_0.45_symWt_0.1_velC"


# echo "Starting training for task: $TASK"
# echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
# do
#         echo "Training with seed: $SEED"
#         python scripts/rsl_rl/train.py --task "$TASK" \
#                                        --num_envs 4096 \
#                                        --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
#                                        --seed "$SEED" \
#                                        --headless \
#                                        --max_iterations "$MAX_ITERATIONS" \
#                                        "agent.algorithm.mirror_symmetry_cfg.weight=0.1" \
#                                        "env.rewards.track_lin_vel_xy_base_exp.params.std=0.45"

#         echo "Testing with seed: $SEED"
#         python scripts/rsl_rl/play.py --task "$TASK" \
#                                      --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
#                                      --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
#                                      --headless \
#                                      --video \
#                                      --num_envs 1 \
#                                      --video_length 399
        
#         # Check if play script succeeded
#         if [ $? -ne 0 ]; then
#             echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
#             exit 1
#         fi
        
# done


# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rewStd_0.50_symWt_0.1_velC"
# echo "Starting training for task: $TASK"
# echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
# do
#         echo "Training with seed: $SEED"
#         python scripts/rsl_rl/train.py --task "$TASK" \
#                                        --num_envs 4096 \
#                                        --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
#                                        --seed "$SEED" \
#                                        --headless \
#                                        --max_iterations "$MAX_ITERATIONS" \
#                                        "agent.algorithm.mirror_symmetry_cfg.weight=0.1" \
#                                        "env.rewards.track_lin_vel_xy_base_exp.params.std=0.5"

#         echo "Testing with seed: $SEED"
#         python scripts/rsl_rl/play.py --task "$TASK" \
#                                      --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
#                                      --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
#                                      --headless \
#                                      --video \
#                                      --num_envs 1 \
#                                      --video_length 399
        
#         # Check if play script succeeded
#         if [ $? -ne 0 ]; then
#             echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
#             exit 1
#         fi
        
# done


# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rewStd_0.55_symWt_0.1_velC"
# echo "Starting training for task: $TASK"
# echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
# do
#         echo "Training with seed: $SEED"
#         python scripts/rsl_rl/train.py --task "$TASK" \
#                                        --num_envs 4096 \
#                                        --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
#                                        --seed "$SEED" \
#                                        --headless \
#                                        --max_iterations "$MAX_ITERATIONS" \
#                                        "agent.algorithm.mirror_symmetry_cfg.weight=0.1" \
#                                        "env.rewards.track_lin_vel_xy_base_exp.params.std=0.55"

#         echo "Testing with seed: $SEED"
#         python scripts/rsl_rl/play.py --task "$TASK" \
#                                      --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
#                                      --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
#                                      --headless \
#                                      --video \
#                                      --num_envs 1 \
#                                      --video_length 399
        
#         # Check if play script succeeded
#         if [ $? -ne 0 ]; then
#             echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
#             exit 1
#         fi
        
# done


# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rewStd_0.60_symWt_0.1_velC"
# echo "Starting training for task: $TASK"
# echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
# do
#         echo "Training with seed: $SEED"
#         python scripts/rsl_rl/train.py --task "$TASK" \
#                                        --num_envs 4096 \
#                                        --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
#                                        --seed "$SEED" \
#                                        --headless \
#                                        --max_iterations "$MAX_ITERATIONS" \
#                                        "agent.algorithm.mirror_symmetry_cfg.weight=0.1" \
#                                        "env.rewards.track_lin_vel_xy_base_exp.params.std=0.6"

#         echo "Testing with seed: $SEED"
#         python scripts/rsl_rl/play.py --task "$TASK" \
#                                      --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
#                                      --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
#                                      --headless \
#                                      --video \
#                                      --num_envs 1 \
#                                      --video_length 399
        
#         # Check if play script succeeded
#         if [ $? -ne 0 ]; then
#             echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
#             exit 1
#         fi
        
# done


# COMMON_FOLDER_WITH_TIMESTAMP="${TIMESTAMP}_rewStd_0.65_symWt_0.1_velC"
# echo "Starting training for task: $TASK"
# echo "Using common folder with timestamp: $COMMON_FOLDER_WITH_TIMESTAMP"

# for SEED in "$@"
# do
#         echo "Training with seed: $SEED"
#         python scripts/rsl_rl/train.py --task "$TASK" \
#                                        --num_envs 4096 \
#                                        --common_folder "$COMMON_FOLDER_WITH_TIMESTAMP" \
#                                        --seed "$SEED" \
#                                        --headless \
#                                        --max_iterations "$MAX_ITERATIONS" \
#                                        "agent.algorithm.mirror_symmetry_cfg.weight=0.1" \
#                                        "env.rewards.track_lin_vel_xy_base_exp.params.std=0.65"

#         echo "Testing with seed: $SEED"
#         python scripts/rsl_rl/play.py --task "$TASK" \
#                                      --load_run "$COMMON_FOLDER_WITH_TIMESTAMP/seed_$SEED" \
#                                      --checkpoint "model_$((MAX_ITERATIONS - 1)).pt" \
#                                      --headless \
#                                      --video \
#                                      --num_envs 1 \
#                                      --video_length 399
        
#         # Check if play script succeeded
#         if [ $? -ne 0 ]; then
#             echo "Play script failed for the experiment with seed $SEED. Terminating all remaining experiments."
#             exit 1
#         fi
        
# done


# echo "All training and testing runs completed successfully"