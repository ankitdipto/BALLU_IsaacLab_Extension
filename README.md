# BALLU Robot Isaac Lab Extension for Morphology Optimization and Locomotion

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab (ankitdipto fork)](https://img.shields.io/badge/IsaacLab%20(fork)-ankitdipto%2FIsaacLab-blue)](https://github.com/ankitdipto/IsaacLab)
[![RSL_RL (ankitdipto fork)](https://img.shields.io/badge/RSL_RL%20(fork)-ankitdipto%2Frsl_rl-blue)](https://github.com/ankitdipto/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)

This repository contains an Isaac Lab extension for the BALLU (Buoyancy-Assisted Legged Locomotion Unit) robot. The goal of this research project is to optimize the morphology of BALLU to achieve stable and dynamic locomotion.

## Overview

This extension allows for:
- Simulating various morphological configurations of the BALLU robot.
- Implementing and testing different sensor suites (e.g., IMUs on various links).
- Training and evaluating locomotion policies using reinforcement learning (RSL-RL PPO implementation).

## Prerequisites

A conda environment (python 3.10) with the following packages installed:
- NVIDIA Isaac Sim (tested with version 4.5.0)
- NVIDIA IsaacLab (**MUST use the forked repo:** [https://github.com/ankitdipto/IsaacLab](https://github.com/ankitdipto/IsaacLab), **ballu_v0 branch**, tested with version 2.0.0)
    - **Do NOT use the original NVIDIA IsaacLab repository. This project requires features only present in the forked repository and the `ballu_v0` branch.**
- RSL_RL (**MUST use the forked repo:** [https://github.com/ankitdipto/rsl_rl](https://github.com/ankitdipto/rsl_rl), **develop branch**)
    - **Do NOT use the original RSL_RL repository. This project requires features only present in the forked repository and the `develop` branch.**

## Installation

1.  Clone and install the forked IsaacLab repository (required):
    ```bash
    git clone -b ballu_v0 https://github.com/ankitdipto/IsaacLab.git
    cd IsaacLab
    # Follow the installation instructions in the forked repo's README.md
    ```
2.  Clone and install the forked RSL_RL repository (required):
    ```bash
    git clone -b develop https://github.com/ankitdipto/rsl_rl.git
    cd rsl_rl
    pip install -e .
    ```
3.  Ensure your Conda environment containing all required packages (including the forked IsaacLab and RSL_RL) is activated.
4.  Clone this repository:
    ```bash
    git clone <repository_url> BALLU_IsaacLab_Extension
    cd BALLU_IsaacLab_Extension
    ```
5.  Install this extension package in editable mode:
    ```bash
    python -m pip install -e source/ballu_isaac_extension
    ```

## Code Structure Highlights

-   `source/ballu_isaac_extension/ballu_isaac_extension/`:
    -   `ballu_assets/`:
        -   `robots/`: Contains USD files for the BALLU robot (e.g., `original/original.usd`) and its configuration layers (`original_base.usd`, `original_physics.usd`, `original_sensor.usd`).
        -   `ballu_config.py`: Defines `ArticulationCfg` for different BALLU robot morphologies (e.g., `BALLU_REAL_CFG`, and commented-out variants for hip/knee actuation experiments).
    -   `tasks/ballu_locomotion/`:
        -   `*_env_cfg.py` (e.g., `tibia_imu_env_cfg.py`, `indirect_act_vel_env_cfg.py`): Python files defining various RL environment configurations, specifying robot assets, sensors, observation/action spaces, and rewards.
        -   `__init__.py`: Registers the RL environments with Gymnasium using IDs (e.g., `Isaac-Vel-BALLU-imu-tibia`).
        -   `mdp/`: Modules for common RL components like reward functions (`rewards.py`) and observation/action processing.
        -   `agents/`: RL agent configurations (e.g., `rsl_rl_ppo_cfg.py` for PPO settings).
-   `scripts/rsl_rl/`:
    -   `train.py`: Main script for training RL policies.
    -   `play.py`: Script for loading and visualizing trained policies.
-   `config/extension.toml`: Metadata for the Isaac Lab extension.

## Robot Configurations & Morphologies

The primary robot configuration is defined in `source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/ballu_config.py` via `ArticulationCfg` instances.
-   `BALLU_REAL_CFG`: The latest configuration that drives the knee joints indirectly through motorarms, simulating the four-bar linkage in the real robot.
-   Commented-out configurations (e.g., `BALLU_HIP_CFG`, `BALLU_HIP_KNEE_CFG`) provide the simpler configurations with direct actuation of *only* hip joint, *only* knee joint, *both* hip and knee joints, and their combinations with the base fixed.

To use a different morphology:
1.  Define or uncomment the desired `ArticulationCfg` in `ballu_config.py`.
2.  Ensure the `SceneCfg` within the target `_env_cfg.py` file (e.g., `tibia_imu_env_cfg.py`) references this `ArticulationCfg` for its `robot` attribute.

## Reinforcement Learning Environments

RL environments are defined by configuration classes (e.g., `BALLU_TibiaIMU_EnvCfg`) in the `source/ballu_isaac_extension/ballu_isaac_extension/tasks/ballu_locomotion/` directory.

Key aspects of an environment configuration (`*_env_cfg.py`):
-   **Scene (`SceneCfg`)**: Defines the robot, ground, sensors (e.g., `ImuCfg`), and lighting.
-   **Commands (`CommandsCfg`)**: Specifies the task commands (e.g., target velocity).
-   **Actions (`ActionsCfg`)**: Defines the RL agent's action space (e.g., target joint positions for `MOTOR_LEFT`, `MOTOR_RIGHT`).
-   **Observations (`ObservationsCfg`)**: Defines the agent's observation space, including sensor data (e.g., IMU readings) and task-relevant information.
-   **Rewards (`RewardsCfg`)**: Specifies the reward functions used to guide learning (references functions in `mdp/rewards.py`).

These environments are registered with Gymnasium in `tasks/ballu_locomotion/__init__.py`, making them accessible via a unique ID.

Example Task IDs (check `tasks/ballu_locomotion/__init__.py` for a full list and their corresponding configurations):
-   `Isaac-Vel-BALLU-real-priv`: Indirect actuation velocity control task.
-   `Isaac-Vel-BALLU-imu-tibia`: Velocity control task using IMU sensors on the tibias.
-   `Isaac-Vel-BALLU-imu-base`: Velocity control task using an IMU sensor on the robot's base.

## Running Experiments

### Training a Policy

To train an RL policy, use the `train.py` script with the desired task ID:

```bash
python scripts/rsl_rl/train.py --task <TASK_ID> --num_envs <NUM_ENVS> --seed <SEED> --max_iterations <MAX_ITERATIONS> 
```
For example, to train with tibia IMU sensors:
```bash
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-imu-tibia --num_envs 16 --seed 0 --max_iterations 2000
```
Training logs and model checkpoints will be saved in the `logs/` directories. The users are requested to take a look at the `automation/automate.sh` file for the different training experiments that can be run.

### Playing/Visualizing a Trained Policy

To load a trained policy and visualize its performance:

```bash
python scripts/rsl_rl/play.py --task <TASK_ID> --load_run <RUN_NAME> --checkpoint <MODEL_NAME> --num_envs <NUM_ENVS> --video 
```


## Customization

-   **New Robot Morphologies**: Modify/create USD files in `ballu_assets/robots/` and define a corresponding `ArticulationCfg` in `ballu_config.py`.
-   **Adding Sensors**: Add sensor configurations (e.g., `ImuCfg`, `CameraCfg`) to the `SceneCfg` in an `_env_cfg.py` file. Update `ObservationsCfg` to include data from the new sensor, potentially adding new processing functions in `mdp/`.
-   **New RL Tasks/Rewards**: Create a new `your_task_env_cfg.py`, define its components, and register it with a unique ID. Implement new reward functions in `mdp/rewards.py` and use them in your task's `RewardsCfg`.

## Troubleshooting

-   Ensure your Isaac Sim and Isaac Lab installations are correct and the Conda environment is properly sourced.
-   If Pylance or other language servers have issues with imports, you might need to configure `python.analysis.extraPaths` in your VSCode settings to point to Isaac Sim/Lab extension directories and this project's source folder.
