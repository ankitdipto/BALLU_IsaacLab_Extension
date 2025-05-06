# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg


def get_periodic_action(step_count, period=200, num_envs=1):
    """Generate alternating motor actions for walking gait"""
    phase = (step_count % period) / period  # Normalized [0,1]
    
    # Sine wave pattern (amplitude 0.5π centered around π/2)
    #if step_count % 400 < 80:
    #    action_motor_left = 0.0
    #    action_motor_right = 0.0
    #else:
    action_motor_left = 0.5 * math.pi * math.sin(2 * math.pi * phase) + (math.pi/2)
    action_motor_right = 0.5 * math.pi * math.cos(2 * math.pi * phase) + (math.pi/2)
    
    # Create tensor with shape (num_envs, 2)
    return torch.tensor(
        [[action_motor_left, action_motor_right] for _ in range(num_envs)],
        device="cuda:0"
    )

def bang_bang_control(step_count, num_envs=1):
    """
    Bang-bang controller for joint actuation.
    Args:
        step_count (int): Current step count
    Returns:
        torch.Tensor: Control actions (num_envs, num_joints)
    """
    min_action = 0.0
    max_action = math.pi
    actions = torch.full((num_envs, 2), max_action, device="cuda:0") if (step_count % 2 == 0) else torch.full((num_envs, 2), min_action, device="cuda:0")
    return actions

def main():
    """Main function."""
    # create environment configuration
    env_cfg = BalluIndirectActEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # Wrap for recording video
    # env = gymnasium.wrappers.RecordVideo(env, video_folder=".",
    #                                     name_prefix='ballu_manual',
    #                                     step_trigger=lambda step: step == 0,
    #                                     video_length=68,
    #                                     disable_logger=True)
    # print environment information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max episode length: {env.max_episode_length}")
    # simulate physics
    # Initialize list to store torque data
    torque_history = []
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Periodic action: alternate every 200 steps
            # period = 200  # total period length
            # half_period = period // 2
            # deg90 = 90 * torch.pi / 180.0
            # minus_deg90 = -90 * torch.pi / 180.0
            # joint_pos_targets = torch.zeros_like(env.action_manager.action)
            # if (count % period) < half_period:
            #     # LEFT_HIP = 90 deg, RIGHT_HIP = -90 deg
            #     joint_pos_targets[:, 0] = deg90
            #     joint_pos_targets[:, 1] = minus_deg90
            # else:
            #     # LEFT_HIP = -90 deg, RIGHT_HIP = 90 deg
            #     joint_pos_targets[:, 0] = minus_deg90
            #     joint_pos_targets[:, 1] = deg90
            # --- Legacy/random action code below (commented out for reference) ---
            # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            # sample random actions from the normal distribution
            # joint_pos_targets = torch.randn_like(env.action_manager.action)
            # Sample random actions from the uniform distribution
            # Convert 0 to 99 degrees to radians (uniform distribution)
            # min_angle_rad = 0
            # max_angle_rad = 200 * torch.pi / 180.0  # Convert 99 degrees to radians
            # joint_pos_targets = torch.rand_like(env.action_manager.action) * (max_angle_rad - min_angle_rad) + min_angle_rad
            # joint_pos_targets = torch.zeros_like(env.action_manager.action)
            # Set the 2nd column values to max_angle_rad
            # joint_pos_targets[:, 0] = max_angle_rad
            # joint_pos_targets = max_angle_rad * torch.ones_like(env.action_manager.action)
            # print("[INFO]: Random action: ", joint_pos_targets)
            actions = get_periodic_action(count, period = 200, num_envs=args_cli.num_envs)
            #actions = torch.zeros_like(env.action_manager.action)
            #actions = bang_bang_control(count, num_envs=args_cli.num_envs)
            obs, rew, terminated, truncated, info = env.step(actions)
            
            robots = env.unwrapped.scene["robot"]
            knee_indices = robots.actuators["knee_effort_actuators"].joint_indices
            torques_applied_on_knees = robots.data.applied_torque[:, knee_indices]
            print("Torques applied on knees: ", torques_applied_on_knees)
            # Store torque data
            torque_history.append(torques_applied_on_knees.cpu().numpy())
            
            count += 1
            #if count == 600:
            #    break
            # if terminated.any() or truncated.any():
            #     print(f"[INFO]: Environments terminated after {count} steps.")
            #     break

    # close the environment
    env.close()

    # Convert to numpy array
    torque_history = np.array(torque_history)  # Shape: (timesteps, num_envs, num_knees)
    
    # Create figure with subplots
    num_envs = torque_history.shape[1]
    num_knees = torque_history.shape[2]
    fig, axes = plt.subplots(num_envs, 1, figsize=(10, 6*num_envs), squeeze=False)
    
    # Plot for each environment
    for env_idx in range(num_envs):
        ax = axes[env_idx, 0]
        for knee_idx in range(num_knees):
            ax.plot(torque_history[:, env_idx, knee_idx], 
                   label=f'Knee {knee_idx}')
        
        ax.set_title(f'Environment {env_idx}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Torque (Nm)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("knee_torques_plot.png", dpi=300, bbox_inches='tight')
    plt.close()  # Free memory by closing the figure


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()