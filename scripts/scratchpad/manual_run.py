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
import numpy as np
import matplotlib.pyplot as plt
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--buoyancy_offset", type=float, nargs=3, help="Buoyancy offset.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg
from action_generators import *
import os
#from ..rsl_rl.plotters import plot_root_com_xy

def plot_root_com_xy(root_com_xyz_hist_tch, num_envs, save_dir):
    """Plot root COM X,Y positions for all environments in a single figure.
    
    Args:
        root_com_xyz_hist_tch: Tensor of root COM XYZ history [timesteps, envs, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Calculate figure size based on number of environments
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    fig_size = max(8, grid_size * 4)  # Minimum 8 inches, scale with env count
    
    plt.figure(figsize=(fig_size, fig_size))
        
    # Extract all X,Y coordinates
    all_xy = np.array([[pos[env_idx][:2] for pos in root_com_xyz_hist_tch] 
                      for env_idx in range(num_envs)])
        
    # Plot each environment's trajectory (swap axes, invert x for Y-left, X-up)
    for env_idx in range(num_envs):
        plt.plot(all_xy[env_idx][:, 1], all_xy[env_idx][:, 0], 
                label=f'Env {env_idx}', alpha=0.7)
    plt.gca().invert_xaxis()  # Invert X so positive Y points left
    
    plt.title(f'Root COM (Y,X) Positions (X↑, Y←) (All {num_envs} Environments)')
    plt.xlabel('Y Position')  # Now horizontal, but inverted
    plt.ylabel('X Position')  # Now vertical
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
        
    plot_path = os.path.join(save_dir, "root_com_xy_positions_all.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved combined root COM XY plot to {plot_path}")

def main():
    """Main function."""
    # create environment configuration
    env_cfg = BalluIndirectActEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    #buoyancy_offset = [float(x) for x in args_cli.buoyancy_offset]
    print("buoyancy_offset: ", args_cli.buoyancy_offset)
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg) #render_mode="rgb_array", buoyancy_offset=args_cli.buoyancy_offset)
    
    # print environment information
    print(f"Observation space: {env.unwrapped.observation_space}")
    print(f"Action space: {env.unwrapped.action_space}")
    print(f"Max episode length: {env.unwrapped.max_episode_length}")
    
    robots = env.unwrapped.scene["robot"]
    neck_indices, neck_names = robots.find_joints("NECK")
    # simulate physics
    # Initialize list to store torque data
    torque_history = []
    neck_joint_pos_history = []
    root_com_xyz_history = []
    base_speed_history = []
    count = 0
    cum_rewards = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            
            #actions = get_periodic_action(count, period = 500, num_envs=args_cli.num_envs)
            actions = stepper(count, period = 40, num_envs=args_cli.num_envs)
            #actions = left_leg_1_right_leg_0(num_envs=args_cli.num_envs)
            #actions = both_legs_1(num_envs=args_cli.num_envs)
            #actions = both_legs_0(num_envs=args_cli.num_envs)
            #actions = torch.zeros_like(env.action_manager.action)
            #actions = bang_bang_control(count, period=40, num_envs=args_cli.num_envs)
            obs, rew, terminated, truncated, info = env.step(actions)
            base_velocity = robots.data.root_lin_vel_b.clone().detach().cpu()
            base_speed_history.append(base_velocity)
            #env_idx = 0
            #neck_joint_pos = robots.data.joint_pos[env_idx, neck_indices]
            #neck_joint_pos_history.append(neck_joint_pos.item())
            #root_com_xyz = robots.data.root_com_state_w.detach().cpu()[..., :3]
            #root_com_xyz_history.append(root_com_xyz)
            #robots = env.unwrapped.scene["robot"]
            #knee_indices = robots.actuators["knee_effort_actuators"].joint_indices
            #torques_applied_on_knees = robots.data.applied_torque[:, knee_indices]
            #print("Torques applied on knees at step: ", count, " are: ", torques_applied_on_knees)
            # Store torque data
            #torque_history.append(torques_applied_on_knees.cpu().numpy())
            cum_rewards += rew
            count += 1
            if count == 400:
                break
            # if terminated.any() or truncated.any():
            #     print(f"[INFO]: Environments terminated after {count} steps.")
            #     break

    # close the environment
    env.close()
    print(f"Cumulative rewards: {cum_rewards.item()}")
    base_speed_history = torch.stack(base_speed_history)
    
    base_vel_mean = base_speed_history.mean(dim=0)
    base_vel_std = base_speed_history.std(dim=0)
    print("base_vel_mean: ", base_vel_mean)
    print("base_vel_std: ", base_vel_std)
    #neck_joint_pos_history = np.array(neck_joint_pos_history)
    #print("neck_joint_pos_history: ", neck_joint_pos_history)
    # with open("neck_joint_pos_history.jsonl", "a") as f:
    #     f.write(json.dumps({
    #         str(args_cli.buoyancy_offset) : neck_joint_pos_history
    #     }) + "\n")

    # print("Saved data to neck_joint_pos_history.jsonl")
    # root_com_xyz_history = np.array(root_com_xyz_history)
    # plot_root_com_xy(root_com_xyz_history, 1, "")
    # Convert to numpy array
    #torque_history = np.array(torque_history)  # Shape: (timesteps, num_envs, num_knees)
    
    # Create figure with subplots
    #num_envs = torque_history.shape[1]
    #num_knees = torque_history.shape[2]
    #fig, axes = plt.subplots(num_envs, 1, figsize=(10, 6*num_envs), squeeze=False)
    
    # Plot for each environment
    #for env_idx in range(num_envs):
    #    ax = axes[env_idx, 0]
    #    for knee_idx in range(num_knees):
    #        ax.plot(torque_history[:, env_idx, knee_idx], 
    #               label=f'Knee {knee_idx}')
        
    #    ax.set_title(f'Environment {env_idx}')
    #    ax.set_xlabel('Timestep')
    #    ax.set_ylabel('Torque (Nm)')
    #    ax.legend()
    #    ax.grid(True)
    
    #plt.tight_layout()
    #plt.savefig("knee_torques_plot.png", dpi=300, bbox_inches


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()