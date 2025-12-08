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
import pandas as pd

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--buoyancy_offset", type=float, nargs=3, help="Buoyancy offset.")
parser.add_argument("--GCR", type=float, default=0.84, help="Gravity compensation ratio.")

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
import isaaclab.utils.math as math_utils
from ballu_isaac_extension.tasks.ballu_locomotion.flat_env_cfg import BalluFlatEnvCfg
from ballu_isaac_extension.tasks.ballu_locomotion.rough_env_cfg import BalluRoughEnvCfg
from ballu_isaac_extension.tasks.ballu_locomotion.single_obstacle_env_cfg import BalluSingleObstacleEnvCfg
from ballu_isaac_extension.tasks.ballu_locomotion.mdp.geometry_utils import get_robot_dimensions
from action_generators import *
import os
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Usd, UsdGeom, Gf

from ballu_isaac_extension.morphology.modifiers import BalluMorphologyModifier
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
    # env_cfg = BalluRoughEnvCfg()
    # modifier = BalluMorphologyModifier()
    # modifier.adjust_femur_to_limb_ratio(0.7)
    # modifier.convert_to_usd()
    # print("Morphology name: ", modifier.morphology_name)
    # os.environ['BALLU_PATH'] = f"{modifier.morphology_name}/{modifier.morphology_name}.usd"

    env_cfg = BalluFlatEnvCfg()
    # env_cfg = BalluSingleObstacleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    #env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.0)
    # env_cfg.scene.robot.spawn.usd_path = f"{modifier.usd_root_dir}/{modifier.morphology_name}/{modifier.morphology_name}.usd"
    #env_cfg.scene.robot.spawn.usd_path = \
    #    "/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/FT_37/ballu_modified_FT_37.usd"
    #buoyancy_offset = [float(x) for x in args_cli.buoyancy_offset]
    print("buoyancy_offset: ", args_cli.buoyancy_offset)
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg, GCR=args_cli.GCR) #render_mode="rgb_array", buoyancy_offset=args_cli.buoyancy_offset)
    # print environment information
    print(f"\n=== ENVIRONMENT INFO ===")
    print(f"Observation space: {env.unwrapped.observation_space}")
    print(f"Action space: {env.unwrapped.action_space}")
    print(f"Max episode length: {env.unwrapped.max_episode_length}")
    
    robots = env.unwrapped.scene["robot"]
    neck_indices, neck_names = robots.find_joints("NECK")
    
    print(f"\n=== ROBOT INFO ===")
    print(f"Robot body names: {robots.body_names}")
    print(f"Robot joint names: {robots.joint_names}")

    pelvis_idx, _ = robots.find_bodies("PELVIS")
    print("Pelvis index: ", pelvis_idx)

    masses = robots.root_physx_view.get_masses()
    print(f"Pelvis mass: {masses[0, pelvis_idx].item() * 1000:.6f} g")

    # masses[:, pelvis_idx] *= 10
    # robots.root_physx_view.set_masses(
    #     masses, 
    #     torch.arange(env.num_envs, device="cpu")
    # )
    # print(f"Pelvis mass after override: {masses[0, pelvis_idx].item() * 1000:.6f} g")
    # print("Viewer object: ", env.viewport_camera_controller.__dict__)
    
    # simulate physics
    # Initialize list to store torque data
    torque_history = []
    # neck_joint_pos_history = []
    # root_com_xyz_history = []
    base_speed_history = []
    # Track tibia endpoints (world frame). Uses URDF foot offset (0, 0.38485, 0) in tibia link frame.
    # Get tibia body indices once.
    left_tibia_ids, _ = robots.find_bodies("TIBIA_LEFT")
    right_tibia_ids, _ = robots.find_bodies("TIBIA_RIGHT")
    left_tibia_idx = left_tibia_ids[0]
    right_tibia_idx = right_tibia_ids[0]
    tibia_endpoints_world_history = []  # list of tensors shape (num_envs, 2, 3) per step [left,right]
    tibia_startpoints_world_history = []  # list of tensors shape (num_envs, 2, 3) per step [left,right]
    obs_history = []
    count = 0
    cum_rewards = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            
            #actions = get_periodic_action(count, period = 500, num_envs=args_cli.num_envs)
            actions = stepper(count, period = 80, num_envs=args_cli.num_envs)
            #actions = left_leg_1_right_leg_0(num_envs=args_cli.num_envs)
            #actions = both_legs_1(num_envs=args_cli.num_envs)
            #actions = both_legs_0(num_envs=args_cli.num_envs)
            # actions = both_legs_theta(theta=0.3, num_envs=args_cli.num_envs)
            # if count % env.max_episode_length <= 150:
            #     actions = both_legs_theta(theta=5, num_envs=args_cli.num_envs)
            # else:
            #     actions = both_legs_theta(theta=0.0, num_envs=args_cli.num_envs)
            # ---- Best action sequence for jumping ----
            # if count % env.max_episode_length <= 80:
            #     actions = both_legs_theta(theta=1.0, num_envs=args_cli.num_envs)
            # elif count % env.max_episode_length <= 85:
            #     actions = both_legs_0(num_envs=args_cli.num_envs)
            # else:
            #     actions = both_legs_1(num_envs = args_cli.num_envs)

            # if count % 200 <= 90:
            #     actions = both_legs_theta(theta=0.8, num_envs=args_cli.num_envs)
            # else:
            #     actions = both_legs_0(num_envs=args_cli.num_envs)
            # if count % 140 <= 80:
            #     actions = both_legs_0(num_envs=args_cli.num_envs)
            # else:
            #     actions = left_leg_1_right_leg_0(num_envs=args_cli.num_envs)
            #actions = torch.zeros_like(env.action_manager.action)
            #actions = bang_bang_control(count, period=40, num_envs=args_cli.num_envs)
            obs, rew, terminated, truncated, info = env.step(actions)

            # print("episode length buffer: ", env.episode_length_buf)
            base_velocity = robots.data.root_lin_vel_b.clone().detach().cpu()
            base_speed_history.append(base_velocity)
            obs_history.append(obs["policy"])
            # Compute tibia endpoints (world frame) for left/right tibias across all envs
            # Link poses in world frame
            link_pos_w = robots.data.body_link_pos_w  # (num_envs, num_bodies, 3)
            link_quat_w = robots.data.body_link_quat_w  # (num_envs, num_bodies, 4) wxyz
            # Select tibias
            tibia_pos_w = torch.stack([
                link_pos_w[:, left_tibia_idx, :],
                link_pos_w[:, right_tibia_idx, :]
            ], dim=1)  # (num_envs, 2, 3)
            tibia_quat_w = torch.stack([
                link_quat_w[:, left_tibia_idx, :],
                link_quat_w[:, right_tibia_idx, :]
            ], dim=1)  # (num_envs, 2, 4)
            # URDF foot endpoint offset in tibia link frame
            foot_offset_b = torch.tensor([0.0, 0.38485 + 0.004, 0.0], device=tibia_pos_w.device, dtype=tibia_pos_w.dtype)
            foot_offset_b = foot_offset_b.unsqueeze(0).unsqueeze(0).expand(tibia_pos_w.shape)  # (num_envs, 2, 3)
            # Rotate offset into world and translate by link position
            rot_offset_w = math_utils.quat_apply(tibia_quat_w.reshape(-1, 4), foot_offset_b.reshape(-1, 3)).reshape_as(tibia_pos_w)
            tibia_endpoints_w = tibia_pos_w + rot_offset_w  # (num_envs, 2, 3)
            tibia_endpoints_world_history.append(tibia_endpoints_w.detach().cpu())
            tibia_startpoints_world_history.append(tibia_pos_w.detach().cpu())
            #env_idx = 0
            #neck_joint_pos = robots.data.joint_pos[env_idx, neck_indices]
            #neck_joint_pos_history.append(neck_joint_pos.item())
            #root_com_xyz = robots.data.root_com_state_w.detach().cpu()[..., :3]
            #root_com_xyz_history.append(root_com_xyz)
            #robots = env.unwrapped.scene["robot"]
            knee_indices = robots.actuators["knee_effort_actuators"].joint_indices
            torques_applied_on_knees = robots.data.applied_torque[:, knee_indices]
            #print("Torques applied on knees at step: ", count, " are: ", torques_applied_on_knees)
            # Store torque data
            torque_history.append(torques_applied_on_knees.detach().cpu())
            cum_rewards += rew
            count += 1
            # print("count: ", count)
            # if count == env.max_episode_length:
            #    break
            # if terminated.any() or truncated.any():
            #     print(f"[INFO]: Environments terminated after {count} steps.")
            #     break
            # Print robot geometry using new utilities
            if count == 1:
                try:
                    # Extract robot dimensions using new geometry utilities
                    dims = get_robot_dimensions(0)
                    
                    print("=== ROBOT GEOMETRY (via geometry_utils) ===")
                    print(f"Pelvis cylinder - radius: {dims.pelvis.radius.item():.6f}, height: {dims.pelvis.height.item():.6f}")
                    print(f"Femur left cylinder - radius: {dims.femur_left.radius.item():.6f}, height: {dims.femur_left.height.item():.6f}")
                    print(f"Femur right cylinder - radius: {dims.femur_right.radius.item():.6f}, height: {dims.femur_right.height.item():.6f}")
                    
                    print(f"Tibia left cylinder - radius: {dims.tibia_left.radius.item():.6f}, height: {dims.tibia_left.height.item():.6f}")
                    print(f"Tibia right cylinder - radius: {dims.tibia_right.radius.item():.6f}, height: {dims.tibia_right.height.item():.6f}")
                    
                    print(f"Electronics left box - size: {dims.electronics_left.box.size.squeeze()}")
                    print(f"Electronics left sphere - radius: {dims.electronics_left.sphere.radius.item():.6f}")
                    
                    print(f"Electronics right box - size: {dims.electronics_right.box.size.squeeze()}")
                    print(f"Electronics right sphere - radius: {dims.electronics_right.sphere.radius.item():.6f}")
                    
                    # Demonstrate batch extraction for multiple environments (if available)
                    if args_cli.num_envs > 1:
                        print("\n=== BATCH GEOMETRY EXTRACTION ===")
                        batch_dims = get_robot_dimensions(slice(0, min(3, args_cli.num_envs)))
                        print(f"Batch pelvis radii: {batch_dims.pelvis.radius}")
                        print(f"Batch tibia left sphere radii: {batch_dims.tibia_left.sphere.radius}")
                    
                    # Demonstrate concatenation for RL observations
                    print("\n=== RL OBSERVATION FEATURES ===")
                    print(f"electronics left box size: {dims.electronics_left.box.size}")
                    print(f"electronics right box size: {dims.electronics_right.box.size}")
                    print(f"electronics left sphere radius: {dims.electronics_left.sphere.radius}")
                    print(f"electronics right sphere radius: {dims.electronics_right.sphere.radius}")
                    obs_features = torch.cat([
                        dims.pelvis.radius,
                        dims.femur_left.radius,
                        dims.femur_right.radius,
                        dims.tibia_left.height,
                        dims.tibia_right.height,
                        dims.electronics_left.box.size[:, 1],
                        dims.electronics_right.box.size[:, 1],
                        dims.electronics_left.sphere.radius,
                        dims.electronics_right.sphere.radius,
                    ], dim=-1)
                    print(f"Concatenated geometry features: {obs_features}")
                    print(f"Feature shape: {obs_features.shape}")

                except Exception as e:
                    print(f"[ERROR] Failed to extract robot geometry: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("obs['policy'].shape: ", obs["policy"].shape)
            #     print("Obstacle height list: ", env.unwrapped.obstacle_height_list)
            # if count % env.max_episode_length == 0:
            #     env.unwrapped.scene._default_env_origins = torch.rand(env.unwrapped.num_envs, 3, device=env.device) * 6.0
            #     env.unwrapped.scene._default_env_origins[:, 2] = 0.7

    # close the environment
    env.close()
    print(f"Cumulative rewards: {cum_rewards.item()}")
    base_speed_history = torch.stack(base_speed_history)
    obs_history = torch.stack(obs_history)
    base_vel_mean = base_speed_history.mean(dim=0)
    base_vel_std = base_speed_history.std(dim=0)
    print("base_vel_mean: ", base_vel_mean)
    print("base_vel_std: ", base_vel_std)

    print("Shape of base_speed_history: ", base_speed_history.shape)

    obs_hist_np = obs_history.detach().cpu().numpy().squeeze(axis=1)
    print("Shape of obs_hist_np: ", obs_hist_np.shape)
    obs_hist_df = pd.DataFrame(obs_hist_np, columns=[
        "joint_pos_hip_left",
        "joint_pos_hip_right",
        "joint_pos_neck",
        "joint_pos_knee_left",
        "joint_pos_knee_right",
        "joint_pos_motor_left",
        "joint_pos_motor_right",
        "joint_vel_hip_left",
        "joint_vel_hip_right",
        "joint_vel_neck",
        "joint_vel_knee_left",
        "joint_vel_knee_right",
        "joint_vel_motor_left",
        "joint_vel_motor_right",
        "left_elctx_quat_1",
        "left_elctx_quat_2",
        "left_elctx_quat_3",
        "left_elctx_quat_4",
        "left_elctx_ang_vel_x",
        "left_elctx_ang_vel_y",
        "left_elctx_ang_vel_z",
        "left_elctx_lin_acc_x",
        "left_elctx_lin_acc_y",
        "left_elctx_lin_acc_z",
        "right_elctx_quat_1",
        "right_elctx_quat_2",
        "right_elctx_quat_3",
        "right_elctx_quat_4",
        "right_elctx_ang_vel_x",
        "right_elctx_ang_vel_y",
        "right_elctx_ang_vel_z",
        "right_elctx_lin_acc_x",
        "right_elctx_lin_acc_y",
        "right_elctx_lin_acc_z",
        "gt_left_elctx_quat_1",
        "gt_left_elctx_quat_2",
        "gt_left_elctx_quat_3",
        "gt_left_elctx_quat_4",
        "gt_left_elctx_ang_vel_x",
        "gt_left_elctx_ang_vel_y",
        "gt_left_elctx_ang_vel_z",
        "gt_left_elctx_lin_acc_x",
        "gt_left_elctx_lin_acc_y",
        "gt_left_elctx_lin_acc_z",
    ])
    obs_hist_df.to_csv(os.path.join("logs/results/manual_run", 'obs_history.csv'), index=False)
    print("Saved observation history to obs_history.csv")

    # results_dir = "logs/results/manual_run"
    # # Plot base speed components
    # plt.figure(figsize=(10, 6))
    # timesteps = base_speed_history.shape[0]
    # components = ['X', 'Y', 'Z']
    # colors = ['b', 'g', 'r']
    
    # for i in range(3):
    #     plt.subplot(3, 1, i+1)
    #     plt.plot(range(timesteps), base_speed_history[:, 0, i], color=colors[i])
    #     plt.ylabel(f'{components[i]} Speed (m/s)')
    #     plt.grid(True)
    
    # plt.xlabel('Timesteps')
    # plt.suptitle('Base Speed Components Over Time')
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'base_speed_components_jump.png'))
    # plt.close()
    
    # torque_history = torch.stack(torque_history)
    # plt.figure(figsize=(10, 6))
    # timesteps = torque_history.shape[0]
    # components = ['Left Knee', 'Right Knee']
    # colors = ['b', 'g']
    
    # for i in range(2):
    #     plt.subplot(2, 1, i+1)
    #     plt.plot(range(timesteps), torque_history[:, 0, i], color=colors[i], alpha=0.67)
    #     plt.ylabel(f'{components[i]} Torque (Nm)')
    #     plt.grid(True)
    
    # plt.xlabel('Timesteps')
    # plt.suptitle('Torque Applied on Knees Over Time')
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'knee_torque_jump.png'))
    # plt.close()

    # Summarize tibia endpoint tracking
    if len(tibia_endpoints_world_history) > 0:
        tibia_endpoints_world_history = torch.stack(tibia_endpoints_world_history)  # (T, num_envs, 2, 3)
        # Example: print last positions for env 0
        last_left = tibia_endpoints_world_history[-1, 0, 0]
        last_right = tibia_endpoints_world_history[-1, 0, 1]
        print("Last tibia endpoints (world) for env 0 — Left:", last_left.numpy(), ", Right:", last_right.numpy())
        # Optionally save to disk
        # np.save("tibia_endpoints_world.npy", tibia_endpoints_world_history.numpy())
        # print("Saved tibia endpoint history to tibia_endpoints_world.npy with shape:", tibia_endpoints_world_history.shape)
        plt.figure(figsize=(10, 6))
        plt.plot(range(timesteps), tibia_endpoints_world_history[:, 0, 0, 2], label='Left toe Z pos')
        plt.plot(range(timesteps), tibia_endpoints_world_history[:, 0, 1, 2], label='Right toe Z pos')
        plt.axhline(y = 0.055, label = "y = 0.055", color="tab:green", linestyle="--")
        plt.xlabel('Timesteps')
        plt.ylabel('Toe Z pos (m)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'toe_z_pos_jump.png'))
        plt.close()

    if len(tibia_startpoints_world_history) > 0:
        tibia_startpoints_world_history = torch.stack(tibia_startpoints_world_history)  # (T, num_envs, 2, 3)
        plt.figure(figsize=(10, 6))
        plt.plot(range(timesteps), tibia_startpoints_world_history[:, 0, 0, 2], label='Left knee Z pos')
        plt.plot(range(timesteps), tibia_startpoints_world_history[:, 0, 1, 2], label='Right knee Z pos')
        plt.xlabel('Timesteps')
        plt.ylabel('Knee Z pos (m)')    
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'knee_z_pos_jump.png'))
        plt.close()
    
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