# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task
and create video overlays with joint data using only OpenCV, matplotlib, and numpy.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tests/scratchpad_video_edit.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment with video overlay.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video_length", type=int, default=399, help="Length of the video to record.")
parser.add_argument("--video_destination", type=str, default="videos", help="Destination folder for the video.")

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


def create_joint_overlay_video(video_path, joint_data, output_path, method='side_by_side'):
    """
    Create video overlay using only OpenCV, matplotlib, and numpy.
    
    Args:
        video_path: Path to input video
        joint_data: Dict with joint names as keys, numpy arrays as values
        output_path: Path for output video  
        method: 'side_by_side' or 'corner_overlay'
    """
    
    print(f"Processing video: {video_path}")
    
    # Step 1: Open input video and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames, {duration:.2f}s")
    
    # Step 2: Prepare output video writer
    if method == 'side_by_side':
        # Double the width for side-by-side layout
        out_width = width * 2
        out_height = height
    else:  # corner_overlay
        out_width = width
        out_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Step 3: Prepare matplotlib figure (reuse for efficiency)
    plt.style.use('dark_background')
    
    if method == 'side_by_side':
        # Calculate figure size to match video aspect ratio and ensure readable text
        plot_width_inches = width / 100  # Convert pixels to inches (assuming ~100 DPI)
        plot_height_inches = height / 100
        # Number of subplots is half the number of joints (pairing LEFT and RIGHT)
        num_subplots = len(joint_data) // 2
        print(f"Creating side-by-side plot: {plot_width_inches:.1f}x{plot_height_inches:.1f} inches, {num_subplots} subplots")
        fig, axes = plt.subplots(num_subplots, 1, 
                               figsize=(plot_width_inches, plot_height_inches), 
                               facecolor='black', dpi=100)
        if num_subplots == 1:
            axes = [axes]
    else:  # corner overlay
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), facecolor='black', dpi=100)
        fig.patch.set_alpha(0.8)
    
    # Step 4: Process each frame
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate current time
        current_time = frame_idx / fps
        
        # Clear previous plots
        if method == 'side_by_side':
            for ax in axes:
                ax.clear()
                ax.set_facecolor('black')
        else:
            ax.clear()
            ax.set_facecolor('black')
            ax.patch.set_alpha(0.8)
        
        # Generate plots for current timestamp
        if method == 'side_by_side':
            # Plot joints in pairs (LEFT and RIGHT in same subplot)
            joint_items = list(joint_data.items())
            colors = ['red', 'blue']  # First joint red, second joint blue
            
            for subplot_idx in range(num_subplots):
                ax = axes[subplot_idx]
                
                # Get the pair of joints for this subplot
                joint_pair = joint_items[subplot_idx * 2:(subplot_idx + 1) * 2]
                joint_names_in_plot = []
                
                for pair_idx, (joint_name, joint_values) in enumerate(joint_pair):
                    color = colors[pair_idx]
                    joint_names_in_plot.append(joint_name)
                    
                    # Create time array for this joint
                    joint_time = np.linspace(0, duration, len(joint_values))
                    
                    # Plot full trajectory (dim)
                    ax.plot(joint_time, joint_values, 'gray', alpha=0.3, linewidth=1)
                    
                    # Plot trajectory up to current time
                    current_idx = int((current_time / duration) * len(joint_values))
                    current_idx = min(current_idx, len(joint_values) - 1)
                    
                    if current_idx > 0:
                        ax.plot(joint_time[:current_idx], joint_values[:current_idx], 
                               color, linewidth=2, label=joint_name)
                    
                    # Current point
                    if current_idx < len(joint_values):
                        ax.plot(current_time, joint_values[current_idx], 
                               color, marker='o', markersize=6)
                
                # Styling with larger, more readable fonts
                ax.set_xlim(0, duration)
                # Use both joint names in the ylabel
                ylabel = f'{" vs ".join(joint_names_in_plot)}\n(rad)'
                ax.set_ylabel(ylabel, color='white', fontsize=14, fontweight='bold')
                ax.tick_params(colors='white', labelsize=12, width=2)
                ax.grid(True, alpha=0.3, linewidth=1)
                ax.legend(fontsize=10, framealpha=0.9, edgecolor='white')
                
                # Make the plot border more visible
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(2)
                
                # Only show x-axis on bottom plot
                if subplot_idx == num_subplots - 1:
                    ax.set_xlabel('Time (s)', color='white', fontsize=14, fontweight='bold')
                else:
                    ax.set_xticklabels([])
        
        else:  # corner_overlay
            # Plot all joints on same plot with rolling window
            colors = ['red', 'blue']
            
            for i, (joint_name, joint_values) in enumerate(joint_data.items()):
                color = colors[i % len(colors)]
                
                joint_time = np.linspace(0, duration, len(joint_values))
                current_idx = int((current_time / duration) * len(joint_values))
                current_idx = min(current_idx, len(joint_values) - 1)
                
                if current_idx > 0:
                    ax.plot(joint_time[:current_idx], joint_values[:current_idx], 
                           color=color, linewidth=2, label=joint_name, alpha=0.8)
                
                if current_idx < len(joint_values):
                    ax.plot(current_time, joint_values[current_idx], 
                           color=color, marker='o', markersize=4)
            
            # Add horizontal reference line at Y = 0.04
            ax.axhline(y=0.05, color='yellow', linestyle='--', linewidth=1, alpha=0.8, label='Refer (0.05)')
            
            # Rolling window (last 3 seconds)
            window_start = max(0, current_time - 3.0)
            ax.set_xlim(window_start, current_time + 0.5)
            ax.set_ylabel('Joint Angle (rad)', color='white', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', color='white', fontsize=10, fontweight='bold')
            ax.tick_params(colors='white', labelsize=8, width=1.5)
            ax.legend(fontsize=8, framealpha=0.9, edgecolor='white')
            ax.grid(True, alpha=0.3, linewidth=1)
            
            # Make corner overlay border more visible
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(1.5)
        
        # Use tight_layout with padding for better text spacing
        plt.tight_layout(pad=2.0)
        
        # Step 5: Convert matplotlib plot to image array
        fig.canvas.draw()
        
        # Get the plot as RGB array (compatible with different matplotlib versions)
        try:
            # Try newer matplotlib method first
            plot_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_img = plot_buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            plot_img = plot_img[:, :, :3]
        except AttributeError:
            try:
                # Try older matplotlib method
                plot_buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Fallback method using tostring_argb
                plot_buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                plot_img = plot_buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert ARGB to RGB (skip alpha channel and reorder)
                plot_img = plot_img[:, :, 1:4]  # Skip alpha, take RGB
        
        # Convert RGB to BGR for OpenCV
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        # Step 6: Combine video frame with plot
        if method == 'side_by_side':
            # Since we already created the plot with correct dimensions, minimal resize needed
            # Only resize if dimensions don't match exactly
            if plot_img.shape[:2] != (height, width):
                # Use high-quality interpolation to preserve text clarity
                plot_resized = cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            else:
                plot_resized = plot_img
            
            # Concatenate horizontally
            combined_frame = np.hstack([frame, plot_resized])
            
        else:  # corner_overlay
            # Resize plot for corner overlay (smaller)
            overlay_width = int(width // 2.5)
            overlay_height = int(height // 2.5)
            plot_resized = cv2.resize(plot_img, (overlay_width, overlay_height))
            
            # Create copy of original frame
            combined_frame = frame.copy()
            
            # Overlay plot in bottom-right corner
            y_offset = height - overlay_height - 20
            x_offset = width - overlay_width - 20
            
            # Blend the overlay
            alpha = 0.8
            combined_frame[y_offset:y_offset+overlay_height, 
                          x_offset:x_offset+overlay_width] = \
                alpha * plot_resized + (1-alpha) * combined_frame[y_offset:y_offset+overlay_height, 
                                                                  x_offset:x_offset+overlay_width]
        
        # Step 7: Write frame to output video
        out.write(combined_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    plt.close(fig)
    
    print(f"Video overlay completed: {output_path}")


def main():
    """Main function."""
    # create environment configuration
    env_cfg = BalluIndirectActEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env,
                                   video_folder=args_cli.video_destination,
                                   name_prefix="ballu",
                                   step_trigger=lambda step: step == 400,
                                   video_length=args_cli.video_length,
                                   disable_logger=True)
    
    
    # print environment information
    print(f"Observation space: {env.unwrapped.observation_space}")
    print(f"Action space: {env.unwrapped.action_space}")
    print(f"Max episode length: {env.unwrapped.max_episode_length}")
    
    # Initialize data collection for joint states
    # First, let's find the joint indices by name
    robots = env.unwrapped.scene["robot"]
    
    # Debug: Print all available joint names
    print("Available joint names in the robot:")
    for i, name in enumerate(robots.joint_names):
        print(f"  Index {i}: {name}")
    print()
    
    # Method 1: Using find_joints() - most flexible
    left_knee_indices, left_knee_names = robots.find_joints("KNEE_LEFT")
    right_knee_indices, right_knee_names = robots.find_joints("KNEE_RIGHT")
    left_motorarm_indices, left_motorarm_names = robots.find_joints("MOTOR_LEFT")
    right_motorarm_indices, right_motorarm_names = robots.find_joints("MOTOR_RIGHT")
    
    print(f"Left knee indices: {left_knee_indices}")
    print(f"Right knee indices: {right_knee_indices}")
    print(f"Left motorarm indices: {left_motorarm_indices}")
    print(f"Right motorarm indices: {right_motorarm_indices}")
    
    # Extract the first index from each list (or use 0 as fallback)
    left_knee_idx = left_knee_indices[0] if left_knee_indices else 0
    right_knee_idx = right_knee_indices[0] if right_knee_indices else 1
    left_motorarm_idx = left_motorarm_indices[0] if left_motorarm_indices else 2
    right_motorarm_idx = right_motorarm_indices[0] if right_motorarm_indices else 3

    joint_data_history = {
        'KNEE_LEFT': [],
        'KNEE_RIGHT': [],
        #'MOTOR_LEFT': [],
        #'MOTOR_RIGHT': [],
        'ACT_LEFT': [],
        'ACT_RIGHT': [],
    }
    
    count = 0
    
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
            
            # Collect joint data using the found indices
            joint_positions = robots.data.joint_pos  # Shape: (num_envs, num_joints)
            
            # Store data for environment 0 using the correct joint indices
            if count >= 400:
                env_idx = 0
                joint_data_history['KNEE_LEFT'].append(joint_positions[env_idx, left_knee_idx].cpu().numpy())
                joint_data_history['KNEE_RIGHT'].append(joint_positions[env_idx, right_knee_idx].cpu().numpy())
                #joint_data_history['MOTOR_LEFT'].append(joint_positions[env_idx, left_motorarm_idx].cpu().numpy())
                #joint_data_history['MOTOR_RIGHT'].append(joint_positions[env_idx, right_motorarm_idx].cpu().numpy())
                joint_data_history['ACT_LEFT'].append(actions[env_idx, 0].cpu().numpy())
                joint_data_history['ACT_RIGHT'].append(actions[env_idx, 1].cpu().numpy())
            
            #robots = env.unwrapped.scene["robot"]
            #knee_indices = robots.actuators["knee_effort_actuators"].joint_indices
            #torques_applied_on_knees = robots.data.applied_torque[:, knee_indices]
            #print("Torques applied on knees at step: ", count, " are: ", torques_applied_on_knees)
            # Store torque data
            #torque_history.append(torques_applied_on_knees.cpu().numpy())
            
            count += 1
            if count == 799:
                break


    # close the environment
    env.close()

    # Convert to numpy arrays
    for joint_name in joint_data_history:
        joint_data_history[joint_name] = np.array(joint_data_history[joint_name])
    
    print("Joint data collected.")
    print(f"Data shapes: KNEE_LEFT: {joint_data_history['KNEE_LEFT'].shape}, KNEE_RIGHT: {joint_data_history['KNEE_RIGHT'].shape}, ACT_LEFT: {joint_data_history['ACT_LEFT'].shape}, ACT_RIGHT: {joint_data_history['ACT_RIGHT'].shape}")
    
    # Find the latest video file
    video_files = glob.glob(f"{args_cli.video_destination}/ballu*.mp4")
    if video_files:
        latest_video = max(video_files, key=os.path.getctime)
        print(f"Found video: {latest_video}")
        
        # Create overlay videos
        print("Creating side-by-side overlay...")
        create_joint_overlay_video(
            latest_video, 
            joint_data_history, 
            f"{args_cli.video_destination}/ballu_with_joints_side_by_side.mp4",
            method='side_by_side'
        )
        
        print("Creating corner overlay (knee joints only)...")
        # Use only knee joints for corner overlay to avoid clutter
        knee_only_data = {
            'KNEE_LEFT': joint_data_history['KNEE_LEFT'],
            'KNEE_RIGHT': joint_data_history['KNEE_RIGHT']
        }
        create_joint_overlay_video(
            latest_video, 
            knee_only_data, 
            f"{args_cli.video_destination}/ballu_with_joints_corner.mp4", 
            method='corner_overlay'
        )
        
        print(f"Done! Check the {args_cli.video_destination}/ folder for output files:")
        print(f"- {args_cli.video_destination}/ballu_with_joints_side_by_side.mp4")
        print(f"- {args_cli.video_destination}/ballu_with_joints_corner.mp4")
    else:
        print("No video files found!")

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