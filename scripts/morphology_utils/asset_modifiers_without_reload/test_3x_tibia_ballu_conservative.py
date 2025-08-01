#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for the 3x Tibia BALLU robot (CONSERVATIVE PHYSICS VERSION).

This script loads the conservative physics 3x tibia BALLU robot with:
- Moderate mass scaling (1.5x)
- Conservative inertia adjustments
- Properly positioned collision geometry
- Stable control parameters

Usage:
    python test_3x_tibia_ballu_conservative.py --num_envs 1 --livestream 2

"""

import argparse
import numpy as np
import os
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test the 3x Tibia BALLU robot (Conservative Physics).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, SpringPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg
import math

def degree_to_radian(degree):
    return degree * math.pi / 180.0

# Create robot configuration for 3x tibia BALLU (CONSERVATIVE PHYSICS)
BALLU_3X_TIBIA_CONSERVATIVE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/ballu_3x_tibia_conservative.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # Increased for stability
            solver_velocity_iteration_count=1,  # Enabled for better contact handling
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Moderate height increase for ground clearance (original: 6.0 → 7.0)
        pos=(0.0, 0.0, 7.0), 
        joint_pos={"NECK": 0.0, 
                   "HIP_LEFT": degree_to_radian(1),
                   "HIP_RIGHT": degree_to_radian(1),
                   "KNEE_LEFT": degree_to_radian(27.35),
                   "KNEE_RIGHT": degree_to_radian(27.35),
                   "MOTOR_LEFT": degree_to_radian(10),
                   "MOTOR_RIGHT": degree_to_radian(10)}
    ),
    actuators={
        # Define actuators for MOTOR joints
        "motor_actuators": ImplicitActuatorCfg(
            joint_names_expr=["MOTOR_LEFT", "MOTOR_RIGHT"],
            effort_limit_sim=1.44 * 9.81 * 1e-2, # 0.1412 Nm
            velocity_limit_sim=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            stiffness=1.0,
            damping=0.01,
        ),
        # Define effort-control actuator for KNEE joints with adjusted gains
        "knee_effort_actuators": SpringPDActuatorCfg(
            joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
            effort_limit=1.44 * 9.81 * 1e-2, # 0.141264 Nm
            velocity_limit=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            spring_coeff=0.1409e-3 / degree_to_radian(1.0), # 0.4021 Nm/rad
            spring_damping=1.0e-2,
            spring_preload=degree_to_radian(180 - 135 + 27.35),
            # Moderate gains for conservative physics
            pd_p=0.06,  # Slightly reduced from original 0.08
            pd_d=0.009, # Slightly reduced from original 0.01
            stiffness=float("inf"), 
            damping=float("inf"), 
        ),
        # Keep other joints passive
        "other_passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT"], 
            stiffness=0.0,
            damping=0.001,
        ),
    },
)

# Action generators with moderate parameters
def stepper_conservative(count, period=80, num_envs=1):
    """Generate stepping control pattern (moderate speed for stability)."""
    phase = (count % period) / period
    
    # Generate stepper pattern
    if phase < 0.5:  # First half of period
        left_action = 0.8
        right_action = -0.8
    else:  # Second half of period
        left_action = -0.8
        right_action = 0.8
    
    actions = torch.zeros((num_envs, 2), dtype=torch.float32)
    actions[:, 0] = left_action   # Left motor
    actions[:, 1] = right_action  # Right motor
    
    return actions

def both_legs_oscillate_conservative(count, period=120, num_envs=1):
    """Generate synchronized oscillation for both legs (moderate amplitude)."""
    phase = (count % period) / period
    action_value = 0.7 * math.sin(2 * math.pi * phase)  # Reduced amplitude
    
    actions = torch.zeros((num_envs, 2), dtype=torch.float32)
    actions[:, 0] = action_value  # Left motor
    actions[:, 1] = action_value  # Right motor
    
    return actions

def gentle_control_conservative(count, period=150, num_envs=1):
    """Generate very gentle control for stable testing."""
    phase = (count % period) / period
    action_value = 0.4 * math.sin(2 * math.pi * phase)  # Very reduced amplitude
    
    actions = torch.zeros((num_envs, 2), dtype=torch.float32)
    actions[:, 0] = action_value  # Left motor
    actions[:, 1] = action_value  # Right motor
    
    return actions

def main():
    """Main function."""
    print("="*60)
    print("🤖 3x TIBIA BALLU ROBOT TEST (CONSERVATIVE PHYSICS)")
    print("="*60)
    print(f"[INFO] Creating environment with {args_cli.num_envs} environments")
    print(f"[INFO] Robot: 3x Tibia Length BALLU (Conservative Physics)")
    print(f"[INFO] Initial height: 7.0m (moderate ground clearance)")
    print(f"[INFO] Physics: Conservative scaling for numerical stability")
    print("="*60)
    
    # Create environment configuration
    env_cfg = BalluIndirectActEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Replace the robot configuration with our conservative 3x tibia version
    env_cfg.scene.robot = BALLU_3X_TIBIA_CONSERVATIVE_CFG
    
    # Setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Print environment information
    print(f"\n=== ENVIRONMENT INFO ===")
    print(f"Observation space: {env.unwrapped.observation_space}")
    print(f"Action space: {env.unwrapped.action_space}")
    print(f"Max episode length: {env.unwrapped.max_episode_length}")
    
    robots = env.unwrapped.scene["robot"]
    
    print(f"\n=== ROBOT INFO ===")
    print(f"Robot body names: {robots.body_names}")
    print(f"Robot joint names: {robots.joint_names}")
    print(f"Using 3x tibia length with conservative physics:")
    print(f"  • Conservative mass: 0.065 kg per tibia (1.5x scaling)")
    print(f"  • Moderate inertia scaling (√scale_factor)")
    print(f"  • Properly repositioned collision geometry")
    print(f"  • Stable control gains and solver settings")
    
    # Initialize tracking data
    base_velocity_history = []
    cumulative_rewards = 0
    count = 0
    target_steps = 1500
    
    print(f"\n=== STARTING SIMULATION ===")
    print(f"Running simulation for {target_steps} steps with conservative control...")
    print(f"Control strategy: Gentle oscillations for stability testing")
    
    stable_simulation = True
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # Generate conservative control actions
            actions = gentle_control_conservative(count, period=150, num_envs=args_cli.num_envs)
            
            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # Check for NaN values in observations
            if torch.isnan(obs["policy"]).any():
                print(f"[WARNING] NaN detected in observations at step {count}")
                stable_simulation = False
                break
            
            # Track robot state
            base_velocity = robots.data.root_lin_vel_b.clone().detach().cpu()
            base_velocity_history.append(base_velocity)
            
            cumulative_rewards += rew
            count += 1
            
            # Print periodic status updates
            if count % 300 == 0:
                avg_reward = cumulative_rewards.mean().item()
                avg_velocity = base_velocity.mean().item()
                robot_height = robots.data.root_com_state_w[0, 2].item()
                print(f"[INFO] Step {count}/{target_steps} - Reward: {avg_reward:.3f}, Velocity: {avg_velocity:.3f} m/s, Height: {robot_height:.3f}m")
            
            # Stop after target steps
            if count >= target_steps:
                print(f"\n[INFO] Completed {target_steps} simulation steps!")
                break
            
            # Handle episode termination
            if terminated.any() or truncated.any():
                print(f"[INFO] Episode terminated after {count} steps")
                # Check why terminated
                robot_height = robots.data.root_com_state_w[0, 2].item()
                if robot_height < 1.0:
                    print(f"[INFO] Robot fell (height: {robot_height:.3f}m)")
                break

    # Close the environment
    env.close()
    
    # Print final results
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Completed {count} simulation steps")
    print(f"Numerical stability: {'✅ STABLE' if stable_simulation else '❌ UNSTABLE'}")
    
    if not torch.isnan(cumulative_rewards).any():
        print(f"Final cumulative rewards: {cumulative_rewards.mean().item():.6f}")
    else:
        print(f"Final cumulative rewards: NaN (unstable)")
    
    if base_velocity_history and not torch.isnan(torch.stack(base_velocity_history)).any():
        base_velocity_tensor = torch.stack(base_velocity_history)
        base_vel_mean = base_velocity_tensor.mean(dim=0)
        base_vel_std = base_velocity_tensor.std(dim=0)
        final_height = robots.data.root_com_state_w[0, 2].item()
        
        print(f"Average base velocity: {base_vel_mean.mean().item():.6f} m/s")
        print(f"Base velocity std: {base_vel_std.mean().item():.6f} m/s")
        print(f"Final robot height: {final_height:.3f}m")
        
        if final_height > 3.0:
            print(f"✅ Robot maintained good height - no ground penetration!")
        elif final_height > 1.0:
            print(f"⚠️  Robot height acceptable but may need adjustment")
        else:
            print(f"❌ Robot fell or penetrated ground")
    else:
        print(f"❌ Velocity tracking failed due to numerical issues")
    
    print(f"\n🎉 3x Tibia BALLU (Conservative Physics) test completed!")
    print("="*60)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 