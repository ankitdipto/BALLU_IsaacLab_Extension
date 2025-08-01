#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for BALLU morphology modification.

This script tests the tibia scaling functionality with minimal setup.
Usage:
    python test_morphology_modification.py --tibia_scale 1.5
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test BALLU morphology modification")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--tibia_scale", type=float, default=1.5, help="Tibia scale factor")
# Note: --headless is automatically added by AppLauncher.add_app_launcher_args()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports after Isaac Sim launch
import torch
import time
from isaaclab.envs import ManagerBasedRLEnv
from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg
from BALLU_IsaacLab_Extension.scripts.morphology_utils.asset_modifiers_without_reload.ballu_morphology_modifier import BalluMorphologyModifier


def test_morphology_modification():
    """Test the morphology modification functionality."""
    
    print("="*60)
    print("BALLU MORPHOLOGY MODIFICATION TEST")
    print("="*60)
    
    # Create environment
    print(f"[1/6] Creating environment with {args.num_envs} environments...")
    env_cfg = BalluIndirectActEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"[2/6] Starting simulation and resetting environment...")
    # Reset the environment to ensure everything is properly initialized
    obs, _ = env.reset()
    
    # Step simulation a few times to ensure robots are visible
    for i in range(5):
        actions = torch.zeros_like(env.action_manager.action)
        obs, rew, terminated, truncated, info = env.step(actions)
    
    print(f"[3/6] Robot should now be visible in the viewport!")
    print(f"      You should see the BALLU robot with normal proportions.")
    
    # Wait for user to see the original robot
    print(f"[4/6] Waiting 3 seconds for you to see the original robot...")
    time.sleep(3)
    
    # Create morphology modifier
    print(f"[5/6] Creating morphology modifier and applying scaling...")
    modifier = BalluMorphologyModifier()
    
    # Get baseline information (but handle errors gracefully)
    try:
        baseline_info = modifier.get_tibia_scale_info(env)
        print("Baseline robot info:")
        for key, value in baseline_info.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Warning: Could not get baseline info: {e}")
        baseline_info = {}
    
    # Apply morphology modification
    print(f"Applying tibia scaling (factor: {args.tibia_scale})...")
    success = modifier.scale_tibia_links(env, scale_factor=args.tibia_scale, apply_physics_updates=False)
    
    if success:
        print("✓ Morphology modification successful!")
        print(f"  The tibia links should now be {args.tibia_scale}x longer!")
        print(f"  Look at the robot's shins - they should appear visibly longer.")
        
    else:
        print("✗ Morphology modification failed!")
        return False
    
    # Run simulation to show the modified robot
    print(f"[6/6] Running simulation with modified robot (30 steps)...")
    print(f"      You should now see the BALLU robot with longer tibia links!")
    
    for step in range(30):
        with torch.inference_mode():
            # Apply zero actions (passive simulation)
            actions = torch.zeros_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"      Simulation step {step}/30")
                
        # Check for any critical issues
        if torch.any(torch.isnan(obs)):
            print(f"✗ Simulation instability detected at step {step}")
            break
            
    print("✓ Simulation completed!")
    
    # Keep the simulation running for visual inspection
    print("\n" + "="*60)
    print("MORPHOLOGY MODIFICATION APPLIED!")
    print("="*60)
    print(f"The robot's tibia links have been scaled by {args.tibia_scale}x")
    print("You should see longer shin bones on the BALLU robot.")
    print("\nSimulation will continue running for visual inspection...")
    print("Press Ctrl+C to stop when you're satisfied with the results.")
    print("="*60)
    
    try:
        # Keep running for visual inspection
        step_count = 0
        while True:
            with torch.inference_mode():
                # Apply some simple actions to make the robot move a bit
                if step_count < 100:
                    # Static pose for first 100 steps
                    actions = torch.zeros_like(env.action_manager.action)
                else:
                    # Small oscillating movement to show the robot is alive
                    t = (step_count - 100) * 0.1
                    action_val = 0.1 * torch.sin(torch.tensor(t))
                    actions = torch.full_like(env.action_manager.action, action_val)
                
                obs, rew, terminated, truncated, info = env.step(actions)
                step_count += 1
                
                # Print status every 100 steps
                if step_count % 100 == 0:
                    print(f"Simulation running... step {step_count}")
                    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    # Clean up
    print("Cleaning up...")
    modifier.revert_modifications()
    env.close()
    
    print("\n" + "="*60)
    print("TEST COMPLETED!")
    print("="*60)
    
    return True


def main():
    """Main function."""
    try:
        success = test_morphology_modification()
        if not success:
            print("TEST FAILED!")
            exit(1)
    except Exception as e:
        print(f"TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main() 