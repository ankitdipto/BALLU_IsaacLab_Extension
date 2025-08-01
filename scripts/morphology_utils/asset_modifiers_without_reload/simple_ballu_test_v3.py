#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
BALLU morphology test with manual physics stepping.

This script applies morphology modification and handles physics issues gracefully.
Usage:
    python simple_ballu_test_v3.py --livestream 2 --tibia_scale 1.5
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="BALLU test with robust morphology modification")
parser.add_argument("--tibia_scale", type=float, default=1.5, help="Tibia scale factor")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import time
import omni

from BALLU_IsaacLab_Extension.scripts.morphology_utils.asset_modifiers_without_reload.ballu_morphology_modifier import BalluMorphologyModifier

# Import environment
import ballu_isaac_extension  # noqa: F401
from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

def safe_step_environment(env, actions, max_retries=3):
    """Safely step the environment with retry logic."""
    for attempt in range(max_retries):
        try:
            return env.step(actions)
        except Exception as e:
            print(f"⚠️  Step failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("❌ Max retries reached, stopping simulation stepping")
                return None
            time.sleep(0.1)  # Brief pause before retry
    
def main():
    """Main function."""
    
    print("="*50)
    print("🤖 BALLU Morphology Test (Robust Version)")
    print("="*50)
    print(f"📏 Tibia scale factor: {args.tibia_scale}x")
    print("⏱️  Extended observation period: 5 minutes")
    print("="*50)
    
    # Create environment configuration
    cfg = BalluIndirectActEnvCfg()
    cfg.scene.num_envs = 1  # Single environment for easier observation
    cfg.sim.dt = 1/60  # Slower simulation for better observation
    cfg.decimation = 2  # Every 2 simulation steps
    cfg.episode_length_s = 300.0  # 5 minutes = 300 seconds
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=cfg)
    
    # Initialize the environment
    print("🚀 Initializing environment...")
    env.reset()
    print("✅ Environment initialized successfully!")
    
    # Wait a few steps for things to settle
    print("⏳ Allowing environment to settle...")
    for _ in range(10):
        try:
            env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        except:
            break
    
    # Apply morphology modification
    print("🔧 Applying morphology modification...")
    modifier = BalluMorphologyModifier()
    
    try:
        success = modifier.scale_tibia_links(env, scale_factor=args.tibia_scale)
        if success:
            print(f"✅ TIBIA SCALING SUCCESSFUL! Scaled by {args.tibia_scale}x")
            print("📺 The tibia (shin) links should now be visibly longer")
        else:
            print("⚠️  Partial scaling applied - some issues occurred")
    except Exception as e:
        print(f"❌ Morphology modification failed: {e}")
        print("🔄 Continuing with original morphology for observation...")
    
    # Set camera position for better viewing
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5])
        print("📷 Camera positioned for optimal viewing")
    except Exception as e:
        print(f"⚠️  Could not set camera position: {e}")
    
    print("\n" + "="*60)
    print("🎯 MORPHOLOGY OBSERVATION MODE")
    print("="*60)
    print("📱 Isaac Sim GUI should now be accessible via livestream")
    print("👀 Examine the BALLU robot carefully:")
    print("   • Compare leg proportions")
    print("   • Look for 1.5x longer tibia (shin) bones")
    print("   • Notice the scaling applied to both left and right legs")
    print("")
    print("⏰ This observation period will last 5 minutes")
    print("🛑 Press Ctrl+C to stop early if needed")
    print("="*60)
    
    # Extended observation period with minimal physics stepping
    start_time = time.time()
    target_duration = cfg.episode_length_s
    step_count = 0
    last_step_time = start_time
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= target_duration:
                print(f"\n⏰ {target_duration} seconds completed!")
                break
            
            # Very gentle stepping - only every 2 seconds to minimize physics issues
            if current_time - last_step_time >= 2.0:
                # Create minimal action (tiny oscillation)
                t = elapsed * 0.1  # Very slow time progression
                gentle_action = 0.05 * torch.sin(torch.tensor(t))  # Very small oscillation
                actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                if env.action_manager.total_action_dim >= 2:
                    actions[:, :2] = gentle_action
                
                # Try to step safely
                result = safe_step_environment(env, actions)
                if result is None:
                    print("🔄 Continuing in observation-only mode (no physics stepping)")
                    # Just wait without physics stepping
                    time.sleep(2.0)
                else:
                    step_count += 1
                
                last_step_time = current_time
            
            # Progress indicator every 30 seconds
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                remaining = target_duration - elapsed
                print(f"⏰ Time: {elapsed:.0f}s | Remaining: {remaining:.0f}s | Steps: {step_count}")
            
            # Small sleep to prevent busy waiting
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Observation stopped by user")
    
    elapsed_final = time.time() - start_time
    print(f"\n✅ Observation completed! Total time: {elapsed_final:.1f} seconds")
    print(f"🔢 Physics steps executed: {step_count}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 MORPHOLOGY TEST SUMMARY")
    print("="*50)
    print(f"Target scale factor: {args.tibia_scale}x")
    print(f"Observation duration: {elapsed_final:.1f} seconds")
    print("Please provide feedback on:")
    print("• Did you observe longer tibia links?")
    print("• Were the leg proportions visibly different?")
    print("• Any issues with the robot appearance?")
    print("="*50)
    
    # Cleanup
    try:
        env.close()
        print("🧹 Environment cleaned up successfully")
    except:
        print("⚠️  Environment cleanup had minor issues")

if __name__ == "__main__":
    main()
    simulation_app.close() 