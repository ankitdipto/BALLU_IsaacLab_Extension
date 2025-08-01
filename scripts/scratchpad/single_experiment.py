# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Single experiment runner for stepper policy testing.
This script is designed to be called as a subprocess by multi_run_stepper.py
"""

import argparse
import numpy as np
import json
import sys
import time
import torch

def main():
    """Run a single stepper experiment."""
    parser = argparse.ArgumentParser(description="Single stepper experiment")
    parser.add_argument("--period", type=int, required=True, help="Stepper period")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps")
    parser.add_argument("--experiment_id", type=int, required=True, help="Experiment ID")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    print(f"Starting experiment {args.experiment_id} with period {args.period}")
    
    try:
        # Import Isaac Lab modules
        from isaaclab.app import AppLauncher
        
        # Create argument parser for AppLauncher
        app_parser = argparse.ArgumentParser(description=f"Experiment {args.experiment_id}")
        app_parser.add_argument("--num_envs", type=int, default=args.num_envs)
        app_parser.add_argument("--max_steps", type=int, default=args.max_steps)
        
        # Add AppLauncher specific arguments
        AppLauncher.add_app_launcher_args(app_parser)
        
        # Create arguments for this experiment
        app_args = [
            "--num_envs", str(args.num_envs),
            "--max_steps", str(args.max_steps)
        ]
        
        if args.headless:
            app_args.append("--headless")
        
        # Parse arguments
        app_cli_args = app_parser.parse_args(app_args)
        
        # Initialize simulator
        app_launcher = AppLauncher(app_cli_args)
        simulation_app = app_launcher.app
        
        # Import environment modules after simulator initialization
        import torch
        import gymnasium as gym
        from isaaclab.envs import ManagerBasedRLEnv
        from ballu_isaac_extension.tasks.ballu_locomotion.indirect_act_vel_env_cfg import BalluIndirectActEnvCfg
        from action_generators import stepper
        
        # Create environment configuration
        env_cfg = BalluIndirectActEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        
        # Setup RL environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print(f"Environment initialized successfully")
        
        # Initialize data collection
        results = {
            'experiment_id': args.experiment_id,
            'period': args.period,
            'num_envs': args.num_envs,
            'max_steps': args.max_steps,
            'start_time': time.time(),
            'status': 'running'
        }
        
        robots = env.unwrapped.scene["robot"]
        step_count = 0
        cumulative_rewards = torch.zeros(args.num_envs, device="cuda:0")
        base_speed_history = []
        
        print(f"Starting simulation loop for {args.max_steps} steps...")
        
        # Main simulation loop
        while simulation_app.is_running() and step_count < args.max_steps:
            with torch.inference_mode():
                # Generate actions using stepper with current period
                actions = stepper(step_count, period=args.period, num_envs=args.num_envs)
                
                # Step environment
                obs, rew, terminated, truncated, info = env.step(actions)
                
                # Collect data
                base_velocity = robots.data.root_lin_vel_b.clone().detach().cpu()
                base_speed_history.append(base_velocity)
                cumulative_rewards += rew
                
                step_count += 1
                
                # Check for termination
                if terminated.any() or truncated.any():
                    print(f"Environment terminated at step {step_count}")
                    break
                
                # Progress update every 100 steps
                if step_count % 100 == 0:
                    print(f"Step {step_count}/{args.max_steps}, "
                          f"Avg Reward: {cumulative_rewards.mean().item():.3f}")
        
        # Calculate final statistics
        base_speed_history = torch.stack(base_speed_history)
        base_vel_mean = base_speed_history.mean(dim=0)
        base_vel_std = base_speed_history.std(dim=0)
        
        # Store results
        results.update({
            'end_time': time.time(),
            'actual_steps': step_count,
            'total_reward': cumulative_rewards.mean().item(),
            'final_base_vel_mean': base_vel_mean.numpy().tolist(),
            'final_base_vel_std': base_vel_std.numpy().tolist(),
            'rewards': cumulative_rewards.cpu().numpy().tolist(),
            'status': 'completed'
        })
        
        print(f"Experiment {args.experiment_id} completed successfully!")
        print(f"Steps: {step_count}, Total Reward: {results['total_reward']:.3f}")
        
        # Clean up environment first
        env.close()
        
    except Exception as e:
        print(f"Error in experiment {args.experiment_id}: {str(e)}")
        results = {
            'experiment_id': args.experiment_id,
            'period': args.period,
            'error': str(e),
            'status': 'failed'
        }
    
    # Save results to file BEFORE closing simulation app
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")
    
    # Close simulation app last (this may hang, but results are already saved)
    try:
        simulation_app.close()
        print("Simulation app closed successfully")
    except Exception as e:
        print(f"Warning: Error closing simulation app: {e}")
        # Force exit if close hangs
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main() 