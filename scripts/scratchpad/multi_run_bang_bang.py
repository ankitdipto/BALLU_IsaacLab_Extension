# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script runs multiple experiments with the stepper policy using different parameters.
It uses subprocess to run each experiment in a separate Python process to avoid 
Isaac Sim restart issues.

Usage:
    python multi_run_stepper_v2.py --num_envs 1 --max_steps 300
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

def run_single_experiment_subprocess(period: int, num_envs: int, max_steps: int, 
                                    experiment_id: int, output_dir: str, 
                                    headless: bool = True) -> Dict[str, Any]:
    """
    Run a single experiment in a subprocess.
    
    Args:
        period: Stepper period parameter
        num_envs: Number of environments
        max_steps: Maximum simulation steps
        experiment_id: Unique experiment identifier
        output_dir: Directory to save results
        headless: Whether to run in headless mode
        
    Returns:
        Dictionary containing experiment results
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_id}: Running stepper with period={period}")
    print(f"{'='*60}")
    
    # Create output file for this experiment
    output_file = os.path.join(output_dir, f"experiment_{experiment_id:02d}.json")
    
    # Construct command to run single experiment
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "single_experiment.py",
        "--period", str(period),
        "--num_envs", str(num_envs),
        "--max_steps", str(max_steps),
        "--experiment_id", str(experiment_id),
        "--output_file", output_file
    ]
    
    if headless:
        cmd.append("--headless")
    
    try:
        # Run the experiment in a subprocess
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per experiment (reduced)
            cwd=os.path.dirname(os.path.abspath(__file__))  # Run in same directory
        )
        
        if result.returncode == 0:
            print("Subprocess completed successfully!")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            
            # Load results from file
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    experiment_results = json.load(f)
                print(f"Loaded results: {experiment_results.get('status', 'unknown')} - "
                      f"Reward: {experiment_results.get('total_reward', 'N/A')}")
                return experiment_results
            else:
                print(f"Output file not found: {output_file}")
                return {
                    'experiment_id': experiment_id,
                    'period': period,
                    'error': 'Output file not created',
                    'status': 'failed'
                }
        else:
            print(f"Subprocess failed with return code: {result.returncode}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return {
                'experiment_id': experiment_id,
                'period': period,
                'error': f"Subprocess failed: {result.stderr}",
                'status': 'failed'
            }
            
    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_id} timed out after 2 minutes")
        return {
            'experiment_id': experiment_id,
            'period': period,
            'error': 'Experiment timed out',
            'status': 'timeout'
        }
    except Exception as e:
        print(f"Error running subprocess for experiment {experiment_id}: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'period': period,
            'error': str(e),
            'status': 'subprocess_error'
        }

def save_combined_results(output_dir: str):
    """Combine individual experiment results and create visualizations."""
    print(f"\nCombining results from {output_dir}...")
    
    # Load all individual experiment results
    all_results = []
    experiment_files = [f for f in os.listdir(output_dir) if f.startswith("experiment_") and f.endswith(".json")]
    experiment_files.sort()
    
    for exp_file in experiment_files:
        try:
            with open(os.path.join(output_dir, exp_file), 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {exp_file}: {e}")
    
    # Save combined results
    combined_file = os.path.join(output_dir, "combined_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Combined results saved to {combined_file}")
    
    # Create summary plot
    successful_results = [r for r in all_results if r.get('status') == 'completed']
    
    if successful_results:
        periods = [r['period'] for r in successful_results]
        rewards = [r['total_reward'] for r in successful_results]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Reward vs Period
        plt.subplot(1, 2, 1)
        plt.plot(periods, rewards, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Stepper Period')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Stepper Period')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on points
        for period, reward in zip(periods, rewards):
            plt.annotate(f'{reward:.1f}', (period, reward), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Base velocity magnitude
        plt.subplot(1, 2, 2)
        base_vels = []
        for r in successful_results:
            if r.get('final_base_vel_mean'):
                # Calculate magnitude of base velocity
                vel_array = np.array(r['final_base_vel_mean'])
                mean_vel_magnitude = np.linalg.norm(vel_array.mean(axis=0))
                base_vels.append(mean_vel_magnitude)
            else:
                base_vels.append(0)
        
        plt.plot(periods, base_vels, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Stepper Period')
        plt.ylabel('Base Velocity Magnitude')
        plt.title('Base Velocity vs Stepper Period')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on points
        for period, vel in zip(periods, base_vels):
            plt.annotate(f'{vel:.3f}', (period, vel), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "experiment_summary.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to {plot_file}")
        
        # Print summary table
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Period':<8} {'Reward':<10} {'Base Vel':<12} {'Status':<10}")
        print("-" * 50)
        
        for r in all_results:
            period = r.get('period', 'N/A')
            reward = f"{r.get('total_reward', 0):.3f}" if r.get('total_reward') else 'N/A'
            status = r.get('status', 'unknown')
            
            if r.get('final_base_vel_mean'):
                vel_array = np.array(r['final_base_vel_mean'])
                vel_mag = np.linalg.norm(vel_array.mean(axis=0))
                base_vel = f"{vel_mag:.3f}"
            else:
                base_vel = 'N/A'
            
            print(f"{period:<8} {reward:<10} {base_vel:<12} {status:<10}")
        
        if successful_results:
            best = max(successful_results, key=lambda x: x['total_reward'])
            print(f"\n🏆 Best Performance: Period {best['period']} with reward {best['total_reward']:.3f}")

def main():
    """Main function to run multiple stepper experiments using subprocesses."""
    parser = argparse.ArgumentParser(description="Run multiple stepper experiments using subprocesses")
    parser.add_argument("--num_envs", type=int, default=1, 
                       help="Number of environments to spawn")
    parser.add_argument("--max_steps", type=int, default=300,
                       help="Maximum steps per experiment")
    parser.add_argument("--output_dir", type=str, 
                       default=f"stepper_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for results")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run experiments in headless mode")
    parser.add_argument("--periods", type=int, nargs='+', 
                       default=[20, 40, 60, 80, 100, 120, 150, 200, 300, 500],
                       help="List of stepper periods to test")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running {len(args.periods)} experiments with periods: {args.periods}")
    print(f"Each experiment: {args.num_envs} envs, {args.max_steps} steps")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run experiments sequentially using subprocesses
    for i, period in enumerate(args.periods):
        try:
            result = run_single_experiment_subprocess(
                period=period,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                experiment_id=i+1,
                output_dir=args.output_dir,
                headless=args.headless
            )
            
            print(f"Experiment {i+1} status: {result.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"Critical error in experiment {i+1}: {str(e)}")
    
    # Combine and analyze results
    save_combined_results(args.output_dir)
    print(f"\nAll experiments completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 