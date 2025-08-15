# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script runs multiple training experiments with RSL-RL using different parameters.
It uses subprocess to run each training session in a separate Python process to avoid 
simulator conflicts and enable parallel hyperparameter sweeps.

Usage:
    python multi_run_training.py --task Isaac-Ballu-Indirect-Act-v0 --seeds 42 123 456 --max_iterations 1000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import subprocess
import sys
import time
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# TensorBoard support
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        # Fallback to tensorflow if tensorboard not available
        import tensorflow as tf
        from tensorflow.python.summary.summary_iterator import summary_iterator
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: TensorBoard/TensorFlow not available. Will use fallback text file parsing.")
        TENSORBOARD_AVAILABLE = False

def run_single_training_subprocess(
    task: str,
    seed: int, 
    max_iterations: int,
    num_envs: int,
    experiment_id: int,
    output_dir: str,
    common_folder: str,
    headless: bool = True,
    world_frame: bool = False,
    balloon_buoyancy_mass: float = 0.24,
    additional_args: List[str] = None
) -> Dict[str, Any]:
    """
    Run a single training experiment in a subprocess.
    
    Args:
        task: Task name for training
        seed: Random seed for the experiment
        max_iterations: Maximum training iterations
        num_envs: Number of environments
        experiment_id: Unique experiment identifier
        output_dir: Directory to save results
        common_folder: Common folder name for this experiment batch
        headless: Whether to run in headless mode
        world_frame: Whether to use world frame for velocity tracking
        additional_args: Additional command line arguments
        
    Returns:
        Dictionary containing experiment results
    """
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING EXPERIMENT {experiment_id}: Task={task}, Seed={seed}, Iterations={max_iterations}")
    print(f"{'='*70}")
    
    # Construct command to run training
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "scripts/rsl_rl/train.py",
        "--task", task,
        "--seed", str(seed),
        "--max_iterations", str(max_iterations),
        "--num_envs", str(num_envs),
        "--common_folder", common_folder,
        "--balloon_buoyancy_mass", str(balloon_buoyancy_mass)
    ]
    
    if headless:
        cmd.append("--headless")
    
    if world_frame:
        cmd.append("--world")
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    # Create metadata for this experiment
    experiment_metadata = {
        'experiment_id': experiment_id,
        'task': task,
        'seed': seed,
        'max_iterations': max_iterations,
        'num_envs': num_envs,
        'world_frame': world_frame,
        'start_time': time.time(),
        'command': ' '.join(cmd),
        'status': 'running'
    }
    
    try:
        # Run the training in a subprocess with real-time output streaming
        print(f"Running command: {' '.join(cmd)}")
        print(f"{'='*50} TRAINING OUTPUT START {'='*50}")
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        
        # Capture output while streaming it live
        captured_output = []
        
        try:
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')  # Print to console in real-time
                captured_output.append(line)
            
            # Wait for process to complete
            process.wait(timeout=3600 * 4)  # 4 hour timeout
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(cmd, 3600 * 4)
        
        print(f"{'='*50} TRAINING OUTPUT END {'='*52}")
        
        if process.returncode == 0:
            print("✅ Training subprocess completed successfully!")
            
            # Update metadata
            experiment_metadata.update({
                'end_time': time.time(),
                'status': 'completed',
                'duration_minutes': (time.time() - experiment_metadata['start_time']) / 60.0,
                'captured_output_lines': len(captured_output)
            })
            
            # Try to extract training results from logs
            try:
                training_results = extract_training_results(common_folder, task, seed)
                experiment_metadata.update(training_results)
                print(f"📊 Extracted training results: Final reward mean: {training_results.get('final_reward_mean', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Warning: Could not extract training results: {e}")
                experiment_metadata['extraction_error'] = str(e)
            
            return experiment_metadata
            
        else:
            print(f"❌ Training subprocess failed with return code: {process.returncode}")
            # Show last part of captured output for debugging
            if captured_output:
                print("Last output lines:")
                for line in captured_output[-10:]:
                    print(f"  {line.rstrip()}")
            
            return {
                'experiment_id': experiment_id,
                'task': task,
                'seed': seed,
                'error': f"Training failed with return code {process.returncode}",
                'status': 'failed',
                'end_time': time.time()
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Training experiment {experiment_id} timed out after 4 hours")
        return {
            'experiment_id': experiment_id,
            'task': task,
            'seed': seed,
            'error': 'Training timed out',
            'status': 'timeout',
            'end_time': time.time()
        }
    except Exception as e:
        print(f"Error running training subprocess for experiment {experiment_id}: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'task': task,
            'seed': seed,
            'error': str(e),
            'status': 'subprocess_error',
            'end_time': time.time()
        }

def extract_tensorboard_data(log_dir: str) -> Dict[str, Any]:
    """
    Extract training metrics from TensorBoard event files.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        Dictionary containing extracted metrics from TensorBoard
    """
    if not TENSORBOARD_AVAILABLE:
        return {'tensorboard_error': 'TensorBoard not available'}
    
    results = {}
    
    # Find TensorBoard event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        return {'tensorboard_error': 'No TensorBoard event files found'}
    
    try:
        # Use EventAccumulator to read the events
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Get available scalar tags
        scalar_tags = ea.Tags()['scalars']
        print(f"📊 Found TensorBoard scalars: {scalar_tags}")
        
        # Extract user-specified RL training metrics
        metrics_mapping = {
            'Loss/entropy': 'entropy_loss',
            'Loss/learning_rate': 'learning_rate',
            'Loss/mirror_symmetry': 'mirror_symmetry_loss',
            'Loss/surrogate': 'surrogate_loss',
            'Loss/value_function': 'value_function_loss',
            'Policy/mean_noise_std': 'policy_noise_std',
            'Train/mean_reward': 'mean_reward',
        }
        
        extracted_metrics = {}
        
        for tb_tag, metric_name in metrics_mapping.items():
            if tb_tag in scalar_tags:
                try:
                    scalar_events = ea.Scalars(tb_tag)
                    if scalar_events:
                        steps = [event.step for event in scalar_events]
                        values = [event.value for event in scalar_events]
                        
                        extracted_metrics[f'{metric_name}_progression'] = values
                        extracted_metrics[f'{metric_name}_steps'] = steps
                        extracted_metrics[f'final_{metric_name}'] = values[-1] if values else None
                        extracted_metrics[f'initial_{metric_name}'] = values[0] if values else None
                        
                except Exception as e:
                    extracted_metrics[f'{metric_name}_error'] = str(e)
        
        # Calculate training summary statistics
        if 'mean_reward_progression' in extracted_metrics:
            rewards = extracted_metrics['mean_reward_progression']
            if len(rewards) > 0:
                extracted_metrics['final_reward_mean'] = rewards[-1]
                extracted_metrics['max_reward_achieved'] = max(rewards)
                extracted_metrics['training_iterations'] = len(rewards)
                extracted_metrics['reward_progression'] = rewards
                
                # Convergence analysis
                if len(rewards) > 10:
                    last_20_percent = int(len(rewards) * 0.2)
                    if last_20_percent > 0:
                        final_rewards = rewards[-last_20_percent:]
                        extracted_metrics['convergence_stability'] = np.std(final_rewards)
                        extracted_metrics['improvement_rate'] = (rewards[-1] - rewards[0]) / len(rewards)
        
        # Training loss metrics
        if 'value_function_loss_progression' in extracted_metrics and extracted_metrics['value_function_loss_progression']:
            value_losses = extracted_metrics['value_function_loss_progression']
            extracted_metrics['final_value_function_loss'] = value_losses[-1]
            extracted_metrics['initial_value_function_loss'] = value_losses[0]
            
        if 'surrogate_loss_progression' in extracted_metrics and extracted_metrics['surrogate_loss_progression']:
            surrogate_losses = extracted_metrics['surrogate_loss_progression']
            extracted_metrics['final_surrogate_loss'] = surrogate_losses[-1]
            extracted_metrics['initial_surrogate_loss'] = surrogate_losses[0]
            
        if 'entropy_loss_progression' in extracted_metrics and extracted_metrics['entropy_loss_progression']:
            entropy_values = extracted_metrics['entropy_loss_progression']
            extracted_metrics['final_entropy_loss'] = entropy_values[-1]
            extracted_metrics['initial_entropy_loss'] = entropy_values[0]
        
        results['tensorboard_data'] = extracted_metrics
        results['tensorboard_source'] = 'EventAccumulator'
        
    except Exception as e:
        results['tensorboard_error'] = f"Error reading TensorBoard data: {str(e)}"
        
        # Fallback: try to read using tensorflow summary_iterator
        try:
            if 'tf' in globals():
                event_file = event_files[0]  # Use the first event file
                metrics = {}
                
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    if event.summary:
                        for value in event.summary.value:
                            tag = value.tag
                            if tag not in metrics:
                                metrics[tag] = {'steps': [], 'values': []}
                            metrics[tag]['steps'].append(event.step)
                            metrics[tag]['values'].append(value.simple_value)
                
                if metrics:
                    results['tensorboard_data'] = metrics
                    results['tensorboard_source'] = 'tensorflow_summary_iterator'
                    
                    # Extract final values for user-specified metrics
                    if 'Train/mean_reward' in metrics:
                        results['final_reward_mean'] = metrics['Train/mean_reward']['values'][-1]
                        results['reward_progression'] = metrics['Train/mean_reward']['values']
                        results['training_iterations'] = len(metrics['Train/mean_reward']['values'])
                        
        except Exception as tf_error:
            results['tensorboard_fallback_error'] = str(tf_error)
    
    return results

def extract_training_results(common_folder: str, task: str, seed: int) -> Dict[str, Any]:
    """
    Extract training results from RSL-RL log files and TensorBoard events.
    
    Args:
        common_folder: Common folder name for the experiment batch
        task: Task name
        seed: Random seed
        
    Returns:
        Dictionary containing extracted training metrics
    """
    # Find the log directory for this specific run
    log_pattern = f"logs/rsl_rl/*/{common_folder}/seed_{seed}"
    log_dirs = glob.glob(log_pattern)
    
    if not log_dirs:
        raise FileNotFoundError(f"No log directory found for pattern: {log_pattern}")
    
    log_dir = log_dirs[0]  # Take the first match
    
    results = {}
    
    # Primary: Try to extract from TensorBoard event files
    print(f"🔍 Extracting TensorBoard data from: {log_dir}")
    tensorboard_results = extract_tensorboard_data(log_dir)
    results.update(tensorboard_results)
    
    # Fallback: Try to read training progress from summaries.txt
    summaries_file = os.path.join(log_dir, "summaries.txt")
    if os.path.exists(summaries_file) and 'final_reward_mean' not in results:
        print(f"📄 Fallback: Reading summaries.txt")
        try:
            # Read the summaries file
            df = pd.read_csv(summaries_file, sep='\t')
            
            if len(df) > 0:
                fallback_data = {
                    'final_iteration': df['Iteration'].iloc[-1] if 'Iteration' in df.columns else None,
                    'final_reward_mean': df['Reward/mean'].iloc[-1] if 'Reward/mean' in df.columns else None,
                    'final_reward_std': df['Reward/std'].iloc[-1] if 'Reward/std' in df.columns else None,
                    'final_episode_length': df['Episode Length/mean'].iloc[-1] if 'Episode Length/mean' in df.columns else None,
                    'training_iterations': len(df),
                    'reward_progression': df['Reward/mean'].tolist() if 'Reward/mean' in df.columns else [],
                    'episode_length_progression': df['Episode Length/mean'].tolist() if 'Episode Length/mean' in df.columns else []
                }
                
                # Calculate convergence metrics
                if 'Reward/mean' in df.columns and len(df) > 10:
                    reward_series = df['Reward/mean']
                    # Simple convergence check: stability over last 20% of training
                    last_20_percent = int(len(reward_series) * 0.2)
                    if last_20_percent > 0:
                        final_rewards = reward_series.iloc[-last_20_percent:]
                        fallback_data['convergence_stability'] = final_rewards.std()
                        fallback_data['improvement_rate'] = (reward_series.iloc[-1] - reward_series.iloc[0]) / len(reward_series)
                
                results['summaries_data'] = fallback_data
                # If no TensorBoard data, promote summaries data to main level
                if 'final_reward_mean' not in results:
                    results.update(fallback_data)
        
        except Exception as e:
            results['summaries_read_error'] = str(e)
    
    # Additional: Try to read other log files if available
    progress_file = os.path.join(log_dir, "progress.csv")
    if os.path.exists(progress_file):
        try:
            df_progress = pd.read_csv(progress_file)
            if len(df_progress) > 0:
                results['progress_file_data'] = True
        except Exception as e:
            results['progress_read_error'] = str(e)
    
    return results

def save_combined_results(all_results: List[Dict[str, Any]], output_dir: str, common_folder: str):
    """Save and analyze combined training results."""
    print(f"\nAnalyzing results from {len(all_results)} training experiments...")
    
    # Save raw results
    results_file = os.path.join(output_dir, "training_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Raw results saved to {results_file}")
    
    # Filter successful experiments
    successful_results = [r for r in all_results if r.get('status') == 'completed' and 'final_reward_mean' in r]
    
    if len(successful_results) == 0:
        print("No successful experiments with reward data found.")
        return
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'Training Results Analysis - {common_folder}', fontsize=16)
    
    # Plot 1: Final rewards by seed
    seeds = [r['seed'] for r in successful_results]
    final_rewards = [r['final_reward_mean'] for r in successful_results]
    
    axes[0, 0].bar(range(len(seeds)), final_rewards, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('Experiment (Seed)')
    axes[0, 0].set_ylabel('Final Reward Mean')
    axes[0, 0].set_title('Final Reward by Seed')
    axes[0, 0].set_xticks(range(len(seeds)))
    axes[0, 0].set_xticklabels([f"Seed {s}" for s in seeds], rotation=45)
    
    # Add value labels on bars
    for i, reward in enumerate(final_rewards):
        axes[0, 0].text(i, reward + max(final_rewards) * 0.01, f'{reward:.2f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Training convergence curves
    axes[0, 1].set_title('Training Convergence Curves')
    for i, result in enumerate(successful_results):
        if 'reward_progression' in result and result['reward_progression']:
            progression = result['reward_progression']
            axes[0, 1].plot(progression, label=f"Seed {result['seed']}", alpha=0.7)
    
    axes[0, 1].set_xlabel('Training Iteration')
    axes[0, 1].set_ylabel('Reward Mean')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy Evolution
    axes[1, 0].set_title('Entropy Evolution')
    entropy_found = False
    for i, result in enumerate(successful_results):
        if 'final_entropy' in result and result.get('entropy_progression'):
            progression = result['entropy_progression']
            axes[1, 0].plot(progression, label=f"Seed {result['seed']}", alpha=0.7)
            entropy_found = True
    
    if entropy_found:
        axes[1, 0].set_xlabel('Training Iteration')
        axes[1, 0].set_ylabel('Loss/entropy')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Entropy Data\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    
    # Plot 4: Surrogate Loss Evolution
    axes[1, 1].set_title('Surrogate Loss Evolution')
    surrogate_found = False
    for i, result in enumerate(successful_results):
        if 'final_surrogate' in result and result.get('surrogate_progression'):
            progression = result['surrogate_progression']
            axes[1, 1].plot(progression, label=f"Seed {result['seed']}", alpha=0.7)
            surrogate_found = True
    
    if surrogate_found:
        axes[1, 1].set_xlabel('Training Iteration')
        axes[1, 1].set_ylabel('Loss/surrogate')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Surrogate Loss Data\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    # Plot 5: Value Function Loss Evolution
    axes[2, 0].set_title('Value Function Loss Evolution')
    value_function_found = False
    for i, result in enumerate(successful_results):
        if 'final_value_function' in result and result.get('value_function_progression'):
            progression = result['value_function_progression']
            axes[2, 0].plot(progression, label=f"Seed {result['seed']}", alpha=0.7)
            value_function_found = True
    
    if value_function_found:
        axes[2, 0].set_xlabel('Training Iteration')
        axes[2, 0].set_ylabel('Loss/value_function')
        axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'Value Function Data\nNot Available', 
                       ha='center', va='center', transform=axes[2, 0].transAxes, fontsize=12)
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
    
    # Plot 6: Learning Rate and Mirror Symmetry
    axes[2, 1].set_title('Learning Rate Evolution')
    lr_found = False
    for i, result in enumerate(successful_results):
        if 'final_learning_rate' in result and result.get('learning_rate_progression'):
            progression = result['learning_rate_progression']
            axes[2, 1].plot(progression, label=f"Seed {result['seed']}", alpha=0.7)
            lr_found = True
    
    if lr_found:
        axes[2, 1].set_xlabel('Training Iteration')
        axes[2, 1].set_ylabel('Loss/learning_rate')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'Learning Rate Data\nNot Available', 
                       ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=12)
        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "training_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comprehensive analysis plots saved to {plot_file}")
    
    # Print comprehensive summary statistics
    print(f"\n{'='*90}")
    print("COMPREHENSIVE TRAINING RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Seed':<6} {'Reward':<10} {'Duration':<10} {'Iter':<6} {'Surrogate':<12} {'Value Func':<12} {'Status':<10}")
    print("-" * 90)
    
    for r in all_results:
        seed = r.get('seed', 'N/A')
        reward = f"{r.get('final_reward_mean', 0):.3f}" if r.get('final_reward_mean') else 'N/A'
        duration = f"{r.get('duration_minutes', 0):.1f}m" if r.get('duration_minutes') else 'N/A'
        iterations = r.get('training_iterations', r.get('final_iteration', 'N/A'))
        surrogate_loss = f"{r.get('final_surrogate', 0):.4f}" if r.get('final_surrogate') else 'N/A'
        value_func_loss = f"{r.get('final_value_function', 0):.4f}" if r.get('final_value_function') else 'N/A'
        status = r.get('status', 'unknown')
        
        print(f"{seed:<6} {reward:<10} {duration:<10} {iterations:<6} {surrogate_loss:<12} {value_func_loss:<12} {status:<10}")
    
    if successful_results:
        rewards = [r['final_reward_mean'] for r in successful_results]
        
        print(f"\n🎯 Performance Statistics:")
        print(f"   Mean Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"   Best Reward: {np.max(rewards):.3f} (Seed {successful_results[np.argmax(rewards)]['seed']})")
        print(f"   Worst Reward: {np.min(rewards):.3f} (Seed {successful_results[np.argmin(rewards)]['seed']})")
        print(f"   Success Rate: {len(successful_results)}/{len(all_results)} ({100*len(successful_results)/len(all_results):.1f}%)")
        
        # Additional TensorBoard-based statistics
        surrogate_losses = [r.get('final_surrogate') for r in successful_results if r.get('final_surrogate') is not None]
        value_function_losses = [r.get('final_value_function') for r in successful_results if r.get('final_value_function') is not None]
        entropy_values = [r.get('final_entropy') for r in successful_results if r.get('final_entropy') is not None]
        noise_std_values = [r.get('final_noise_std') for r in successful_results if r.get('final_noise_std') is not None]
        
        if surrogate_losses:
            print(f"\n🔍 Training Loss Analysis:")
            print(f"   Surrogate Loss: {np.mean(surrogate_losses):.4f} ± {np.std(surrogate_losses):.4f}")
            print(f"   Best (lowest) Surrogate Loss: {np.min(surrogate_losses):.4f}")
            
        if value_function_losses:
            print(f"   Value Function Loss: {np.mean(value_function_losses):.4f} ± {np.std(value_function_losses):.4f}")
            print(f"   Best (lowest) Value Function Loss: {np.min(value_function_losses):.4f}")
            
        if entropy_values:
            print(f"   Entropy: {np.mean(entropy_values):.4f} ± {np.std(entropy_values):.4f}")
            
        if noise_std_values:
            print(f"   Policy Noise Std: {np.mean(noise_std_values):.4f} ± {np.std(noise_std_values):.4f}")
        
        # Convergence analysis
        convergence_data = [r.get('convergence_stability') for r in successful_results if r.get('convergence_stability') is not None]
        if convergence_data:
            print(f"\n📈 Convergence Analysis:")
            print(f"   Mean Stability (lower=better): {np.mean(convergence_data):.4f}")
            print(f"   Most Stable Run: {np.min(convergence_data):.4f}")
        
        # Data source information
        tensorboard_count = sum(1 for r in successful_results if 'tensorboard_data' in r or 'tensorboard_source' in r)
        summaries_count = sum(1 for r in successful_results if 'summaries_data' in r)
        
        print(f"\n📊 Data Sources:")
        print(f"   TensorBoard data: {tensorboard_count}/{len(successful_results)} experiments")
        print(f"   Summaries.txt data: {summaries_count}/{len(successful_results)} experiments")

def main():
    """Main function to run multiple training experiments."""
    parser = argparse.ArgumentParser(description="Run multiple RSL-RL training experiments")
    parser.add_argument("--task", type=str, default="Isc-Vel-BALLU-encoder",
                       help="Task name for training")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456, 789, 999],
                       help="List of random seeds to test")
    parser.add_argument("--max_iterations", type=int, default=6000,
                       help="Maximum training iterations per experiment")
    parser.add_argument("--num_envs", type=int, default=4096,
                       help="Number of environments for training")
    parser.add_argument("--output_dir", type=str, 
                       default=f"multi_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for results")
    parser.add_argument("--common_folder", type=str,
                       default=f"multi_run_test",
                       help="Common folder name for all training runs")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run training in headless mode")
    parser.add_argument("--world", action="store_true", default=False,
                       help="Use world frame for velocity tracking reward")
    parser.add_argument("--balloon_buoyancy_masses", type=float, nargs='+', 
                        default=[0.19,
                                 0.20,
                                 0.21,
                                 0.22,
                                 0.23, 
                                 0.24, 
                                 0.25, 
                                 0.26, 
                                 0.27, 
                                 0.28,  
                                 0.29],
                       help="List of buoyancy masses to test")
    parser.add_argument("--additional_args", type=str, nargs='*', default=[],
                       help="Additional arguments to pass to train.py")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running {len(args.seeds)} training experiments")
    print(f"Task: {args.task}")
    print(f"Seeds: {args.seeds}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Environments: {args.num_envs}")
    print(f"Common folder: {args.common_folder}")
    print(f"Results will be saved to: {args.output_dir}")
    
    all_results = []
    # Add the timestamp to common_folder before launching the experiments
    args.common_folder = f"{datetime.now().strftime('%m%d_%H%M%S')}_{args.common_folder}"
    # Hardcoding the folder to store aggregate results
    args.output_dir = os.path.join("logs/rsl_rl/lab_7.29.2025", args.common_folder)
    # Run experiments sequentially
    #for i, seed in enumerate(args.seeds):
    for i, balloon_buoyancy_mass in enumerate(args.balloon_buoyancy_masses):
        try:
            # TODO: Restore to common_folder passed through CLI
            args.common_folder = f"07_27_12_55_09_buoyMass_{balloon_buoyancy_mass}"
            result = run_single_training_subprocess(
                task=args.task,
                seed=2, # TODO: Restore to seeds passed through CLI
                max_iterations=args.max_iterations,
                num_envs=args.num_envs,
                experiment_id=i+1,
                output_dir=args.output_dir,
                common_folder=args.common_folder,
                headless=args.headless,
                world_frame=args.world,
                balloon_buoyancy_mass=balloon_buoyancy_mass,
                additional_args=args.additional_args
            )
            all_results.append(result)
            
            status = result.get('status', 'unknown')
            status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏰" if status == "timeout" else "❓"
            duration = result.get('duration_minutes', 0)
            print(f"{status_emoji} Experiment {i+1} status: {status} (Duration: {duration:.1f} min)")
            
            # Save intermediate results
            # temp_results_file = os.path.join(args.output_dir, "partial_results.json")
            # with open(temp_results_file, 'w') as f:
            #    json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"Critical error in experiment {i+1}: {str(e)}")
            all_results.append({
                'experiment_id': i+1,
                'seed': 2,#seed,
                'error': str(e),
                'status': 'critical_failure'
            })
    
    # Analyze and save final results
    save_combined_results(all_results, args.output_dir, args.common_folder)
    print(f"\nAll training experiments completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 