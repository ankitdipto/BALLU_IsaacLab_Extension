#!/usr/bin/env python3
"""
W&B Sweep Agent for BALLU Morphology Optimization.

This script is called by W&B sweep to run individual training experiments
with different morphology and buoyancy mass configurations.
"""

import argparse
import os
import sys
import subprocess
import time
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent))

from sweep_utils import (
    get_robot_usd_full_path,
    extract_morphology_name,
    extract_max_reward_from_logs,
    find_log_directory,
    validate_experiment_config,
    create_experiment_summary
)


def convert_ratio_to_usd_path(femur_to_tibia_ratio: float) -> str:
    """
    Convert femur-to-tibia ratio decimal to USD path.
    
    Args:
        femur_to_tibia_ratio: Decimal ratio (e.g., 4.2)
        
    Returns:
        USD path string (e.g., "FT_4_2_5_8/ballu_modified_FT_4_2_5_8.usd")
    """
    # Calculate tibia length (total length is 10.0)
    tibia_length = 10.0 - femur_to_tibia_ratio
    
    # Convert to integer and decimal parts
    femur_int = int(femur_to_tibia_ratio)
    femur_decimal = int(round((femur_to_tibia_ratio - femur_int) * 10))
    
    tibia_int = int(tibia_length)
    tibia_decimal = int(round((tibia_length - tibia_int) * 10))
    
    # Create the path components
    folder_name = f"FT_{femur_int}_{femur_decimal}_{tibia_int}_{tibia_decimal}"
    file_name = f"ballu_modified_FT_{femur_int}_{femur_decimal}_{tibia_int}_{tibia_decimal}.usd"
    
    return f"{folder_name}/{file_name}"


def run_training_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single training experiment with the given configuration.
    
    Args:
        config: Configuration dictionary from W&B containing all hyperparameters
        
    Returns:
        Dictionary containing experiment results
    """

    MASS_ROBOT = 0.289
    # Extract parameters from config
    gravity_comp_ratio = config['gravity_comp_ratio']
    buoyancy_mass = gravity_comp_ratio * MASS_ROBOT #config['buoyancy_mass']
    femur_to_tibia_ratio = config['femur_to_tibia_ratio']
    motor_limit = config['motor_limit']
    task = config['task']
    seed = config['seed']
    max_iterations = config['max_iterations']
    num_envs = config['num_envs']
    
    # Convert decimal ratio to USD path
    robot_morphology = convert_ratio_to_usd_path(femur_to_tibia_ratio)
    
    # Get full USD path
    usd_path = get_robot_usd_full_path(robot_morphology)
    morphology_name = extract_morphology_name(robot_morphology)
    
    # Create unique experiment ID
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    experiment_id = f"wb_sweep_{timestamp}_{morphology_name}_buoy{buoyancy_mass:.3f}_kneeEffLim{motor_limit:.3f}"
    
    print(f"\n🚀 Starting W&B Sweep Experiment")
    print(f"🤖 Morphology: {morphology_name}")
    print(f"🎈 Buoyancy Mass: {buoyancy_mass:.3f}")
    print(f"🎯 Task: {task}")
    print(f"🎲 Seed: {seed}")
    print(f"🔄 Max Iterations: {max_iterations}")
    print(f"🏭 Num Envs: {num_envs}")
    print(f"📁 USD Path: {usd_path}")
    
    # Build training command
    train_script = script_dir.parent / "rsl_rl" / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--task", task,
        "--seed", str(seed),
        "--max_iterations", str(max_iterations),
        "--num_envs", str(num_envs),
        "--balloon_buoyancy_mass", str(buoyancy_mass),
        "--common_folder", experiment_id,
        f"env.scene.robot.actuators.knee_effort_actuators.effort_limit={motor_limit}",
        "--headless"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env['BALLU_MORPHOLOGY_USD_PATH'] = usd_path
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'
    
    # Validate parameters before running
    if not os.path.exists(usd_path):
        error_msg = f"USD file not found: {usd_path}"
        print(f"❌ {error_msg}")
        return {
            'experiment_id': experiment_id,
            'error': error_msg,
            'status': 'validation_failed',
            'max_reward': 0.0,
            'max_mean_reward': 0.0
        }
    
    if not (0.1 <= buoyancy_mass <= 0.5):
        error_msg = f"Buoyancy mass {buoyancy_mass} outside reasonable range [0.1, 0.5]"
        print(f"⚠️  Warning: {error_msg}")
    
    # Track experiment metadata
    experiment_metadata = {
        'experiment_id': experiment_id,
        'morphology_name': morphology_name,
        'usd_path': usd_path,
        'task': task,
        'seed': seed,
        'max_iterations': max_iterations,
        'num_envs': num_envs,
        'buoyancy_mass': buoyancy_mass,
        'start_time': time.time(),
        'command': ' '.join(cmd)
    }
    
    try:
        print(f"\n⚡ Running training command:")
        print(f"   {' '.join(cmd)}")
        print(f"   USD Override: {usd_path}")
        print(f"   Buoyancy Mass: {buoyancy_mass}")
        
        # Run training with timeout (2 hours max)
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor process with real-time output and W&B logging
        captured_output = []
        #last_log_time = time.time()
        # log_interval = 30  # Log to W&B every 30 seconds
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                captured_output.append(output.strip())
                
        #         # Log intermediate progress to W&B
        #         current_time = time.time()
        #         if current_time - last_log_time > log_interval:
        #             elapsed_minutes = (current_time - experiment_metadata['start_time']) / 60.0
        #             wandb.log({
        #                 'training_elapsed_minutes': elapsed_minutes,
        #                 'training_status': 'running'
        #             })
        #             last_log_time = current_time
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Update metadata
            experiment_metadata.update({
                'end_time': time.time(),
                'status': 'completed',
                'duration_minutes': (time.time() - experiment_metadata['start_time']) / 60.0,
                'captured_output_lines': len(captured_output)
            })
            
            # Extract training results
            try:
                max_reward = extract_max_reward_from_logs(experiment_id, task, seed)
                if max_reward is not None:
                    experiment_metadata['max_reward'] = max_reward  # For W&B optimization
                    print(f"📊 Extracted max reward: {max_reward:.4f}")
                else:
                    # Fallback: try to find log directory and extract directly
                    log_dir = find_log_directory(experiment_id, seed)
                    if log_dir:
                        from sweep_utils import extract_max_reward_from_tensorboard
                        max_reward = extract_max_reward_from_tensorboard(log_dir)
                        if max_reward is not None:
                            experiment_metadata['max_reward'] = max_reward
                            
                if 'max_reward' not in experiment_metadata:
                    print("⚠️  Warning: Could not extract max reward, using 0.0")
                    experiment_metadata['max_reward'] = 0.0
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not extract training results: {e}")
                experiment_metadata['extraction_error'] = str(e)
                experiment_metadata['max_reward'] = 0.0
            
            return experiment_metadata
            
        else:
            print(f"❌ Training failed with return code: {process.returncode}")
            # Show last part of output for debugging
            if captured_output:
                print("Last output lines:")
                for line in captured_output[-10:]:
                    print(f"  {line}")
            
            return {
                **experiment_metadata,
                'error': f"Training failed with return code {process.returncode}",
                'status': 'failed',
                'end_time': time.time(),
                'max_reward': 0.0,
            }
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 2 hours")
        process.kill()
        process.wait()
        return {
            **experiment_metadata,
            'error': 'Training timed out',
            'status': 'timeout',
            'end_time': time.time(),
            'max_reward': 0.0,
        }
    except Exception as e:
        print(f"❌ Error running training: {str(e)}")
        return {
            **experiment_metadata,
            'error': str(e),
            'status': 'subprocess_error',
            'end_time': time.time(),
            'max_reward': 0.0,
        }


def main():
    """Main function for W&B sweep agent."""
    parser = argparse.ArgumentParser(description="W&B Sweep Agent for BALLU Morphology Optimization")
    parser.add_argument("--project", type=str, default="ballu-morphology-sweep", 
                       help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, 
                       help="W&B entity (username or team)")
    args, unknown = parser.parse_known_args()  # Ignore unknown arguments from W&B
    
    # Initialize W&B run
    run = wandb.init(project=args.project, entity=args.entity)
    
    try:
        # Get configuration from W&B
        config = dict(wandb.config)
        
        print(f"\n🔍 W&B Sweep Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Validate configuration
        if not validate_experiment_config(config):
            print("❌ Invalid configuration, marking run as failed")
            wandb.log({'max_reward': 0.0, 'status': 'invalid_config'})
            return
        
        # Convert decimal ratio to morphology path and get name
        robot_morphology = convert_ratio_to_usd_path(config['femur_to_tibia_ratio'])
        morphology_name = extract_morphology_name(robot_morphology)
        wandb.log({
            'femur_to_tibia_ratio': config['femur_to_tibia_ratio'],
            'gravity_comp_ratio': config['gravity_comp_ratio'],
            'motor_limit': config['motor_limit'],
            'task': config['task'],
            'seed': config['seed'],
            'max_iterations': config['max_iterations'],
            'num_envs': config['num_envs']
        })
        
        # Run the training experiment
        results = run_training_experiment(config)
        
        # Create comprehensive summary
        summary = create_experiment_summary(config, results)
        
        # Log results to W&B
        wandb.log(summary)
        
        # Log the key metric for optimization
        max_reward = results.get('max_reward', 0.0)
        wandb.log({'max_reward': max_reward})
        
        print(f"\n🎯 Final Results:")
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Max Reward: {max_reward:.4f}")
        print(f"   Duration: {results.get('duration_minutes', 0):.2f} minutes")
        print(f"   Morphology: {morphology_name}")
        print(f"   Gravity Comp Ratio: {config['gravity_comp_ratio']:.3f}")
        print(f"   Motor Limit: {config['motor_limit']:.3f}")
        
        # Set run summary
        wandb.run.summary.update({
            'max_reward': max_reward,
            'morphology_name': morphology_name,
            'gravity_comp_ratio': config['gravity_comp_ratio'],
            'motor_limit': config['motor_limit'],
            'status': results.get('status', 'unknown'),
            'duration_minutes': results.get('duration_minutes', 0)
        })
        
    except Exception as e:
        print(f"❌ Critical error in sweep agent: {str(e)}")
        wandb.log({'max_reward': 0.0, 'status': 'critical_error', 'error': str(e)})
    
    finally:
        # Finish W&B run
        wandb.finish()


if __name__ == "__main__":
    main()