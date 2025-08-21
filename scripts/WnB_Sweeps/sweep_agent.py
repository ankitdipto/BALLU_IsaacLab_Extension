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


def run_full_experiment(config: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
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
    spring_coefficient = config['spring_coefficient']
    
    # Convert decimal ratio to USD path
    robot_morphology = convert_ratio_to_usd_path(femur_to_tibia_ratio)
    
    # Get full USD path
    usd_path = get_robot_usd_full_path(robot_morphology)
    morphology_name = extract_morphology_name(robot_morphology)
    
    # Create unique experiment ID
    # timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    experiment_id = f"wb_sweep_{timestamp}_{morphology_name}_buoy{buoyancy_mass:.3f}_sprCoef{spring_coefficient:.5f}"
    
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
        f"env.scene.robot.actuators.knee_effort_actuators.spring_coeff={spring_coefficient}",
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
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                captured_output.append(output.strip())
                
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
            
            # Run play script to evaluate the trained model
            print(f"\n🎮 Starting play script evaluation...")
            play_results = run_play_script(config, experiment_id, task, seed, buoyancy_mass, usd_path)
            
            # Integrate play results into experiment metadata
            experiment_metadata.update(play_results)
            
            # Log play reward separately for analysis
            if play_results['play_status'] == 'completed':
                print(f"✅ Play evaluation completed with reward: {play_results['play_reward']:.4f}")
            else:
                print(f"❌ Play evaluation failed: {play_results.get('play_error', 'Unknown error')}")
            
            return experiment_metadata
        
        else:
            print("❌ Training failed with return code: ", process.returncode, "but still trying to run play script")
            experiment_metadata.update({
                'end_time': time.time(),
                'status': 'failed',
                'duration_minutes': (time.time() - experiment_metadata['start_time']) / 60.0,
                'captured_output_lines': len(captured_output)
            })

            # Run play script to evaluate the trained model upto the iterations where it failed
            play_results = run_play_script(config, experiment_id, task, seed, buoyancy_mass, usd_path)
            
            # Integrate play results into experiment metadata
            experiment_metadata.update(play_results)
            
            # Log play reward separately for analysis
            if play_results['play_status'] == 'completed':
                print(f"✅ Play evaluation completed with reward: {play_results['play_reward']:.4f}")
            else:
                print(f"❌ Play evaluation failed: {play_results.get('play_error', 'Unknown error')}")
            
            return experiment_metadata
            
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
        }


def run_play_script(config: Dict[str, Any], experiment_id: str, task: str, seed: int, 
                   buoyancy_mass: float, usd_path: str) -> Dict[str, Any]:
    """
    Run the play script to test the trained model.
    
    Args:
        config: Configuration dictionary from W&B
        experiment_id: Unique experiment identifier
        task: Task name
        seed: Random seed used in training
        buoyancy_mass: Buoyancy mass parameter
        usd_path: Path to the robot USD file
        
    Returns:
        Dictionary containing play script results
    """
    print(f"\n🎮 Starting play script evaluation...")
    
    # Build play command
    play_script = script_dir.parent / "rsl_rl" / "play_simple.py"
    
    # Find the checkpoint path based on experiment_id. 
    # The training script creates a date-based folder, so we need to search for the experiment_id directory.
    # log_root_path = Path("logs") / "rsl_rl"
    # checkpoint_pattern = f"{log_root_path}/**/{experiment_id}/**/nn/*.pth"
    
    # print(f"🔍 Searching for checkpoints with pattern: {checkpoint_pattern}")
    # checkpoint_files = list(Path(".").glob(checkpoint_pattern))
    # checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    # print(f"Found checkpoint files (after sorting): {checkpoint_files}")
    
    # if not checkpoint_files:
    #     print(f"❌ No checkpoint found for experiment: {experiment_id}")
    #     return {
    #         'play_status': 'checkpoint_not_found',
    #         'play_error': f'No checkpoint found matching pattern: {checkpoint_pattern}',
    #         'play_reward': 0.0
    #     }
    
    # Define the run name and checkpoint name for the play script.
    # The play_simple.py script will handle locating the exact path.
    run_name = f"{experiment_id}/seed_{seed}"
    checkpoint_name = "model_best.pt"
    motor_limit = config["motor_limit"]

    print(f"▶️ Preparing to run play script with:")
    print(f"   Run Name: {run_name}")
    print(f"   Checkpoint: {checkpoint_name}")
    print(f"   Motor Limit: {motor_limit}")
    
    cmd = [
        sys.executable,
        str(play_script),
        "--task", task,
        "--num_envs", "1",  # Single environment for evaluation
        "--balloon_buoyancy_mass", str(buoyancy_mass),
        "--load_run", run_name,
        "--checkpoint", checkpoint_name,
        "--headless",
        "--video",
        "--video_length", str(399),
        f"env.scene.robot.actuators.knee_effort_actuators.effort_limit={motor_limit}"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env['BALLU_MORPHOLOGY_USD_PATH'] = usd_path
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'
    
    play_metadata = {
        'play_start_time': time.time(),
        'play_command': ' '.join(cmd),
        'play_checkpoint': checkpoint_name,
        'play_run': run_name
    }
    
    try:
        print(f"⚡ Running play command:")
        print(f"   {' '.join(cmd)}")
        
        # Run play script with timeout (30 minutes max)
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor process output
        captured_output = []
        play_reward = None
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                captured_output.append(output.strip())
                
                # Parse cumulative reward from output
                if "Mean cumulative reward:" in output:
                    try:
                        reward_str = output.split("Mean cumulative reward:")[1].strip()
                        play_reward = float(reward_str)
                        print(f"🎯 Play reward extracted: {play_reward:.4f}")
                    except (IndexError, ValueError):
                        pass
        
        process.wait(timeout=7 * 60)
        
        if process.returncode == 0:
            print("✅ Play script completed successfully!")
            
            # Default to 0.0 if reward couldn't be parsed
            if play_reward is None:
                print("⚠️  Could not parse play reward from output, using 0.0")
                play_reward = 0.0
            
            return {
                **play_metadata,
                'play_status': 'completed',
                'play_reward': play_reward,
                'play_duration_minutes': (time.time() - play_metadata['play_start_time']) / 60.0,
                'play_captured_output_lines': len(captured_output)
            }
        else:
            print(f"❌ Play script failed with return code: {process.returncode}")
            return {
                **play_metadata,
                'play_status': 'failed',
                'play_error': f'Play script failed with return code {process.returncode}',
                'play_reward': -3.0
            }
            
    except subprocess.TimeoutExpired:
        print("⏰ Play script timed out after 7 minutes")
        process.kill()
        process.wait()
        return {
            **play_metadata,
            'play_status': 'timeout',
            'play_error': 'Play script timed out',
            'play_reward': -2.0
        }
    except Exception as e:
        print(f"❌ Error running play script: {str(e)}")
        return {
            **play_metadata,
            'play_status': 'error',
            'play_error': str(e),
            'play_reward': -1.0
        }


def main():
    """Main function for W&B sweep agent."""
    parser = argparse.ArgumentParser(description="W&B Sweep Agent for BALLU Morphology Optimization")
    parser.add_argument("--project", type=str, default="ballu-morphology-sweep", 
                       help="W&B project name")
    parser.add_argument("--entity", type=str, default="ankitdipto", 
                       help="W&B entity (username or team)")
    args, unknown = parser.parse_known_args()  # Ignore unknown arguments from W&B
    
    # Initialize W&B run
    run = wandb.init(project=args.project, entity=args.entity)
    
    try:
        # Get configuration from W&B
        config = dict(wandb.config)
        # config = {
        #     'femur_to_tibia_ratio': 2.0,
        #     'gravity_comp_ratio': 0.91,
        #     'motor_limit': 0.1,
        #     'task': 'Isc-Vel-BALLU-encoder',
        #     'max_iterations': 100,
        #     'num_envs': 4096,
        # }
        
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
            'max_iterations': config['max_iterations'],
            'num_envs': config['num_envs']
        })
        
        play_rewards = []
        train_durations = []
        play_durations = []
        timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

        # for seed in range(3):
        # config['seed'] = seed
        results = run_full_experiment(config, timestamp)
        print("Printing experiment results: ", results)
        # Create comprehensive summary
        # summary = create_experiment_summary(config, results)
            
        # Log results to W&B
        # wandb.log(summary)
        
        play_reward = results.get('play_reward', -1.0)
        play_rewards.append(play_reward)
        train_durations.append(results.get('duration_minutes', -1))
        play_durations.append(results.get('play_duration_minutes', -1))
            
        # Log the key metric for optimization
        avg_play_reward = sum(play_rewards) / len(play_rewards)
        avg_train_duration = sum(train_durations) / len(train_durations)
        avg_play_duration = sum(play_durations) / len(play_durations)
        
        wandb.log({
            'experiment_id': results.get('experiment_id', 'N/A'),
            'avg_velocity': avg_play_reward / 1600,
            'avg_play_reward': avg_play_reward,
            'avg_train_duration': avg_train_duration,
            'avg_play_duration': avg_play_duration
        })
        
        print(f"\n🎯 Final Results:")
        print(f"   Avg Play Reward: {avg_play_reward:.4f}")
        print(f"   Avg Train Duration: {avg_train_duration:.2f} minutes")
        print(f"   Avg Play Duration: {avg_play_duration:.2f} minutes")
        print(f"   Morphology: {morphology_name}")
        print(f"   Gravity Comp Ratio: {config['gravity_comp_ratio']:.3f}")
        print(f"   Motor Limit: {config['motor_limit']:.3f}")
        
        # Set run summary
        wandb.run.summary.update({
            'avg_play_reward': avg_play_reward,
            'avg_train_duration': avg_train_duration,
            'avg_play_duration': avg_play_duration,
            'morphology_name': morphology_name,
            'gravity_comp_ratio': config['gravity_comp_ratio'],
            'motor_limit': config['motor_limit']
        })
        
    except Exception as e:
        print(f"❌ Critical error in sweep agent: {str(e)}")
        wandb.log({'max_reward': 0.0, 'status': 'critical_error', 'error': str(e)})
    
    finally:
        # Finish W&B run
        wandb.finish()


if __name__ == "__main__":
    main()