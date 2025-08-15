#!/usr/bin/env python3
"""
Utility functions for W&B morphology sweeps.
"""

import os
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent directories to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parent.parent))
sys.path.append(str(script_dir.parent / "rsl_rl"))

try:
    # Try relative import first
    from ..rsl_rl.multi_run_training import extract_training_results, extract_tensorboard_data
except ImportError:
    # Fallback to absolute import
    sys.path.append(str(script_dir.parent / "rsl_rl"))
    from multi_run_training import extract_training_results, extract_tensorboard_data


def get_robot_usd_full_path(robot_morphology: str) -> str:
    """
    Convert relative robot morphology path to full USD path.
    
    Args:
        robot_morphology: Relative path like "original/original.usd" or "FT_5_0_5_0/ballu_modified_FT_5_0_5_0.usd"
        
    Returns:
        Full absolute path to the USD file
    """
    # Get the base robots directory
    script_dir = Path(__file__).parent
    robots_dir = script_dir.parent.parent / "source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots"
    
    # Construct full path
    full_path = robots_dir / robot_morphology
    
    if not full_path.exists():
        raise FileNotFoundError(f"Robot USD file not found: {full_path}")
    
    return str(full_path.absolute())


def extract_morphology_name(robot_morphology: str) -> str:
    """
    Extract a clean morphology name from the USD path for logging.
    
    Args:
        robot_morphology: Relative path like "original/original.usd"
        
    Returns:
        Clean name like "original" or "FT_5_0_5_0"
    """
    if robot_morphology.startswith("original/"):
        return "original"
    
    # Extract from patterns like "FT_5_0_5_0/ballu_modified_FT_5_0_5_0.usd"
    morphology_dir = robot_morphology.split("/")[0]
    return morphology_dir


def extract_max_reward_from_logs(common_folder: str, task: str, seed: int) -> Optional[float]:
    """
    Extract the maximum mean reward achieved during training from log files.
    
    Args:
        common_folder: Common folder name for the experiment
        task: Task name
        seed: Random seed
        
    Returns:
        Maximum mean reward value or None if extraction fails
    """
    try:
        # Use existing extraction function
        results = extract_training_results(common_folder, task, seed)['tensorboard_data']
        
        # Try to get max reward from different possible keys
        max_reward = None
        
        # Look for max_reward in results (from TensorBoard data)
        if 'max_reward_achieved' in results:
            max_reward = results['max_reward_achieved']
        elif 'final_reward_mean' in results:
            # Fallback to final reward if max not available
            max_reward = results['final_reward_mean']
        
        if max_reward is not None:
            print(f"✅ Extracted max reward: {max_reward:.4f}")
            return float(max_reward)
        else:
            print("⚠️  No reward data found in training results")
            return None
            
    except Exception as e:
        print(f"❌ Failed to extract reward from logs: {e}")
        return None


# def extract_max_reward_from_tensorboard(log_dir: str) -> Optional[float]:
#     """
#     Directly extract maximum reward from TensorBoard event files.
    
#     Args:
#         log_dir: Path to the log directory containing TensorBoard events
        
#     Returns:
#         Maximum mean reward or None if extraction fails
#     """
#     try:
#         # Use existing TensorBoard extraction function
#         tensorboard_results = extract_tensorboard_data(log_dir)
        
#         if 'max_reward' in tensorboard_results:
#             max_reward = float(tensorboard_results['max_reward'])
#             print(f"✅ Extracted max reward from TensorBoard: {max_reward:.4f}")
#             return max_reward
#         else:
#             print("⚠️  No max reward found in TensorBoard data")
#             return None
            
#     except Exception as e:
#         print(f"❌ Failed to extract reward from TensorBoard: {e}")
#         return None


def find_log_directory(common_folder: str, seed: int) -> Optional[str]:
    """
    Find the log directory for a specific experiment.
    
    Args:
        common_folder: Common folder name for the experiment
        seed: Random seed
        
    Returns:
        Path to log directory or None if not found
    """
    # Pattern to find log directory
    log_pattern = f"logs/rsl_rl/*/{common_folder}/seed_{seed}"
    log_dirs = glob.glob(log_pattern)
    
    if log_dirs:
        log_dir = log_dirs[0]  # Take the first match
        print(f"📁 Found log directory: {log_dir}")
        return log_dir
    else:
        print(f"❌ No log directory found for pattern: {log_pattern}")
        return None


def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """
    Validate the experiment configuration from W&B.
    
    Args:
        config: Configuration dictionary from wandb.config
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ['gravity_comp_ratio', 'femur_to_tibia_ratio', 'motor_limit', 'task', 'max_iterations', 'num_envs']
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required configuration key: {key}")
            return False
    
    # Validate gravity compensation ratio range
    gravity_comp_ratio = config['gravity_comp_ratio']
    if not (0.30 <= gravity_comp_ratio <= 1.15):
        print(f"❌ Gravity comp ratio {gravity_comp_ratio} outside valid range [0.50, 1.15]")
        return False
    
    # Validate femur_to_tibia_ratio range
    femur_to_tibia_ratio = config['femur_to_tibia_ratio']
    if not (1.0 <= femur_to_tibia_ratio <= 9.0):
        print(f"❌ Femur to tibia ratio {femur_to_tibia_ratio} outside valid range [1.0, 9.0]")
        return False
    
    # Import and use the conversion function to validate the resulting path
    try:
        # Import the conversion function from sweep_agent
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.append(str(script_dir))
        from sweep_agent import convert_ratio_to_usd_path
        
        robot_morphology = convert_ratio_to_usd_path(femur_to_tibia_ratio)
        get_robot_usd_full_path(robot_morphology)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    except ImportError as e:
        print(f"⚠️  Warning: Could not validate USD path due to import error: {e}")
        # Continue validation without USD path check
    
    print("✅ Experiment configuration is valid")
    return True


def create_experiment_summary(config: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the experiment.
    
    Args:
        config: Configuration used for the experiment
        results: Results from the training experiment
        
    Returns:
        Summary dictionary
    """
    # Convert decimal ratio to morphology path and extract name
    try:
        from sweep_agent import convert_ratio_to_usd_path
        robot_morphology = convert_ratio_to_usd_path(config['femur_to_tibia_ratio'])
        morphology_name = extract_morphology_name(robot_morphology)
    except ImportError:
        # Fallback: create morphology name from ratio
        morphology_name = f"FT_ratio_{config['femur_to_tibia_ratio']}"
    
    summary = {
        'morphology_name': morphology_name,
        'gravity_comp_ratio': config['gravity_comp_ratio'],
        'motor_limit': config['motor_limit'],
        'task': config['task'],
        'seed': config['seed'],
        'max_iterations': config['max_iterations'],
        'num_envs': config['num_envs'],
        'status': results.get('status', 'unknown'),
        'max_mean_reward': results.get('max_reward', None),
        'final_reward_mean': results.get('final_reward_mean', None),
        'duration_minutes': results.get('duration_minutes', None),
        'experiment_id': results.get('experiment_id', None),
        
        # Play script results
        'play_status': results.get('play_status', 'not_run'),
        'play_reward': results.get('play_reward', None),
        'play_duration_minutes': results.get('play_duration_minutes', None),
        'play_checkpoint': results.get('play_checkpoint', None)
    }
    
    # Add additional metrics if available
    for key in ['final_surrogate_loss', 'final_value_function_loss', 'final_entropy_loss', 'final_noise_std_loss']:
        if key in results:
            summary[key] = results[key]
    
    # Add play script error if available
    if 'play_error' in results:
        summary['play_error'] = results['play_error']
    
    return summary