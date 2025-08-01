#!/usr/bin/env python3
"""
Script to extract Train/mean_reward from TensorFlow event files.
Processes multiple directories and finds the maximum mean reward for each event file.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_tf_event_files(directory):
    """
    Find all TensorFlow event files in the given directory and its subdirectories.
    
    Args:
        directory (str): Root directory to search for event files
        
    Returns:
        list: List of paths to TensorFlow event files
    """
    event_files = []
    
    # Search for files matching TensorFlow event file pattern
    pattern = os.path.join(directory, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    
    return event_files


def extract_mean_reward_data(event_file_path):
    """
    Extract Train/mean_reward data from a TensorFlow event file.
    
    Args:
        event_file_path (str): Path to the TensorFlow event file
        
    Returns:
        tuple: (max_reward, all_rewards, steps) or (None, None, None) if no data found
    """
    try:
        # Load the event file
        ea = EventAccumulator(event_file_path)
        ea.Reload()
        
        # Check if Train/mean_reward tag exists
        if 'Train/mean_reward' not in ea.Tags()['scalars']:
            print(f"  Warning: 'Train/mean_reward' not found in {event_file_path}")
            return None, None, None
        
        # Extract the scalar data
        scalar_events = ea.Scalars('Train/mean_reward')
        
        if not scalar_events:
            print(f"  Warning: No data found for 'Train/mean_reward' in {event_file_path}")
            return None, None, None
        
        # Extract values and steps
        rewards = [event.value for event in scalar_events]
        steps = [event.step for event in scalar_events]
        
        max_reward = max(rewards)
        
        return max_reward, rewards, steps
        
    except Exception as e:
        print(f"  Error reading {event_file_path}: {e}")
        return None, None, None


def process_directories(directories):
    """
    Process multiple directories and extract mean reward data from all TensorFlow event files.
    
    Args:
        directories (list): List of directory paths to process
        
    Returns:
        dict: Dictionary mapping event file paths to their max reward values
    """
    results = {}
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        
        if not os.path.exists(directory):
            print(f"  Warning: Directory {directory} does not exist, skipping...")
            continue
        
        # Find all TensorFlow event files in this directory
        event_files = find_tf_event_files(directory)
        
        if not event_files:
            print(f"  No TensorFlow event files found in {directory}")
            continue
        
        print(f"  Found {len(event_files)} event file(s)")
        
        # Process each event file
        for event_file in event_files:
            print(f"  Processing: {event_file}")
            
            max_reward, rewards, steps = extract_mean_reward_data(event_file)
            
            if max_reward is not None:
                results[event_file] = {
                    'max_reward': max_reward,
                    'all_rewards': rewards,
                    'steps': steps,
                    'num_episodes': len(rewards)
                }
                print(f"    Max mean_reward: {max_reward:.4f} (from {len(rewards)} episodes)")
            else:
                print(f"    No valid data found")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract Train/mean_reward from TensorFlow event files across multiple directories"
    )
    parser.add_argument(
        'directories', 
        nargs='+', 
        help='Directories containing TensorFlow event files'
    )
    parser.add_argument(
        '--output', 
        '-o', 
        type=str, 
        help='Output file to save results (optional)'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true', 
        help='Verbose output with detailed information'
    )
    
    args = parser.parse_args()
    
    print("TensorFlow Event File Analysis")
    print("=" * 50)
    
    # Process all directories
    results = process_directories(args.directories)
    
    if not results:
        print("\nNo valid TensorFlow event files found or no data extracted.")
        return
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    max_rewards = []
    for event_file, data in results.items():
        max_reward = data['max_reward']
        max_rewards.append(max_reward)
        
        # Extract directory and seed information from path
        path_parts = Path(event_file).parts
        experiment_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        seed_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        
        print(f"Experiment: {experiment_name}")
        print(f"Seed: {seed_name}")
        print(f"Event file: {os.path.basename(event_file)}")
        print(f"Max mean_reward: {max_reward:.4f}")
        print(f"Total episodes: {data['num_episodes']}")
        
        if args.verbose:
            print(f"All rewards: {data['all_rewards'][:10]}{'...' if len(data['all_rewards']) > 10 else ''}")
            print(f"Steps: {data['steps'][:10]}{'...' if len(data['steps']) > 10 else ''}")
        
        print("-" * 30)
    
    # Overall statistics
    if max_rewards:
        print(f"\nOverall Statistics:")
        print(f"Total event files processed: {len(results)}")
        print(f"Average max reward: {np.mean(max_rewards):.4f}")
        print(f"Std dev of max rewards: {np.std(max_rewards):.4f}")
        print(f"Min max reward: {np.min(max_rewards):.4f}")
        print(f"Max max reward: {np.max(max_rewards):.4f}")
    
    # Save results to file if requested
    if args.output:
        try:
            import json
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for event_file, data in results.items():
                serializable_results[event_file] = {
                    'max_reward': float(data['max_reward']),
                    'all_rewards': [float(r) for r in data['all_rewards']],
                    'steps': [int(s) for s in data['steps']],
                    'num_episodes': data['num_episodes']
                }
            
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")
            
        except Exception as e:
            print(f"Error saving results to {args.output}: {e}")


if __name__ == "__main__":
    main() 