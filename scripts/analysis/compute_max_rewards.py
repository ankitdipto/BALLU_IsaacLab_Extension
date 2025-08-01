#!/usr/bin/env python3
"""
Script to compute the maximum of the Train/mean_reward timeseries for each event file
in multiple directories and write all maximums to a text file.
"""

import os
import sys
import argparse
import glob
import csv
from pathlib import Path
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


def extract_max_mean_reward(event_file_path):
    """
    Extract the maximum Train/mean_reward value from a TensorFlow event file.
    
    Args:
        event_file_path (str): Path to the TensorFlow event file
        
    Returns:
        float or None: Maximum mean reward value or None if no data found
    """
    try:
        # Load the event file
        ea = EventAccumulator(event_file_path)
        ea.Reload()
        
        # Check if Train/mean_reward tag exists
        if 'Train/mean_reward' not in ea.Tags()['scalars']:
            print(f"  Warning: 'Train/mean_reward' not found in {event_file_path}")
            return None
        
        # Extract the scalar data
        scalar_events = ea.Scalars('Train/mean_reward')
        
        if not scalar_events:
            print(f"  Warning: No data found for 'Train/mean_reward' in {event_file_path}")
            return None
        
        # Extract values and find maximum
        rewards = [event.value for event in scalar_events]
        max_reward = max(rewards)
        
        return max_reward
        
    except Exception as e:
        print(f"  Error reading {event_file_path}: {e}")
        return None


def process_directories(directories):
    """
    Process multiple directories and extract maximum mean reward from all TensorFlow event files.
    
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
            
            max_reward = extract_max_mean_reward(event_file)
            
            if max_reward is not None:
                results[event_file] = max_reward
                print(f"    Max mean_reward: {max_reward:.4f}")
            else:
                print(f"    No valid data found")
    
    return results


def write_max_rewards_to_csv(results, output_file, subdir_name):
    """
    Write maximum rewards to a CSV file with experiment names and rewards.
    
    Args:
        results (dict): Dictionary mapping event file paths to their max reward values
        output_file (str): Path to the output CSV file
        subdir_name (str): Name of the subdirectory for column header
    """
    try:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [f'{subdir_name}_max_rewards', 'Experiment name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Sort results by max reward (descending)
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            for event_file, max_reward in sorted_results:
                # Extract experiment name from path
                path_parts = Path(event_file).parts
                # Find the experiment folder name (should be the one with buoyMass)
                experiment_name = None
                for part in reversed(path_parts):
                    if 'buoyMass' in part:
                        experiment_name = part
                        break
                
                if experiment_name is None:
                    # Fallback: use the parent directory name
                    experiment_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
                
                writer.writerow({
                    f'{subdir_name}_max_rewards': f'{max_reward:.6f}',
                    'Experiment name': experiment_name
                })
        
        print(f"\nResults written to: {output_file}")
        
    except Exception as e:
        print(f"Error writing results to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute maximum Train/mean_reward from TensorFlow event files across multiple directories"
    )
    parser.add_argument(
        'directories', 
        nargs='+', 
        help='Base directories containing experiment folders'
    )
    parser.add_argument(
        '--subdir', 
        type=str, 
        required=True,
        help='Subdirectory name to search within each experiment folder (e.g., seed_1, seed_3)'
    )
    parser.add_argument(
        '--output', 
        '-o', 
        type=str, 
        default='max_rewards.csv',
        help='Output CSV file to save maximum rewards (default: max_rewards.csv)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Keyword to filter experiment directories (e.g., buoyMass)'
    )
    
    args = parser.parse_args()
    
    print("TensorFlow Event File Max Reward Analysis")
    print("=" * 50)
    
    # Update directories to include subdirectory
    updated_directories = []
    for base_dir in args.directories:
        # Find all experiment folders in the base directory
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                full_path = os.path.join(base_dir, item)
                if os.path.isdir(full_path) and (args.filter is None or args.filter in item):
                    subdir_path = os.path.join(full_path, args.subdir)
                    if os.path.exists(subdir_path):
                        updated_directories.append(subdir_path)
    
    if not updated_directories:
        print(f"No directories found with subdirectory '{args.subdir}'")
        return
    
    print(f"Found {len(updated_directories)} directories with subdirectory '{args.subdir}'")
    
    # Process all updated directories
    results = process_directories(updated_directories)
    
    if not results:
        print("\nNo valid TensorFlow event files found or no data extracted.")
        return
    
    # Write results to CSV file
    write_max_rewards_to_csv(results, args.output, args.subdir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    max_rewards = list(results.values())
    
    print(f"Total event files processed: {len(results)}")
    print(f"Average max reward: {sum(max_rewards)/len(max_rewards):.4f}")
    print(f"Min max reward: {min(max_rewards):.4f}")
    print(f"Max max reward: {max(max_rewards):.4f}")


if __name__ == "__main__":
    main()
