#!/usr/bin/env python3
"""
Script to summarize morphology study results from the JSON output.
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction


def extract_morphology_params(experiment_name):
    """
    Extract morphology parameters from experiment name.
    Format: FT_X_Y_Z_W where X,Y,Z,W are the morphology parameters.
    """
    match = re.search(r'FT_(\d+)_(\d+)_(\d+)_(\d+)_morphology_study', experiment_name)
    if match:
        return [int(x) for x in match.groups()]
    return None


def calculate_morphology_ratio(experiment_name):
    """
    Calculate morphology ratio from experiment name.
    Format: FT_X_Y_Z_W -> (X.Y)/(Z.W) reduced to 2 decimal places
    """
    match = re.search(r'FT_(\d+)_(\d+)_(\d+)_(\d+)_morphology_study', experiment_name)
    if match:
        int1, int2, int3, int4 = map(int, match.groups())
        numerator = int1 + int2 / 10.0
        denominator = int3 + int4 / 10.0
        
        if denominator == 0:
            return None
        
        ratio = numerator / denominator
        return round(ratio, 2)
    return None


def calculate_morphology_decimal(experiment_name):
    """
    Calculate morphology decimal from experiment name.
    Format: FT_X_Y_Z_W -> X.Y (first two parameters as decimal)
    """
    match = re.search(r'FT_(\d+)_(\d+)_(\d+)_(\d+)_morphology_study', experiment_name)
    if match:
        int1, int2 = map(int, match.groups()[:2])
        decimal = int1 + int2 / 10.0
        return round(decimal, 1)
    return None


def analyze_results(json_file):
    """
    Analyze the morphology study results from JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for event_file, event_data in data.items():
        # Extract experiment name from file path
        path_parts = Path(event_file).parts
        experiment_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        
        # Extract morphology parameters
        params = extract_morphology_params(experiment_name)
        
        # Calculate morphology ratio
        ratio = calculate_morphology_ratio(experiment_name)
        
        # Calculate morphology decimal
        decimal = calculate_morphology_decimal(experiment_name)
        
        results.append({
            'experiment': experiment_name,
            'max_reward': event_data['max_reward'],
            'num_episodes': event_data['num_episodes'],
            'morphology_params': params,
            'morphology_ratio': ratio,
            'morphology_decimal': decimal
        })
    
    # Sort by max reward
    results.sort(key=lambda x: x['max_reward'], reverse=True)
    
    print("Morphology Study Results Summary")
    print("=" * 60)
    print(f"Total experiments processed: {len(results)}")
    print()
    
    print("Top 10 Best Performing Morphologies:")
    print("-" * 60)
    for i, result in enumerate(results[:10]):
        params = result['morphology_params']
        param_str = f"[{params[0]}, {params[1]}, {params[2]}, {params[3]}]" if params else "Unknown"
        ratio_str = f"Ratio: {result['morphology_ratio']:.2f}" if result['morphology_ratio'] is not None else "Ratio: N/A"
        print(f"{i+1:2d}. {result['experiment']}")
        print(f"     Parameters: {param_str}")
        print(f"     {ratio_str}")
        print(f"     Max Reward: {result['max_reward']:.4f}")
        print(f"     Episodes: {result['num_episodes']}")
        print()
    
    print("Bottom 10 Worst Performing Morphologies:")
    print("-" * 60)
    for i, result in enumerate(results[-10:]):
        params = result['morphology_params']
        param_str = f"[{params[0]}, {params[1]}, {params[2]}, {params[3]}]" if params else "Unknown"
        ratio_str = f"Ratio: {result['morphology_ratio']:.2f}" if result['morphology_ratio'] is not None else "Ratio: N/A"
        print(f"{len(results)-9+i:2d}. {result['experiment']}")
        print(f"     Parameters: {param_str}")
        print(f"     {ratio_str}")
        print(f"     Max Reward: {result['max_reward']:.4f}")
        print(f"     Episodes: {result['num_episodes']}")
        print()
    
    # Statistics
    rewards = [r['max_reward'] for r in results]
    print("Overall Statistics:")
    print("-" * 60)
    print(f"Average max reward: {sum(rewards)/len(rewards):.4f}")
    print(f"Median max reward: {sorted(rewards)[len(rewards)//2]:.4f}")
    print(f"Min max reward: {min(rewards):.4f}")
    print(f"Max max reward: {max(rewards):.4f}")
    print(f"Std dev: {sum((x - sum(rewards)/len(rewards))**2 for x in rewards)**0.5/len(rewards)**0.5:.4f}")
    
    # Parameter analysis
    print("\nParameter Analysis:")
    print("-" * 60)
    if results[0]['morphology_params']:
        param_analysis = {0: [], 1: [], 2: [], 3: []}
        for result in results:
            if result['morphology_params']:
                for i, param in enumerate(result['morphology_params']):
                    param_analysis[i].append((param, result['max_reward']))
        
        for param_idx in range(4):
            param_data = param_analysis[param_idx]
            if param_data:
                # Group by parameter value
                param_groups = {}
                for param_val, reward in param_data:
                    if param_val not in param_groups:
                        param_groups[param_val] = []
                    param_groups[param_val].append(reward)
                
                print(f"Parameter {param_idx + 1}:")
                for param_val in sorted(param_groups.keys()):
                    avg_reward = sum(param_groups[param_val]) / len(param_groups[param_val])
                    print(f"  Value {param_val}: Avg reward = {avg_reward:.4f} (n={len(param_groups[param_val])})")
                print()
    
    # Create plots
    create_morphology_plot(results)
    create_morphology_decimal_plot(results)


def create_morphology_plot(results):
    """
    Create a plot of max mean reward vs morphology ratio.
    """
    # Filter out results with None morphology ratio
    valid_results = [r for r in results if r['morphology_ratio'] is not None]
    
    if not valid_results:
        print("No valid morphology ratios found for plotting.")
        return
    
    # Extract data for plotting
    ratios = [r['morphology_ratio'] for r in valid_results]
    rewards = [r['max_reward'] for r in valid_results]
    experiments = [r['experiment'] for r in valid_results]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(ratios, rewards, alpha=0.7, s=100, edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for i, (ratio, reward, experiment) in enumerate(zip(ratios, rewards, experiments)):
        # Extract the FT parameters for annotation
        match = re.search(r'FT_(\d+)_(\d+)_(\d+)_(\d+)_morphology_study', experiment)
        if match:
            ft_params = f"FT_{match.group(1)}_{match.group(2)}_{match.group(3)}_{match.group(4)}"
            plt.annotate(ft_params, (ratio, reward), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Add trend line
    if len(ratios) > 1:
        z = np.polyfit(ratios, rewards, 1)
        p = np.poly1d(z)
        plt.plot(ratios, p(ratios), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.2f})')
        plt.legend()
    
    # Customize the plot
    plt.xlabel('Morphology Ratio (X.Y/Z.W)', fontsize=12)
    plt.ylabel('Max Mean Reward', fontsize=12)
    plt.title('Morphology Study: Max Mean Reward vs Morphology Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    avg_reward = np.mean(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    plt.text(0.02, 0.98, f'Average Reward: {avg_reward:.2f}\nMax Reward: {max_reward:.2f}\nMin Reward: {min_reward:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    plot_filename = 'morphology_ratio_plot.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()


def create_morphology_decimal_plot(results):
    """
    Create a plot of max mean reward vs morphology decimal (int1.int2).
    """
    # Filter out results with None morphology decimal
    valid_results = [r for r in results if r['morphology_decimal'] is not None]
    
    if not valid_results:
        print("No valid morphology decimals found for plotting.")
        return
    
    # Extract data for plotting
    decimals = [r['morphology_decimal'] for r in valid_results]
    rewards = [r['max_reward'] for r in valid_results]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with reduced intensity
    plt.scatter(decimals, rewards, alpha=0.4, s=100, edgecolors='black', linewidth=1, color='blue')
    
    # Add vertical dotted lines from points to X-axis
    for decimal, reward in zip(decimals, rewards):
        plt.vlines(x=decimal, ymin=0, ymax=reward, colors='gray', linestyles='dotted', alpha=0.5, linewidth=1)
    
    # Customize the plot
    plt.xlabel('Morphology Decimal (int1.int2)', fontsize=12)
    plt.ylabel('Max Mean Reward', fontsize=12)
    plt.title('Morphology Study: Max Mean Reward vs Morphology Decimal', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set Y-axis to start from 0 for better visualization of vertical lines
    plt.ylim(bottom=0)
    
    # Set X-axis ticks to include both original range and specific values
    all_ticks = list(range(1, 10)) + [4.4, 4.6, 4.8, 5.0, 5.2, 5.4]
    plt.xticks(all_ticks)
    
    # Add horizontal line at Y = 430
    plt.axhline(y=430, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Y = 430')
    plt.legend()
    
    # Add statistics text
    avg_reward = np.mean(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    plt.text(0.02, 0.98, f'Average Reward: {avg_reward:.2f}\nMax Reward: {max_reward:.2f}\nMin Reward: {min_reward:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save the plot
    plot_filename = 'morphology_decimal_plot.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python summarize_morphology_results.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"Error: File {json_file} does not exist")
        sys.exit(1)
    
    analyze_results(json_file) 