#!/usr/bin/env python3
"""
Script to read ft_ratio_max_rewards.csv and plot femur proportion vs maximum rewards.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
import re


def extract_femur_proportion(experiment_name):
    """
    Extract femur proportion from experiment name.
    
    The experiment name format is expected to contain a substring like:
    "FT_<int1>_<int2>_<int3>_<int4>" where we extract <int1>.<int2>
    as the femur proportion.
    
    Args:
        experiment_name (str): Experiment folder name
        
    Returns:
        float: Femur proportion value (int1.int2)
    """
    try:
        # Look for pattern FT_<int1>_<int2>_<int3>_<int4>
        pattern = r'FT_(\d)_(\d)_(\d)_(\d)'
        match = re.search(pattern, experiment_name)
        
        if match:
            int1 = int(match.group(1))
            int2 = int(match.group(2))
            return float(f"{int1}.{int2}")
        else:
            return None
    except (IndexError, ValueError):
        return None


def plot_femur_proportion_rewards(input_csv, output_file=None):
    """
    Read CSV file and plot femur proportion vs max rewards.
    
    Args:
        input_csv (str): Path to input CSV file
        output_file (str, optional): Path to save the plot. If None, displays plot.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} experiments from {input_csv}")
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Check if required columns exist
    if 'seed_4_max_rewards' not in df.columns or 'Experiment name' not in df.columns:
        print("Error: CSV must contain 'seed_2_max_rewards' and 'Experiment name' columns")
        return
    
    # Extract femur proportion values
    df['femur_proportion'] = df['Experiment name'].apply(extract_femur_proportion)
    
    # Remove any rows where we couldn't extract femur proportion
    df = df.dropna(subset=['femur_proportion'])
    
    if df.empty:
        print("Error: No valid femur proportion values found in experiment names")
        return
    
    # Convert max rewards to float
    df['max_reward'] = pd.to_numeric(df['seed_4_max_rewards'], errors='coerce')
    df = df.dropna(subset=['max_reward'])
    
    # Sort by femur proportion
    df = df.sort_values('femur_proportion')
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, constrained_layout=True)
    
    # Define colors for a professional look
    line_color = '#1f77b4'      # A standard blue
    marker_face = '#ffdfba'     # Light orange
    marker_edge = '#ff7f0e'     # Darker orange
    text_color = '#333333'       # Dark gray for text
    
    # Plot the data with styled triangles
    ax.plot(df['femur_proportion'], df['max_reward'], 
            marker='^', 
            markersize=10, 
            markerfacecolor=marker_face,
            markeredgewidth=1.5, 
            markeredgecolor=marker_edge,
            color=line_color, 
            linewidth=2.5, 
            zorder=3,  
            clip_on=False)
    
    # Customize axes and spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d1d3d4')
    ax.spines['bottom'].set_color('#d1d3d4')
    
    # Add labels and title with improved typography
    ax.set_xlabel('Femur Proportion (on a scale of 0 to 10)', 
                 fontsize=16, labelpad=12, color=text_color)
    ax.set_ylabel('Cumulative Reward = 4 * Velocity', 
                 fontsize=16, labelpad=12, color=text_color)
    ax.set_title('Impact of Femur Proportion on Locomotion Speed', 
                fontsize=20, pad=18, color=text_color)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', colors=text_color, 
                  labelsize=14, width=1.2, length=6)

    # Draw a dashed horizontal line at Y = 430 and annotate it
    ax.axhline(y=430, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    # Annotate the line near the right
    xmax = df['femur_proportion'].max()
    ax.annotate('Y = 430', xy=(xmax, 430), xytext=(-10, 8), textcoords='offset points',
                ha='right', va='bottom', fontsize=13, color='black',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    
    # Add a subtle grid
    ax.grid(True, color='#f0f0f0', linestyle='-', linewidth=1, zorder=0)

    # Highlight the important region and add vertical lines
    ax.axvspan(4.4, 5.4, color='#fff9c4', alpha=0.7, zorder=1)
    ax.axvline(x=4.4, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=2)
    ax.axvline(x=5.4, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=2)

    # Ensure 4.4 and 5.4 are shown as x-ticks
    xticks = list(ax.get_xticks())
    for special_tick in [4.4, 5.4]:
        if special_tick not in xticks:
            xticks.append(special_tick)
    xticks = sorted(xticks)
    ax.set_xticks(xticks)

    # Legend (optional)
    # ax.legend(loc='best', frameon=False)

    # Add value labels on data points with better positioning
    # for femur_prop, reward in zip(df['femur_proportion'], df['max_reward']):
    #     # Alternate label positions for better readability
    #     va = 'bottom' if reward < max(df['max_reward']) * 0.7 else 'top'
    #     y_offset = 15 if va == 'bottom' else -15
        
    #     ax.annotate(f'{reward:.1f}', 
    #                xy=(femur_prop, reward),
    #                xytext=(0, y_offset), 
    #                textcoords='offset points',
    #                ha='center', va=va,
    #                fontsize=13, 
    #                color=text_color,
    #                bbox=dict(boxstyle='round,pad=0.3', 
    #                         fc='white', 
    #                         ec=marker_color, 
    #                         lw=1.2,
    #                         alpha=0.95),
    #                zorder=4)
    
    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save in multiple formats
        base_name = os.path.splitext(output_file)[0]
        
        # High-res PNG
        plt.savefig(f"{base_name}.png", dpi=600, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        
        print(f"Plot saved to: {base_name}.png")
    else:
        # Apply tight layout for display
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print("\n=== Femur Proportion Analysis Summary ===")
    print(f"Number of experiments: {len(df)}")
    print(f"Femur proportion range: {df['femur_proportion'].min():.2f} - {df['femur_proportion'].max():.2f}")
    print(f"Max reward range: {df['max_reward'].min():.2f} - {df['max_reward'].max():.2f}")
    
    # Find optimal femur proportion
    best_idx = df['max_reward'].idxmax()
    print(f"Best performance: {df.loc[best_idx, 'max_reward']:.2f} at femur proportion {df.loc[best_idx, 'femur_proportion']:.2f}")


def test_extraction():
    """
    Test the femur proportion extraction function.
    """
    test_cases = [
        ("07_24_02_07_03_FT_5_4_4_6_morphology_study", 5.4),
        ("07_24_02_07_03_FT_4_6_5_4_morphology_study", 4.6),
        ("07_24_02_07_03_FT_3_2_6_8_morphology_study", 3.2),
        ("07_24_02_07_03_FT_5_0_5_0_morphology_study", 5.0),
        ("invalid_name", None),
        ("FT_9_9_1_2_morphology_study", 9.9),
    ]
    
    print("Testing femur proportion extraction...")
    all_passed = True
    
    for experiment_name, expected in test_cases:
        result = extract_femur_proportion(experiment_name)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: '{experiment_name}' -> {result} (expected: {expected})")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Plot femur proportion vs maximum rewards from CSV file"
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='ft_ratio_max_rewards.csv',
        help='Path to input CSV file (default: ft_ratio_max_rewards.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file to save the plot (e.g., femur_proportion_plot.png). If not provided, displays plot.',
        default=None
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run extraction tests only'
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_extraction()
        return
    
    # Use default filename if not provided
    input_file = args.input_csv
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)
    
    # Generate output filename if not provided
    output_file = args.output
    if output_file is None and not args.output:
        output_file = os.path.join(os.path.dirname(input_file), 'femur_proportion_rewards_plot.png')
    
    plot_femur_proportion_rewards(input_file, output_file)


if __name__ == "__main__":
    main()
