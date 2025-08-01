#!/usr/bin/env python3
"""
Script to read buoyancy_max_rewards.csv and plot the maximum rewards.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


def extract_buoy_mass(experiment_name):
    """
    Extract buoyancy mass value from experiment name.
    
    Args:
        experiment_name (str): Experiment folder name
        
    Returns:
        float: Buoyancy mass value
    """
    try:
        # Look for the buoyMass value in the experiment name
        parts = experiment_name.split('_')
        for i, part in enumerate(parts):
            if part == 'buoyMass':
                return float(parts[i + 1])
        return None
    except (IndexError, ValueError):
        return None


def plot_buoyancy_rewards(input_csv, output_file=None):
    """
    Read CSV file and plot buoyancy mass vs max rewards.
    
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
    if 'seed_2_max_rewards' not in df.columns or 'Experiment name' not in df.columns:
        print("Error: CSV must contain 'seed_2_max_rewards' and 'Experiment name' columns")
        return
    
    # Extract buoyancy mass values
    df['buoy_mass'] = df['Experiment name'].apply(extract_buoy_mass)
    
    # Remove any rows where we couldn't extract buoyancy mass
    df = df.dropna(subset=['buoy_mass'])
    
    if df.empty:
        print("Error: No valid buoyancy mass values found in experiment names")
        return
    
    # Convert max rewards to float
    df['max_reward'] = pd.to_numeric(df['seed_2_max_rewards'], errors='coerce')
    df = df.dropna(subset=['max_reward'])

    # Create gravity compensation ratio column
    massRobot = 0.289
    df["gravity_comp_ratio"] = df["buoy_mass"] / massRobot
    
    # Sort by buoyancy mass
    df = df.sort_values('buoy_mass')
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, constrained_layout=True)
    
    # Define colors to match femur plot style
    line_color = '#1f77b4'      # Blue
    marker_face = '#ffdfba'     # Light orange
    marker_edge = '#ff7f0e'     # Darker orange
    text_color = '#333333'      # Dark gray for text
    
    # Plot the data with styled triangles
    ax.plot(df['gravity_comp_ratio'], df['max_reward'], 
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
    ax.set_xlabel('Gravity Compensation Ratio', 
                 fontsize=16, labelpad=12, color=text_color)
    ax.set_ylabel('Cumulative Reward = 4 * Velocity', 
                 fontsize=16, labelpad=12, color=text_color)
    ax.set_title('Impact of Buoyancy on Locomotion Speed', 
                fontsize=20, pad=18, color=text_color)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', colors=text_color, 
                  labelsize=14, width=1.2, length=6)
    
    # Add a subtle grid
    ax.grid(True, color='#f0f0f0', linestyle='-', linewidth=1, zorder=0)

    # Highlight region x=[0.75, 0.84] with a light yellow shade
    ax.axvspan(0.75, 0.84, color='#fff9c4', alpha=0.7, zorder=1)

    # Add vertical dotted lines at X=0.75 and X=0.84
    ax.axvline(x=0.75, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=2)
    ax.axvline(x=0.84, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=2)

    # Ensure 0.75 and 0.84 are shown as x-ticks
    xticks = list(ax.get_xticks())
    for special_tick in [0.75]:
        if special_tick not in xticks:
            xticks.append(special_tick)
    xticks = sorted(xticks)
    ax.set_xticks(xticks)

    # Add value labels on data points with better positioning
    # for gcRatio, reward in zip(df['gravity_comp_ratio'], df['max_reward']):
    #     # Alternate label positions for better readability
    #     va = 'bottom' if reward < max(df['max_reward']) * 0.7 else 'top'
    #     y_offset = 15 if va == 'bottom' else -15
        
    #     ax.annotate(f'{reward:.1f}', 
    #                xy=(gcRatio, reward),
    #                xytext=(0, y_offset), 
    #                textcoords='offset points',
    #                ha='center', va=va,
    #                fontsize=13, 
    #                color=text_color,
    #                bbox=dict(boxstyle='round,pad=0.3', 
    #                         fc=marker_face, 
    #                         ec=marker_edge, 
    #                         lw=1.2,
    #                         alpha=0.95),
    #                zorder=4)
    
    # Draw a dashed horizontal line at Y = 420 and annotate it
    ax.axhline(y=420, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    xmax = df['gravity_comp_ratio'].max()
    ax.annotate('Y = 420', xy=(xmax, 420), xytext=(-10, 8), textcoords='offset points',
                ha='right', va='bottom', fontsize=13, color='black',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

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
    print("\n=== Buoyancy Analysis Summary ===")
    print(f"Number of experiments: {len(df)}")
    print(f"Buoyancy mass range: {df['buoy_mass'].min():.2f} - {df['buoy_mass'].max():.2f} kg")
    print(f"Max reward range: {df['max_reward'].min():.2f} - {df['max_reward'].max():.2f}")
    
    # Find optimal buoyancy mass
    best_idx = df['max_reward'].idxmax()
    print(f"Best performance: {df.loc[best_idx, 'max_reward']:.2f} at buoyancy mass {df.loc[best_idx, 'buoy_mass']:.2f} kg")


def main():
    parser = argparse.ArgumentParser(
        description="Plot buoyancy mass vs maximum rewards from CSV file"
    )
    parser.add_argument(
        'input_csv',
        help='Path to input CSV file (e.g., buoyancy_max_rewards.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file to save the plot (e.g., buoyancy_plot.png). If not provided, displays plot.',
        default=None
    )
    
    args = parser.parse_args()
    
    # Use default filename if not provided
    input_file = args.input_csv
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)
    
    # Generate output filename if not provided
    output_file = args.output
    if output_file is None and not args.output:
        output_file = os.path.join(os.path.dirname(input_file), 'buoyancy_rewards_plot.png')
    
    plot_buoyancy_rewards(input_file, output_file)


if __name__ == "__main__":
    main()
