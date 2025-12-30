#!/usr/bin/env python3
"""
Velocity Analyzer Script

This script reads a CSV file containing robot simulation results and plots rolling averages
of the velocity components (VEL_X, VEL_Y, VEL_Z) on the same figure.

Usage:
    python velocity_analyzer.py <path_to_csv_file> [--window <window_size>] [--output <output_path>]

Example:
    python velocity_analyzer.py results.csv --window 50
    python velocity_analyzer.py /path/to/results.csv --window 20 --output /custom/output/path
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze velocity data from CSV files and plot rolling averages"
    )
    
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing the results data"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Window size for rolling average (default: 30)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for saving plots (default: same as input file)"
    )
    
    return parser.parse_args()


def load_csv_data(csv_file):
    """Load CSV data and validate required columns."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV file: {csv_file}")
        print(f"Data shape: {df.shape}")
        
        # Check if required velocity columns exist
        required_columns = ['VEL_X', 'VEL_Y', 'VEL_Z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Found velocity columns: {required_columns}")
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")


def compute_rolling_averages(df, window_size):
    """Compute rolling averages for velocity columns."""
    velocity_cols = ['VEL_X', 'VEL_Y', 'VEL_Z']
    
    rolling_data = {}
    for col in velocity_cols:
        rolling_data[col] = df[col].rolling(window=window_size, center=True).mean()
    
    print(f"Computed rolling averages with window size: {window_size}")
    return rolling_data


def create_velocity_plot(df, rolling_data, window_size, output_path):
    """Create and save velocity plot with rolling averages."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors for each velocity component
    colors = {'VEL_X': 'red', 'VEL_Y': 'green', 'VEL_Z': 'blue'}
    labels = {'VEL_X': 'X-axis', 'VEL_Y': 'Y-axis', 'VEL_Z': 'Z-axis'}
    
    # Create time axis (assuming sequential data points)
    time_steps = np.arange(len(df))
    
    # Plot rolling averages
    for col in ['VEL_X', 'VEL_Y', 'VEL_Z']:
        ax.plot(time_steps, rolling_data[col], 
                color=colors[col], 
                label=f'{labels[col]} (Rolling Avg)', 
                linewidth=2,
                alpha=0.6)
    
    # Add horizontal dashed line at y = 0.25
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.9, linewidth=1.5, label='Ref (0.25 m/s)')
    
    # Customize the plot
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Robot Velocity Components - Rolling Average (Window: {window_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    # stats_text = []
    # for col in ['VEL_X', 'VEL_Y', 'VEL_Z']:
    #     mean_val = rolling_data[col].mean()
    #     std_val = rolling_data[col].std()
    #     stats_text.append(f'{labels[col]}: μ={mean_val:.3f}, σ={std_val:.3f}')
    
    # ax.text(0.02, 0.98, '\n'.join(stats_text), 
    #         transform=ax.transAxes, 
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    output_file = os.path.join(output_path, f'velocity_rolling_average_w{window_size}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()


def print_summary_statistics(df, rolling_data):
    """Print summary statistics for velocity data."""
    print("\n" + "="*50)
    print("VELOCITY ANALYSIS SUMMARY")
    print("="*50)
    
    velocity_cols = ['VEL_X', 'VEL_Y', 'VEL_Z']
    
    print(f"{'Component':<10} {'Raw Mean':<12} {'Raw Std':<12} {'Roll Mean':<12} {'Roll Std':<12}")
    print("-" * 60)
    
    for col in velocity_cols:
        raw_mean = df[col].mean()
        raw_std = df[col].std()
        roll_mean = rolling_data[col].mean()
        roll_std = rolling_data[col].std()
        
        print(f"{col:<10} {raw_mean:<12.4f} {raw_std:<12.4f} {roll_mean:<12.4f} {roll_std:<12.4f}")
    
    print("\nOverall velocity magnitude:")
    velocity_magnitude = np.sqrt(df['VEL_X']**2 + df['VEL_Y']**2 + df['VEL_Z']**2)
    print(f"Mean magnitude: {velocity_magnitude.mean():.4f} m/s")
    print(f"Max magnitude: {velocity_magnitude.max():.4f} m/s")
    print(f"Min magnitude: {velocity_magnitude.min():.4f} m/s")


def main():
    """Main function to execute the velocity analysis."""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file does not exist: {args.csv_file}")
        return
    
    # Determine output path
    if args.output is None:
        output_path = os.path.dirname(os.path.abspath(args.csv_file))
    else:
        output_path = args.output
        os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load and process data
        df = load_csv_data(args.csv_file)
        rolling_data = compute_rolling_averages(df, args.window)
        
        # Create and save plot
        create_velocity_plot(df, rolling_data, args.window, output_path)
        
        # Print summary statistics
        print_summary_statistics(df, rolling_data)
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return


if __name__ == "__main__":
    main() 