#!/usr/bin/env python3
"""
CSV Data Analysis and Plotting Script

This script parses CSV files containing robot joint and actuator data and generates
professional plots suitable for research papers.

Author: AI Assistant
Date: 2025
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def setup_matplotlib_style():
    """Configure matplotlib for professional research paper quality plots."""
    plt.style.use('default')  # Start with clean slate
    
    # Set professional parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def get_professional_colors():
    """Return a set of professional colors suitable for research papers."""
    return {
        'motor_left': '#1f77b4',    # Muted blue
        'motor_right': '#ff7f0e',   # Safety orange
        'knee_left': '#2ca02c',     # Cooked asparagus green
        'knee_right': '#d62728',    # Brick red
        'act_left': '#e74c3c',      # Bright red
        'act_right': '#3498db',     # Bright blue
    }


def validate_csv_file(csv_path):
    """Validate that the CSV file exists and has required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    required_columns = ['MOTOR_LEFT', 'MOTOR_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'ACT_LEFT', 'ACT_RIGHT']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    return df


def plot_motor_and_knee_data(df, output_dir):
    """
    Create a professional plot of motor and knee joint data.
    
    Args:
        df: pandas DataFrame containing the data
        output_dir: Directory to save the plot
    """
    colors = get_professional_colors()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create time axis (assuming sequential timesteps)
    time_steps = range(len(df))
    
    # Plot motor data
    ax.plot(time_steps, df['MOTOR_LEFT'], 
            color=colors['motor_left'], 
            label='Motor Left', 
            linestyle='--', 
            alpha=0.8)
    
    ax.plot(time_steps, df['MOTOR_RIGHT'], 
            color=colors['motor_right'], 
            label='Motor Right', 
            linestyle='--', 
            alpha=0.8)
    
    # Plot knee data
    ax.plot(time_steps, df['KNEE_LEFT'], 
            color=colors['knee_left'], 
            label='Knee Left', 
            linestyle='-', 
            alpha=0.8)
    
    ax.plot(time_steps, df['KNEE_RIGHT'], 
            color=colors['knee_right'], 
            label='Knee Right', 
            linestyle='-', 
            alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Joint Angle (rad)')
    ax.set_title('Motor and Knee Joint Angles Over Time')
    
    # Place legend in upper right, outside plot area if necessary
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'motor_knee_joints_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Motor and knee joints plot saved to: {output_path}")


def plot_actuator_data(df, output_dir):
    """
    Create a professional plot of actuator data.
    
    Args:
        df: pandas DataFrame containing the data
        output_dir: Directory to save the plot
    """
    colors = get_professional_colors()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create time axis (assuming sequential timesteps)
    time_steps = range(len(df))
    
    # Plot actuator data
    ax.plot(time_steps, df['ACT_LEFT'], 
            color=colors['act_left'], 
            label='Actuator Left', 
            linestyle='-', 
            alpha=0.8,
            linewidth=2.5)
    
    ax.plot(time_steps, df['ACT_RIGHT'], 
            color=colors['act_right'], 
            label='Actuator Right', 
            linestyle='-', 
            alpha=0.8,
            linewidth=2.5)
    
    # Formatting
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Actuator Value')
    ax.set_title('Actuator Values Over Time')
    
    # Place legend in best location to avoid clutter
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'actuator_values_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Actuator values plot saved to: {output_path}")


def plot_torque_data(df, output_dir, side):
    """
    Plot computed and applied torques for a given actuator side ('LEFT' or 'RIGHT').
    """
    assert side in ("LEFT", "RIGHT")
    comp_col = f"COMP_TORQ_{side}_KNEE"
    appl_col = f"APPLIED_TORQ_{side}_KNEE"
    if comp_col not in df.columns or appl_col not in df.columns:
        print(f"[WARN] Columns {comp_col} or {appl_col} not found in data. Skipping {side.lower()} torque plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    time_steps = range(len(df))
    ax.plot(time_steps, df[comp_col], label=f"Computed Torque {side.title()}", color="#1f77b4", linestyle="-", linewidth=2.0)
    ax.plot(time_steps, df[appl_col], label=f"Applied Torque {side.title()}", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title(f"Computed vs Applied Torque ({side.title()} Actuator)")
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"torque_{side.lower()}_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] {side.title()} actuator torque plot saved to: {output_path}")


def compute_rolling_averages(df, window_size):
    """Compute rolling averages for velocity columns."""
    velocity_cols = ['VEL_X', 'VEL_Y', 'VEL_Z']
    
    # Check if velocity columns exist
    existing_vel_cols = [col for col in velocity_cols if col in df.columns]
    if not existing_vel_cols:
        return None
    
    rolling_data = {}
    for col in existing_vel_cols:
        rolling_data[col] = df[col].rolling(window=window_size, center=True).mean()
    
    print(f"[INFO] Computed rolling averages with window size: {window_size}")
    return rolling_data


def plot_velocity_data(df, rolling_data, window_size, output_dir):
    """Create and save velocity plot with rolling averages."""
    if rolling_data is None:
        print("[WARN] No velocity data available for plotting.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each velocity component
    colors = {'VEL_X': '#e74c3c', 'VEL_Y': '#2ecc71', 'VEL_Z': '#3498db'}
    labels = {'VEL_X': 'X-axis', 'VEL_Y': 'Y-axis', 'VEL_Z': 'Z-axis'}
    
    # Create time axis (assuming sequential data points)
    time_steps = np.arange(len(df))
    
    # Plot rolling averages
    for col in rolling_data.keys():
        ax.plot(time_steps, rolling_data[col], 
                color=colors[col], 
                label=f'{labels[col]} (Rolling Avg)', 
                linewidth=2.0,
                alpha=0.8)
    
    # Add horizontal reference line at y = 0.25
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.9, linewidth=1.5, label='Ref (0.25 m/s)')
    
    # Formatting
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Robot Velocity Components - Rolling Average (Window: {window_size})')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'velocity_rolling_average_w{window_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Velocity rolling average plot saved to: {output_path}")


def write_velocity_statistics(df, rolling_data, output_dir):
    """Write comprehensive velocity statistics to a text file."""
    if rolling_data is None:
        print("[WARN] No velocity data available for statistics.")
        return
    
    velocity_cols = list(rolling_data.keys())
    
    # Create statistics file path
    stats_file_path = os.path.join(output_dir, 'velocity_statistics.txt')
    
    with open(stats_file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VELOCITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Raw velocity statistics
        f.write("RAW VELOCITY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Component':<10} {'Mean':<12} {'Median':<12} {'Std':<12} {'Q1 (25%)':<12} {'Q3 (75%)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for col in velocity_cols:
            raw_mean = df[col].mean()
            raw_median = df[col].median()
            raw_std = df[col].std()
            raw_q1 = df[col].quantile(0.25)
            raw_q3 = df[col].quantile(0.75)
            
            f.write(f"{col:<10} {raw_mean:<12.4f} {raw_median:<12.4f} {raw_std:<12.4f} {raw_q1:<12.4f} {raw_q3:<12.4f}\n")
        
        # Rolling average statistics
        f.write("\nROLLING AVERAGE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Component':<10} {'Roll Mean':<12} {'Roll Std':<12}\n")
        f.write("-" * 80 + "\n")
        
        for col in velocity_cols:
            roll_mean = rolling_data[col].mean()
            roll_std = rolling_data[col].std()
            
            f.write(f"{col:<10} {roll_mean:<12.4f} {roll_std:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[INFO] Velocity statistics saved to: {stats_file_path}")


def print_data_summary(df):
    """Print a summary of the loaded data."""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Number of time steps: {len(df)}")
    print(f"Available columns: {list(df.columns)}")
    print("\nData ranges:")
    
    # Joint and actuator data
    for col in ['MOTOR_LEFT', 'MOTOR_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'ACT_LEFT', 'ACT_RIGHT']:
        if col in df.columns:
            print(f"  {col:12s}: [{df[col].min():8.3f}, {df[col].max():8.3f}]")
    
    # Velocity data
    for col in ['VEL_X', 'VEL_Y', 'VEL_Z']:
        if col in df.columns:
            print(f"  {col:12s}: [{df[col].min():8.3f}, {df[col].max():8.3f}]")
    
    print("="*50 + "\n")


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Analyze CSV data and generate professional plots for research papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python results_analyzer.py --csv_file data/results.csv --output_dir plots/
  python results_analyzer.py --csv_file /path/to/data.csv --output_dir /path/to/output/ --window 50
        """
    )
    
    parser.add_argument('--csv_file', 
                       help='Path to the input CSV file')
    
    parser.add_argument('--output_dir', 
                       help='Directory to save the generated plots (defaults to parent directory of input file)')
    
    parser.add_argument('--window',
                       type=int,
                       default=30,
                       help='Window size for rolling average of velocity data (default: 30)')
    
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Print detailed information during processing')
    
    args = parser.parse_args()
    
    # Setup matplotlib style
    setup_matplotlib_style()
    
    try:
        # Validate and load CSV file
        if args.verbose:
            print(f"Loading CSV file: {args.csv_file}")
        
        if not os.path.exists(args.csv_file):
            raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
        
        try:
            df = pd.read_csv(args.csv_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        if args.verbose:
            print_data_summary(df)
        
        # Set default output directory to parent directory of input file if not specified
        if args.output_dir is None:
            args.output_dir = str(Path(args.csv_file).parent)
            if args.verbose:
                print(f"Using default output directory: {args.output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.verbose:
            print(f"Output directory: {output_dir}")
        
        # Check what data is available and generate appropriate plots
        joint_columns = ['MOTOR_LEFT', 'MOTOR_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'ACT_LEFT', 'ACT_RIGHT']
        velocity_columns = ['VEL_X', 'VEL_Y', 'VEL_Z']
        
        has_joint_data = any(col in df.columns for col in joint_columns)
        has_velocity_data = any(col in df.columns for col in velocity_columns)
        
        # Generate joint and actuator plots if data is available
        if has_joint_data:
            required_columns = ['MOTOR_LEFT', 'MOTOR_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'ACT_LEFT', 'ACT_RIGHT']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if not missing_columns:
                print("Generating motor and knee joints plot...")
                plot_motor_and_knee_data(df, str(output_dir))
                
                print("Generating actuator values plot...")
                plot_actuator_data(df, str(output_dir))
            else:
                print(f"[WARN] Missing some joint/actuator columns for complete analysis: {missing_columns}")
        
        # Generate torque plots if columns exist
        print("Generating computed/applied torque plots for actuators...")
        plot_torque_data(df, str(output_dir), side="LEFT")
        plot_torque_data(df, str(output_dir), side="RIGHT")
        
        # Generate velocity plots if data is available
        if has_velocity_data:
            print("Generating velocity analysis...")
            rolling_data = compute_rolling_averages(df, args.window)
            plot_velocity_data(df, rolling_data, args.window, str(output_dir))
            write_velocity_statistics(df, rolling_data, str(output_dir))
        else:
            print("[INFO] No velocity data found in CSV. Skipping velocity analysis.")
        
        if not has_joint_data and not has_velocity_data:
            print("[WARN] No recognized data columns found. Please check your CSV file format.")
        
        print("\n✓ All available analyses completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 