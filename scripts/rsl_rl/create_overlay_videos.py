#!/usr/bin/env python3
"""
Create video overlays with contact force and actuator value animations.
This script generates two videos:
1. Contact force overlay (LEFT/RIGHT X and Z forces)
2. Actuator values overlay (ACT_LEFT and ACT_RIGHT)

Usage: python video_contact_force_overlay.py --results_dir <path> [--contact_force_output <path>] [--actuator_output <path>]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import os

def validate_csv(csv_path, required_cols):
    """Validate CSV has required columns."""
    df = pd.read_csv(csv_path)
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    print(f"CSV validated: {len(df)} rows, columns OK")
    return df


def create_contact_force_overlay_video(video_path, csv_path, output_path):
    """Create video with contact force overlay."""
    
    # Load and validate data
    required_cols = ['CONTACT_FORCE_LEFT_X', 'CONTACT_FORCE_RIGHT_X', 
                     'CONTACT_FORCE_LEFT_Z', 'CONTACT_FORCE_RIGHT_Z']
    df = validate_csv(csv_path, required_cols)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"Video: {width}x{height}, {fps}FPS, {frame_count} frames, {duration:.2f}s")
    
    # Setup output video (video on left, plots on right)
    out_width = width * 2
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Setup matplotlib figure
    plt.style.use('dark_background')
    plot_width_inches = width / 100
    plot_height_inches = height / 100
    fig, (ax_z, ax_x) = plt.subplots(2, 1, 
                                      figsize=(plot_width_inches, plot_height_inches),
                                      facecolor='black', dpi=100)
    
    # Extract contact force data
    time_array = np.linspace(0, duration, len(df))
    cf_data = {
        'LEFT_Z': df['CONTACT_FORCE_LEFT_Z'].values,
        'RIGHT_Z': df['CONTACT_FORCE_RIGHT_Z'].values,
        'LEFT_X': df['CONTACT_FORCE_LEFT_X'].values,
        'RIGHT_X': df['CONTACT_FORCE_RIGHT_X'].values,
    }
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        data_idx = int((current_time / duration) * len(df))
        data_idx = min(data_idx, len(df) - 1)
        
        # Clear axes
        ax_z.clear()
        ax_x.clear()
        for ax in [ax_z, ax_x]:
            ax.set_facecolor('black')
        
        # Plot Z-axis forces (upper half)
        ax_z.plot(time_array, cf_data['LEFT_Z'], 'gray', alpha=0.3, linewidth=1)
        ax_z.plot(time_array, cf_data['RIGHT_Z'], 'gray', alpha=0.3, linewidth=1)
        
        if data_idx > 0:
            ax_z.plot(time_array[:data_idx], cf_data['LEFT_Z'][:data_idx], 
                     'red', linewidth=2, label='LEFT_Z')
            ax_z.plot(time_array[:data_idx], cf_data['RIGHT_Z'][:data_idx], 
                     'blue', linewidth=2, label='RIGHT_Z')
        
        ax_z.plot(current_time, cf_data['LEFT_Z'][data_idx], 'ro', markersize=6)
        ax_z.plot(current_time, cf_data['RIGHT_Z'][data_idx], 'bo', markersize=6)
        
        ax_z.set_xlim(0, duration)
        ax_z.set_ylabel('Contact Force Z (N)', color='white', fontsize=14, fontweight='bold')
        ax_z.tick_params(colors='white', labelsize=12, width=2)
        ax_z.grid(True, alpha=0.3, linewidth=1)
        ax_z.legend(fontsize=10, framealpha=0.9, edgecolor='white', loc='upper right')
        ax_z.set_xticklabels([])
        
        for spine in ax_z.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        
        # Plot X-axis forces (lower half)
        ax_x.plot(time_array, cf_data['LEFT_X'], 'gray', alpha=0.3, linewidth=1)
        ax_x.plot(time_array, cf_data['RIGHT_X'], 'gray', alpha=0.3, linewidth=1)
        
        if data_idx > 0:
            ax_x.plot(time_array[:data_idx], cf_data['LEFT_X'][:data_idx], 
                     'red', linewidth=2, label='LEFT_X')
            ax_x.plot(time_array[:data_idx], cf_data['RIGHT_X'][:data_idx], 
                     'blue', linewidth=2, label='RIGHT_X')
        
        ax_x.plot(current_time, cf_data['LEFT_X'][data_idx], 'ro', markersize=6)
        ax_x.plot(current_time, cf_data['RIGHT_X'][data_idx], 'bo', markersize=6)
        
        ax_x.set_xlim(0, duration)
        ax_x.set_ylabel('Contact Force X (N)', color='white', fontsize=14, fontweight='bold')
        ax_x.set_xlabel('Time (s)', color='white', fontsize=14, fontweight='bold')
        ax_x.tick_params(colors='white', labelsize=12, width=2)
        ax_x.grid(True, alpha=0.3, linewidth=1)
        ax_x.legend(fontsize=10, framealpha=0.9, edgecolor='white', loc='upper right')
        
        for spine in ax_x.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        
        plt.tight_layout(pad=2.0)
        
        # Convert plot to image
        fig.canvas.draw()
        plot_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # Resize plot if needed
        if plot_img.shape[:2] != (height, width):
            plot_img = cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Combine frames horizontally
        combined_frame = np.hstack([frame, plot_img])
        out.write(combined_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Progress: {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    plt.close(fig)
    
    print(f"Output saved: {output_path}")


def create_actuator_overlay_video(video_path, csv_path, output_path):
    """Create video with actuator values overlay."""
    
    # Load and validate data
    required_cols = ['ACT_LEFT', 'ACT_RIGHT']
    df = validate_csv(csv_path, required_cols)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"Video: {width}x{height}, {fps}FPS, {frame_count} frames, {duration:.2f}s")
    
    # Setup output video (video on left, plot on right)
    out_width = width * 2
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Setup matplotlib figure
    plt.style.use('dark_background')
    plot_width_inches = width / 100
    plot_height_inches = height / 100
    fig, ax = plt.subplots(1, 1, 
                           figsize=(plot_width_inches, plot_height_inches),
                           facecolor='black', dpi=100)
    
    # Extract actuator data
    time_array = np.linspace(0, duration, len(df))
    act_data = {
        'LEFT': df['ACT_LEFT'].values,
        'RIGHT': df['ACT_RIGHT'].values,
    }
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        data_idx = int((current_time / duration) * len(df))
        data_idx = min(data_idx, len(df) - 1)
        
        # Clear axis
        ax.clear()
        ax.set_facecolor('black')
        
        # Plot actuator values
        ax.plot(time_array, act_data['LEFT'], 'gray', alpha=0.3, linewidth=1)
        ax.plot(time_array, act_data['RIGHT'], 'gray', alpha=0.3, linewidth=1)
        
        if data_idx > 0:
            ax.plot(time_array[:data_idx], act_data['LEFT'][:data_idx], 
                   'red', linewidth=2, label='ACT_LEFT')
            ax.plot(time_array[:data_idx], act_data['RIGHT'][:data_idx], 
                   'blue', linewidth=2, label='ACT_RIGHT')
        
        ax.plot(current_time, act_data['LEFT'][data_idx], 'ro', markersize=6)
        ax.plot(current_time, act_data['RIGHT'][data_idx], 'bo', markersize=6)
        
        ax.set_xlim(0, duration)
        ax.set_ylabel('Actuator Value', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white', labelsize=12, width=2)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.legend(fontsize=10, framealpha=0.9, edgecolor='white', loc='upper right')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        
        plt.tight_layout(pad=2.0)
        
        # Convert plot to image
        fig.canvas.draw()
        plot_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # Resize plot if needed
        if plot_img.shape[:2] != (height, width):
            plot_img = cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Combine frames horizontally
        combined_frame = np.hstack([frame, plot_img])
        out.write(combined_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Progress: {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    plt.close(fig)
    
    print(f"Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create video overlays with contact force and actuator data')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--contact_force_output', type=str, default=None, 
                       help='Path to contact force output video (default: video_with_contact_forces.mp4)')
    parser.add_argument('--actuator_output', type=str, default=None,
                       help='Path to actuator output video (default: video_with_actuator_values.mp4)')
    
    args = parser.parse_args()
    
    # Set default output paths
    if args.contact_force_output is None:
        args.contact_force_output = os.path.join(args.results_dir, 'video_with_contact_forces.mp4')
    if args.actuator_output is None:
        args.actuator_output = os.path.join(args.results_dir, 'video_with_actuator_values.mp4')
    
    # Get input paths
    video_path = os.path.join(args.results_dir, 'ballu-step-0.mp4')
    csv_path = os.path.join(args.results_dir, 'results.csv')
    
    # Create contact force overlay video
    print("\n" + "="*60)
    print("Creating Contact Force Overlay Video")
    print("="*60)
    try:
        create_contact_force_overlay_video(video_path, csv_path, args.contact_force_output)
    except Exception as e:
        print(f"Error creating contact force video: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create actuator overlay video
    print("\n" + "="*60)
    print("Creating Actuator Values Overlay Video")
    print("="*60)
    try:
        create_actuator_overlay_video(video_path, csv_path, args.actuator_output)
    except Exception as e:
        print(f"Error creating actuator video: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n" + "="*60)
    print("All videos created successfully!")
    print("="*60)
    print(f"Contact Force Video: {args.contact_force_output}")
    print(f"Actuator Values Video: {args.actuator_output}")
    print("="*60)


if __name__ == "__main__":
    main()

