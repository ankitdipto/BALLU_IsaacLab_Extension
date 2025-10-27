#!/usr/bin/env python3
"""
Create video overlay with contact force animations.
Usage: python video_contact_force_overlay.py --video_path <path> --csv_path <path> [--output_path <path>]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import os

def validate_csv(csv_path):
    """Validate CSV has required contact force columns."""
    df = pd.read_csv(csv_path)
    required_cols = ['CONTACT_FORCE_LEFT_X', 'CONTACT_FORCE_RIGHT_X', 
                     'CONTACT_FORCE_LEFT_Z', 'CONTACT_FORCE_RIGHT_Z']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    print(f"CSV validated: {len(df)} rows, columns OK")
    return df


def create_overlay_video(video_path, csv_path, output_path):
    """Create video with contact force overlay."""
    
    # Load and validate data
    df = validate_csv(csv_path)
    
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


def main():
    parser = argparse.ArgumentParser(description='Create video overlay with contact force data')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Path to output video (default: input_with_forces.mp4)')
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join(args.results_dir, 'video_with_contact_forces.mp4')
    
    try:
        video_path = os.path.join(args.results_dir, 'ballu-step-0.mp4')
        csv_path = os.path.join(args.results_dir, 'results.csv')
        create_overlay_video(video_path, csv_path, args.output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

