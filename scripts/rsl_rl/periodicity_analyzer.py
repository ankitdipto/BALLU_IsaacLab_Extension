#!/usr/bin/env python3
"""
Periodicity Analysis Script for Time-Series Data

This script analyzes periodicity in KNEE_LEFT and KNEE_RIGHT joint data using:
- Autocorrelation Function (ACF) analysis
- Fast Fourier Transform (FFT) with Power Spectral Density
- Spectral Entropy calculation
- Comprehensive visualizations

Author: AI Assistant
Date: 2025
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
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
        'knee_left': '#2ca02c',     # Cooked asparagus green
        'knee_right': '#d62728',    # Brick red
        'acf_left': '#1f77b4',      # Muted blue
        'acf_right': '#ff7f0e',     # Safety orange
        'fft_left': '#9467bd',      # Muted purple
        'fft_right': '#8c564b',     # Chestnut brown
    }


def validate_csv_file(csv_path):
    """Validate that the CSV file exists and has required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    required_columns = ['KNEE_LEFT', 'KNEE_RIGHT']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    return df


def calculate_autocorrelation(data, max_lags=None):
    """
    Calculate autocorrelation function for the given data.
    
    Args:
        data: Time series data
        max_lags: Maximum number of lags (default: len(data)//3)
    
    Returns:
        lags: Lag values
        autocorr: Autocorrelation values
        significant_peaks: Significant autocorrelation peaks
    """
    if max_lags is None:
        max_lags = min(len(data)//3, 100)  # Limit for better visualization
    
    # Calculate autocorrelation
    autocorr = acf(data, nlags=max_lags, fft=True)
    lags = np.arange(max_lags + 1)
    
    # Find significant peaks (above certain threshold)
    threshold = 0.2  # Adjust based on data characteristics
    peak_indices, _ = signal.find_peaks(autocorr[1:], height=threshold)
    peak_indices += 1  # Adjust for skipping lag 0
    
    return lags, autocorr, peak_indices


def calculate_fft_spectrum(data, sampling_rate=1.0):
    """
    Calculate FFT power spectrum and related metrics.
    
    Args:
        data: Time series data
        sampling_rate: Sampling rate (default: 1.0)
    
    Returns:
        freqs: Frequency values
        power_spectrum: Power spectral density
        dominant_freqs: Dominant frequency components
        spectral_entropy: Spectral entropy value
    """
    # Remove DC component (mean)
    data_centered = data - np.mean(data)
    
    # Apply window to reduce spectral leakage
    windowed_data = data_centered * signal.windows.hann(len(data_centered))
    
    # Calculate FFT
    fft_values = fft(windowed_data)
    freqs = fftfreq(len(windowed_data), 1/sampling_rate)
    
    # Calculate power spectrum (only positive frequencies)
    n = len(windowed_data)
    power_spectrum = np.abs(fft_values[:n//2])**2
    freqs_positive = freqs[:n//2]
    
    # Find dominant frequencies
    # Remove DC component and find peaks
    peak_indices, properties = signal.find_peaks(
        power_spectrum[1:], 
        height=np.max(power_spectrum[1:]) * 0.1,  # 10% of max power
        distance=max(1, len(power_spectrum) // 50)  # Minimum distance between peaks
    )
    peak_indices += 1  # Adjust for skipping DC
    dominant_freqs = freqs_positive[peak_indices]
    
    # Calculate spectral entropy
    # Normalize power spectrum to get probability distribution
    power_normalized = power_spectrum / np.sum(power_spectrum)
    # Remove zeros to avoid log(0)
    power_normalized = power_normalized[power_normalized > 0]
    spectral_entropy = entropy(power_normalized)
    
    return freqs_positive, power_spectrum, dominant_freqs, spectral_entropy


def plot_time_series(df, output_dir):
    """
    Plot the raw time series data for KNEE_LEFT and KNEE_RIGHT.
    
    Args:
        df: DataFrame containing the data
        output_dir: Output directory for plots
    """
    colors = get_professional_colors()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    time_steps = np.arange(len(df))
    
    # Plot KNEE_LEFT
    ax1.plot(time_steps, df['KNEE_LEFT'], 
             color=colors['knee_left'], 
             label='KNEE_LEFT', 
             linewidth=1.5)
    ax1.set_ylabel('Knee Left Angle (rad)')
    ax1.set_title('KNEE_LEFT Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot KNEE_RIGHT
    ax2.plot(time_steps, df['KNEE_RIGHT'], 
             color=colors['knee_right'], 
             label='KNEE_RIGHT', 
             linewidth=1.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Knee Right Angle (rad)')
    ax2.set_title('KNEE_RIGHT Time Series')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'knee_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Time series plot saved to: {output_path}")


def plot_autocorrelation_analysis(df, output_dir):
    """
    Plot autocorrelation analysis for both knee joints.
    
    Args:
        df: DataFrame containing the data
        output_dir: Output directory for plots
    """
    colors = get_professional_colors()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ACF for KNEE_LEFT
    lags_left, acf_left, peaks_left = calculate_autocorrelation(df['KNEE_LEFT'])
    ax1.plot(lags_left, acf_left, color=colors['acf_left'], linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.2, color='red', linestyle=':', alpha=0.7, label='Significance threshold')
    if len(peaks_left) > 0:
        ax1.scatter(lags_left[peaks_left], acf_left[peaks_left], 
                   color='red', s=50, zorder=5, label=f'Peaks: {len(peaks_left)}')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('ACF - KNEE_LEFT')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ACF for KNEE_RIGHT
    lags_right, acf_right, peaks_right = calculate_autocorrelation(df['KNEE_RIGHT'])
    ax2.plot(lags_right, acf_right, color=colors['acf_right'], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.2, color='red', linestyle=':', alpha=0.7, label='Significance threshold')
    if len(peaks_right) > 0:
        ax2.scatter(lags_right[peaks_right], acf_right[peaks_right], 
                   color='red', s=50, zorder=5, label=f'Peaks: {len(peaks_right)}')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('ACF - KNEE_RIGHT')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Detailed view of first 50 lags for KNEE_LEFT
    max_detailed_lags = min(50, len(lags_left))
    ax3.plot(lags_left[:max_detailed_lags], acf_left[:max_detailed_lags], 
             color=colors['acf_left'], linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.2, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('ACF - KNEE_LEFT (First 50 lags)')
    ax3.grid(True, alpha=0.3)
    
    # Detailed view of first 50 lags for KNEE_RIGHT
    ax4.plot(lags_right[:max_detailed_lags], acf_right[:max_detailed_lags], 
             color=colors['acf_right'], linewidth=2, marker='o', markersize=3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.2, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('ACF - KNEE_RIGHT (First 50 lags)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'autocorrelation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Autocorrelation analysis plot saved to: {output_path}")
    
    return {
        'knee_left': {'lags': lags_left, 'acf': acf_left, 'peaks': peaks_left},
        'knee_right': {'lags': lags_right, 'acf': acf_right, 'peaks': peaks_right}
    }


def plot_fft_analysis(df, output_dir, sampling_rate=1.0):
    """
    Plot FFT power spectrum analysis for both knee joints.
    
    Args:
        df: DataFrame containing the data
        output_dir: Output directory for plots
        sampling_rate: Sampling rate of the data
    """
    colors = get_professional_colors()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # FFT for KNEE_LEFT
    freqs_left, power_left, dom_freqs_left, entropy_left = calculate_fft_spectrum(
        df['KNEE_LEFT'], sampling_rate)
    
    ax1.semilogy(freqs_left, power_left, color=colors['fft_left'], linewidth=2)
    if len(dom_freqs_left) > 0:
        # Find power values for dominant frequencies
        dom_indices = np.searchsorted(freqs_left, dom_freqs_left)
        dom_indices = np.clip(dom_indices, 0, len(power_left)-1)
        ax1.scatter(dom_freqs_left, power_left[dom_indices], 
                   color='red', s=50, zorder=5, label=f'Dominant freqs: {len(dom_freqs_left)}')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'Power Spectrum - KNEE_LEFT (Entropy: {entropy_left:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FFT for KNEE_RIGHT
    freqs_right, power_right, dom_freqs_right, entropy_right = calculate_fft_spectrum(
        df['KNEE_RIGHT'], sampling_rate)
    
    ax2.semilogy(freqs_right, power_right, color=colors['fft_right'], linewidth=2)
    if len(dom_freqs_right) > 0:
        dom_indices = np.searchsorted(freqs_right, dom_freqs_right)
        dom_indices = np.clip(dom_indices, 0, len(power_right)-1)
        ax2.scatter(dom_freqs_right, power_right[dom_indices], 
                   color='red', s=50, zorder=5, label=f'Dominant freqs: {len(dom_freqs_right)}')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title(f'Power Spectrum - KNEE_RIGHT (Entropy: {entropy_right:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Linear scale plots for better visualization of lower frequencies
    ax3.plot(freqs_left, power_left, color=colors['fft_left'], linewidth=2)
    if len(dom_freqs_left) > 0:
        dom_indices = np.searchsorted(freqs_left, dom_freqs_left)
        dom_indices = np.clip(dom_indices, 0, len(power_left)-1)
        ax3.scatter(dom_freqs_left, power_left[dom_indices], 
                   color='red', s=50, zorder=5)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Power Spectrum - KNEE_LEFT (Linear Scale)')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(freqs_right, power_right, color=colors['fft_right'], linewidth=2)
    if len(dom_freqs_right) > 0:
        dom_indices = np.searchsorted(freqs_right, dom_freqs_right)
        dom_indices = np.clip(dom_indices, 0, len(power_right)-1)
        ax4.scatter(dom_freqs_right, power_right[dom_indices], 
                   color='red', s=50, zorder=5)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power Spectral Density')
    ax4.set_title('Power Spectrum - KNEE_RIGHT (Linear Scale)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fft_power_spectrum.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] FFT power spectrum plot saved to: {output_path}")
    
    return {
        'knee_left': {
            'freqs': freqs_left, 
            'power': power_left, 
            'dominant_freqs': dom_freqs_left,
            'spectral_entropy': entropy_left
        },
        'knee_right': {
            'freqs': freqs_right, 
            'power': power_right, 
            'dominant_freqs': dom_freqs_right,
            'spectral_entropy': entropy_right
        }
    }


def plot_combined_analysis(df, acf_results, fft_results, output_dir):
    """
    Create a combined plot showing key periodicity metrics.
    
    Args:
        df: DataFrame containing the data
        acf_results: Results from ACF analysis
        fft_results: Results from FFT analysis
        output_dir: Output directory for plots
    """
    colors = get_professional_colors()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Raw data comparison
    time_steps = np.arange(len(df))
    ax1.plot(time_steps, df['KNEE_LEFT'], 
             color=colors['knee_left'], label='KNEE_LEFT', linewidth=1.5, alpha=0.8)
    ax1.plot(time_steps, df['KNEE_RIGHT'], 
             color=colors['knee_right'], label='KNEE_RIGHT', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Knee Angle (rad)')
    ax1.set_title('Raw Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ACF comparison
    max_lags = min(50, len(acf_results['knee_left']['lags']))
    ax2.plot(acf_results['knee_left']['lags'][:max_lags], 
             acf_results['knee_left']['acf'][:max_lags],
             color=colors['acf_left'], label='KNEE_LEFT ACF', linewidth=2)
    ax2.plot(acf_results['knee_right']['lags'][:max_lags], 
             acf_results['knee_right']['acf'][:max_lags],
             color=colors['acf_right'], label='KNEE_RIGHT ACF', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Autocorrelation Function Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Power spectrum comparison (log scale)
    ax3.semilogy(fft_results['knee_left']['freqs'], 
                 fft_results['knee_left']['power'],
                 color=colors['fft_left'], label='KNEE_LEFT', linewidth=2)
    ax3.semilogy(fft_results['knee_right']['freqs'], 
                 fft_results['knee_right']['power'],
                 color=colors['fft_right'], label='KNEE_RIGHT', linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Power Spectrum Comparison (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Spectral entropy comparison
    joint_names = ['KNEE_LEFT', 'KNEE_RIGHT']
    entropies = [fft_results['knee_left']['spectral_entropy'], 
                 fft_results['knee_right']['spectral_entropy']]
    bars = ax4.bar(joint_names, entropies, 
                   color=[colors['knee_left'], colors['knee_right']], alpha=0.7)
    ax4.set_ylabel('Spectral Entropy')
    ax4.set_title('Spectral Entropy Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_periodicity_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Combined analysis plot saved to: {output_path}")


def print_analysis_summary(acf_results, fft_results):
    """Print a comprehensive summary of the periodicity analysis."""
    print("\n" + "="*60)
    print("PERIODICITY ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n--- AUTOCORRELATION ANALYSIS ---")
    for joint_name, data in [('KNEE_LEFT', acf_results['knee_left']), 
                             ('KNEE_RIGHT', acf_results['knee_right'])]:
        print(f"\n{joint_name}:")
        print(f"  • Number of significant peaks: {len(data['peaks'])}")
        if len(data['peaks']) > 0:
            peak_lags = data['lags'][data['peaks']]
            peak_values = data['acf'][data['peaks']]
            print(f"  • Peak lags: {peak_lags}")
            print(f"  • Peak values: {[f'{val:.3f}' for val in peak_values]}")
            # Estimate period from first significant peak
            if len(peak_lags) > 0:
                estimated_period = peak_lags[0]
                print(f"  • Estimated period: {estimated_period} time steps")
        else:
            print("  • No significant periodic patterns detected")
    
    print("\n--- FOURIER ANALYSIS ---")
    for joint_name, data in [('KNEE_LEFT', fft_results['knee_left']), 
                             ('KNEE_RIGHT', fft_results['knee_right'])]:
        print(f"\n{joint_name}:")
        print(f"  • Spectral entropy: {data['spectral_entropy']:.4f}")
        print(f"  • Number of dominant frequencies: {len(data['dominant_freqs'])}")
        if len(data['dominant_freqs']) > 0:
            print(f"  • Dominant frequencies: {[f'{freq:.4f}' for freq in data['dominant_freqs'][:5]]}")
            # Convert to periods
            periods = 1.0 / data['dominant_freqs'][data['dominant_freqs'] > 0]
            print(f"  • Corresponding periods: {[f'{period:.2f}' for period in periods[:5]]}")
        else:
            print("  • No dominant frequencies detected")
    
    print("\n--- INTERPRETATION ---")
    left_entropy = fft_results['knee_left']['spectral_entropy']
    right_entropy = fft_results['knee_right']['spectral_entropy']
    
    print(f"• Spectral entropy comparison:")
    if abs(left_entropy - right_entropy) < 0.1:
        print("  - Similar complexity in both knee joints")
    elif left_entropy > right_entropy:
        print("  - KNEE_LEFT shows more complex/irregular behavior")
    else:
        print("  - KNEE_RIGHT shows more complex/irregular behavior")
    
    print(f"• Periodicity assessment:")
    avg_entropy = (left_entropy + right_entropy) / 2
    if avg_entropy < 2.0:
        print("  - Strong periodic patterns detected")
    elif avg_entropy < 4.0:
        print("  - Moderate periodic patterns with some irregularity")
    else:
        print("  - Highly irregular/chaotic behavior with weak periodicity")


def main():
    """Main function to run the periodicity analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze periodicity in KNEE_LEFT and KNEE_RIGHT time-series data"
    )
    parser.add_argument(
        "--csv_file", 
        help="Path to CSV file containing time-series data"
    )
    parser.add_argument(
        "--output_dir", 
        default="./periodicity_analysis_output",
        help="Output directory for plots and results (default: ./periodicity_analysis_output)"
    )
    parser.add_argument(
        "--sampling_rate", 
        type=float, 
        default=1.0,
        help="Sampling rate of the data in Hz (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Setup matplotlib style
    setup_matplotlib_style()
    
    # Validate input file
    try:
        df = validate_csv_file(args.csv_file)
        print(f"[INFO] Successfully loaded data from: {args.csv_file}")
        print(f"[INFO] Data shape: {df.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    try:
        # Generate plots and analyses
        print("\n[INFO] Generating time series plot...")
        plot_time_series(df, output_dir)
        
        print("[INFO] Performing autocorrelation analysis...")
        acf_results = plot_autocorrelation_analysis(df, output_dir)
        
        print("[INFO] Performing FFT power spectrum analysis...")
        fft_results = plot_fft_analysis(df, output_dir, args.sampling_rate)
        
        print("[INFO] Generating combined analysis plot...")
        plot_combined_analysis(df, acf_results, fft_results, output_dir)
        
        # Print summary
        print_analysis_summary(acf_results, fft_results)
        
        print(f"\n[SUCCESS] Periodicity analysis completed successfully!")
        print(f"[INFO] All plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()