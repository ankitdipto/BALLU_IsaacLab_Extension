#!/usr/bin/env python3
"""Real-time visualization of MoE gate probabilities during training.

This script reads the gate_probs.jsonl file written by the on_policy_runner
and displays/saves plots showing how gate probabilities evolve
across training iterations for the first N environments.

Usage (with display):
    python visualize_gate_probs.py --log-dir /path/to/experiment/logs

Usage (headless - saves to file):
    python visualize_gate_probs.py --log-dir /path/to/logs --headless
    python visualize_gate_probs.py --log-dir /path/to/logs --headless --output-dir ./plots
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Use non-interactive backend for headless mode (must be set before importing pyplot)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize MoE gate probabilities during training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to the experiment log directory containing gate_probs.jsonl",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=2.0,
        help="How often to refresh the plot (seconds)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Maximum number of data points to display (older points are dropped)",
    )
    parser.add_argument(
        "--show-morphology",
        action="store_true",
        default=True,
        help="Display morphology vectors in subplot titles",
    )
    parser.add_argument(
        "--no-show-morphology",
        action="store_false",
        dest="show_morphology",
        help="Hide morphology vectors from subplot titles",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode (save plots to files instead of displaying)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot images (default: same as log-dir)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Generate plot once and exit (don't loop)",
    )
    return parser.parse_args()


class GateProbsVisualizer:
    """Real-time visualizer for MoE gate probabilities."""
    
    def __init__(
        self,
        log_file: str,
        refresh_rate: float = 2.0,
        max_points: int = 500,
        show_morphology: bool = True,
        headless: bool = False,
        output_dir: str = None,
        run_once: bool = False,
    ):
        """Initialize the visualizer.
        
        Args:
            log_file: Path to the gate_probs.jsonl file.
            refresh_rate: How often to refresh the plot (seconds).
            max_points: Maximum number of data points to display.
            show_morphology: Whether to display morphology vectors in titles.
            headless: If True, save plots to files instead of displaying.
            output_dir: Directory to save plot images (headless mode).
            run_once: If True, generate plot once and exit.
        """
        self.log_file = log_file
        self.refresh_rate = refresh_rate
        self.max_points = max_points
        self.show_morphology = show_morphology
        self.headless = headless
        self.output_dir = output_dir or os.path.dirname(log_file)
        self.run_once = run_once
        
        # Data storage
        self.header = None
        self.iterations = []
        self.temperatures = []
        self.gate_probs = {}  # {env_idx: {expert_idx: [probs]}}
        self.gate_logits = {}  # {env_idx: {expert_idx: [logits]}}
        self.gumbel_noise = {}  # {env_idx: {expert_idx: [noise]}}
        self.morphologies = {}  # {env_idx: [morph_vector]}
        
        # File reading state
        self.last_file_pos = 0
        self.last_read_time = 0
        
        # Matplotlib setup - gate probs figure
        self.fig = None
        self.axes = None
        self.lines = {}  # {env_idx: {expert_idx: line}}
        self.temp_text = None
        
        # Matplotlib setup - gate logits figure
        self.fig_logits = None
        self.axes_logits = None
        self.lines_logits = {}  # {env_idx: {expert_idx: line}}
        self.temp_text_logits = None
        
        # Matplotlib setup - gumbel noise figure
        self.fig_gumbel = None
        self.axes_gumbel = None
        self.lines_gumbel = {}  # {env_idx: {expert_idx: line}}
        self.temp_text_gumbel = None
        
        # Colors for experts
        self.expert_colors = plt.cm.tab10.colors
        
    def read_log_file(self):
        """Read new entries from the log file."""
        if not os.path.exists(self.log_file):
            return False
        
        with open(self.log_file, "r") as f:
            f.seek(self.last_file_pos)
            new_lines = f.readlines()
            self.last_file_pos = f.tell()
        
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if entry.get("type") == "header":
                self.header = entry
                self._init_data_storage()
            elif entry.get("type") == "data":
                self._process_data_entry(entry)
        
        return len(new_lines) > 0
    
    def _init_data_storage(self):
        """Initialize data storage based on header info."""
        num_envs = self.header["num_envs_logged"]
        num_experts = self.header["num_experts"]
        
        for env_idx in range(num_envs):
            self.gate_probs[env_idx] = {exp: [] for exp in range(num_experts)}
            self.gate_logits[env_idx] = {exp: [] for exp in range(num_experts)}
            self.gumbel_noise[env_idx] = {exp: [] for exp in range(num_experts)}
            self.morphologies[env_idx] = None
    
    def _process_data_entry(self, entry: dict):
        """Process a data entry from the log file."""
        iteration = entry["iteration"]
        temperature = entry.get("temperature", 1.0)
        
        self.iterations.append(iteration)
        self.temperatures.append(temperature)
        
        # Trim to max_points
        if len(self.iterations) > self.max_points:
            trim_count = len(self.iterations) - self.max_points
            self.iterations = self.iterations[trim_count:]
            self.temperatures = self.temperatures[trim_count:]
            for env_idx in self.gate_probs:
                for exp_idx in self.gate_probs[env_idx]:
                    self.gate_probs[env_idx][exp_idx] = \
                        self.gate_probs[env_idx][exp_idx][trim_count:]
                for exp_idx in self.gate_logits[env_idx]:
                    self.gate_logits[env_idx][exp_idx] = \
                        self.gate_logits[env_idx][exp_idx][trim_count:]
                for exp_idx in self.gumbel_noise[env_idx]:
                    self.gumbel_noise[env_idx][exp_idx] = \
                        self.gumbel_noise[env_idx][exp_idx][trim_count:]
        
        # Extract per-environment data
        for env_idx in self.gate_probs:
            probs_key = f"env_{env_idx}_gate_probs"
            logits_key = f"env_{env_idx}_gate_logits"
            gumbel_key = f"env_{env_idx}_gumbel_noise"
            morph_key = f"env_{env_idx}_morphology"
            
            if probs_key in entry:
                probs = entry[probs_key]
                for exp_idx, prob in enumerate(probs):
                    self.gate_probs[env_idx][exp_idx].append(prob)
            
            if logits_key in entry:
                logits = entry[logits_key]
                for exp_idx, logit in enumerate(logits):
                    self.gate_logits[env_idx][exp_idx].append(logit)
            
            if gumbel_key in entry:
                gumbel = entry[gumbel_key]
                for exp_idx, noise in enumerate(gumbel):
                    self.gumbel_noise[env_idx][exp_idx].append(noise)
            
            if morph_key in entry:
                self.morphologies[env_idx] = entry[morph_key]
    
    def setup_plot(self):
        """Set up the matplotlib figure and axes."""
        if self.header is None:
            return False
        
        num_envs = self.header["num_envs_logged"]
        num_experts = self.header["num_experts"]
        routing_type = self.header.get("routing_type", "unknown")
        
        # Create figure with subplots
        nrows = 2
        ncols = (num_envs + 1) // 2
        self.fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )
        self.axes = axes.flatten()
        
        # Set up each subplot
        for env_idx in range(num_envs):
            ax = self.axes[env_idx]
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gate Probability")
            ax.set_title(f"Environment {env_idx}")
            ax.grid(True, alpha=0.3)
            
            # Create lines for each expert
            self.lines[env_idx] = {}
            for exp_idx in range(num_experts):
                color = self.expert_colors[exp_idx % len(self.expert_colors)]
                line, = ax.plot([], [], label=f"Expert {exp_idx}", color=color, linewidth=1.5)
                self.lines[env_idx][exp_idx] = line
            
            ax.legend(loc="upper right", fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_envs, len(self.axes)):
            self.axes[idx].set_visible(False)
        
        # Add temperature display
        self.temp_text = self.fig.text(
            0.02, 0.98,
            f"Temperature: --  |  Routing: {routing_type}",
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        
        self.fig.suptitle("MoE Gate Probabilities Over Training", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return True
    
    def setup_plot_logits(self):
        """Set up the matplotlib figure and axes for gate logits."""
        if self.header is None:
            return False
        
        num_envs = self.header["num_envs_logged"]
        num_experts = self.header["num_experts"]
        routing_type = self.header.get("routing_type", "unknown")
        
        # Create figure with subplots
        nrows = 2
        ncols = (num_envs + 1) // 2
        self.fig_logits, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )
        self.axes_logits = axes.flatten()
        
        # Set up each subplot
        for env_idx in range(num_envs):
            ax = self.axes_logits[env_idx]
            ax.set_xlim(0, 100)
            ax.set_ylim(-5, 5)  # Logits can be negative, start with reasonable range
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gate Logit")
            ax.set_title(f"Environment {env_idx}")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # Zero line
            
            # Create lines for each expert
            self.lines_logits[env_idx] = {}
            for exp_idx in range(num_experts):
                color = self.expert_colors[exp_idx % len(self.expert_colors)]
                line, = ax.plot([], [], label=f"Expert {exp_idx}", color=color, linewidth=1.5)
                self.lines_logits[env_idx][exp_idx] = line
            
            ax.legend(loc="upper right", fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_envs, len(self.axes_logits)):
            self.axes_logits[idx].set_visible(False)
        
        # Add temperature display
        self.temp_text_logits = self.fig_logits.text(
            0.02, 0.98,
            f"Temperature: --  |  Routing: {routing_type}",
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        
        self.fig_logits.suptitle("MoE Gate Logits Over Training", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return True
    
    def setup_plot_gumbel(self):
        """Set up the matplotlib figure and axes for Gumbel noise."""
        if self.header is None:
            return False
        
        num_envs = self.header["num_envs_logged"]
        num_experts = self.header["num_experts"]
        routing_type = self.header.get("routing_type", "unknown")
        
        # Create figure with subplots
        nrows = 2
        ncols = (num_envs + 1) // 2
        self.fig_gumbel, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )
        self.axes_gumbel = axes.flatten()
        
        # Set up each subplot
        for env_idx in range(num_envs):
            ax = self.axes_gumbel[env_idx]
            ax.set_xlim(0, 100)
            ax.set_ylim(-3, 5)  # Gumbel noise typical range
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gumbel Noise")
            ax.set_title(f"Environment {env_idx}")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)  # Zero line
            
            # Create lines for each expert
            self.lines_gumbel[env_idx] = {}
            for exp_idx in range(num_experts):
                color = self.expert_colors[exp_idx % len(self.expert_colors)]
                line, = ax.plot([], [], label=f"Expert {exp_idx}", color=color, linewidth=1.5)
                self.lines_gumbel[env_idx][exp_idx] = line
            
            ax.legend(loc="upper right", fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_envs, len(self.axes_gumbel)):
            self.axes_gumbel[idx].set_visible(False)
        
        # Add temperature display
        self.temp_text_gumbel = self.fig_gumbel.text(
            0.02, 0.98,
            f"Temperature: --  |  Routing: {routing_type}",
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        
        self.fig_gumbel.suptitle("MoE Gumbel Noise Over Training", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return True
    
    def update_plot(self, frame=None):
        """Update the plot with new data."""
        # Read new data
        self.read_log_file()
        
        if not self.iterations:
            return list(self.lines.values())
        
        # Update x-axis limits
        min_iter = min(self.iterations)
        max_iter = max(self.iterations)
        x_margin = max(10, (max_iter - min_iter) * 0.05)
        
        for env_idx, ax in enumerate(self.axes):
            if env_idx >= len(self.gate_probs):
                continue
            
            ax.set_xlim(min_iter - x_margin, max_iter + x_margin)
            
            # Update lines
            for exp_idx in self.gate_probs[env_idx]:
                probs = self.gate_probs[env_idx][exp_idx]
                if probs:
                    self.lines[env_idx][exp_idx].set_data(
                        self.iterations[:len(probs)],
                        probs,
                    )
            
            # Update title with morphology
            if self.show_morphology and self.morphologies.get(env_idx) is not None:
                morph = self.morphologies[env_idx]
                # Format morphology vector (show first few values)
                morph_str = ", ".join([f"{v:.2f}" for v in morph[:5]])
                if len(morph) > 5:
                    morph_str += ", ..."
                ax.set_title(f"Env {env_idx} | Morph: [{morph_str}]", fontsize=10)
        
        # Update temperature text
        if self.temperatures:
            current_temp = self.temperatures[-1]
            current_iter = self.iterations[-1]
            routing_type = self.header.get("routing_type", "unknown")
            self.temp_text.set_text(
                f"Iteration: {current_iter}  |  Temperature: {current_temp:.4f}  |  Routing: {routing_type}"
            )
        
        self.fig.canvas.draw_idle()
        
        # Update gate logits figure
        self._update_logits_plot()
        
        # Update gumbel noise figure
        self._update_gumbel_plot()
        
        return []
    
    def _update_logits_plot(self):
        """Update the gate logits plot."""
        if self.fig_logits is None or not self.iterations:
            return
        
        min_iter = min(self.iterations)
        max_iter = max(self.iterations)
        x_margin = max(10, (max_iter - min_iter) * 0.05)
        
        # Track global min/max logits for y-axis scaling
        all_logits = []
        
        for env_idx, ax in enumerate(self.axes_logits):
            if env_idx >= len(self.gate_logits):
                continue
            
            ax.set_xlim(min_iter - x_margin, max_iter + x_margin)
            
            # Update lines
            for exp_idx in self.gate_logits[env_idx]:
                logits = self.gate_logits[env_idx][exp_idx]
                if logits:
                    self.lines_logits[env_idx][exp_idx].set_data(
                        self.iterations[:len(logits)],
                        logits,
                    )
                    all_logits.extend(logits)
            
            # Update title with morphology
            if self.show_morphology and self.morphologies.get(env_idx) is not None:
                morph = self.morphologies[env_idx]
                morph_str = ", ".join([f"{v:.2f}" for v in morph[:5]])
                if len(morph) > 5:
                    morph_str += ", ..."
                ax.set_title(f"Env {env_idx} | Morph: [{morph_str}]", fontsize=10)
        
        # Auto-scale y-axis based on logit values
        if all_logits:
            min_logit = min(all_logits)
            max_logit = max(all_logits)
            y_margin = max(0.5, (max_logit - min_logit) * 0.1)
            for env_idx, ax in enumerate(self.axes_logits):
                if env_idx < len(self.gate_logits):
                    ax.set_ylim(min_logit - y_margin, max_logit + y_margin)
        
        # Update temperature text
        if self.temperatures:
            current_temp = self.temperatures[-1]
            current_iter = self.iterations[-1]
            routing_type = self.header.get("routing_type", "unknown")
            self.temp_text_logits.set_text(
                f"Iteration: {current_iter}  |  Temperature: {current_temp:.4f}  |  Routing: {routing_type}"
            )
        
        self.fig_logits.canvas.draw_idle()
    
    def _update_gumbel_plot(self):
        """Update the Gumbel noise plot."""
        if self.fig_gumbel is None or not self.iterations:
            return
        
        min_iter = min(self.iterations)
        max_iter = max(self.iterations)
        x_margin = max(10, (max_iter - min_iter) * 0.05)
        
        # Track global min/max noise for y-axis scaling
        all_noise = []
        
        for env_idx, ax in enumerate(self.axes_gumbel):
            if env_idx >= len(self.gumbel_noise):
                continue
            
            ax.set_xlim(min_iter - x_margin, max_iter + x_margin)
            
            # Update lines
            for exp_idx in self.gumbel_noise[env_idx]:
                noise = self.gumbel_noise[env_idx][exp_idx]
                if noise:
                    self.lines_gumbel[env_idx][exp_idx].set_data(
                        self.iterations[:len(noise)],
                        noise,
                    )
                    all_noise.extend(noise)
            
            # Update title with morphology
            if self.show_morphology and self.morphologies.get(env_idx) is not None:
                morph = self.morphologies[env_idx]
                morph_str = ", ".join([f"{v:.2f}" for v in morph[:5]])
                if len(morph) > 5:
                    morph_str += ", ..."
                ax.set_title(f"Env {env_idx} | Morph: [{morph_str}]", fontsize=10)
        
        # Auto-scale y-axis based on noise values
        if all_noise:
            min_noise = min(all_noise)
            max_noise = max(all_noise)
            y_margin = max(0.5, (max_noise - min_noise) * 0.1)
            for env_idx, ax in enumerate(self.axes_gumbel):
                if env_idx < len(self.gumbel_noise):
                    ax.set_ylim(min_noise - y_margin, max_noise + y_margin)
        
        # Update temperature text
        if self.temperatures:
            current_temp = self.temperatures[-1]
            current_iter = self.iterations[-1]
            routing_type = self.header.get("routing_type", "unknown")
            self.temp_text_gumbel.set_text(
                f"Iteration: {current_iter}  |  Temperature: {current_temp:.4f}  |  Routing: {routing_type}"
            )
        
        self.fig_gumbel.canvas.draw_idle()
    
    def save_plot(self):
        """Save the current plot to a file."""
        if self.iterations:
            current_iter = self.iterations[-1]
        else:
            current_iter = 0
        
        # Save gate probs figure
        output_path_probs = os.path.join(self.output_dir, "gate_probs_latest.png")
        self.fig.savefig(output_path_probs, dpi=100, bbox_inches='tight')
        
        # Save gate logits figure
        output_path_logits = os.path.join(self.output_dir, "gate_logits_latest.png")
        if self.fig_logits is not None:
            self.fig_logits.savefig(output_path_logits, dpi=100, bbox_inches='tight')
        
        # Save gumbel noise figure
        output_path_gumbel = os.path.join(self.output_dir, "gumbel_noise_latest.png")
        if self.fig_gumbel is not None:
            self.fig_gumbel.savefig(output_path_gumbel, dpi=100, bbox_inches='tight')
        
        return output_path_probs, output_path_logits, output_path_gumbel
    
    def run(self):
        """Run the visualization."""
        print(f"Waiting for log file: {self.log_file}")
        
        # Wait for the log file to appear
        while not os.path.exists(self.log_file):
            time.sleep(0.5)
        
        print("Log file found. Reading initial data...")
        
        # Read initial data and wait for header
        while self.header is None:
            self.read_log_file()
            time.sleep(0.5)
        
        print(f"Header loaded: {self.header}")
        
        # Set up the plots
        if not self.setup_plot():
            print("Error: Could not set up gate probs plot. Check the log file format.")
            return
        
        if not self.setup_plot_logits():
            print("Error: Could not set up gate logits plot. Check the log file format.")
            return
        
        if not self.setup_plot_gumbel():
            print("Error: Could not set up gumbel noise plot. Check the log file format.")
            return
        
        if self.headless:
            # Headless mode: save plots to files
            print(f"Running in headless mode. Saving plots to: {self.output_dir}")
            print(f"Refresh rate: {self.refresh_rate}s")
            if self.run_once:
                print("Running once and exiting...")
            else:
                print("Press Ctrl+C to stop.")
            
            try:
                while True:
                    self.update_plot()
                    output_probs, output_logits, output_gumbel = self.save_plot()
                    
                    if self.iterations:
                        print(f"[Iter {self.iterations[-1]}] Saved: probs, logits, gumbel")
                    
                    if self.run_once:
                        break
                    
                    time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            # Interactive mode: use matplotlib animation
            import matplotlib.animation as animation
            
            # Switch to interactive backend
            plt.switch_backend('TkAgg')
            
            ani = animation.FuncAnimation(
                self.fig,
                self.update_plot,
                interval=int(self.refresh_rate * 1000),
                blit=False,
                cache_frame_data=False,
            )
            
            print(f"Visualization started. Refreshing every {self.refresh_rate}s.")
            print("Close the window to exit.")
            
            plt.show()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Construct log file path
    log_file = os.path.join(args.log_dir, "gate_probs.jsonl")
    
    # Check if log directory exists
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory does not exist: {args.log_dir}")
        print("Make sure to provide the correct path to the experiment log directory.")
        sys.exit(1)
    
    # Determine output directory
    output_dir = args.output_dir or args.log_dir
    
    print(f"Log directory: {args.log_dir}")
    print(f"Log file: {log_file}")
    print(f"Refresh rate: {args.refresh_rate}s")
    print(f"Max points: {args.max_points}")
    print(f"Show morphology: {args.show_morphology}")
    print(f"Headless mode: {args.headless}")
    if args.headless:
        print(f"Output directory: {output_dir}")
    
    # Create and run visualizer
    visualizer = GateProbsVisualizer(
        log_file=log_file,
        refresh_rate=args.refresh_rate,
        max_points=args.max_points,
        show_morphology=args.show_morphology,
        headless=args.headless,
        output_dir=output_dir,
        run_once=args.once,
    )
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")


if __name__ == "__main__":
    main()
