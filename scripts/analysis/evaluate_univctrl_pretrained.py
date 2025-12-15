"""Script to evaluate RL policy across different GCR (Gravity Compensation Ratio) values.

This script calls play_universal.py as a subprocess for each GCR value and aggregates the success rates.
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional
import random

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)


def run_play_universal(
    gcr: float,
    task: str,
    load_run: str,
    cmdir: str,
    checkpoint: str = "model_best.pt",
    num_envs: int = 1,
    video_length: int = 399,
    device: str = "cuda:0",
    difficulty_level: int = -1,
    spcf: float = None,
) -> Tuple[bool, float, str]:
    """Run play_universal.py for a given GCR value and return success status, success rate, and output.
    
    Args:
        gcr: Gravity compensation ratio value to test
        task: Task name
        experiment_name: Experiment name for loading checkpoint
        load_run: Run name to load checkpoint from
        checkpoint: Checkpoint file name (default: model_best.pt)
        num_envs: Number of environments to simulate
        video_length: Length of video recording in steps
        device: Device to run on
        difficulty_level: Difficulty level (-1 for auto from checkpoint)
        cmdir: Common directory name
        video: Whether to record video
        headless: Whether to run in headless mode
        
    Returns:
        Tuple of (success_status, success_rate, output_text)
    """
    play_script_path = os.path.join(project_dir, "scripts", "rsl_rl", "play_universal.py")
    
    cmd = [
        sys.executable,
        play_script_path,
        "--task", task,
        "--GCR", str(gcr),
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--video",
        "--video_length", str(video_length),
        "--device", device,
        "--difficulty_level", str(difficulty_level),
        "--headless",
        "--cmdir", cmdir,
        "--spcf", str(spcf),
    ]
    
    env = os.environ.copy()
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'
    
    print(f"\n{'='*80}")
    print(f"[GCR={gcr:.4f}] Running play_universal.py...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    output_lines = []
    success_rate = None
    
    try:
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
            # Parse success rate from output
            if "SUCCESS_RATE:" in line:
                try:
                    success_rate_str = line.split("SUCCESS_RATE:")[1].strip()
                    success_rate = float(success_rate_str)
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse success rate from line: {line.strip()}")
        
        process.wait(timeout=2 * 3600)  # 2 hour timeout
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Process timeout (2h) for GCR={gcr:.4f}, killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), ''.join(output_lines)
    
    output_text = ''.join(output_lines)
    success = process.returncode == 0
    
    if success_rate is None:
        print(f"[WARNING] Could not find SUCCESS_RATE in output for GCR={gcr:.4f}")
        success_rate = float('-inf')
    
    return success, success_rate, output_text


def main():
    """Main function to run GCR sweep evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL policy across different GCR values"
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=True,
        help="Number of test cases to run"
    )
    parser.add_argument(
        "--gcr_range",
        type=float,
        nargs=2,
        default=[0.75, 0.90],
        help="Range of GCR values to test (default: [0.75, 0.90])"
    )
    parser.add_argument(
        "--spcf_range",
        type=float,
        nargs=2,
        default=[1e-3, 1e-2],
        help="Range of spring coefficient values to test (default: [1e-3, 1e-2])"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isc-BALLU-hetero-pretrain",
        help="Task name (default: Isc-BALLU-hetero-pretrain)"
    )
    parser.add_argument(
        "--load_run",
        type=str,
        required=True,
        help="Run name to load checkpoint from"
    )
    parser.add_argument(
        "--cmdir",
        type=str,
        default=None,
        help="Common directory name (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_best.pt",
        help="Checkpoint file name (default: model_best.pt)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to simulate (default: 1)"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=399,
        help="Length of video recording in steps (default: 399)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--difficulty_level",
        type=int,
        default=-1,
        help="Difficulty level (-1 for auto from checkpoint, default: -1)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file to save results (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    if args.cmdir is None:
        timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        args.cmdir = f"{timestamp}_univctrl_test_Ht{args.difficulty_level}"

    print(f"\n{'='*80}")
    print("GCR SWEEP EVALUATION")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Common directory: {args.cmdir}")
    print(f"Run: {args.load_run}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GCR range: {args.gcr_range}")
    print(f"Spring coefficient range: {args.spcf_range}")
    print(f"Number of trials: {args.trials}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Video length: {args.video_length}")
    print(f"Device: {args.device}")
    print(f"Difficulty level: {args.difficulty_level}")
    print(f"{'='*80}\n")
    
    # Run evaluation for each GCR value
    results = []
    for _ in range(args.trials):
        gcr = random.uniform(args.gcr_range[0], args.gcr_range[1])
        spcf = random.uniform(args.spcf_range[0], args.spcf_range[1])
        success, success_rate, output = run_play_universal(
            gcr=gcr,
            task=args.task,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
            num_envs=args.num_envs,
            video_length=args.video_length,
            device=args.device,
            difficulty_level=args.difficulty_level,
            cmdir=args.cmdir,
            spcf=spcf,
        )
        
        results.append({
            "gcr": gcr,
            "success": success,
            "success_rate": success_rate,
            "output": output if not success else None,  # Only store output on failure
        })
        
        print(f"\n[GCR={gcr:.4f}] Success: {success}, Success Rate: {success_rate:.6f}")
    
    # Aggregate results
    successful_runs = [r for r in results if r["success"]]
    success_rates = [r["success_rate"] for r in successful_runs if r["success_rate"] != float('-inf')]
    
    if success_rates:
        avg_success_rate = sum(success_rates) / len(success_rates)
        max_success_rate = max(success_rates)
        min_success_rate = min(success_rates)
        best_gcr = max(successful_runs, key=lambda x: x["success_rate"])["gcr"]
    else:
        avg_success_rate = float('-inf')
        max_success_rate = float('-inf')
        min_success_rate = float('-inf')
        best_gcr = None
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")
    print(f"Total GCR values tested: {len(results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(results) - len(successful_runs)}")
    if success_rates:
        print(f"\nSuccess Rate Statistics:")
        print(f"  Average: {avg_success_rate:.6f}")
        print(f"  Maximum: {max_success_rate:.6f} (GCR={best_gcr:.4f})")
        print(f"  Minimum: {min_success_rate:.6f}")
        print(f"\nBest GCR: {best_gcr:.4f} with success rate: {max_success_rate:.6f}")
    else:
        print("\nNo successful runs to aggregate.")
    print(f"{'='*80}\n")
    
    # Print detailed results
    print("Detailed Results:")
    print("-" * 80)
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} GCR={result['gcr']:.4f}: Success Rate={result['success_rate']:.6f}")
    print("-" * 80)
    
    # Save results to JSON file
    if args.output_file is None:
        output_dir = os.path.join(project_dir, "logs", "rsl_rl", "lab_12.02.2025", args.load_run, args.cmdir)
        # os.makedirs(output_dir, exist_ok=True)
        assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"
        args.output_file = os.path.join(output_dir, "univctrl_test.json")
    
    summary = {
        "aggregated_results": {
            "total_tested": len(results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(results) - len(successful_runs),
            "average_success_rate": avg_success_rate if success_rates else None,
            "max_success_rate": max_success_rate if success_rates else None,
            "min_success_rate": min_success_rate if success_rates else None,
            "best_gcr": best_gcr,
        },
        "detailed_results": [
            {
                "gcr": r["gcr"],
                "success": r["success"],
                "success_rate": r["success_rate"],
            }
            for r in results
        ],
    }
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")
    print(f"{'='*80}\n✓ GCR sweep evaluation completed!")


if __name__ == "__main__":
    main()

