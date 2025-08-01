#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Single Morphology Multi-Seed Training Experiments

This script generates a single BALLU morphology and runs multi-seed training experiments on it.
It's a simplified version of the full morphology study workflow for focused experiments.

Usage:
    python single_morphology_multi_seed.py --ratio "3:7" --seeds 42 123 456 --max_iterations 2000
    python single_morphology_multi_seed.py --ratio "1:1" --seeds 1 2 3 4 5 --task Isaac-Vel-BALLU-imu-tibia
"""

import argparse
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add the morphology workflow to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from morphology_study_workflow import MorphologyStudyWorkflow


class SingleMorphologyMultiSeed:
    """
    Manages multi-seed training experiments for a single BALLU morphology.
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the single morphology multi-seed workflow."""
        self.workflow = MorphologyStudyWorkflow(base_dir=base_dir)
        self.base_dir = self.workflow.base_dir
        
    def run_multi_seed_experiment(self, ratio: str, seeds: List[int], task: str,
                                max_iterations: int, num_envs: int,
                                additional_args: List[str] = None,
                                run_testing: bool = True) -> Dict[str, Any]:
        """
        Run multi-seed training experiments on a single morphology.
        
        Args:
            ratio: Femur:tibia ratio string (e.g., "3:7")
            seeds: List of random seeds
            task: Task name for training
            max_iterations: Maximum training iterations
            num_envs: Number of environments
            additional_args: Additional arguments for training
            run_testing: Whether to run testing/evaluation after training (default: True)
            
        Returns:
            Dictionary containing all results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        femur_ratio, tibia_ratio = self.workflow.parse_ratio(ratio)
        # Handle decimal ratios by replacing dots with underscores
        femur_str = str(femur_ratio).replace('.', '_')
        tibia_str = str(tibia_ratio).replace('.', '_')
        ratio_name = f"FT_{femur_str}_{tibia_str}"
        experiment_id = f"single_morphology_{ratio_name}_{timestamp}"
        experiment_dir = os.path.join(self.base_dir, "morphology_studies", experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🧬 SINGLE MORPHOLOGY MULTI-SEED EXPERIMENT")
        print(f"{'='*80}")
        print(f"📅 Experiment ID: {experiment_id}")
        print(f"📂 Output Directory: {experiment_dir}")
        print(f"📏 Morphology Ratio: {ratio} ({ratio_name})")
        print(f"🎯 Task: {task}")
        print(f"🎲 Seeds: {seeds}")
        print(f"🔄 Max Iterations: {max_iterations}")
        print(f"🤖 Num Environments: {num_envs}")
        print(f"{'='*80}")
        
        all_results = {
            'experiment_id': experiment_id,
            'experiment_dir': experiment_dir,
            'ratio': ratio,
            'ratio_name': ratio_name,
            'femur_ratio': femur_ratio,
            'tibia_ratio': tibia_ratio,
            'task': task,
            'seeds': seeds,
            'max_iterations': max_iterations,
            'num_envs': num_envs,
            'start_time': time.time(),
            'morphology': None,
            'experiments': [],
            'testing_results': []
        }
        
        # Step 1: Generate morphology
        print(f"\n🔧 STEP 1: Generating morphology {ratio}...")
        
        try:
            # Create morphology directory
            morphology_dir = os.path.join(self.workflow.robots_dir, ratio_name)
            
            # Generate URDF
            urdf_path, _ = self.workflow.generate_morphology_urdf(
                femur_ratio, tibia_ratio, morphology_dir
            )
            
            # Convert to USD
            usd_path = self.workflow.convert_urdf_to_usd(
                urdf_path, ratio_name, morphology_dir
            )
            
            morphology_result = {
                'ratio_str': ratio,
                'ratio_name': ratio_name,
                'femur_ratio': femur_ratio,
                'tibia_ratio': tibia_ratio,
                'urdf_path': urdf_path,
                'usd_path': usd_path,
                'morphology_dir': morphology_dir,
                'status': 'generated'
            }
            
            all_results['morphology'] = morphology_result
            print(f"✅ Morphology {ratio_name} generated successfully")
            
        except Exception as e:
            print(f"❌ Failed to generate morphology {ratio}: {e}")
            all_results['morphology'] = {
                'ratio_str': ratio,
                'error': str(e),
                'status': 'failed'
            }
            return all_results
        
        # Step 2: Run training experiments for each seed
        print(f"\n🚀 STEP 2: Running {len(seeds)} training experiments...")
        
        for i, seed in enumerate(seeds):
            try:
                seed_experiment_id = f"{ratio_name}_seed_{seed}"
                
                print(f"\n--- Experiment {i+1}/{len(seeds)}: Seed {seed} ---")
                
                experiment_result = self.workflow.run_training_experiment(
                    usd_path=morphology_result['usd_path'],
                    ratio_name=ratio_name,
                    task=task,
                    seed=seed,
                    max_iterations=max_iterations,
                    num_envs=num_envs,
                    experiment_id=seed_experiment_id,
                    output_dir=experiment_dir,
                    additional_args=additional_args
                )
                
                all_results['experiments'].append(experiment_result)
                
                # If training was successful and testing is enabled, run testing experiments
                if experiment_result['status'] == 'completed' and run_testing:
                    print(f"\n🎮 STEP 3: Running testing for seed {seed}...")
                    
                    # Test final model
                    final_test_result = self.workflow.run_testing_experiment(
                        usd_path=morphology_result['usd_path'],
                        ratio_name=ratio_name,
                        task=task,
                        seed=seed,
                        max_iterations=max_iterations,
                        experiment_id=seed_experiment_id,
                        checkpoint_type="final"
                    )
                    all_results['testing_results'].append(final_test_result)
                    
                    # Test best model
                    best_test_result = self.workflow.run_testing_experiment(
                        usd_path=morphology_result['usd_path'],
                        ratio_name=ratio_name,
                        task=task,
                        seed=seed,
                        max_iterations=max_iterations,
                        experiment_id=seed_experiment_id,
                        checkpoint_type="best"
                    )
                    all_results['testing_results'].append(best_test_result)
                
                # Save intermediate results
                results_file = os.path.join(experiment_dir, "multi_seed_results.json")
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                print(f"❌ Failed to run experiment for seed {seed}: {e}")
                experiment_result = {
                    'experiment_id': f"{ratio_name}_seed_{seed}",
                    'ratio_name': ratio_name,
                    'seed': seed,
                    'error': str(e),
                    'status': 'failed'
                }
                all_results['experiments'].append(experiment_result)
                continue
        
        # Finalize results
        all_results['end_time'] = time.time()
        all_results['total_duration_minutes'] = (all_results['end_time'] - all_results['start_time']) / 60.0
        
        successful_experiments = [e for e in all_results['experiments'] if e['status'] == 'completed']
        successful_tests = [t for t in all_results['testing_results'] if t['status'] == 'completed']
        
        print(f"\n{'='*80}")
        print(f"🎉 MULTI-SEED EXPERIMENT COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Results Summary:")
        print(f"   • Morphology: {ratio_name} ({ratio})")
        print(f"   • Successful training experiments: {len(successful_experiments)}/{len(seeds)}")
        if run_testing:
            print(f"   • Successful testing experiments: {len(successful_tests)}/{len(successful_experiments) * 2}")
        print(f"   • Total duration: {all_results['total_duration_minutes']:.1f} minutes")
        print(f"   • Results saved to: {experiment_dir}")
        
        # Show per-seed results
        if successful_experiments:
            print(f"\n📈 Per-Seed Results:")
            for exp in all_results['experiments']:
                seed = exp.get('seed', 'N/A')
                status = exp.get('status', 'unknown')
                duration = exp.get('duration_minutes', 0)
                status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏰" if status == "timeout" else "❓"
                print(f"   {status_emoji} Seed {seed}: {status} ({duration:.1f} min)")
        
        print(f"{'='*80}")
        
        # Save final results
        results_file = os.path.join(experiment_dir, "multi_seed_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Single Morphology Multi-Seed Training Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ratio "3:7" --seeds 42 123 456 --max_iterations 2000
  %(prog)s --ratio "1:1" --seeds 1 2 3 4 5 --task Isaac-Vel-BALLU-imu-tibia
  %(prog)s --ratio "1:2" --seeds 42 123 456 789 999 --max_iterations 5000 --num_envs 8192
        """
    )
    
    # Core arguments
    parser.add_argument(
        '--ratio', 
        type=str, 
        required=True,
        help='Femur:tibia ratio to study (e.g., "3:7")'
    )
    parser.add_argument(
        '--seeds', 
        type=int, 
        nargs='+', 
        required=True,
        help='List of random seeds for experiments'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='Isc-Vel-BALLU-encoder',
        help='Task name for training (default: Isc-Vel-BALLU-encoder)'
    )
    parser.add_argument(
        '--max_iterations', 
        type=int, 
        default=2000,
        help='Maximum training iterations (default: 2000)'
    )
    parser.add_argument(
        '--num_envs', 
        type=int, 
        default=4096,
        help='Number of environments (default: 4096)'
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        help='Base directory for BALLU extension (auto-detected if not provided)'
    )
    parser.add_argument(
        '--additional_args', 
        type=str, 
        nargs='*', 
        default=[],
        help='Additional arguments to pass to train.py'
    )
    parser.add_argument(
        '--no_testing', 
        action='store_true',
        help='Skip testing/evaluation after training (training only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize workflow
        workflow = SingleMorphologyMultiSeed(base_dir=args.base_dir)
        
        # Run the multi-seed experiment
        results = workflow.run_multi_seed_experiment(
            ratio=args.ratio,
            seeds=args.seeds,
            task=args.task,
            max_iterations=args.max_iterations,
            num_envs=args.num_envs,
            additional_args=args.additional_args,
            run_testing=not args.no_testing
        )
        
        print(f"\n🎉 Multi-seed experiment completed successfully!")
        print(f"📊 Check results in: {results['experiment_dir']}")
        
    except Exception as e:
        print(f"❌ Multi-seed experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 