#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive Morphology Study Workflow for BALLU Robot

This script automates the complete workflow for conducting training and testing experiments 
on different BALLU morphologies:

1. Generate different morphology URDF files using create_leg_ratio_ballu.py
2. Convert URDF files to USD format using convert_urdf.py  
3. Run training experiments with each morphology
4. Run testing/evaluation experiments with both final and best models (optional)

Usage:
    python morphology_study_workflow.py --ratios "1:1" "3:7" "1:2" --seed 42 --max_iterations 2000
    python morphology_study_workflow.py --ratios "1:1" "1:2" "2:3" "3:7" "1:3" --task Isaac-Vel-BALLU-imu-tibia --seed 123
    python morphology_study_workflow.py --ratios "4.1:5.9" "4.2:5.8" --seed 42 --max_iterations 80 --no_testing
"""

import argparse
import os
import sys
import subprocess
import time
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class MorphologyStudyWorkflow:
    """
    Orchestrates the complete morphology study workflow for BALLU robot.
    
    This class manages the generation of different robot morphologies,
    their conversion to Isaac Sim format, and automated training experiments.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the morphology study workflow.
        
        Args:
            base_dir: Base directory for the BALLU extension (auto-detected if None)
        """
        if base_dir is None:
            # Auto-detect base directory (assuming script is in scripts/morphology_utils/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.dirname(os.path.dirname(script_dir))
        else:
            self.base_dir = base_dir
        
        print(f"Base directory: {self.base_dir}")
        # Define key paths
        self.morphology_utils_dir = os.path.join(self.base_dir, "scripts", "morphology_utils")
        self.rsl_rl_scripts_dir = os.path.join(self.base_dir, "scripts", "rsl_rl")
        self.assets_dir = os.path.join(self.base_dir, "source", "ballu_isaac_extension", 
                                     "ballu_isaac_extension", "ballu_assets")
        self.robots_dir = os.path.join(self.assets_dir, "robots")
        self.original_urdf = os.path.join(self.assets_dir, "old", "urdf", "urdf", "original.urdf")
        
        # Verify paths exist
        self._verify_paths()
        
        print(f"🤖 BALLU Morphology Study Workflow Initialized")
        print(f"📂 Base directory: {self.base_dir}")
        print(f"🔧 Morphology utils: {self.morphology_utils_dir}")
        print(f"🏃 RSL-RL scripts: {self.rsl_rl_scripts_dir}")
        print(f"🤖 Robot assets: {self.robots_dir}")
        
    def _verify_paths(self):
        """Verify that all required paths exist."""
        required_paths = [
            self.morphology_utils_dir,
            self.rsl_rl_scripts_dir,
            self.assets_dir,
            self.robots_dir
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")
        
        # Check for required scripts
        required_scripts = [
            os.path.join(self.morphology_utils_dir, "create_leg_ratio_ballu.py"),
            os.path.join(self.morphology_utils_dir, "convert_urdf.py"),
            os.path.join(self.rsl_rl_scripts_dir, "train.py")
        ]
        
        for script in required_scripts:
            if not os.path.exists(script):
                raise FileNotFoundError(f"Required script not found: {script}")
                
        print("✅ All required paths and scripts verified")
    
    def parse_ratio(self, ratio_str: str) -> Tuple[float, float]:
        """Parse ratio string like '3:7' into (3.0, 7.0)."""
        try:
            parts = ratio_str.split(':')
            if len(parts) != 2:
                raise ValueError("Ratio must be in format 'X:Y'")
            
            femur_ratio = float(parts[0])
            tibia_ratio = float(parts[1])
            
            if femur_ratio <= 0 or tibia_ratio <= 0:
                raise ValueError("Ratio values must be positive")
                
            return femur_ratio, tibia_ratio
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid ratio format '{ratio_str}': {e}")
    
    def _copy_mesh_files(self, morphology_dir: str) -> None:
        """
        Copy mesh files to the morphology directory so URDF can find them.
        
        Args:
            morphology_dir: Path to the morphology directory
        """
        try:
            # Source mesh directory 
            source_mesh_dir = os.path.join(
                self.base_dir, 
                "source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/old/urdf/meshes"
            )
            
            # Create target directory structure: morphology_dir/urdf/meshes/
            target_mesh_dir = os.path.join(morphology_dir, "urdf", "meshes")
            os.makedirs(target_mesh_dir, exist_ok=True)
            
            # Copy all .STL files
            import shutil
            import glob
            mesh_files = glob.glob(os.path.join(source_mesh_dir, "*.STL"))
            
            for mesh_file in mesh_files:
                filename = os.path.basename(mesh_file)
                target_path = os.path.join(target_mesh_dir, filename)
                shutil.copy2(mesh_file, target_path)
                
            print(f"✅ Copied {len(mesh_files)} mesh files to {target_mesh_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to copy mesh files: {e}")
            print("   Robot may appear invisible in visualization")
    
    def generate_morphology_urdf(self, femur_ratio: float, tibia_ratio: float, 
                               output_dir: str) -> Tuple[str, str]:
        """
        Generate URDF file for a specific morphology ratio.
        
        Args:
            femur_ratio: Femur portion of the ratio
            tibia_ratio: Tibia portion of the ratio  
            output_dir: Directory to save the URDF file
            
        Returns:
            Tuple of (urdf_path, ratio_name)
        """
        # Handle decimal ratios by replacing dots with underscores
        femur_str = str(femur_ratio).replace('.', '_')
        tibia_str = str(tibia_ratio).replace('.', '_')
        ratio_name = f"FT_{femur_str}_{tibia_str}"
        urdf_filename = f"ballu_modified_{ratio_name}.urdf"
        urdf_path = os.path.join(output_dir, urdf_filename)
        
        print(f"\n🔧 Generating URDF for ratio {femur_ratio}:{tibia_ratio}")
        print(f"📁 Output: {urdf_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run create_leg_ratio_ballu.py
        cmd = [
            sys.executable,
            os.path.join(self.morphology_utils_dir, "create_leg_ratio_ballu.py"),
            "--input", self.original_urdf,
            "--femur-ratio", str(femur_ratio),
            "--tibia-ratio", str(tibia_ratio),
            "--output", urdf_path
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ URDF generation successful")
                if os.path.exists(urdf_path):
                    # Copy mesh files to the morphology directory
                    print(f"📂 Copying mesh files for {ratio_name}")
                    self._copy_mesh_files(output_dir)
                    return urdf_path, ratio_name
                else:
                    raise FileNotFoundError(f"URDF file not created: {urdf_path}")
            else:
                print(f"❌ URDF generation failed")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
                
        except subprocess.TimeoutExpired:
            raise TimeoutError("URDF generation timed out after 5 minutes")
    
    def convert_urdf_to_usd(self, urdf_path: str, ratio_name: str, 
                          morphology_dir: str) -> str:
        """
        Convert URDF file to USD format.
        
        Args:
            urdf_path: Path to the URDF file
            ratio_name: Name of the morphology ratio (e.g., "FT_37")
            morphology_dir: Directory to save the USD file
            
        Returns:
            Path to the generated USD file
        """
        usd_filename = f"ballu_modified_{ratio_name}.usd"
        usd_path = os.path.join(morphology_dir, usd_filename)
        
        print(f"\n🔄 Converting URDF to USD for {ratio_name}")
        print(f"📂 Input URDF: {urdf_path}")
        print(f"💾 Output USD: {usd_path}")
        
        # Create morphology directory if it doesn't exist
        os.makedirs(morphology_dir, exist_ok=True)
        
        # Run convert_urdf.py
        cmd = [
            sys.executable,
            os.path.join(self.morphology_utils_dir, "convert_urdf.py"),
            urdf_path,
            usd_path,
            "--headless"
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ USD conversion successful")
                if os.path.exists(usd_path):
                    return usd_path
                else:
                    raise FileNotFoundError(f"USD file not created: {usd_path}")
            else:
                print(f"❌ USD conversion failed")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
                
        except subprocess.TimeoutExpired:
            raise TimeoutError("USD conversion timed out after 10 minutes")
    
    def run_testing_experiment(self, usd_path: str, ratio_name: str,
                             task: str, seed: int, max_iterations: int,
                             experiment_id: str, checkpoint_type: str = "final") -> Dict[str, Any]:
        """
        Run a testing/evaluation experiment with the specified morphology.
        
        Args:
            usd_path: Path to the USD file for this morphology
            ratio_name: Name of the morphology ratio
            task: Task name for testing
            seed: Random seed
            max_iterations: Max iterations (to determine checkpoint name)
            experiment_id: Unique experiment identifier
            checkpoint_type: Type of checkpoint ("final" or "best")
            
        Returns:
            Dictionary containing testing results
        """
        print(f"\n🎮 Testing Experiment: {ratio_name} ({checkpoint_type} model)")
        
        # Determine checkpoint filename
        if checkpoint_type == "final":
            checkpoint_file = f"model_{max_iterations - 1}.pt"
        elif checkpoint_type == "best":
            checkpoint_file = "model_best.pt"
        else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
        
        # Build load_run path to match training folder structure
        load_run = f"{experiment_id}_morphology_study/seed_{seed}"
        
        cmd = [
            sys.executable,
            os.path.join(self.rsl_rl_scripts_dir, "play.py"),
            "--task", task,
            "--load_run", load_run,
            "--checkpoint", checkpoint_file,
            "--headless",
            "--video",
            "--num_envs", "1",
            "--video_length", "399"
        ]
        
        # Set environment variable to override robot USD path
        env = os.environ.copy()
        env['BALLU_MORPHOLOGY_USD_PATH'] = usd_path
        # Add Isaac Sim specific environment variables for subprocess execution
        env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
        env['FORCE_GPU'] = '1'
        
        testing_metadata = {
            'experiment_id': experiment_id,
            'ratio_name': ratio_name,
            'task': task,
            'seed': seed,
            'checkpoint_type': checkpoint_type,
            'checkpoint_file': checkpoint_file,
            'load_run': load_run,
            'start_time': time.time(),
            'command': ' '.join(cmd),
            'status': 'running'
        }
        
        try:
            print(f"Running: {' '.join(cmd)}")
            
            # Run the testing with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=self.base_dir
            )
            
            # Capture output while streaming it live
            captured_output = []
            
            try:
                # Guard for Optional stdout (type-checker)
                if process.stdout is None:
                    raise RuntimeError("Subprocess stdout is None; cannot stream output.")
                for line in process.stdout:
                    print(line, end='')
                    captured_output.append(line)
                
                process.wait(timeout=1800)  # 30 minute timeout
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, 1800)
            
            if process.returncode == 0:
                print(f"✅ Testing ({checkpoint_type}) completed successfully!")
                
                testing_metadata.update({
                    'end_time': time.time(),
                    'status': 'completed',
                    'duration_minutes': (time.time() - testing_metadata['start_time']) / 60.0
                })
                
                return testing_metadata
            else:
                print(f"❌ Testing ({checkpoint_type}) failed with return code: {process.returncode}")
                return {
                    **testing_metadata,
                    'error': f"Testing failed with return code {process.returncode}",
                    'status': 'failed',
                    'end_time': time.time()
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Testing experiment {experiment_id} ({checkpoint_type}) timed out after 30 minutes")
            return {
                **testing_metadata,
                'error': 'Testing timed out',
                'status': 'timeout',
                'end_time': time.time()
            }
        except Exception as e:
            print(f"Error running testing experiment {experiment_id} ({checkpoint_type}): {str(e)}")
            return {
                **testing_metadata,
                'error': str(e),
                'status': 'subprocess_error',
                'end_time': time.time()
            }

    def run_training_experiment(self, usd_path: str, ratio_name: str, 
                              task: str, seed: int, max_iterations: int,
                              num_envs: int, experiment_id: str,
                              output_dir: str, additional_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a training experiment with the specified morphology.
        
        Args:
            usd_path: Path to the USD file for this morphology
            ratio_name: Name of the morphology ratio
            task: Task name for training
            seed: Random seed
            max_iterations: Maximum training iterations
            num_envs: Number of environments
            experiment_id: Unique experiment identifier
            output_dir: Directory to save results
            additional_args: Additional command line arguments
            
        Returns:
            Dictionary containing experiment results
        """
        print(f"\n🚀 Training Experiment: {ratio_name}")
        print(f"🎯 Task: {task}")
        print(f"🎲 Seed: {seed}")
        print(f"🔄 Iterations: {max_iterations}")
        print(f"🤖 USD Path: {usd_path}")
        
        # Create a modified training script call
        cmd = [
            sys.executable,
            os.path.join(self.rsl_rl_scripts_dir, "train.py"),
            "--task", task,
            "--seed", str(seed),
            "--max_iterations", str(max_iterations),
            "--num_envs", str(num_envs),
            "--common_folder", f"{experiment_id}_morphology_study",
            "--headless"
        ]
        
        # Add additional arguments
        if additional_args:
            cmd.extend(additional_args)
        
        # Set environment variable to override robot USD path
        env = os.environ.copy()
        env['BALLU_MORPHOLOGY_USD_PATH'] = usd_path
        # Add Isaac Sim specific environment variables for subprocess execution
        env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
        env['FORCE_GPU'] = '1'
        
        experiment_metadata = {
            'experiment_id': experiment_id,
            'ratio_name': ratio_name,
            'usd_path': usd_path,
            'task': task,
            'seed': seed,
            'max_iterations': max_iterations,
            'num_envs': num_envs,
            'start_time': time.time(),
            'command': ' '.join(cmd),
            'status': 'running'
        }
        
        try:
            print(f"Running: {' '.join(cmd)}")
            
            # Run the training with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=self.base_dir
            )
            
            # Capture output while streaming it live
            captured_output = []
            
            try:
                # Guard for Optional stdout (type-checker)
                if process.stdout is None:
                    raise RuntimeError("Subprocess stdout is None; cannot stream output.")
                for line in process.stdout:
                    print(line, end='')
                    captured_output.append(line)
                
                process.wait(timeout=3600 * 6)  # 6 hour timeout
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, 3600 * 6)
            
            if process.returncode == 0:
                print("✅ Training completed successfully!")
                
                experiment_metadata.update({
                    'end_time': time.time(),
                    'status': 'completed',
                    'duration_minutes': (time.time() - experiment_metadata['start_time']) / 60.0
                })
                
                return experiment_metadata
            else:
                print(f"❌ Training failed with return code: {process.returncode}")
                return {
                    **experiment_metadata,
                    'error': f"Training failed with return code {process.returncode}",
                    'status': 'failed',
                    'end_time': time.time()
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Training experiment {experiment_id} timed out after 6 hours")
            return {
                **experiment_metadata,
                'error': 'Training timed out',
                'status': 'timeout',
                'end_time': time.time()
            }
        except Exception as e:
            print(f"Error running training experiment {experiment_id}: {str(e)}")
            return {
                **experiment_metadata,
                'error': str(e),
                'status': 'subprocess_error',
                'end_time': time.time()
            }
    
    def run_morphology_study(self, ratios: List[str], task: str, seed: int,
                           max_iterations: int, num_envs: int,
                           additional_args: Optional[List[str]] = None, 
                           run_testing: bool = True) -> Dict[str, Any]:
        """
        Run the complete morphology study workflow.
        
        Args:
            ratios: List of ratio strings (e.g., ["1:1", "3:7", "1:2"])
            task: Task name for training
            seed: Random seed for experiments
            max_iterations: Maximum training iterations
            num_envs: Number of environments
            additional_args: Additional arguments for training
            run_testing: Whether to run testing/evaluation after training (default: True)
            
        Returns:
            Dictionary containing all results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_id = f"morphology_study_{timestamp}"
        study_dir = os.path.join(self.base_dir, "morphology_studies", study_id)
        os.makedirs(study_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🧬 BALLU MORPHOLOGY STUDY WORKFLOW")
        print(f"{'='*80}")
        print(f"📅 Study ID: {study_id}")
        print(f"📂 Output Directory: {study_dir}")
        print(f"🎯 Task: {task}")
        print(f"🎲 Seed: {seed}")
        print(f"🔄 Max Iterations: {max_iterations}")
        print(f"🤖 Num Environments: {num_envs}")
        print(f"📏 Morphology Ratios: {ratios}")
        print(f"{'='*80}")
        
        all_results = {
            'study_id': study_id,
            'study_dir': study_dir,
            'task': task,
            'seed': seed,
            'max_iterations': max_iterations,
            'num_envs': num_envs,
            'ratios': ratios,
            'start_time': time.time(),
            'morphologies': [],
            'experiments': [],
            'testing_results': []
        }
        
        # Step 1: Generate all morphologies
        print(f"\n🔧 STEP 1: Generating {len(ratios)} morphologies...")
        
        for i, ratio_str in enumerate(ratios):
            try:
                femur_ratio, tibia_ratio = self.parse_ratio(ratio_str)
                # Handle decimal ratios by replacing dots with underscores
                femur_str = str(femur_ratio).replace('.', '_')
                tibia_str = str(tibia_ratio).replace('.', '_')
                ratio_name = f"FT_{femur_str}_{tibia_str}"
                
                # Create morphology subdirectory
                morphology_dir = os.path.join(self.robots_dir, ratio_name)
                
                print(f"\n--- Morphology {i+1}/{len(ratios)}: {ratio_str} ---")
                
                # Generate URDF
                urdf_path, _ = self.generate_morphology_urdf(
                    femur_ratio, tibia_ratio, morphology_dir
                )
                
                # Convert to USD
                usd_path = self.convert_urdf_to_usd(
                    urdf_path, ratio_name, morphology_dir
                )
                
                morphology_result = {
                    'ratio_str': ratio_str,
                    'ratio_name': ratio_name,
                    'femur_ratio': femur_ratio,
                    'tibia_ratio': tibia_ratio,
                    'urdf_path': urdf_path,
                    'usd_path': usd_path,
                    'morphology_dir': morphology_dir,
                    'status': 'generated'
                }
                
                all_results['morphologies'].append(morphology_result)
                
                print(f"✅ Morphology {ratio_name} generated successfully")
                
            except Exception as e:
                print(f"❌ Failed to generate morphology {ratio_str}: {e}")
                morphology_result = {
                    'ratio_str': ratio_str,
                    'error': str(e),
                    'status': 'failed'
                }
                all_results['morphologies'].append(morphology_result)
                continue
        
        # Filter successful morphologies
        successful_morphologies = [m for m in all_results['morphologies'] if m['status'] == 'generated']
        
        if not successful_morphologies:
            print("❌ No morphologies were generated successfully. Stopping workflow.")
            return all_results
        
        print(f"\n✅ Generated {len(successful_morphologies)}/{len(ratios)} morphologies successfully")
        
        # Step 2: Run training experiments
        print(f"\n🚀 STEP 2: Running training experiments...")
        
        # TODO: enable the automated timestamp-based folder creation
        # morph_study_start_time_fmt = datetime.now().strftime('%m_%d_%H_%M_%S') 
        morph_study_start_time_fmt = "07_24_02_07_03"
        for i, morphology in enumerate(successful_morphologies):
            try:
                experiment_id = f"{morph_study_start_time_fmt}_{morphology['ratio_name']}"
                
                print(f"\n--- Experiment {i+1}/{len(successful_morphologies)}: {morphology['ratio_name']} ---")
                
                experiment_result = self.run_training_experiment(
                    usd_path=morphology['usd_path'],
                    ratio_name=morphology['ratio_name'],
                    task=task,
                    seed=seed,
                    max_iterations=max_iterations,
                    num_envs=num_envs,
                    experiment_id=experiment_id,
                    output_dir=study_dir,
                    additional_args=additional_args
                )
                
                all_results['experiments'].append(experiment_result)
                
                # If training was successful and testing is enabled, run testing experiments
                if experiment_result['status'] == 'completed' and run_testing:
                    print(f"\n🎮 STEP 3: Running testing for {morphology['ratio_name']}...")
                    
                    # Test final model
                    final_test_result = self.run_testing_experiment(
                        usd_path=morphology['usd_path'],
                        ratio_name=morphology['ratio_name'],
                        task=task,
                        seed=seed,
                        max_iterations=max_iterations,
                        experiment_id=experiment_id,
                        checkpoint_type="final"
                    )
                    all_results['testing_results'].append(final_test_result)
                    
                    # Test best model
                    best_test_result = self.run_testing_experiment(
                        usd_path=morphology['usd_path'],
                        ratio_name=morphology['ratio_name'],
                        task=task,
                        seed=seed,
                        max_iterations=max_iterations,
                        experiment_id=experiment_id,
                        checkpoint_type="best"
                    )
                    all_results['testing_results'].append(best_test_result)
                
                # Save intermediate results
                results_file = os.path.join(study_dir, "morphology_study_results.json")
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                print(f"❌ Failed to run experiment for {morphology['ratio_name']}: {e}")
                experiment_result = {
                    'experiment_id': f"{morphology['ratio_name']}_seed_{seed}",
                    'ratio_name': morphology['ratio_name'],
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
        print(f"🎉 MORPHOLOGY STUDY COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Results Summary:")
        print(f"   • Generated morphologies: {len(successful_morphologies)}/{len(ratios)}")
        print(f"   • Successful training experiments: {len(successful_experiments)}/{len(successful_morphologies)}")
        print(f"   • Successful testing experiments: {len(successful_tests)}/{len(successful_experiments) * 2}")
        print(f"   • Total duration: {all_results['total_duration_minutes']:.1f} minutes")
        print(f"   • Results saved to: {study_dir}")
        print(f"{'='*80}")
        
        # Print detailed testing summary
        if successful_tests:
            print(f"\n🎮 Testing Summary:")
            for test in successful_tests:
                checkpoint_emoji = "🏆" if test['checkpoint_type'] == "best" else "📝"
                print(f"   {checkpoint_emoji} {test['ratio_name']} ({test['checkpoint_type']} model): ✅ Completed in {test['duration_minutes']:.1f}min")
        
        if len(successful_tests) < len(successful_experiments) * 2:
            failed_tests = [t for t in all_results['testing_results'] if t['status'] != 'completed']
            print(f"\n⚠️ Failed Tests:")
            for test in failed_tests:
                print(f"   ❌ {test['ratio_name']} ({test['checkpoint_type']} model): {test.get('error', 'Unknown error')}")
        
        print(f"{'='*80}")
        
        # Save final results
        results_file = os.path.join(study_dir, "morphology_study_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive BALLU Morphology Study Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ratios "1:1" "3:7" "1:2" --seed 42 --max_iterations 2000
  %(prog)s --ratios "1:1" "1:2" "2:3" "3:7" "1:3" --task Isaac-Vel-BALLU-imu-tibia --seed 123
  %(prog)s --ratios "3:7" "1:1" --task Isc-Vel-BALLU-encoder --max_iterations 5000 --num_envs 8192
  %(prog)s --ratios "4.1:5.9" "4.2:5.8" --seed 42 --max_iterations 80 --no_testing
        """
    )
    
    # Core arguments
    parser.add_argument(
        '--ratios', 
        type=str, 
        nargs='+', 
        required=True,
        help='List of femur:tibia ratios to study (e.g., "1:1" "3:7" "1:2")'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='Isc-Vel-BALLU-encoder',
        help='Task name for training (default: Isc-Vel-BALLU-encoder)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for experiments (default: 42)'
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
        workflow = MorphologyStudyWorkflow(base_dir=args.base_dir)
        
        # Run the complete study
        results = workflow.run_morphology_study(
            ratios=args.ratios,
            task=args.task,
            seed=args.seed,
            max_iterations=args.max_iterations,
            num_envs=args.num_envs,
            additional_args=args.additional_args,
            run_testing=not args.no_testing
        )
        
        print(f"\n🎉 Morphology study completed successfully!")
        print(f"📊 Check results in: {results['study_dir']}")
        
    except Exception as e:
        print(f"❌ Morphology study failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 