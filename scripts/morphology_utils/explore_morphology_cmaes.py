import os
import sys
import random
import re
import numpy as np
from datetime import datetime
import subprocess
import json
import argparse
import cma
import torch
from typing import Dict, List, Tuple, Optional

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
ext_dir = os.path.join(project_dir, "source", "ballu_isaac_extension", "ballu_isaac_extension")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# --- Import Morphology Tools ---
try:
    from morphology import BalluMorphology, BalluRobotGenerator, create_morphology_variant
except Exception as exc:
    print(f"[ERROR] Failed to import morphology tools: {exc}")
    print(f"[HINT] Ensure PYTHONPATH includes: {ext_dir}")
    sys.exit(1)


def run_evaluation_experiment(
        morph_id: str,
        task: str = "Isc-BALLU-hetero-general",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 64,
        num_episodes: int = 30,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
        difficulty_level: int = -1
    ) -> tuple[bool, float, str]:
    """Run evaluation with universal controller and return success status, curriculum level, and eval info."""
    eval_script_path = f"{project_dir}/scripts/rsl_rl/evaluate_design_cmaes.py"
    cmd = [
        sys.executable,
        eval_script_path,
        "--task", task,
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff),
        "--difficulty_level", str(difficulty_level),
        "--headless"
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/12.02.2025/{morph_id}/{morph_id}.usd"
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    curriculum_level = None
    eval_info = ""
    try:
        for line in process.stdout:
            print(line, end='')
            eval_info += line
            # Parse BEST_CRCLM_LEVEL from output
            if line.startswith("BEST_CRCLM_LEVEL:"):
                try:
                    curriculum_level = float(line.split(":")[1].strip())
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse curriculum level: {e}")
        process.wait(timeout=30 * 60)  # 30 minute timeout
    except subprocess.TimeoutExpired:
        print(f"Evaluation timeout (30 min), killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), ""

    if process.returncode != 0:
        return False, float('-inf'), ""
    
    if curriculum_level is None:
        print(f"[WARNING] Could not extract curriculum level from output")
        return False, float('-inf'), eval_info
    
    return True, curriculum_level, eval_info


def run_testing_experiment(
        morph_id: str, 
        task: str = "Isc-Vel-BALLU-1-obstacle",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 1,
        video_length: int = 399,
        device: str = "cuda:0",
        difficulty_level: int = 0,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
        cmdir: str = "test"
    ) -> tuple[bool, str]:
    """Run testing experiment for a trained morphology and return success status."""
    test_script_path = f"{project_dir}/scripts/rsl_rl/play_universal.py"
    cmd = [
        sys.executable,
        test_script_path,
        "--task", task,
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--video",
        "--video_length", str(video_length),
        "--device", device,
        "--headless",
        "--difficulty_level", str(difficulty_level),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff),
        "--cmdir", cmdir,
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/12.02.2025/{morph_id}/{morph_id}.usd"
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    try:
        for line in process.stdout:
            print(line, end='')
        process.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        print(f"Testing timeout (2h), killing process")
        process.kill()
        process.wait()
        return False, ""

    test_command = f"env BALLU_USD_REL_PATH={env['BALLU_USD_REL_PATH']}" + " " + ' '.join(cmd)
    return process.returncode == 0, test_command


def generate_morphology_from_params(sampled_config: dict, iteration: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate morphology from sampled parameters and return (urdf_path, usd_path, morphology_id)."""
    print(f"\n[Iteration {iteration}] Generating morphology: {sampled_config['morphology_id']}")
    
    try:
        morph = create_morphology_variant(**sampled_config)
    except Exception as e:
        print(f"[ERROR] Failed to create morphology: {e}")
        return None, None, None
    
    is_valid, errors = morph.validate()
    if not is_valid:
        print(f"[ERROR] Validation failed: {errors}")
        return None, None, None
    
    try:
        generator = BalluRobotGenerator(morph)
        urdf_path = generator.generate_urdf()
        return_code, usd_file_path = generator.generate_usd(urdf_path)
        
        if return_code == 0 and os.path.exists(usd_file_path):
            print(f"[Iteration {iteration}] Generated USD: {usd_file_path}")
            return urdf_path, usd_file_path, morph.morphology_id
        else:
            print(f"[ERROR] USD conversion failed (code: {return_code})")
            return None, None, None
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return None, None, None


class CMAESOptimizer:
    """CMA-ES based morphology optimizer for BALLU robot."""
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        initial_params: Optional[Dict[str, float]] = None,
        sigma0: float = 0.3,
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 64,
        num_episodes: int = 30,
        seed: int = 42,
        task: str = "Isc-BALLU-hetero-general",
        device: str = "cuda:0",
        test_video_length: int = 399,
        test_num_envs: int = 1,
        results_dir: str = None,
        popsize: Optional[int] = None,
        difficulty_level: int = 10,
        cmdir: str = "cmaes"
    ):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            initial_params: Optional initial parameter values (if None, uses midpoint of bounds)
            sigma0: Initial standard deviation (step size)
            load_run: Run name of universal controller (required)
            checkpoint: Checkpoint name of universal controller
            num_envs: Number of environments for evaluation
            num_episodes: Number of episodes for evaluation
            seed: Random seed
            task: Task name for evaluation
            device: Device for testing
            test_video_length: Video length for testing
            test_num_envs: Number of environments for testing
            results_dir: Directory to save results
            popsize: Population size (if None, CMA-ES uses default)
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.load_run = load_run
        self.checkpoint = checkpoint
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.seed = seed
        self.task = task
        self.device = device
        self.test_video_length = test_video_length
        self.test_num_envs = test_num_envs
        self.results_dir = results_dir or f"{project_dir}/logs/results"
        self.difficulty_level = difficulty_level
        self.cmdir = cmdir
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Normalize bounds to [0, 1] for CMA-ES
        self.bounds_lower = np.array([bounds[0] for bounds in param_bounds.values()])
        self.bounds_upper = np.array([bounds[1] for bounds in param_bounds.values()])
        self.bounds_range = self.bounds_upper - self.bounds_lower
        
        # Set initial point (normalized to [0, 1])
        if initial_params is None:
            # Use midpoint of bounds
            x0_unnormalized = (self.bounds_lower + self.bounds_upper) / 2
        else:
            x0_unnormalized = np.array([initial_params[name] for name in self.param_names])
        
        self.x0 = self._normalize(x0_unnormalized)
        
        # CMA-ES options
        self.cmaes_options = {
            'bounds': [0, 1],  # Normalized bounds
            'seed': seed,
            'verbose': 1,
            'verb_disp': 1,
            'verb_log': 1,
        }
        
        if popsize is not None:
            self.cmaes_options['popsize'] = popsize
        
        self.sigma0 = sigma0
        
        # History tracking
        self.iteration = 0
        self.history = []
        self.best_result = {
            'params': None,
            'fitness': float('-inf'),
            'morphology_id': None,
            'log_dir': None
        }
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize parameters from original bounds to [0, 1]."""
        return (x - self.bounds_lower) / self.bounds_range
    
    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] to original bounds."""
        return x * self.bounds_range + self.bounds_lower
    
    def _params_array_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        x_denorm = self._denormalize(x)
        return {name: float(val) for name, val in zip(self.param_names, x_denorm)}
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for CMA-ES (to be minimized).
        
        Args:
            x: Normalized parameter vector
            
        Returns:
            Negative best curriculum level (for minimization)
        """
        self.iteration += 1
        print(f"\n{'='*80}\n[CMA-ES ITERATION {self.iteration}]\n{'='*80}")
        
        # Convert to parameter dictionary
        params_dict = self._params_array_to_dict(x)
        
        # Extract specific parameters
        femur_length = params_dict['femur_length']
        tibia_length = params_dict['tibia_length']
        gravity_comp_ratio = params_dict['gravity_comp_ratio']
        spring_coeff = params_dict['spring_coeff']
        
        # Create morphology ID
        morph_id = f"cmaes_iter{self.iteration:03d}_fl{femur_length:.3f}_tl{tibia_length:.3f}_spc{spring_coeff:.5f}_gcr{gravity_comp_ratio:.3f}"
        
        sampled_config = {
            "morphology_id": morph_id,
            "femur_length": femur_length,
            "tibia_length": tibia_length
        }
        
        # Generate morphology
        urdf_path, usd_path, final_morph_id = generate_morphology_from_params(sampled_config, self.iteration)
        
        if usd_path is None:
            print(f"[Iteration {self.iteration}] FAILED: Morphology generation")
            fitness = float('-inf')
            self._record_result(params_dict, fitness, None, None, None, False, 0)
            return -fitness  # Return positive infinity for minimization
        
        # Run evaluation with universal controller
        print(f"[Iteration {self.iteration}] Starting evaluation with universal controller (task={self.task})...")
        success, curriculum_level, eval_info = run_evaluation_experiment(
            final_morph_id,
            task=self.task,
            load_run=self.load_run,
            checkpoint=self.checkpoint,
            num_envs=self.num_envs,
            num_episodes=self.num_episodes,
            spring_coeff=spring_coeff,
            gravity_comp_ratio=gravity_comp_ratio,
            difficulty_level=self.difficulty_level  # Auto-detect from checkpoint
        )
        
        if not success:
            print(f"[Iteration {self.iteration}] FAILED: Evaluation")
            fitness = float('-inf')
            self._record_result(params_dict, fitness, final_morph_id, None, urdf_path, False, 0)
            return -fitness
        
        fitness = curriculum_level
        print(f"[Iteration {self.iteration}] COMPLETED - Curriculum level: {fitness:.4f}")
        
        # Run testing with universal controller (record video)
        difficulty = int(curriculum_level * 100) - 1 if curriculum_level > 0 else 0
        print(f"[Iteration {self.iteration}] Starting testing with universal controller (difficulty={difficulty})...")
        test_success, test_command = run_testing_experiment(
            morph_id=final_morph_id,
            task=self.task,
            load_run=self.load_run,
            checkpoint=self.checkpoint,
            num_envs=self.test_num_envs,
            video_length=self.test_video_length,
            device=self.device,
            difficulty_level=difficulty,
            spring_coeff=spring_coeff,
            gravity_comp_ratio=gravity_comp_ratio,
            cmdir=self.cmdir
        )
        print(f"[Iteration {self.iteration}] Testing {'SUCCESS' if test_success else 'FAILED'}")
        
        # Write test rerun script to results directory
        test_script_dir = os.path.join(self.results_dir, f"iter_{self.iteration:03d}_{final_morph_id}")
        os.makedirs(test_script_dir, exist_ok=True)
        script_path = os.path.join(test_script_dir, f"rerun_test.sh")
        try:
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(test_command + "\n")
            os.chmod(script_path, 0o755)
            print(f"[Iteration {self.iteration}] Wrote test rerun script to {script_path}")
        except Exception as e:
            print(f"[Iteration {self.iteration}] WARNING: Failed to write rerun script: {e}")
        
        # Record result
        self._record_result(params_dict, fitness, final_morph_id, test_script_dir, urdf_path, test_success, difficulty)
        
        # Update best result
        if fitness > self.best_result['fitness']:
            self.best_result = {
                'params': params_dict.copy(),
                'fitness': fitness,
                'morphology_id': final_morph_id,
                'log_dir': test_script_dir,
                'iteration': self.iteration
            }
            print(f"[Iteration {self.iteration}] *** NEW BEST FITNESS: {fitness:.4f} ***")
        
        # CMA-ES minimizes, so return negative fitness
        return -fitness
    
    def _record_result(
        self,
        params: Dict[str, float],
        fitness: float,
        morphology_id: Optional[str],
        log_dir: Optional[str],
        urdf_path: Optional[str],
        test_success: bool,
        test_difficulty: int
    ):
        """Record iteration result to history."""
        result = {
            'iteration': self.iteration,
            'params': params,
            'fitness': fitness,
            'morphology_id': morphology_id,
            'log_dir': log_dir,
            'urdf_path': urdf_path,
            'test_success': test_success,
            'test_difficulty': test_difficulty,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(result)
        
        # Save history incrementally
        history_path = os.path.join(self.results_dir, "cmaes_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def optimize(self, max_fevals: int = 100) -> Dict:
        """
        Run CMA-ES optimization.
        
        Args:
            max_fevals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}\nSTARTING CMA-ES OPTIMIZATION\n{'='*80}")
        print(f"Parameters to optimize: {self.param_names}")
        print(f"Bounds: {self.param_bounds}")
        print(f"Initial point (denormalized): {self._params_array_to_dict(self.x0)}")
        print(f"Sigma0: {self.sigma0}")
        print(f"Max function evaluations: {max_fevals}")
        print(f"Results directory: {self.results_dir}\n{'='*80}\n")
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, self.cmaes_options)
        
        # Optimization loop
        iteration_count = 0
        generation_count = 0
        try:
            while not es.stop() and iteration_count < max_fevals:
                solutions = es.ask()
                fitness_values = [self.objective_function(x) for x in solutions]
                es.tell(solutions, fitness_values)
                es.disp()
                iteration_count += len(solutions)
                generation_count += 1
                
                # Save CMA-ES state periodically (every 5 generations)
                if generation_count % 5 == 0:
                    self._save_cmaes_state(es)
                    
        except KeyboardInterrupt:
            print("\n[INFO] Optimization interrupted by user")
        
        # Final results
        print(f"\n{'='*80}\nOPTIMIZATION COMPLETE\n{'='*80}")
        print(f"Total function evaluations: {iteration_count}")
        print(f"Best fitness found: {self.best_result['fitness']:.4f}")
        print(f"Best parameters: {self.best_result['params']}")
        if self.best_result['morphology_id']:
            print(f"Best morphology ID: {self.best_result['morphology_id']}")
        if self.best_result['log_dir']:
            print(f"Best log directory: {self.best_result['log_dir']}")
        
        # Save final summary
        self._save_summary(es)
        
        # Save CMA-ES plots
        try:
            import matplotlib.pyplot as plt
            # Use cma.plot() function instead of es.result.plot()
            cma.plot()
            plt.savefig(os.path.join(self.results_dir, "cmaes_convergence.png"))
            plt.close()
            print(f"Saved convergence plot to {self.results_dir}/cmaes_convergence.png")
        except Exception as e:
            print(f"[WARNING] Could not save convergence plot: {e}")
        
        return {
            'best_params': self.best_result['params'],
            'best_fitness': self.best_result['fitness'],
            'best_morphology_id': self.best_result['morphology_id'],
            'best_log_dir': self.best_result['log_dir'],
            'total_evaluations': iteration_count,
            'history': self.history,
            'cmaes_result': es.result
        }
    
    def _save_cmaes_state(self, es: cma.CMAEvolutionStrategy):
        """Save CMA-ES state for potential resume."""
        state_path = os.path.join(self.results_dir, "cmaes_state.pkl")
        try:
            # pickle_dumps() returns the pickled string, we need to write it to file
            pickled_state = es.pickle_dumps()
            with open(state_path, 'wb') as f:
                f.write(pickled_state)
        except Exception as e:
            print(f"[WARNING] Could not save CMA-ES state: {e}")
    
    def _save_summary(self, es: cma.CMAEvolutionStrategy):
        """Save optimization summary."""
        summary = {
            'best_params': self.best_result['params'],
            'best_fitness': self.best_result['fitness'],
            'best_morphology_id': self.best_result['morphology_id'],
            'best_log_dir': self.best_result['log_dir'],
            'best_iteration': self.best_result.get('iteration'),
            'total_iterations': self.iteration,
            'param_bounds': self.param_bounds,
            'evaluation_config': {
                'load_run': self.load_run,
                'checkpoint': self.checkpoint,
                'num_envs': self.num_envs,
                'num_episodes': self.num_episodes,
                'seed': self.seed,
                'task': self.task
            },
            'cmaes_config': {
                'sigma0': self.sigma0,
                'options': self.cmaes_options
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.results_dir, "cmaes_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")
        
        # Also save top 5 results
        sorted_history = sorted(
            [h for h in self.history if h['fitness'] != float('-inf')],
            key=lambda x: x['fitness'],
            reverse=True
        )[:5]
        
        top5_path = os.path.join(self.results_dir, "cmaes_top5.json")
        with open(top5_path, 'w') as f:
            json.dump(sorted_history, f, indent=2)
        print(f"Saved top 5 results to {top5_path}")


def main():
    """Run morphology optimization with CMA-ES using universal controller."""
    parser = argparse.ArgumentParser(description="BALLU Morphology Optimization with CMA-ES (Universal Controller)")
    parser.add_argument("--max_fevals", type=int, default=40, help="Maximum function evaluations (default: 40)")
    parser.add_argument("--load_run", type=str, required=True, help="Run name of universal controller (required)")
    parser.add_argument("--checkpoint", type=str, default="model_best.pt", help="Checkpoint name (default: model_best.pt)")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments for evaluation (default: 64)")
    parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes for evaluation (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CMA-ES (default: 42)")
    parser.add_argument("--task", type=str, default="Isc-BALLU-hetero-general", help="Task name (default: Isc-BALLU-hetero-general)")
    parser.add_argument("--sigma0", type=float, default=0.2, help="Initial step size for CMA-ES (default: 0.2)")
    parser.add_argument("--popsize", type=int, default=None, help="Population size (default: CMA-ES default)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory (default: logs/results/<timestamp>)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for testing (default: cuda:0)")
    parser.add_argument("--test_video_length", type=int, default=399, help="Test video length in frames (default: 399)")
    parser.add_argument("--test_num_envs", type=int, default=1, help="Number of envs for testing (default: 1)")
    
    # Initial parameter values (optional)
    parser.add_argument("--init_femur_length", type=float, default=None, help="Initial femur length (default: midpoint)")
    parser.add_argument("--init_tibia_length", type=float, default=None, help="Initial tibia length (default: midpoint)")
    parser.add_argument("--init_gravity_comp_ratio", type=float, default=None, help="Initial gravity comp ratio (default: midpoint)")
    parser.add_argument("--init_spring_coeff", type=float, default=None, help="Initial spring coefficient (default: midpoint)")
    parser.add_argument("--difficulty_level", type=int, default=10, help="Difficulty level (default: 10)")
    parser.add_argument("--cmdir", type=str, default="cmaes", help="Common directory name (default: cmaes)")
    
    args = parser.parse_args()
    
    # Define parameter bounds (same as Optuna version)
    param_bounds = {
        'femur_length': (0.30, 0.48),
        'tibia_length': (0.30, 0.43),
        'gravity_comp_ratio': (0.75, 0.87),
        'spring_coeff': (1e-3, 1e-2)
    }
    
    # Set initial parameters if provided
    initial_params = None
    if any([args.init_femur_length, args.init_tibia_length, args.init_gravity_comp_ratio, args.init_spring_coeff]):
        initial_params = {}
        if args.init_femur_length is not None:
            initial_params['femur_length'] = args.init_femur_length
        if args.init_tibia_length is not None:
            initial_params['tibia_length'] = args.init_tibia_length
        if args.init_gravity_comp_ratio is not None:
            initial_params['gravity_comp_ratio'] = args.init_gravity_comp_ratio
        if args.init_spring_coeff is not None:
            initial_params['spring_coeff'] = args.init_spring_coeff
        
        # Fill in missing values with midpoints
        for param_name, bounds in param_bounds.items():
            if param_name not in initial_params:
                initial_params[param_name] = (bounds[0] + bounds[1]) / 2
    
    # Set results directory
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    results_dir = args.results_dir if args.results_dir else f"{project_dir}/logs/results/{timestamp}_CMAES"
    
    print(f"\n{'='*80}\nBALLU MORPHOLOGY OPTIMIZATION WITH CMA-ES (Universal Controller)\n{'='*80}")
    print(f"Max function evaluations: {args.max_fevals}")
    print(f"Universal Controller: run={args.load_run}, checkpoint={args.checkpoint}")
    print(f"Evaluation: num_envs={args.num_envs}, num_episodes={args.num_episodes}, task={args.task}")
    print(f"CMA-ES: sigma0={args.sigma0}, popsize={args.popsize if args.popsize else 'default'}, seed={args.seed}")
    print(f"Results directory: {results_dir}")
    print(f"Parameter bounds: {param_bounds}")
    if initial_params:
        print(f"Initial parameters: {initial_params}")
    print(f"{'='*80}\n")
    
    args.cmdir = f"{timestamp}_CMAES"
    # Create optimizer
    optimizer = CMAESOptimizer(
        param_bounds=param_bounds,
        initial_params=initial_params,
        sigma0=args.sigma0,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        seed=args.seed,
        task=args.task,
        device=args.device,
        test_video_length=args.test_video_length,
        test_num_envs=args.test_num_envs,
        results_dir=results_dir,
        popsize=args.popsize,
        difficulty_level=args.difficulty_level,
        cmdir=args.cmdir
    )
    
    # Run optimization
    results = optimizer.optimize(max_fevals=args.max_fevals)
    
    print(f"\n{'='*80}\n✓ CMA-ES Optimization completed!\n{'='*80}\n")


if __name__ == "__main__":
    main()

