import os
import sys
import random
from datetime import datetime
import subprocess
import json
import argparse
import optuna
from optuna.trial import TrialState
# from optuna.integration.wandb import WeightsAndBiasesCallback
import torch

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

def run_training_experiment(
        morph_id: str, 
        task: str = "Isc-Vel-BALLU-1-obstacle",
        max_iterations: int = 50,
        seed: int = 42,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65
    ) -> tuple[bool, float]:
    """Run training experiment and return success status and best curriculum level."""
    train_script_path = f"{project_dir}/scripts/rsl_rl/train.py"
    cmd = [
        sys.executable, 
        train_script_path,
        "--task", task,
        "--num_envs", "4096",
        "--max_iterations", str(max_iterations),
        "--run_name", morph_id,
        "--headless",
        "--seed", str(seed),
        "--gravity_compensation_ratio", str(gravity_comp_ratio),
        f"env.scene.robot.actuators.knee_effort_actuators.spring_coeff={spring_coeff:.6f}"
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/11.11.2025/{morph_id}/{morph_id}.usd"
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    log_dir = None
    try:
        for line in process.stdout:
            print(line, end='')
            if line.startswith("EXP_DIR:"):
                log_dir = line.split("EXP_DIR:")[1].strip()
        process.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        print(f"Training timeout (2h), killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), ""

    if process.returncode != 0:
        return False, float('-inf'), ""
    
    ckpt_dict = torch.load(os.path.join(log_dir, "model_best.pt"))
    best_crclm_level = ckpt_dict["best_crclm_level"]
    
    return True, best_crclm_level, log_dir


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
        gravity_comp_ratio: float = 0.65
    ) -> tuple[bool, str]:
    """Run testing experiment for a trained morphology and return success status."""
    test_script_path = f"{project_dir}/scripts/rsl_rl/play_single_obstacle.py"
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
        "--gravity_compensation_ratio", str(gravity_comp_ratio),
        f"env.scene.robot.actuators.knee_effort_actuators.spring_coeff={spring_coeff:.6f}",
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/11.11.2025/{morph_id}/{morph_id}.usd"
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


def generate_morphology_from_params(sampled_config: dict, trial_number: int):
    """Generate morphology from sampled parameters and return (urdf_path, usd_path, morphology_id)."""
    print(f"\n[Trial {trial_number}] Generating morphology: {sampled_config['morphology_id']}")
    
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
            print(f"[Trial {trial_number}] Generated USD: {usd_file_path}")
            return urdf_path, usd_file_path, morph.morphology_id
        else:
            print(f"[ERROR] USD conversion failed (code: {return_code})")
            return None, None, None
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return None, None, None


def objective(
    trial: optuna.Trial,
    max_iterations: int,
    seed: int,
    task: str,
    device: str,
    test_video_length: int,
    test_num_envs: int,
) -> float:
    """Optuna objective: samples parameters, generates morphology, trains, and returns best curriculum level."""
    print(f"\n{'='*80}\n[TRIAL {trial.number}]\n{'='*80}")
    
    # Sample parameters
    femur_length = trial.suggest_float("femur_length", 0.30, 0.48)
    tibia_length = trial.suggest_float("tibia_length", 0.30, 0.43)
    # hip_width = trial.suggest_float("hip_width", 0.08, 0.14)
    # femur_to_limb_ratio = trial.suggest_float("femur_to_limb_ratio", 0.20, 0.70)
    # knee_damping = trial.suggest_float("Kd_knee", 0.06, 0.50)
    # spring_damping = trial.suggest_float("Kd_spring", 0.001, 0.08)
    gravity_comp_ratio = trial.suggest_float("gravity_comp_ratio", 0.75, 0.87)
    spring_coeff = trial.suggest_float("spring_coeff", 1e-3, 1e-2)
    # morph_id = f"trial{trial.number:02d}_f{femur_length:.2f}_t{tibia_length:.2f}_knKd{knee_damping:.2f}"
    morph_id = f"trial{trial.number:02d}_fl{femur_length:.3f}_tl{tibia_length:.3f}_spc{spring_coeff:.3f}_gcr{gravity_comp_ratio:.3f}"

    sampled_config = {
        "morphology_id": morph_id,
        "femur_length": femur_length,
        "tibia_length": tibia_length
        # "femur_to_limb_ratio": femur_to_limb_ratio
    }
    
    # Generate morphology
    urdf_path, usd_path, final_morph_id = generate_morphology_from_params(sampled_config, trial.number)
    
    if usd_path is None:
        print(f"[Trial {trial.number}] FAILED: Morphology generation")
        return float('-inf')
    
    # Run training with passed parameters
    print(f"[Trial {trial.number}] Starting training (max_iter={max_iterations}, seed={seed}, task={task})...")
    success, best_crclm_level, log_dir = run_training_experiment(
        final_morph_id, 
        task=task,
        max_iterations=max_iterations, 
        seed=seed,
        spring_coeff=spring_coeff,
        gravity_comp_ratio=gravity_comp_ratio
    )
    
    if not success:
        print(f"[Trial {trial.number}] FAILED: Training")
        return float('-inf')
    
    print(f"[Trial {trial.number}] COMPLETED - Best curriculum level: {best_crclm_level:.4f}")
    
    # Store trial metadata
    trial.set_user_attr("morphology_id", final_morph_id)
    trial.set_user_attr("urdf_path", urdf_path)
    trial.set_user_attr("usd_path", usd_path)
    trial.set_user_attr("best_crclm_level", best_crclm_level)
    trial.set_user_attr("log_dir", log_dir)

    # Run testing for this morphology
    run_name = log_dir.split("/")[-1] if log_dir else ""
    difficulty = int(best_crclm_level * 100) - 1 if best_crclm_level > 0 else 0
    print(f"[Trial {trial.number}] Starting testing (difficulty={difficulty})...")
    test_success, test_command = run_testing_experiment(
        morph_id=final_morph_id,
        task=task,
        load_run=run_name,
        checkpoint="model_best.pt",
        num_envs=test_num_envs,
        video_length=test_video_length,
        device=device,
        difficulty_level=difficulty,
        spring_coeff=spring_coeff,
        gravity_comp_ratio=gravity_comp_ratio
    )
    print(f"[Trial {trial.number}] Testing {'SUCCESS' if test_success else 'FAILED'}")
    # Write test_command to a shell script in log_dir and make it executable.
    script_path = os.path.join(log_dir, f"rerun_test_{trial.number}.sh")
    try:
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(test_command + "\n")
        os.chmod(script_path, 0o755)
        print(f"[Trial {trial.number}] Wrote test rerun script to {script_path}. Run it to rerun the test.")
    except Exception as e:
        print(f"[Trial {trial.number}] WARNING: Failed to write rerun_test_{trial.number}.sh: {e}")
        
    trial.set_user_attr("test_success", test_success)
    trial.set_user_attr("test_difficulty_level", difficulty)
    
    return best_crclm_level


def main():
    """Run morphology optimization with Optuna."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="BALLU Morphology Optimization with Optuna")
    parser.add_argument("--n_trials", type=int, default=40, help="Number of optimization trials (default: 40)")
    parser.add_argument("--max_iterations", type=int, default=1600, help="Max training iterations per trial (default: 1600)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training (default: 42)")
    parser.add_argument("--task", type=str, default="Isc-Vel-BALLU-1-obstacle", help="Task name (default: Isc-Vel-BALLU-1-obstacle)")
    parser.add_argument("--study_name", type=str, default="TPE", help="Study name (default: TPE)")
    parser.add_argument("--storage", type=str, default=None, help="Database storage path (default: sqlite:///logs/optuna/TPE.db)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing study (default: False)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for testing/play (default: cuda:0)")
    parser.add_argument("--test_video_length", type=int, default=399, help="Test video length in frames (default: 399)")
    parser.add_argument("--test_num_envs", type=int, default=1, help="Number of envs for testing (default: 1)")
    
    args = parser.parse_args()
    
    # Configuration
    N_TRIALS = args.n_trials
    MAX_ITERATIONS = args.max_iterations
    SEED = args.seed
    TASK = args.task
    DEVICE = args.device
    TEST_VIDEO_LENGTH = args.test_video_length
    TEST_NUM_ENVS = args.test_num_envs
    
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    STUDY_NAME = args.study_name if args.resume else f"{timestamp}_{args.study_name}"
    STORAGE = args.storage if args.storage else f"sqlite:///{project_dir}/logs/optuna/TPE.db"
    
    print(f"\n{'='*80}\nBALLU MORPHOLOGY OPTIMIZATION\n{'='*80}")
    print(f"Study: {STUDY_NAME} | Trials: {N_TRIALS}")
    print(f"Training: max_iter={MAX_ITERATIONS}, seed={SEED}, task={TASK}")
    print(f"Storage: {STORAGE}\n{'='*80}")
    
    # Ensure storage directory exists
    if STORAGE and STORAGE.startswith("sqlite:///"):
        db_path = STORAGE.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    # wandb_cb = WeightsAndBiasesCallback(
    #     wandb_kwargs={"project": STUDY_NAME, "entity": "ankitdipto"},
    #     as_multirun=True,
    #     metric_name="best_crclm_level"
    # )

    # Run optimization with lambda to pass additional parameters
    try:
        study.optimize(
            lambda trial: objective(trial, MAX_ITERATIONS, SEED, TASK, DEVICE, TEST_VIDEO_LENGTH, TEST_NUM_ENVS),
            n_trials=N_TRIALS, 
            show_progress_bar=True, 
            catch=(Exception,),
            # callbacks=[wandb_cb]
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Results
    print(f"\n{'='*80}\nRESULTS\n{'='*80}")
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Finished: {len(study.trials)} | Completed: {len(completed_trials)}")
    
    if completed_trials:
        best_trial = study.best_trial
        print(f"\nBest Trial {best_trial.number}: {best_trial.value:.4f}")
        print(f"  Parameters: {best_trial.params}")
        if "morphology_id" in best_trial.user_attrs:
            print(f"  Morphology ID: {best_trial.user_attrs['morphology_id']}")
        if "log_dir" in best_trial.user_attrs:
            print(f"  Log Directory: {best_trial.user_attrs['log_dir']}")
        print("\nTop 5 Trials:")
        for i, trial in enumerate(sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5], 1):
            print(f"  {i}. Trial {trial.number}: {trial.value:.4f} - {trial.params}")
            if "log_dir" in trial.user_attrs:
                print(f"    Log Directory: {trial.user_attrs['log_dir']}")
    else:
        print("No trials completed successfully.")
    
    # Save summary
    summary_path = f"{project_dir}/logs/optuna/10.14.2025/{STUDY_NAME}_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        "study_name": STUDY_NAME,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_value": study.best_value if completed_trials else None,
        "best_params": study.best_params if completed_trials else None,
        "best_trial_number": study.best_trial.number if completed_trials else None,
        "training_config": {
            "max_iterations": MAX_ITERATIONS,
            "seed": SEED,
            "task": TASK
        },
        "best_trial_log_dir": best_trial.user_attrs["log_dir"] if "log_dir" in best_trial.user_attrs else None,
        "best_trial_morphology_id": best_trial.user_attrs["morphology_id"] if "morphology_id" in best_trial.user_attrs else None
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}\n{'='*80}\n✓ Optimization completed!")


if __name__ == "__main__":
    main()