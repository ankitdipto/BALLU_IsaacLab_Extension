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

def run_evaluation_experiment(
        morph_id: str,
        task: str = "Isc-BALLU-fast-walk",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 64,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
    ) -> tuple[bool, float, str]:
    """Run evaluation with universal controller and return success status, performance metric, and eval info."""
    eval_script_path = f"{project_dir}/scripts/rsl_rl/evaluate_design_walking.py"
    cmd = [
        sys.executable,
        eval_script_path,
        "--task", task,
        "--load_run", load_run,
        "--checkpoint", checkpoint, 
        "--num_envs", str(num_envs),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff),
        "--headless"
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/02.10.2026/{morph_id}/{morph_id}.usd"
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(f"Running evaluation experiment with command: {cmd}")

    performance_metric = None
    eval_info = ""
    try:
        for line in process.stdout:
            print(line, end='')
            eval_info += line
            # Parse PERFORMANCE_METRIC from output (e.g., base velocity)
            if line.startswith("PERFORMANCE_METRIC:"):
                try:
                    performance_metric = float(line.split(":")[1].strip())
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse performance metric: {e}")
                    raise e
        process.wait(timeout=30 * 60)  # 30 minute timeout
    except subprocess.TimeoutExpired:
        print(f"Evaluation timeout (30 min), killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), ""

    if process.returncode != 0:
        return False, float('-inf'), ""
    
    if performance_metric is None:
        print(f"[WARNING] Could not extract performance metric from output")
        return False, float('-inf'), eval_info
    
    return True, performance_metric, eval_info


def run_testing_experiment(
        morph_id: str, 
        task: str = "Isc-BALLU-fast-walk",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 1,
        video_length: int = 399,
        device: str = "cuda:0",
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
        difficulty_level: int = 0,
        cmdir: str = "optuna_walking"
    ) -> tuple[bool, str]:
    """Run testing experiment with universal controller and return success status."""
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
    env['BALLU_USD_REL_PATH'] = f"morphologies/02.10.2026/{morph_id}/{morph_id}.usd"
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
    load_run: str,
    checkpoint: str,
    num_envs: int,
    task: str,
    device: str,
    test_video_length: int,
    test_num_envs: int,
    difficulty_level: int,
    cmdir: str,
    results_dir: str,
) -> float:
    """Optuna objective: samples parameters, generates morphology, evaluates with universal controller."""
    print(f"\n{'='*80}\n[TRIAL {trial.number}]\n{'='*80}")
    
    # Sample parameters
    femur_length = trial.suggest_float("femur_length", 0.30, 0.60)
    tibia_length = trial.suggest_float("tibia_length", 0.24, 0.54)
    gravity_comp_ratio = trial.suggest_float("gravity_comp_ratio", 0.65, 0.85)
    spring_coeff = trial.suggest_float("spring_coeff", 1e-3, 4e-2)
    hip_width = trial.suggest_float("hip_width", 0.08, 0.15)
    
    morph_id = f"trial{trial.number:03d}_fl{femur_length:.3f}_tl{tibia_length:.3f}_hw{hip_width:.3f}_spc{spring_coeff:.5f}_gcr{gravity_comp_ratio:.3f}"

    sampled_config = {
        "morphology_id": morph_id,
        "femur_length": femur_length,
        "tibia_length": tibia_length,
        "hip_width": hip_width,
        "pelvis_height": hip_width
    }
    
    # Generate morphology
    urdf_path, usd_path, final_morph_id = generate_morphology_from_params(sampled_config, trial.number)
    
    if usd_path is None:
        print(f"[Trial {trial.number}] FAILED: Morphology generation")
        return float('-inf')
    
    # Run evaluation with universal controller
    print(f"[Trial {trial.number}] Starting evaluation with universal controller (task={task})...")
    success, performance_metric, eval_info = run_evaluation_experiment(
        final_morph_id, 
        task=task,
        load_run=load_run,
        checkpoint=checkpoint,
        num_envs=num_envs,
        spring_coeff=spring_coeff,
        gravity_comp_ratio=gravity_comp_ratio,
    )
    
    if not success:
        print(f"[Trial {trial.number}] FAILED: Evaluation")
        return float('-inf')
    
    print(f"[Trial {trial.number}] COMPLETED - Performance metric: {performance_metric:.4f}")
    
    # Store trial metadata
    trial.set_user_attr("morphology_id", final_morph_id)
    trial.set_user_attr("urdf_path", urdf_path)
    trial.set_user_attr("usd_path", usd_path)
    trial.set_user_attr("performance_metric", performance_metric)

    # Run testing with universal controller (record video)
    print(f"[Trial {trial.number}] Starting testing with universal controller...")
    test_success, test_command = run_testing_experiment(
        morph_id=final_morph_id,
        task=task,
        load_run=load_run,
        checkpoint=checkpoint,
        num_envs=test_num_envs,
        video_length=test_video_length,
        device=device,
        difficulty_level=int(performance_metric * 100),
        spring_coeff=spring_coeff,
        gravity_comp_ratio=gravity_comp_ratio,
        cmdir=cmdir
    )
    print(f"[Trial {trial.number}] Testing {'SUCCESS' if test_success else 'FAILED'}")
    
    # Write test rerun script to results directory
    test_script_dir = os.path.join(results_dir, f"trial_{trial.number:03d}_{final_morph_id}")
    os.makedirs(test_script_dir, exist_ok=True)
    script_path = os.path.join(test_script_dir, f"rerun_test.sh")
    try:
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(test_command + "\n")
        os.chmod(script_path, 0o755)
        print(f"[Trial {trial.number}] Wrote test rerun script to {script_path}")
    except Exception as e:
        print(f"[Trial {trial.number}] WARNING: Failed to write rerun script: {e}")
        
    trial.set_user_attr("test_success", test_success)
    trial.set_user_attr("test_script_dir", test_script_dir)
    
    return performance_metric


def main():
    """Run morphology optimization with Optuna using universal controller."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="BALLU Morphology Optimization with Optuna (Universal Controller)")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials (default: 100)")
    parser.add_argument("--load_run", type=str, required=True, help="Run name of universal controller (required)")
    parser.add_argument("--checkpoint", type=str, default="model_best.pt", help="Checkpoint name (default: model_best.pt)")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments for evaluation (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler (default: 42)")
    parser.add_argument("--task", type=str, default="Isc-BALLU-fast-walk", help="Task name (default: Isc-BALLU-fast-walk-hetero)")
    parser.add_argument("--study_name", type=str, default="TPE_walking", help="Study name (default: TPE_walking)")
    parser.add_argument("--storage", type=str, default=None, help="Database storage path (default: sqlite:///logs/optuna/TPE_walking.db)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing study (default: False)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for testing/play (default: cuda:0)")
    parser.add_argument("--test_video_length", type=int, default=399, help="Test video length in frames (default: 399)")
    parser.add_argument("--test_num_envs", type=int, default=1, help="Number of envs for testing (default: 1)")
    parser.add_argument("--difficulty_level", type=int, default=0, help="Difficulty level (default: 0)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory (default: logs/results/<timestamp>)")
    
    args = parser.parse_args()
    
    # Configuration
    N_TRIALS = args.n_trials
    LOAD_RUN = args.load_run
    CHECKPOINT = args.checkpoint
    NUM_ENVS = args.num_envs
    SEED = args.seed
    TASK = args.task
    DEVICE = args.device
    TEST_VIDEO_LENGTH = args.test_video_length
    TEST_NUM_ENVS = args.test_num_envs
    DIFFICULTY_LEVEL = args.difficulty_level
    
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    STUDY_NAME = args.study_name if args.resume else f"{timestamp}_{args.study_name}"
    STORAGE = args.storage if args.storage else f"sqlite:///{project_dir}/logs/optuna/TPE_walking.db"
    CMDIR = f"{timestamp}_TPE_walking"
    RESULTS_DIR = args.results_dir if args.results_dir else f"{project_dir}/logs/results/{CMDIR}"
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"\n{'='*80}\nBALLU MORPHOLOGY OPTIMIZATION WITH OPTUNA (Universal Controller)\n{'='*80}")
    print(f"Study: {STUDY_NAME} | Trials: {N_TRIALS}")
    print(f"Universal Controller: run={LOAD_RUN}, checkpoint={CHECKPOINT}")
    print(f"Evaluation: num_envs={NUM_ENVS}, task={TASK}")
    print(f"Optuna: sampler=TPE, seed={SEED}")
    print(f"Storage: {STORAGE}")
    print(f"Results directory: {RESULTS_DIR}\n{'='*80}")
    
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
    #     metric_name="performance_metric"
    # )

    # Run optimization with lambda to pass additional parameters
    try:
        study.optimize(
            lambda trial: objective(
                trial, 
                LOAD_RUN, 
                CHECKPOINT, 
                NUM_ENVS, 
                TASK, 
                DEVICE, 
                TEST_VIDEO_LENGTH, 
                TEST_NUM_ENVS,
                DIFFICULTY_LEVEL,
                CMDIR,
                RESULTS_DIR
            ),
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
        if "test_script_dir" in best_trial.user_attrs:
            print(f"  Test Script Directory: {best_trial.user_attrs['test_script_dir']}")
        print("\nTop 5 Trials:")
        for i, trial in enumerate(sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5], 1):
            print(f"  {i}. Trial {trial.number}: {trial.value:.4f} - {trial.params}")
            if "test_script_dir" in trial.user_attrs:
                print(f"    Test Script Directory: {trial.user_attrs['test_script_dir']}")
    else:
        print("No trials completed successfully.")
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, f"{STUDY_NAME}_summary.json")
    
    summary = {
        "study_name": STUDY_NAME,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_value": study.best_value if completed_trials else None,
        "best_params": study.best_params if completed_trials else None,
        "best_trial_number": study.best_trial.number if completed_trials else None,
        "evaluation_config": {
            "load_run": LOAD_RUN,
            "checkpoint": CHECKPOINT,
            "num_envs": NUM_ENVS,
            "seed": SEED,
            "task": TASK
        },
        "best_trial_test_script_dir": best_trial.user_attrs.get("test_script_dir") if completed_trials else None,
        "best_trial_morphology_id": best_trial.user_attrs.get("morphology_id") if completed_trials else None
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save top 5 results
    top5_trials = []
    for trial in sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]:
        top5_trials.append({
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "morphology_id": trial.user_attrs.get("morphology_id"),
            "test_script_dir": trial.user_attrs.get("test_script_dir")
        })
    
    top5_path = os.path.join(RESULTS_DIR, "optuna_top5.json")
    with open(top5_path, 'w') as f:
        json.dump(top5_trials, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"Top 5 results saved: {top5_path}\n{'='*80}\n✓ Optimization completed!")


if __name__ == "__main__":
    main()