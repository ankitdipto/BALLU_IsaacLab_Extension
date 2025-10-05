import os
import sys
import random
from datetime import datetime
import subprocess
import json
import optuna
from optuna.trial import TrialState
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
        seed: int = 42
    ) -> tuple[bool, float]:
    """
    Run training experiment and return success status and final reward.
    
    Returns:
        tuple: (success: bool, final_reward: float)
    """
    train_script_path = f"{project_dir}/scripts/rsl_rl/train.py"
    cmd = [
        sys.executable,
        train_script_path,
        "--task", task,
        "--num_envs", "4096",
        "--max_iterations", str(max_iterations),
        "--run_name", morph_id,
        "--headless",
        "--seed", str(seed)
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = f"morphologies/{morph_id}/{morph_id}.usd"
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
        process.wait(timeout=2 * 3600)  # 2 hours
    
    except subprocess.TimeoutExpired:
        print("Training is taking more than 2 hours, killing the process")
        process.kill()
        process.wait()
        return False, float('-inf')

    success = process.returncode == 0
    
    # Extract final reward from logs
    log_dir = f"{project_dir}/logs/rsl_rl/10.07.2025/{morph_id}"
    ckpt_dict = torch.load(os.path.join(log_dir, "model_best.pt"))
    best_crclm_level = ckpt_dict["best_crclm_level"]
    
    return success, best_crclm_level


def generate_morphology_from_params(sampled_config: dict, trial_number: int):
    """
    Generate a morphology from sampled parameters.
    
    Args:
        sampled_config: Dictionary of sampled parameter values
        trial_number: Optuna trial number for ID generation
    
    Returns:
        tuple: (urdf_path, usd_path, morphology_id) or (None, None, None) if failed
    """
    print("=" * 80)
    print(f"BALLU Morphology Generation - Trial {trial_number}")
    print("=" * 80)
    
    print("\nSampled Parameters:")
    for param, value in sampled_config.items():
        if param != 'morphology_id':
            print(f"  {param}: {value:.6f}")
    
    # Create morphology variant
    print(f"\nCreating morphology with ID: {sampled_config['morphology_id']}")
    
    try:
        morph = create_morphology_variant(**sampled_config)
    except Exception as e:
        print(f"[ERROR] Failed to create morphology: {e}")
        return None, None, None
    
    # Validate morphology
    print("\nValidating morphology...")
    is_valid, errors = morph.validate()
    
    if not is_valid:
        print("✗ Validation FAILED")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return None, None, None
    
    print("✓ Validation PASSED")
    
    # Display computed properties
    props = morph.get_derived_properties()
    print("\nComputed Properties:")
    print(f"  Total Mass: {props['total_mass']:.4f} kg")
    print(f"  Total Leg Length: {props['total_leg_length']:.4f} m")
    print(f"  Femur-to-Limb Ratio: {props['femur_to_limb_ratio']:.4f}")
    print(f"  Balloon Mass Ratio: {props['balloon_mass_ratio']:.4f}")
    
    # Generate URDF
    print("\nGenerating URDF...")
    try:
        generator = BalluRobotGenerator(morph)
        urdf_path = generator.generate_urdf()
        print(f"✓ URDF generated: {urdf_path}")
    except Exception as e:
        print(f"✗ URDF generation failed: {e}")
        return None, None, None
    
    # Generate USD
    print("\nGenerating USD...")
    try:
        return_code, usd_file_path = generator.generate_usd(urdf_path)
        
        if return_code == 0 and os.path.exists(usd_file_path):
            print(f"✓ USD generated: {usd_file_path}")
            print(f"✓ Morphology generation completed!")
            return urdf_path, usd_file_path, morph.morphology_id
        else:
            print(f"✗ USD conversion failed with return code: {return_code}")
            return None, None, None
            
    except Exception as e:
        print(f"✗ USD generation failed: {e}")
        return None, None, None


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for morphology optimization.
    
    This function:
    1. Samples morphology parameters from defined ranges
    2. Generates the robot (URDF + USD)
    3. Trains the robot on the task
    4. Returns the final reward (to be maximized)
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Final mean reward from training (higher is better)
    """
    print("\n" + "=" * 80)
    print(f"OPTUNA TRIAL {trial.number}")
    print("=" * 80)
    
    # Sample parameters from defined ranges
    # You can add more parameters here as needed
    femur_length = trial.suggest_float("femur_length", 0.20, 0.45)
    tibia_length = trial.suggest_float("tibia_length", 0.20, 0.45)
    
    # You can add more parameters to optimize:
    # total_leg_length = trial.suggest_float("total_leg_length", 0.6, 1.0)
    # hip_width = trial.suggest_float("hip_width", 0.08, 0.15)
    # balloon_radius = trial.suggest_float("balloon_radius", 0.25, 0.40)
    
    # Generate unique morphology ID
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    morph_id = f"trial{trial.number:03d}_f{femur_length:.2f}_t{tibia_length:.2f}"
    
    # Prepare config dictionary
    sampled_config = {
        "morphology_id": morph_id,
        "femur_length": femur_length,
        "tibia_length": tibia_length
        # Add other sampled parameters here
    }
    
    # Generate morphology
    urdf_path, usd_path, final_morph_id = generate_morphology_from_params(
        sampled_config, 
        trial.number
    )
    
    if usd_path is None:
        print(f"\n✗ Trial {trial.number} FAILED: Morphology generation failed")
        return float('-inf')
    
    # Run training experiment
    print("\n" + "=" * 80)
    print(f"Starting Training for Trial {trial.number}")
    print("=" * 80)
    
    success, best_crclm_level = run_training_experiment(
        final_morph_id,
        max_iterations=1200,  # Adjust based on your needs
        seed=42
    )
    
    if not success:
        print(f"\n✗ Trial {trial.number} FAILED: Training failed")
        return float('-inf')
    
    print("\n" + "=" * 80)
    print(f"Trial {trial.number} COMPLETED")
    print(f"Best Curriculum Level in this trial: {best_crclm_level:.4f}")
    print("=" * 80)
    
    # Store additional info in trial user attributes
    trial.set_user_attr("morphology_id", final_morph_id)
    trial.set_user_attr("urdf_path", urdf_path)
    trial.set_user_attr("usd_path", usd_path)
    trial.set_user_attr("best_crclm_level", best_crclm_level)
    
    return best_crclm_level


def run_optuna_optimization(
    n_trials: int = 20,
    study_name: str = "ballu_morphology_optimization",
    storage: str = None
):
    """
    Run Optuna optimization study for BALLU morphology.
    
    Args:
        n_trials: Number of optimization trials to run
        study_name: Name of the Optuna study
        storage: Database URL for persistent storage (e.g., "sqlite:///optuna_study.db")
                 If None, study is stored in memory only
    """
    print("\n" + "=" * 80)
    print("BALLU MORPHOLOGY OPTIMIZATION WITH OPTUNA")
    print("=" * 80)
    print(f"Study Name: {study_name}")
    print(f"Number of Trials: {n_trials}")
    print(f"Storage: {storage if storage else 'In-memory (not persistent)'}")
    print("=" * 80)
    
    # Ensure storage directory exists if using SQLite
    if storage and storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            print(f"Ensured storage directory exists: {db_dir}")
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # We want to maximize the best curriculum level
        load_if_exists=True,   # Resume if study already exists
        sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
    )
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Catch exceptions and continue with next trial
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    if completed_trials:
        print(f"Number of completed trials: {len(completed_trials)}")
        
        # Best trial
        best_trial = study.best_trial
        print(f"\nBest Trial: {best_trial.number}")
        print(f"  Best Curriculum Level: {best_trial.value:.4f}")
        print(f"  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.6f}")
        
        if "morphology_id" in best_trial.user_attrs:
            print(f"  Morphology ID: {best_trial.user_attrs['morphology_id']}")
            print(f"  USD Path: {best_trial.user_attrs.get('usd_path', 'N/A')}")
        
        # Top 5 trials
        print("\nTop 5 Trials:")
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
        for i, trial in enumerate(sorted_trials, 1):
            print(f"  {i}. Trial {trial.number}: Best Curriculum Level = {trial.value:.4f}")
            print(f"     Parameters: {trial.params}")
    else:
        print("No trials completed successfully.")
    
    # Save study summary
    summary_path = f"{project_dir}/logs/optuna/10.07.2025/{study_name}_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_value": study.best_value if completed_trials else None, # Best curriculum level
        "best_params": study.best_params if completed_trials else None,
        "best_trial_number": study.best_trial.number if completed_trials else None,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nStudy summary saved to: {summary_path}") # Best curriculum level
    print("=" * 80)
    
    return study


def main():
    """
    Main function to run morphology optimization.
    """
    # ==========================================================================
    # OPTUNA CONFIGURATION
    # ==========================================================================
    
    # Number of optimization trials
    N_TRIALS = 40
    
    # Study name (for tracking and resuming)
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    STUDY_NAME = f"{timestamp}_TPE"
    
    # Database for persistent storage (optional but recommended)
    # Use SQLite for local storage
    STORAGE = f"sqlite:///{project_dir}/logs/optuna/10.07.2025/{STUDY_NAME}.db"
    # Or set to None for in-memory only (not persistent across runs)
    # STORAGE = None
    
    # ==========================================================================
    
    # Run optimization
    study = run_optuna_optimization(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        storage=STORAGE
    )
    
    print("\n✓ Morphology optimization completed!")


if __name__ == "__main__":
    main()