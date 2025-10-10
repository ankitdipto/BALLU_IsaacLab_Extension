import os
import sys
import argparse
from datetime import datetime
import subprocess
import json
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
        max_iterations: int = 1200,
        seed: int = 42,
        device: str = "cuda:0"
    ) -> tuple[bool, float]:
    """
    Run training experiment and return success status and final reward.
    
    Args:
        morph_id: Unique identifier for the morphology
        task: Task identifier
        max_iterations: Maximum number of training iterations
        seed: Random seed for reproducibility
        device: Device to use for training
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
        "--seed", str(seed),
        "--device", device
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
        process.wait(timeout=2 * 3600)  # 2 hours timeout
    
    except subprocess.TimeoutExpired:
        print("Training exceeded 2-hour limit, terminating process")
        process.kill()
        process.wait()
        return False, float('-inf')

    success = process.returncode == 0
    
    # Extract final reward from logs
    log_dir = f"{project_dir}/logs/rsl_rl/10.07.2025/{morph_id}"
    ckpt_dict = torch.load(os.path.join(log_dir, "model_best.pt"))
    best_crclm_level = ckpt_dict["best_crclm_level"]
    
    return success, best_crclm_level

def generate_morphology_variant(femur_ratio: float, total_leg_length: float = 0.75) -> tuple[str, str, str]:
    """
    Generate a morphology variant with specified femur-to-total-leg ratio.
    
    Args:
        femur_ratio: Ratio of femur length to total leg length (0-1)
        total_leg_length: Total leg length in meters
    
    Returns:
        tuple: (urdf_path, usd_path, morphology_id) or (None, None, None) if failed
    """
    # Calculate individual segment lengths
    femur_length = total_leg_length * femur_ratio
    tibia_length = total_leg_length * (1 - femur_ratio)
    
    # Generate unique morphology ID
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    morph_id = f"femur_ratio_{femur_ratio:.3f}_{timestamp}"
    
    # Prepare configuration
    config = {
        "morphology_id": morph_id,
        "femur_length": femur_length,
        "tibia_length": tibia_length
    }
    
    print("\n" + "=" * 80)
    print(f"Generating Morphology - Femur Ratio: {femur_ratio:.3f}")
    print(f"Total Leg Length: {total_leg_length:.3f}m")
    print(f"Femur Length: {femur_length:.3f}m")
    print(f"Tibia Length: {tibia_length:.3f}m")
    print("=" * 80)
    
    try:
        # Create morphology variant
        morph = create_morphology_variant(**config)
        
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
        
        # Display properties
        props = morph.get_derived_properties()
        print("\nComputed Properties:")
        print(f"  Total Mass: {props['total_mass']:.4f} kg")
        print(f"  Total Leg Length: {props['total_leg_length']:.4f} m")
        print(f"  Femur-to-Limb Ratio: {props['femur_to_limb_ratio']:.4f}")
        print(f"  Balloon Mass Ratio: {props['balloon_mass_ratio']:.4f}")
        
        # Generate URDF
        print("\nGenerating URDF...")
        generator = BalluRobotGenerator(morph)
        urdf_path = generator.generate_urdf()
        print(f"✓ URDF generated: {urdf_path}")
        
        # Generate USD
        print("\nGenerating USD...")
        return_code, usd_path = generator.generate_usd(urdf_path)
        
        if return_code == 0 and os.path.exists(usd_path):
            print(f"✓ USD generated: {usd_path}")
            return urdf_path, usd_path, morph.morphology_id
        else:
            print(f"✗ USD conversion failed with return code: {return_code}")
            return None, None, None
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Train BALLU variants with different femur-to-leg length ratios')
    parser.add_argument('--ratios', type=float, nargs='+', required=True,
                      help='List of femur-to-total-leg-length ratios (0-1)')
    parser.add_argument('--total-leg-length', type=float, default=0.75,
                      help='Total leg length in meters (default: 0.75)')
    parser.add_argument('--max-iterations', type=int, default=1200,
                      help='Maximum training iterations per variant (default: 1200)')
    parser.add_argument("--device", type=str, default="cuda:0",
                      help='Device to use for training (default: cuda:0)')
    parser.add_argument("--task", type=str, default="Isc-Vel-BALLU-1-obstacle",
                      help='Task to use for training (default: Isc-Vel-BALLU-1-obstacle)')
    parser.add_argument("--seed", type=int, default=42,
                      help='Random seed for training (default: 42)')
    args = parser.parse_args()
    
    # Validate ratios
    invalid_ratios = [r for r in args.ratios if r <= 0 or r >= 1]
    if invalid_ratios:
        print(f"Error: Ratios must be between 0 and 1. Invalid values: {invalid_ratios}")
        sys.exit(1)
    
    results = []
    
    # Process each ratio
    for ratio in args.ratios:
        print(f"\n{'='*80}\nProcessing Femur Ratio: {ratio:.3f}\n{'='*80}")
        
        # Generate morphology
        urdf_path, usd_path, morph_id = generate_morphology_variant(
            femur_ratio=ratio,
            total_leg_length=args.total_leg_length
        )
        
        if morph_id is None:
            print(f"✗ Skipping ratio {ratio:.3f} due to generation failure")
            results.append({
                "ratio": ratio,
                "status": "failed",
                "error": "Morphology generation failed"
            })
            continue
        
        # Run training
        print(f"\nStarting training for ratio {ratio:.3f}")
        success, best_crclm_level = run_training_experiment(
            morph_id=morph_id,
            max_iterations=args.max_iterations,
            device=args.device,
            seed=args.seed,
            task=args.task
        )
        
        # Store results
        results.append({
            "ratio": ratio,
            "status": "completed" if success else "failed",
            "morphology_id": morph_id,
            "urdf_path": urdf_path,
            "usd_path": usd_path,
            "best_crclm_level": best_crclm_level if success else None
        })
        
        print(f"\nCompleted ratio {ratio:.3f} - Best Curriculum Level: {best_crclm_level:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    results_path = f"{project_dir}/logs/{timestamp}_femur_ratio_study.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            "total_leg_length": args.total_leg_length,
            "max_iterations": args.max_iterations,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for result in results:
        status = "✓" if result["status"] == "completed" else "✗"
        ratio = result["ratio"]
        level = result.get("best_crclm_level", "N/A")
        print(f"{status} Ratio {ratio:.3f}: {level}")
    print("-" * 40)

if __name__ == "__main__":
    main()
