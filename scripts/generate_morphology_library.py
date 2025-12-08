#!/usr/bin/env python3
"""
Generate a diverse library of BALLU robot morphologies for heterogeneous training.

This script uses Latin Hypercube Sampling (LHS) to generate a well-distributed
set of morphologies across the parameter space, ensuring good coverage for
universal policy pretraining.

Usage:
    python generate_morphology_library.py --num_morphologies 100 --output_dir morphologies/hetero_library
    python generate_morphology_library.py --num_morphologies 50 --sampling_strategy lhs --validate
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
ext_dir = os.path.join(project_dir, "source", "ballu_isaac_extension", "ballu_isaac_extension")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# --- Import Morphology Tools ---
try:
    from morphology import (
        BalluRobotGenerator, 
        create_morphology_variant,
        constants
    )
except Exception as exc:
    print(f"[ERROR] Failed to import morphology tools: {exc}")
    print(f"[HINT] Ensure PYTHONPATH includes: {ext_dir}")
    sys.exit(1)


# ==============================================================================
# PARAMETER RANGES - CUSTOMIZE HERE
# ==============================================================================
# Each parameter has (min, max, default) tuple
# To customize ranges, simply edit the values below
#
# Example: To generate only long-legged robots:
#   "femur_length": (0.40, 0.48, 0.44),  # Increase min from 0.30 to 0.40
#   "tibia_length": (0.38, 0.43, 0.40),  # Increase min from 0.30 to 0.38
# ==============================================================================

PARAMETER_RANGES = {
    "femur_length": (0.35, 0.50, 0.36501),      # Upper leg length (m)
    "tibia_length": (0.29, 0.45, 0.32000),      # Lower leg length (m)
    "hip_width": (0.08, 0.15, 0.11605),         # Hip joint spacing (m)
    "balloon_radius": (0.25, 0.40, 0.32000),    # Buoyancy cylinder radius (m)
    "balloon_height": (0.60, 0.85, 0.70000),    # Buoyancy cylinder height (m)
    "limb_radius": (0.004, 0.007, 0.00500),     # Leg cylinder radius (m)
    "foot_radius": (0.003, 0.006, 0.00400),     # Contact sphere radius (m)
}

# Parameters to vary during generation
# To exclude a parameter from variation, remove it from this list
PARAMETERS_TO_VARY = [
    "femur_length",
    "tibia_length",
    # "hip_width"
]

# ==============================================================================


def latin_hypercube_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1]^n_dims.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    np.random.seed(seed)
    
    # Create LHS samples
    samples = np.zeros((n_samples, n_dims))
    
    for dim in range(n_dims):
        # Divide [0, 1] into n_samples intervals
        intervals = np.arange(n_samples) / n_samples
        # Randomly sample within each interval
        samples[:, dim] = intervals + np.random.uniform(0, 1/n_samples, n_samples)
        # Shuffle to break correlation between dimensions
        np.random.shuffle(samples[:, dim])
    
    return samples


def sobol_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate Sobol sequence samples (requires scipy).
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions
        seed: Random seed (for scipy compatibility)
        
    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        samples = sampler.random(n_samples)
        return samples
    except ImportError:
        print("[WARNING] scipy not available, falling back to LHS")
        return latin_hypercube_sampling(n_samples, n_dims, seed)


def random_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate uniform random samples.
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions
        seed: Random seed
        
    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    np.random.seed(seed)
    return np.random.uniform(0, 1, (n_samples, n_dims))


def sample_morphology_parameters(
    n_samples: int,
    sampling_strategy: str = "lhs",
    seed: int = 42
) -> List[Dict[str, float]]:
    """
    Sample morphology parameters using specified strategy.
    
    Args:
        n_samples: Number of morphologies to generate
        sampling_strategy: 'lhs', 'sobol', or 'random'
        seed: Random seed for reproducibility
        
    Returns:
        List of parameter dictionaries
    """
    # Get parameter specs from hardcoded ranges
    param_specs = [
        (param_name, PARAMETER_RANGES[param_name])
        for param_name in PARAMETERS_TO_VARY
    ]
    
    param_names = [name for name, _ in param_specs]
    param_bounds = [bounds for _, bounds in param_specs]
    n_dims = len(param_names)
    
    # Generate samples in [0, 1]^n_dims
    if sampling_strategy == "lhs":
        unit_samples = latin_hypercube_sampling(n_samples, n_dims, seed)
    elif sampling_strategy == "sobol":
        unit_samples = sobol_sampling(n_samples, n_dims, seed)
    elif sampling_strategy == "random":
        unit_samples = random_sampling(n_samples, n_dims, seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Scale samples to parameter bounds
    sampled_params = []
    for i in range(n_samples):
        params = {}
        for j, (param_name, (min_val, max_val, _)) in enumerate(zip(param_names, param_bounds)):
            # Scale from [0, 1] to [min_val, max_val]
            params[param_name] = min_val + unit_samples[i, j] * (max_val - min_val)
        sampled_params.append(params)
    
    return sampled_params


def generate_morphology_library(
    num_morphologies: int,
    output_dir: str,
    sampling_strategy: str = "lhs",
    seed: int = 42,
    validate: bool = True,
    force: bool = False,
    timestamp: str = None
) -> Tuple[List[str], List[Dict], int]:
    """
    Generate a library of diverse BALLU morphologies.
    
    Args:
        num_morphologies: Number of morphologies to generate
        output_dir: Directory to save generated morphologies
        sampling_strategy: Sampling strategy ('lhs', 'sobol', 'random')
        seed: Random seed for reproducibility
        validate: Whether to validate morphologies before generating USD
        force: Overwrite existing morphologies
        
    Returns:
        Tuple of (successful_morphology_ids, metadata_list, num_failed)
    """
    print(f"\n{'='*80}")
    print(f"BALLU MORPHOLOGY LIBRARY GENERATION")
    print(f"{'='*80}")
    print(f"Target: {num_morphologies} morphologies")
    print(f"Strategy: {sampling_strategy.upper()}")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample parameters
    print(f"[1/4] Sampling {num_morphologies} parameter sets using {sampling_strategy.upper()}...")
    sampled_params = sample_morphology_parameters(
        num_morphologies,
        sampling_strategy,
        seed
    )
    print(f"✓ Generated {len(sampled_params)} parameter sets\n")
    
    # Generate morphologies
    print(f"[2/4] Generating morphologies...")
    successful_morphologies = []
    metadata_list = []
    num_failed = 0
    
    for i, params in enumerate(sampled_params):
        morph_id = f"hetero_{i:04d}_fl{params['femur_length']:.3f}_tl{params['tibia_length']:.3f}"
        print(f"  [{i+1}/{num_morphologies}] Generating {morph_id}...", end=" ")
        
        try:
            # Create morphology
            morph = create_morphology_variant(
                morphology_id=morph_id,
                **params
            )
            
            # Validate if requested
            if validate:
                is_valid, errors = morph.validate()
                if not is_valid:
                    print(f"✗ INVALID")
                    print(f"      Errors: {errors}")
                    num_failed += 1
                    continue
            
            # Generate URDF and USD
            # Set output directories to be within the library directory
            generator = BalluRobotGenerator(
                morph,
                usd_output_dir=str(output_path),
                urdf_output_dir=os.path.join(constants.BALLU_ASSETS_DIR, "old", "urdf", "urdf", "morphologies", constants.NEXT_LAB_DATE, timestamp)
            )
            urdf_path = generator.generate_urdf()
            return_code, usd_path = generator.generate_usd(urdf_path)
            
            if return_code != 0 or not os.path.exists(usd_path):
                print(f"✗ USD FAILED (code: {return_code})")
                num_failed += 1
                continue
            
            # Store metadata
            metadata = {
                "morphology_id": morph_id,
                "index": i,
                "usd_path": str(usd_path),
                "urdf_path": str(urdf_path),
                "parameters": params,
                "derived_properties": morph.get_derived_properties(),
                "generated_at": datetime.now().isoformat(),
                "sampling_strategy": sampling_strategy,
                "seed": seed
            }
            
            successful_morphologies.append(morph_id)
            metadata_list.append(metadata)
            
            print(f"✓")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            num_failed += 1
            continue
    
    print(f"\n✓ Generated {len(successful_morphologies)}/{num_morphologies} morphologies")
    print(f"  Failed: {num_failed}\n")
    
    # Save registry
    print(f"[3/4] Saving morphology registry...")
    registry_path = output_path / "morphology_registry.json"
    registry = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "num_morphologies": len(successful_morphologies),
        "sampling_strategy": sampling_strategy,
        "seed": seed,
        "morphologies": metadata_list
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"✓ Registry saved: {registry_path}\n")
    
    # Save summary statistics
    print(f"[4/4] Computing summary statistics...")
    if metadata_list:
        summary = compute_summary_statistics(metadata_list)
        summary_path = output_path / "morphology_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}\n")
        
        # Print summary
        print(f"{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        for param, stats in summary["parameter_statistics"].items():
            print(f"{param:20s}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
        print(f"{'='*80}\n")
    
    return successful_morphologies, metadata_list, num_failed


def compute_summary_statistics(metadata_list: List[Dict]) -> Dict:
    """Compute summary statistics for generated morphologies."""
    # Extract all parameters
    all_params = {}
    for metadata in metadata_list:
        for param, value in metadata["parameters"].items():
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(value)
    
    # Compute statistics
    param_stats = {}
    for param, values in all_params.items():
        param_stats[param] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values))
        }
    
    # Extract derived properties
    all_derived = {}
    for metadata in metadata_list:
        for prop, value in metadata["derived_properties"].items():
            if prop not in all_derived:
                all_derived[prop] = []
            all_derived[prop].append(value)
    
    derived_stats = {}
    for prop, values in all_derived.items():
        derived_stats[prop] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values))
        }
    
    return {
        "num_morphologies": len(metadata_list),
        "parameter_statistics": param_stats,
        "derived_property_statistics": derived_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate diverse library of BALLU morphologies for heterogeneous training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 morphologies using Latin Hypercube Sampling
  python generate_morphology_library.py --num_morphologies 100
  
  # Generate 50 morphologies using Sobol sequence
  python generate_morphology_library.py --num_morphologies 50 --sampling_strategy sobol
  
  # Generate without validation (faster but risky)
  python generate_morphology_library.py --num_morphologies 100 --no-validate
        """
    )
    
    parser.add_argument(
        '--num_morphologies', 
        type=int, 
        default=100,
        help='Number of morphologies to generate (default: 100)'
    )
    
    parser.add_argument(
        '--output_foldername',
        type=str,
        default=None,
        help='Output foldername (default: hetero_library_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--sampling_strategy',
        type=str,
        choices=['lhs', 'sobol', 'random'],
        default='lhs',
        help='Sampling strategy for parameter space (default: lhs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate morphologies before generating USD (default: True)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_false',
        dest='validate',
        help='Skip validation (faster but may generate invalid morphologies)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Overwrite existing morphologies (default: False)'
    )
    
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

    # Set default output directory if not specified
    if args.output_foldername is None:
        args.output_foldername = f"hetero_library_{timestamp}"

    assets_dir = os.path.join(ext_dir, "ballu_assets", "robots", "morphologies")
    OUTPUT_DIR = os.path.join(assets_dir, args.output_foldername)
    
    # Generate library
    try:
        successful_ids, metadata, num_failed = generate_morphology_library(
            num_morphologies=args.num_morphologies,
            output_dir=OUTPUT_DIR,
            sampling_strategy=args.sampling_strategy,
            seed=args.seed,
            validate=args.validate,
            force=args.force,
            timestamp=timestamp
        )
        
        print(f"{'='*80}")
        print(f"✓ GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Success: {len(successful_ids)}/{args.num_morphologies}")
        print(f"Failed: {num_failed}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"{'='*80}\n")
        
        if num_failed > 0:
            sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

