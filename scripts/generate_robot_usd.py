#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import datetime

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

def generate_robot_usd(femur_to_limb_ratio: float, total_leg_length: float = 0.75, name: str = "auto_asset_FL") -> str:
    """
    Generate a robot USD file with specified femur-to-limb ratio.
    
    Args:
        femur_to_limb_ratio: Ratio of femur length to total leg length (0-1)
        total_leg_length: Total leg length in meters
    
    Returns:
        str: USD filename (without path) or empty string if failed
    """
    # Generate unique morphology ID
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    morph_id = f"{name}_{femur_to_limb_ratio:.2f}"
    
    try:
        # Create morphology variant using the direct parameter
        morph = create_morphology_variant(
            morphology_id=morph_id,
            femur_to_limb_ratio=femur_to_limb_ratio,
            total_leg_length=total_leg_length
        )
        
        # Validate morphology
        is_valid, errors = morph.validate()
        if not is_valid:
            print("✗ Validation FAILED")
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
            return ""
        
        # Generate URDF and USD
        generator = BalluRobotGenerator(morph)
        urdf_path = generator.generate_urdf()
        return_code, usd_path = generator.generate_usd(urdf_path)
        
        if return_code == 0 and os.path.exists(usd_path):
            return os.path.basename(usd_path)
        else:
            print(f"✗ USD conversion failed with return code: {return_code}")
            return ""
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='Generate BALLU robot USD with specified femur-to-limb ratio')
    parser.add_argument('--ratio', type=float, required=True,
                      help='Femur-to-total-leg-length ratio (0-1)')
    parser.add_argument('--total-leg-length', type=float, default=0.75,
                      help='Total leg length in meters (default: 0.75)')
    parser.add_argument('--name', type=str, default="auto_asset_FL",
                      help='Name of the robot')
    
    args = parser.parse_args()
    
    # Validate ratio
    if args.ratio <= 0 or args.ratio >= 1:
        print(f"Error: Ratio must be between 0 and 1. Got: {args.ratio}")
        sys.exit(1)
    
    # Generate the robot USD
    usd_filename = generate_robot_usd(
        femur_to_limb_ratio=args.ratio,
        total_leg_length=args.total_leg_length,
        name=args.name
    )
    
    if usd_filename:
        print("*" * 90)
        print("usd_filename: ", usd_filename)
        print("*" * 90)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
