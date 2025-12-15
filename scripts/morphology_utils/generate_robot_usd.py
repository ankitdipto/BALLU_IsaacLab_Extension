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

def generate_robot_usd(femur_length: float, tibia_length: float, name: str = "auto_asset") -> str:
    """
    Generate a robot USD file with specified femur and tibia lengths.
    
    Args:
        femur_length: Femur length in meters
        tibia_length: Tibia length in meters
        name: Name of the robot
    
    Returns:
        str: USD filename (without path) or empty string if failed
    """
    # Generate unique morphology ID
    timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
    morph_id = f"{name}_fl{femur_length:.2f}_tl{tibia_length:.2f}"
    
    try:
        # Create morphology variant using the direct parameter
        morph = create_morphology_variant(
            morphology_id=morph_id,
            femur_length=femur_length,
            tibia_length=tibia_length,
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
    parser.add_argument('--femur_length', type=float, required=True,
                      help='Femur length in meters')
    parser.add_argument("--tibia_length", type=float, required=True,
                      help='Tibia length in meters')
    parser.add_argument('--name', type=str, default="auto_asset",
                      help='Name of the robot')
    args = parser.parse_args()
    
    # Generate the robot USD
    usd_filename = generate_robot_usd(
        femur_length=args.femur_length,
        tibia_length=args.tibia_length,
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
