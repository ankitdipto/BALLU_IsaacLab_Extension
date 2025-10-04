#!/usr/bin/env python3
"""
Example script demonstrating URDF generation from morphology configurations.

This script shows how to:
1. Create morphology configurations
2. Generate URDFs from configurations
3. Generate URDFs for multiple morphology variants
4. Compare generated URDFs
"""

import os
from pathlib import Path

from ballu_morphology_config import (
    BalluMorphology,
    create_morphology_variant,
)
from urdf_generator import BalluURDFGenerator

def example_1_generate_default():
    """Example 1: Generate URDF from default morphology."""
    print("=" * 80)
    print("EXAMPLE 1: Generate URDF from Default Morphology")
    print("=" * 80)
    
    # Create default morphology
    morph = BalluMorphology.default()
    
    # Create generator
    generator = BalluURDFGenerator(morph, use_visual_meshes=True)
    
    # Generate URDF
    output_path = "ballu_urdfs/default.urdf"
    urdf_path = generator.generate_urdf(output_path)
    
    print(f"\n✓ URDF generated successfully!")
    print(f"  Location: {urdf_path}")
    print(f"  File size: {os.path.getsize(urdf_path) / 1024:.1f} KB")
    print()



def example_2_generate_custom():
    """Example 2: Generate URDF from custom morphology."""
    print("=" * 80)
    print("EXAMPLE 2: Generate URDF from Custom Morphology")
    print("=" * 80)
    
    # Create custom morphology with longer legs
    morph = create_morphology_variant(
        morphology_id="long_legs_0.9m",
        total_leg_length=0.9,
        balloon_mass=0.22,
    )
    
    print(f"Morphology: {morph.morphology_id}")
    print(f"  Leg length: {morph.geometry.get_total_leg_length():.4f}m")
    print(f"  Balloon mass: {morph.mass.balloon_mass:.4f}kg")
    
    # Generate URDF (without visual meshes for demonstration)
    generator = BalluURDFGenerator(morph, use_visual_meshes=True)
    
    output_path = "ballu_urdfs/long_legs.urdf"
    urdf_path = generator.generate_urdf(output_path)
    
    print(f"\n✓ URDF generated successfully!")
    print()



def example_3_batch_generation():
    """Example 3: Generate URDFs for multiple morphology variants."""
    print("=" * 80)
    print("EXAMPLE 3: Batch URDF Generation")
    print("=" * 80)
    
    # Define morphology variants
    variants = [
        {"id": "fl_0.35", "femur_to_limb_ratio": 0.35},
        {"id": "fl_0.40", "femur_to_limb_ratio": 0.40},
        {"id": "fl_0.45", "femur_to_limb_ratio": 0.45},
        {"id": "fl_0.50", "femur_to_limb_ratio": 0.50},
        {"id": "fl_0.55", "femur_to_limb_ratio": 0.55},
    ]
    
    output_dir = "ballu_urdfs/batch"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {len(variants)} URDFs...")
    print()
    
    for variant in variants:
        # Create morphology
        morph = create_morphology_variant(
            morphology_id=variant["id"],
            femur_to_limb_ratio=variant["femur_to_limb_ratio"],
        )
        
        # Validate
        is_valid, errors = morph.validate()
        if not is_valid:
            print(f"  ✗ {variant['id']}: INVALID")
            for error in errors:
                print(f"      - {error}")
            continue
        
        # Generate URDF
        generator = BalluURDFGenerator(morph)
        output_path = os.path.join(output_dir, f"{variant['id']}.urdf")
        urdf_path = generator.generate_urdf(output_path)
        
        # Also save morphology config
        morph.to_json(os.path.join(output_dir, f"{variant['id']}_config.json"))
    
    print()
    print(f"✓ All URDFs generated in: {output_dir}")
    print()



def example_4_compare_primitives_vs_meshes():
    """Example 4: Generate URDFs with primitives vs meshes."""
    print("=" * 80)
    print("EXAMPLE 4: Primitives vs Mesh Visuals")
    print("=" * 80)
    
    morph = BalluMorphology.default()
    output_dir = "ballu_urdfs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # With meshes
    print("Generating URDF with visual meshes...")
    gen_meshes = BalluURDFGenerator(morph, use_visual_meshes=True)
    path_meshes = gen_meshes.generate_urdf(
        os.path.join(output_dir, "default_with_meshes.urdf")
    )
    
    # With primitives
    print("\nGenerating URDF with primitive visuals...")
    gen_primitives = BalluURDFGenerator(morph, use_visual_meshes=False)
    path_primitives = gen_primitives.generate_urdf(
        os.path.join(output_dir, "default_with_primitives.urdf")
    )
    
    # Compare file sizes
    size_meshes = os.path.getsize(path_meshes)
    size_primitives = os.path.getsize(path_primitives)
    
    print()
    print("Comparison:")
    print(f"  With meshes:     {size_meshes / 1024:.1f} KB")
    print(f"  With primitives: {size_primitives / 1024:.1f} KB")
    print(f"  Difference:      {(size_primitives - size_meshes) / 1024:+.1f} KB")
    print()



def example_5_inertia_computation():
    """Example 5: Demonstrate inertia computation."""
    print("=" * 80)
    print("EXAMPLE 5: Inertia Computation")
    print("=" * 80)
    
    from urdf_generator import InertiaCalculator
    
    calc = InertiaCalculator()
    
    # Compute femur inertia
    femur_mass = 0.00944
    femur_radius = 0.005
    femur_length = 0.36501
    
    femur_inertia = calc.cylinder_y_axis(femur_mass, femur_radius, femur_length)
    
    print("Femur Inertia Properties:")
    print(f"  Mass: {femur_inertia.mass:.6f} kg")
    print(f"  COM:  ({femur_inertia.com_x:.6f}, {femur_inertia.com_y:.6f}, {femur_inertia.com_z:.6f}) m")
    print(f"  Ixx:  {femur_inertia.ixx:.6e} kg⋅m²")
    print(f"  Iyy:  {femur_inertia.iyy:.6e} kg⋅m²")
    print(f"  Izz:  {femur_inertia.izz:.6e} kg⋅m²")
    
    # Compute foot (sphere) inertia
    foot_mass = 0.005  # Approximate
    foot_radius = 0.004
    
    foot_inertia = calc.sphere(foot_mass, foot_radius)
    
    print("\nFoot (Sphere) Inertia Properties:")
    print(f"  Mass: {foot_inertia.mass:.6f} kg")
    print(f"  I:    {foot_inertia.ixx:.6e} kg⋅m² (all axes)")
    
    # Compute balloon inertia
    balloon_mass = 0.15898
    balloon_radius = 0.32
    balloon_height = 0.7
    
    balloon_inertia = calc.cylinder_y_axis(balloon_mass, balloon_radius, balloon_height, com_y=-0.38)
    
    print("\nBalloon Inertia Properties:")
    print(f"  Mass: {balloon_inertia.mass:.6f} kg")
    print(f"  COM:  ({balloon_inertia.com_x:.6f}, {balloon_inertia.com_y:.6f}, {balloon_inertia.com_z:.6f}) m")
    print(f"  Ixx:  {balloon_inertia.ixx:.6e} kg⋅m²")
    print(f"  Iyy:  {balloon_inertia.iyy:.6e} kg⋅m²")
    print(f"  Izz:  {balloon_inertia.izz:.6e} kg⋅m²")
    print()



def example_6_verify_urdf():
    """Example 6: Basic URDF verification."""
    print("=" * 80)
    print("EXAMPLE 6: URDF Verification")
    print("=" * 80)
    
    # Generate URDF
    morph = BalluMorphology.default()
    generator = BalluURDFGenerator(morph)
    urdf_path = generator.generate_urdf("ballu_urdfs/verify.urdf")
    
    # Parse and verify structure
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Count elements
    links = root.findall(".//link")
    joints = root.findall(".//joint")
    materials = root.findall(".//material")
    
    print("\nURDF Structure:")
    print(f"  Links:     {len(links)}")
    print(f"  Joints:    {len(joints)}")
    print(f"  Materials: {len(materials)}")
    
    print("\nLinks:")
    for link in links:
        name = link.get("name")
        has_inertial = link.find("inertial") is not None
        has_visual = link.find("visual") is not None
        has_collision = link.find("collision") is not None
        collisions = len(link.findall("collision"))
        
        status = []
        if has_inertial:
            status.append("I")
        if has_visual:
            status.append("V")
        if has_collision:
            status.append(f"C×{collisions}")
        
        status_str = ",".join(status) if status else "empty"
        print(f"    {name:20s} [{status_str}]")
    
    print("\nJoints:")
    for joint in joints:
        name = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        print(f"    {name:15s} ({jtype:8s}): {parent:15s} → {child}")
    
    print(f"\n✓ URDF structure verified!")
    print()



def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "BALLU URDF GENERATION EXAMPLES" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    example_1_generate_default()
    example_2_generate_custom()
    example_3_batch_generation()
    example_4_compare_primitives_vs_meshes()
    example_5_inertia_computation()
    example_6_verify_urdf()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print()
    print("Generated URDFs are in: ballu_urdfs/")
    print()


if __name__ == "__main__":
    main()

