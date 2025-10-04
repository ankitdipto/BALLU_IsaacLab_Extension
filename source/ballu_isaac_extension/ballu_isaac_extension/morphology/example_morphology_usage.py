#!/usr/bin/env python3
"""
Example script demonstrating usage of the BALLU morphology configuration system.

This script shows how to:
1. Create default morphologies
2. Create custom morphologies
3. Validate morphologies
4. Save/load morphologies
5. Create morphology variants
6. Access parameter ranges
"""

from ballu_morphology_config import (
    BalluMorphology,
    GeometryParams,
    MassParams,
    MorphologyParameterRanges,
    create_morphology_variant,
)


def example_1_default_morphology():
    """Example 1: Create and inspect default morphology."""
    print("=" * 80)
    print("EXAMPLE 1: Default Morphology")
    print("=" * 80)
    
    # Create default morphology
    morph = BalluMorphology.default()
    
    # Print summary
    print(morph)
    
    # Validate
    is_valid, errors = morph.validate()
    print(f"Valid: {is_valid}")
    if not is_valid:
        print("Errors:", errors)
    
    # Get derived properties
    print("\nDerived Properties:")
    for key, value in morph.get_derived_properties().items():
        print(f"  {key}: {value:.4f}")
    
    print()


def example_2_custom_morphology():
    """Example 2: Create custom morphology with specific parameters."""
    print("=" * 80)
    print("EXAMPLE 2: Custom Morphology")
    print("=" * 80)
    
    # Create custom morphology
    morph = BalluMorphology(
        morphology_id="long_legs_v1",
        description="BALLU variant with longer legs for obstacle clearance",
        geometry=GeometryParams(
            femur_length=0.45,  # Longer femur
            tibia_length=0.45,  # Longer tibia
            balloon_radius=0.35,  # Slightly larger balloon for balance
        ),
        mass=MassParams(
            balloon_mass=0.20,  # More balloon mass for longer legs
        ),
    )
    
    print(morph)
    
    # Validate
    is_valid, errors = morph.validate()
    print(f"Valid: {is_valid}")
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    print()


def example_3_save_load_morphology():
    """Example 3: Save and load morphology from JSON."""
    print("=" * 80)
    print("EXAMPLE 3: Save/Load Morphology")
    print("=" * 80)
    
    # Create a morphology
    morph = BalluMorphology(
        morphology_id="test_save_load",
        description="Testing serialization",
    )
    
    # Save to JSON
    json_path = "/tmp/test_morphology.json"
    morph.to_json(json_path)
    print(f"Saved to: {json_path}")
    
    # Load from JSON
    loaded_morph = BalluMorphology.from_json(json_path)
    print(f"Loaded morphology: {loaded_morph.morphology_id}")
    print(f"Description: {loaded_morph.description}")
    
    # Verify they match
    assert morph.to_dict() == loaded_morph.to_dict()
    print("✓ Save/load successful - morphologies match!")
    
    print()


def example_4_create_variant():
    """Example 4: Create morphology variants using convenience function."""
    print("=" * 80)
    print("EXAMPLE 4: Create Morphology Variants")
    print("=" * 80)
    
    # Create variant with different femur-to-limb ratio
    morph1 = create_morphology_variant(
        morphology_id="fl_ratio_0.40",
        femur_to_limb_ratio=0.40,  # 40% femur, 60% tibia
    )
    print("Variant 1 (FL ratio 0.40):")
    print(f"  Femur: {morph1.geometry.femur_length:.4f}m")
    print(f"  Tibia: {morph1.geometry.tibia_length:.4f}m")
    print(f"  Ratio: {morph1.geometry.get_femur_to_limb_ratio():.2f}")
    
    # Create variant with different total leg length
    morph2 = create_morphology_variant(
        morphology_id="long_legs_0.9m",
        total_leg_length=0.9,  # 0.9m total leg length
    )
    print("\nVariant 2 (Total leg length 0.9m):")
    print(f"  Femur: {morph2.geometry.femur_length:.4f}m")
    print(f"  Tibia: {morph2.geometry.tibia_length:.4f}m")
    print(f"  Total: {morph2.geometry.get_total_leg_length():.4f}m")
    
    # Create variant with balloon size change
    morph3 = create_morphology_variant(
        morphology_id="big_balloon",
        balloon_radius=0.4,
        balloon_height=0.9,
        balloon_mass=0.25,
    )
    print("\nVariant 3 (Bigger balloon):")
    print(f"  Balloon radius: {morph3.geometry.balloon_radius:.3f}m")
    print(f"  Balloon height: {morph3.geometry.balloon_height:.3f}m")
    print(f"  Balloon mass: {morph3.mass.balloon_mass:.4f}kg")
    derived = morph3.get_derived_properties()
    print(f"  Balloon volume: {derived['balloon_volume']:.4f}m³")
    
    print()


def example_5_parameter_ranges():
    """Example 5: Work with parameter ranges for sampling."""
    print("=" * 80)
    print("EXAMPLE 5: Parameter Ranges")
    print("=" * 80)
    
    ranges = MorphologyParameterRanges()
    
    # Display some parameter ranges
    print("Parameter Ranges (min, max, default):")
    print(f"  Femur length: {ranges.femur_length}")
    print(f"  Tibia length: {ranges.tibia_length}")
    print(f"  Balloon radius: {ranges.balloon_radius}")
    print(f"  Balloon mass: {ranges.balloon_mass}")
    print(f"  Foot friction: {ranges.foot_friction}")
    
    # Sample random values
    print("\nRandom samples:")
    for i in range(3):
        femur_len = ranges.sample_uniform("femur_length")
        tibia_len = ranges.sample_uniform("tibia_length")
        print(f"  Sample {i+1}: femur={femur_len:.4f}m, tibia={tibia_len:.4f}m")
    
    # Create morphology from range defaults
    morph = ranges.get_default_morphology()
    print(f"\nMorphology from range defaults: {morph.morphology_id}")
    
    print()


def example_6_batch_morphologies():
    """Example 6: Generate batch of morphologies for exploration."""
    print("=" * 80)
    print("EXAMPLE 6: Batch Morphology Generation")
    print("=" * 80)
    
    # Generate morphologies with varying femur-to-limb ratios
    ratios = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    morphologies = []
    for ratio in ratios:
        morph = create_morphology_variant(
            morphology_id=f"fl_ratio_{ratio:.2f}",
            femur_to_limb_ratio=ratio,
        )
        morphologies.append(morph)
    
    print(f"Generated {len(morphologies)} morphologies:")
    for morph in morphologies:
        is_valid, _ = morph.validate()
        status = "✓" if is_valid else "✗"
        derived = morph.get_derived_properties()
        print(f"  {status} {morph.morphology_id}: "
              f"FL ratio={derived['femur_to_limb_ratio']:.2f}, "
              f"total_leg={derived['total_leg_length']:.4f}m")
    
    # Save all to JSON files
    output_dir = "/tmp/ballu_morphologies"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for morph in morphologies:
        json_path = os.path.join(output_dir, f"{morph.morphology_id}.json")
        morph.to_json(json_path)
    
    print(f"\nAll morphologies saved to: {output_dir}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "BALLU MORPHOLOGY CONFIGURATION EXAMPLES" + " " * 19 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    example_1_default_morphology()
    example_2_custom_morphology()
    example_3_save_load_morphology()
    example_4_create_variant()
    example_5_parameter_ranges()
    example_6_batch_morphologies()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

