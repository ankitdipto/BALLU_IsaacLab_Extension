# BALLU Morphology Configuration System

This directory contains the morphology parameter space definition and generation tools for the BALLU robot.

## 📁 Files Overview

### Core Modules

- **`ballu_morphology_config.py`** - Complete morphology parameter space definition
  - `BalluMorphology`: Main configuration class
  - `GeometryParams`: Geometric parameters (limb lengths, radii, etc.)
  - `MassParams`: Mass and inertial parameters
  - `JointParams`: Joint limits, damping, initial positions
  - `ContactParams`: Friction and contact properties
  - `MorphologyParameterRanges`: Valid parameter ranges for exploration
  - `create_morphology_variant()`: Convenience function for creating variants

- **`ballu_morphology.py`** - URDF modification utilities
  - `BalluMorphologyModifier`: Class for adjusting femur-to-limb ratios in URDF

- **`convert_urdf.py`** - URDF to USD conversion script
  - Converts URDF files to Isaac Sim USD format

- **`example_morphology_usage.py`** - Usage examples and demonstrations

## 🚀 Quick Start

### Create Default Morphology

```python
from ballu_morphology_config import BalluMorphology

# Create default morphology from original URDF
morph = BalluMorphology.default()

# Print summary
print(morph)

# Validate
is_valid, errors = morph.validate()
print(f"Valid: {is_valid}")

# Get derived properties
properties = morph.get_derived_properties()
print(f"Total leg length: {properties['total_leg_length']:.3f}m")
```

### Create Custom Morphology

```python
from ballu_morphology_config import BalluMorphology, GeometryParams, MassParams

morph = BalluMorphology(
    morphology_id="long_legs_v1",
    description="BALLU with longer legs for obstacles",
    geometry=GeometryParams(
        femur_length=0.45,
        tibia_length=0.45,
    ),
    mass=MassParams(
        balloon_mass=0.20,
    )
)

# Validate
is_valid, errors = morph.validate()
```

### Create Morphology Variants

```python
from ballu_morphology_config import create_morphology_variant

# Variant with specific femur-to-limb ratio
morph1 = create_morphology_variant(
    morphology_id="fl_ratio_0.40",
    femur_to_limb_ratio=0.40,  # 40% femur, 60% tibia
)

# Variant with different total leg length
morph2 = create_morphology_variant(
    morphology_id="long_legs_0.9m",
    total_leg_length=0.9,  # 0.9m total
)

# Variant with balloon modifications
morph3 = create_morphology_variant(
    morphology_id="big_balloon",
    balloon_radius=0.4,
    balloon_height=0.9,
    balloon_mass=0.25,
)
```

### Save/Load Morphologies

```python
# Save to JSON
morph.to_json("morphologies/my_morphology.json")

# Load from JSON
morph = BalluMorphology.from_json("morphologies/my_morphology.json")

# Convert to/from dictionary
morph_dict = morph.to_dict()
morph = BalluMorphology.from_dict(morph_dict)
```

### Work with Parameter Ranges

```python
from ballu_morphology_config import MorphologyParameterRanges

ranges = MorphologyParameterRanges()

# View parameter ranges (min, max, default)
print(ranges.femur_length)  # (0.2, 0.6, 0.36501)

# Sample random values
femur_len = ranges.sample_uniform("femur_length")
tibia_len = ranges.sample_uniform("tibia_length")

# Create morphology from range defaults
morph = ranges.get_default_morphology()
```

## 📊 Morphology Parameters

### Geometry Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `femur_length` | 0.365 | m | Upper leg length |
| `tibia_length` | 0.385 | m | Lower leg length (to foot) |
| `limb_radius` | 0.005 | m | Leg cylinder radius |
| `hip_width` | 0.116 | m | Distance between hip joints |
| `foot_radius` | 0.004 | m | Contact sphere radius |
| `pelvis_height` | 0.15 | m | Main body length |
| `balloon_radius` | 0.32 | m | Buoyancy cylinder radius |
| `balloon_height` | 0.70 | m | Buoyancy cylinder height |

### Mass Parameters

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `pelvis_mass` | 0.021 | kg | Main body mass |
| `femur_mass` | 0.009 | kg | Upper leg mass (per leg) |
| `tibia_mass` | 0.044 | kg | Lower leg mass (per leg) |
| `balloon_mass` | 0.159 | kg | Balloon mass (net buoyancy) |
| `motorarm_mass` | 0.0002 | kg | Motor linkage mass |

**Total robot mass**: ~0.30 kg

### Joint Parameters

| Joint | Lower Limit | Upper Limit | Default Init |
|-------|-------------|-------------|--------------|
| HIP | -90° | +90° | 1° |
| KNEE | 0° | 100° | 27.35° |
| NECK | -90° | +90° | 0° |
| MOTOR | 0° | 180° | 10° |

### Contact Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `foot_friction` | 0.9 | Ground contact friction coefficient |
| `leg_friction` | 0.5 | Leg-object contact friction |

## 🔍 Validation

The morphology configuration system includes comprehensive validation:

```python
is_valid, errors = morph.validate()

if not is_valid:
    for error in errors:
        print(f"❌ {error}")
```

**Validation checks:**
- ✓ All geometric parameters are positive
- ✓ Total leg length is realistic (0.1-2.0m)
- ✓ Femur-to-limb ratio is reasonable (0.1-0.9)
- ✓ All masses are positive
- ✓ Balloon mass ratio supports buoyancy (0.1-0.9)
- ✓ Joint limits are properly ordered
- ✓ Initial joint positions within limits
- ✓ Friction coefficients in valid range
- ✓ Initial pose has positive ground clearance

## 📈 Derived Properties

Access computed metrics:

```python
derived = morph.get_derived_properties()
```

**Available properties:**
- `total_mass`: Total robot mass (kg)
- `leg_mass`: Mass of one complete leg (kg)
- `total_leg_length`: Femur + tibia length (m)
- `femur_to_limb_ratio`: Femur / total leg length
- `balloon_mass_ratio`: Balloon mass / total mass
- `balloon_volume`: Balloon volume (m³)
- `leg_slenderness`: Leg length / radius ratio
- `hip_width_to_leg_ratio`: Hip width / leg length

## 🎯 Use Cases

### 1. Morphology Exploration

Generate a batch of morphologies with varying parameters:

```python
morphologies = []
for ratio in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    morph = create_morphology_variant(
        morphology_id=f"fl_ratio_{ratio:.2f}",
        femur_to_limb_ratio=ratio,
    )
    morphologies.append(morph)
```

### 2. Optimization Search Space

Use parameter ranges for optimization:

```python
ranges = MorphologyParameterRanges()

# Define search space
search_space = {
    'femur_length': ranges.femur_length,
    'tibia_length': ranges.tibia_length,
    'balloon_mass': ranges.balloon_mass,
}

# Sample candidates
for i in range(100):
    morph = create_morphology_variant(
        morphology_id=f"candidate_{i:03d}",
        femur_length=ranges.sample_uniform('femur_length'),
        tibia_length=ranges.sample_uniform('tibia_length'),
        balloon_mass=ranges.sample_uniform('balloon_mass'),
    )
    # Train and evaluate...
```

### 3. Task-Specific Morphologies

Design morphologies for specific tasks:

```python
# High obstacle clearance
obstacle_morph = create_morphology_variant(
    morphology_id="high_obstacles",
    total_leg_length=0.9,  # Longer legs
    femur_to_limb_ratio=0.55,  # More femur for hip clearance
    balloon_mass=0.22,  # More buoyancy support
)

# Speed optimization
speed_morph = create_morphology_variant(
    morphology_id="speed_optimized",
    total_leg_length=0.8,
    limb_radius=0.003,  # Lighter legs
    tibia_mass=0.03,
)
```

## 🔗 Integration with URDF Generation

### New: Complete URDF Generation (Stage 2) ✅

Generate complete URDFs from morphology configurations:

```python
from ballu_morphology_config import create_morphology_variant
from urdf_generator import BalluURDFGenerator

# Create morphology configuration
morph = create_morphology_variant(
    morphology_id="long_legs_0.9m",
    total_leg_length=0.9,
    balloon_mass=0.22,
)

# Generate complete URDF
generator = BalluURDFGenerator(morph)
urdf_path = generator.generate_urdf("output.urdf")

# Convert to USD using existing converter
# python convert_urdf.py output.urdf output.usd --merge-joints --headless
```

### Legacy: Femur-Limb Ratio Modifier

For backward compatibility, the original modifier is still available:

```python
from ballu_morphology import BalluMorphologyModifier

# Only supports FL ratio adjustment
modifier = BalluMorphologyModifier(original_urdf_path="path/to/original.urdf")
urdf_path = modifier.adjust_femur_to_limb_ratio(femur_ratio=0.45)
modifier.convert_to_usd()
```

## 📝 Example Scripts

### Morphology Configuration (Stage 1)

Run the morphology configuration examples:

```bash
cd ballu_isclb_extension/source/ballu_isaac_extension/ballu_isaac_extension/morphology
python example_morphology_usage.py
```

This demonstrates:
1. Creating default morphologies
2. Creating custom morphologies
3. Saving/loading from JSON
4. Creating variants
5. Working with parameter ranges
6. Batch generation

### URDF Generation (Stage 2)

Run the URDF generation examples:

```bash
python example_urdf_generation.py
```

This demonstrates:
1. Generate URDF from default morphology
2. Generate URDF from custom morphology
3. Batch URDF generation
4. Primitives vs mesh visuals
5. Inertia computation
6. URDF structure verification

## 🛠️ Implementation Status

### ✅ Completed:
1. **Stage 1: Morphology Configuration** - Complete parameter space definition
2. **Stage 2: URDF Generation** - Auto-generate URDFs with computed inertia

### 🔄 In Progress / Future Extensions:

1. **Morphology Database** - Track morphology performance across experiments
2. **Advanced Sampling** - Latin Hypercube, Sobol sequences, adaptive sampling
3. **Multi-objective Optimization** - Pareto frontier exploration
4. **Morphology Interpolation** - Smooth transitions between morphologies
5. **Constraint Satisfaction** - Complex kinematic and dynamic constraints
6. **Direct USD Generation** - Skip URDF, generate USD directly

## 📚 References

- Original URDF: `ballu_assets/old/urdf/urdf/original.urdf`
- BALLU Config: `ballu_assets/ballu_config.py`
- Project README: `../../../../../README.md`

