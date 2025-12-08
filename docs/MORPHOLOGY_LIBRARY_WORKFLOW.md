# Dynamic Morphology Library Workflow

This document describes the workflow for generating and using large-scale morphology libraries (100+ robots) for heterogeneous BALLU training.

## 🎯 Overview

The dynamic morphology library system enables you to:
- Generate 100+ diverse BALLU morphologies using Latin Hypercube Sampling
- Dynamically load morphologies at runtime (no hardcoding)
- Track morphology metadata and parameters
- Manage libraries with easy-to-use utilities
- Scale your universal policy pretraining to hundreds of morphologies

## 📁 System Components

### 1. **Morphology Generator** (`generate_morphology_library.py`)
Generates diverse morphologies using LHS, Sobol, or random sampling.

### 2. **Dynamic Loader** (`morphology_loader.py`)
Loads morphologies from directories at runtime.

### 3. **Config Integration** (`ballu_config.py`)
Provides `get_ballu_hetero_cfg_dynamic()` function for dynamic configuration.

### 4. **Environment Config** (`single_obstacle_hetero_dynamic_env_cfg.py`)
New environment configuration that uses dynamic loading.

### 5. **Management Utilities** (`manage_morphology_library.py`)
CLI tool for managing morphology libraries.

## 🚀 Quick Start

### Step 1: Generate a Morphology Library

Generate 100 diverse morphologies using Latin Hypercube Sampling:

```bash
cd ballu_isclb_extension/scripts
python generate_morphology_library.py --num_morphologies 100 --sampling_strategy lhs
```

This creates a library at:
```
ballu_assets/robots/morphologies/hetero_library_TIMESTAMP/
├── morphology_registry.json       # Metadata for all morphologies
├── morphology_summary.json        # Statistical summary
├── hetero_0000/
│   ├── hetero_0000.usd
│   └── hetero_0000.urdf
├── hetero_0001/
│   ├── hetero_0001.usd
│   └── hetero_0001.urdf
└── ...
```

**Options:**
```bash
# Generate 50 morphologies using Sobol sequence
python generate_morphology_library.py --num_morphologies 50 --sampling_strategy sobol

# Generate with custom seed
python generate_morphology_library.py --num_morphologies 100 --seed 123

# Generate to custom directory
python generate_morphology_library.py --num_morphologies 100 --output_dir /path/to/my_library

# Skip validation (faster but risky)
python generate_morphology_library.py --num_morphologies 100 --no-validate
```

### Step 2: Set Library Path (Optional)

If you generated the library to a custom location, set the environment variable:

```bash
export BALLU_MORPHOLOGY_LIBRARY_PATH=/path/to/your/library
```

Or rename your library directory to `hetero_library` in the default location:
```bash
mv ballu_assets/robots/morphologies/hetero_library_TIMESTAMP \
   ballu_assets/robots/morphologies/hetero_library
```

### Step 3: Train with Dynamic Morphologies

Use the new dynamic environment configuration:

```bash
python scripts/rsl_rl/train.py \
    --task Isc-Vel-BALLU-1-obstacle-hetero-dynamic \
    --num_envs 4096 \
    --max_iterations 1600 \
    --headless
```

**Customize morphology loading:**

```python
# In your environment config or script
from ballu_isaac_extension.ballu_assets.ballu_config import get_ballu_hetero_cfg_dynamic

# Load all morphologies
robot_cfg = get_ballu_hetero_cfg_dynamic()

# Load first 50 morphologies
robot_cfg = get_ballu_hetero_cfg_dynamic(max_morphologies=50)

# Load from custom library
robot_cfg = get_ballu_hetero_cfg_dynamic(library_name="my_custom_library")

# Customize actuator parameters
robot_cfg = get_ballu_hetero_cfg_dynamic(
    spring_coeff=0.01,
    spring_damping=0.02,
    pd_p=1.5,
    pd_d=0.1
)
```

## 🛠️ Managing Libraries

### List Morphologies

```bash
python manage_morphology_library.py list --library hetero_library
```

Output:
```
================================================================================
MORPHOLOGY LIBRARY: hetero_library
Path: /path/to/ballu_assets/robots/morphologies/hetero_library
================================================================================

Found 100 morphologies:

Index    Morphology ID             USD Path
-------- ------------------------- --------------------------------------------------
0        hetero_0000               .../hetero_library/hetero_0000/hetero_0000.usd
1        hetero_0001               .../hetero_library/hetero_0001/hetero_0001.usd
...
```

### Show Library Info

```bash
python manage_morphology_library.py info --library hetero_library
```

Output:
```
================================================================================
MORPHOLOGY LIBRARY INFO
================================================================================
Library Name:      hetero_library
Library Path:      /path/to/library
Registry Path:     /path/to/library/morphology_registry.json
Version:           1.0
Generated:         2025-11-16T10:30:00
Num Morphologies:  100
Sampling Strategy: lhs
Seed:              42
================================================================================
```

### Validate Library

Check that all USD files exist and are valid:

```bash
python manage_morphology_library.py validate --library hetero_library
```

### Show Statistics

Display parameter distributions:

```bash
python manage_morphology_library.py stats --library hetero_library
```

Output:
```
================================================================================
MORPHOLOGY LIBRARY STATISTICS: hetero_library
================================================================================

Total Morphologies: 100

Parameter Statistics:
--------------------------------------------------------------------------------

femur_length:
  Min:    0.300000
  Max:    0.480000
  Mean:   0.390123
  Std:    0.052341
  Median: 0.389456

tibia_length:
  Min:    0.300000
  Max:    0.430000
  Mean:   0.365234
  ...
```

### Export Metadata

```bash
python manage_morphology_library.py export \
    --library hetero_library \
    --output my_library_metadata.json
```

### Remove Morphology

```bash
python manage_morphology_library.py remove \
    --library hetero_library \
    --morphology_id hetero_0042
```

## 📊 Morphology Parameter Space

The generator samples from these parameter ranges:

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| `femur_length` | 0.30 | 0.48 | Upper leg length (m) |
| `tibia_length` | 0.30 | 0.43 | Lower leg length (m) |
| `hip_width` | 0.08 | 0.14 | Hip joint spacing (m) |
| `balloon_radius` | 0.25 | 0.40 | Buoyancy cylinder radius (m) |
| `balloon_height` | 0.60 | 0.85 | Buoyancy cylinder height (m) |
| `limb_radius` | 0.004 | 0.007 | Leg cylinder radius (m) |
| `foot_radius` | 0.003 | 0.006 | Contact sphere radius (m) |

**Derived properties** (automatically computed):
- Total leg length
- Femur-to-limb ratio
- Total mass
- Balloon volume
- Center of mass

## 🔬 Advanced Usage

### Custom Sampling Strategy

Create your own sampling function:

```python
from generate_morphology_library import generate_morphology_library

# Use Sobol sequence for better space-filling
generate_morphology_library(
    num_morphologies=200,
    output_dir="my_library",
    sampling_strategy="sobol",
    seed=42
)
```

### Filter Morphologies

Load only morphologies matching certain criteria:

```python
from ballu_assets.morphology_loader import (
    load_morphology_library,
    filter_by_leg_length,
    filter_by_parameter
)

# Load morphologies with leg length between 0.7 and 0.8 meters
morphologies = load_morphology_library(
    "hetero_library",
    filter_fn=filter_by_leg_length(0.7, 0.8)
)

# Load morphologies with large balloons
morphologies = load_morphology_library(
    "hetero_library",
    filter_fn=filter_by_parameter("balloon_radius", 0.35, 0.40)
)
```

### Programmatic Library Generation

```python
from generate_morphology_library import generate_morphology_library

successful_ids, metadata, num_failed = generate_morphology_library(
    num_morphologies=100,
    output_dir="my_library",
    sampling_strategy="lhs",
    seed=42,
    validate=True
)

print(f"Generated {len(successful_ids)} morphologies")
print(f"Failed: {num_failed}")
```

### Custom Morphology Creation

Add a single custom morphology to a library:

```python
from morphology import create_morphology_variant, BalluRobotGenerator
import shutil

# Create custom morphology
morph = create_morphology_variant(
    morphology_id="custom_long_legs",
    femur_length=0.45,
    tibia_length=0.40,
    balloon_radius=0.35
)

# Generate USD
generator = BalluRobotGenerator(morph)
urdf_path = generator.generate_urdf()
return_code, usd_path = generator.generate_usd(urdf_path)

# Copy to library
library_path = "ballu_assets/robots/morphologies/hetero_library"
morph_dir = os.path.join(library_path, "custom_long_legs")
shutil.copytree(os.path.dirname(usd_path), morph_dir)

# Update registry manually or regenerate
```

## 🎓 Research Workflow Integration

### Universal Policy Pretraining

1. **Generate diverse library** (100+ morphologies)
2. **Pretrain universal policy** on all morphologies
3. **Save pretrained checkpoint**

```bash
# Generate library
python generate_morphology_library.py --num_morphologies 100

# Pretrain
python scripts/rsl_rl/train.py \
    --task Isc-Vel-BALLU-1-obstacle-hetero-dynamic \
    --num_envs 4096 \
    --max_iterations 5000 \
    --run_name universal_pretrain
```

### Morphology Optimization with Pretraining

Modify `explore_morphology.py` to use pretrained policy:

```python
def run_training_experiment_with_pretrain(
    morph_id: str,
    pretrain_checkpoint: str,
    max_iterations: int = 200  # Much fewer iterations!
):
    """Finetune pretrained policy on new morphology."""
    cmd = [
        sys.executable,
        "scripts/rsl_rl/train.py",
        "--task", "Isc-Vel-BALLU-1-obstacle",
        "--num_envs", "4096",
        "--max_iterations", str(max_iterations),
        "--resume",  # Resume from pretrained checkpoint
        "--load_run", pretrain_checkpoint,
        "--run_name", morph_id,
        "--headless"
    ]
    # ... rest of training code
```

### Curriculum Learning

Start with easier morphologies, gradually increase diversity:

```python
# Stage 1: Train on similar morphologies
robot_cfg = get_ballu_hetero_cfg_dynamic(
    max_morphologies=20,
    filter_fn=filter_by_leg_length(0.70, 0.75)
)

# Stage 2: Expand to more diverse set
robot_cfg = get_ballu_hetero_cfg_dynamic(
    max_morphologies=50
)

# Stage 3: Full diversity
robot_cfg = get_ballu_hetero_cfg_dynamic()  # All morphologies
```

## 🐛 Troubleshooting

### Library Not Found

**Error:** `Library 'hetero_library' not found`

**Solution:**
1. Check library exists: `ls ballu_assets/robots/morphologies/`
2. Set environment variable: `export BALLU_MORPHOLOGY_LIBRARY_PATH=/path/to/library`
3. Or rename library to `hetero_library`

### USD Generation Failed

**Error:** `USD conversion failed`

**Solution:**
1. Ensure Isaac Sim is properly installed
2. Check URDF validity: `python manage_morphology_library.py validate`
3. Try regenerating with validation: `--validate` flag

### Memory Issues with Large Libraries

If loading 100+ morphologies causes memory issues:

```python
# Load in batches
robot_cfg = get_ballu_hetero_cfg_dynamic(max_morphologies=50)
```

### Slow Generation

Speed up generation by skipping validation (use cautiously):

```bash
python generate_morphology_library.py \
    --num_morphologies 100 \
    --no-validate
```

## 📚 API Reference

### `generate_morphology_library.py`

```python
generate_morphology_library(
    num_morphologies: int,
    output_dir: str,
    sampling_strategy: str = "lhs",  # 'lhs', 'sobol', 'random'
    seed: int = 42,
    validate: bool = True,
    force: bool = False
) -> Tuple[List[str], List[Dict], int]
```

### `morphology_loader.py`

```python
load_morphology_library(
    directory: str,
    max_morphologies: Optional[int] = None,
    use_registry: bool = True,
    filter_fn: Optional[callable] = None
) -> List[Dict]

create_hetero_config(
    morphologies: List[Dict],
    init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.9),
    spring_coeff: float = 0.0807,
    spring_damping: float = 1.0e-2,
    pd_p: float = 1.0,
    pd_d: float = 0.08,
    random_choice: bool = True
) -> ArticulationCfg

load_default_hetero_config(
    library_name: str = "hetero_library",
    max_morphologies: Optional[int] = None,
    **config_kwargs
) -> ArticulationCfg
```

### `ballu_config.py`

```python
get_ballu_hetero_cfg_dynamic(
    library_name: str = "hetero_library",
    max_morphologies: int = None,
    spring_coeff: float = 0.0807,
    spring_damping: float = 1.0e-2,
    pd_p: float = 1.0,
    pd_d: float = 0.08,
    **kwargs
) -> ArticulationCfg

has_dynamic_morphology_support() -> bool
```

## 🎉 Benefits

1. **Scalability**: Easily scale from 12 to 100+ morphologies
2. **Reproducibility**: LHS/Sobol sampling ensures consistent parameter coverage
3. **Flexibility**: Load subsets, filter by criteria, customize at runtime
4. **Maintainability**: No hardcoded paths, easy to manage
5. **Research-Ready**: Perfect for universal policy pretraining and morphology optimization

## 📝 Next Steps

1. Generate your first library: `python generate_morphology_library.py --num_morphologies 100`
2. Validate it: `python manage_morphology_library.py validate`
3. Train with it: `python train.py --task Isc-Vel-BALLU-1-obstacle-hetero-dynamic`
4. Integrate into your morphology optimization loop

Happy researching! 🚀

