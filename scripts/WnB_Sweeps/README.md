# BALLU Morphology Optimization with W&B Sweeps

This directory contains tools for running Weights & Biases hyperparameter sweeps to optimize BALLU robot morphologies and buoyancy mass parameters. **Optimized for single GPU sequential execution.**

## 🚀 Quick Start

### Option 1: CLI Approach (Standard)
```bash
# Create sweep
python launch_sweep.py --project ballu-morphology-sweep

# Start agents sequentially
wandb agent entity/project/sweep_id
```

### Option 2: Programmatic Approach (Advanced)
```bash
# Create and run sweep in one command
python launch_sweep.py --start-agent --programmatic --count 5
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `sweep_config.yaml` | Sweep configuration (parameters, method, metric) |
| `launch_sweep.py` | Create sweeps (supports both CLI and programmatic) |
| `sweep_agent.py` | Individual experiment runner (called by W&B) |
| `sweep_utils.py` | Utility functions for data extraction |

## 🔄 Approach Comparison

### CLI Approach
**When to use**: Standard W&B workflows, simple execution

**Pros**:
- ✅ Standard W&B pattern, well-documented
- ✅ Simple setup and execution
- ✅ Optimized by W&B team
- ✅ Robust agent management by W&B
- ✅ Less Python overhead

**Cons**:
- ❌ Limited control over agent lifecycle
- ❌ Harder to integrate with custom workflows
- ❌ Less flexibility for custom monitoring

### Programmatic Approach
**When to use**: Custom workflows, tight integration, advanced monitoring, debugging

**Pros**:
- ✅ Full control over execution
- ✅ Custom retry logic and error handling
- ✅ Integrated monitoring and logging
- ✅ Easy debugging and development
- ✅ Better integration with existing code
- ✅ Sequential execution optimized for single GPU

**Cons**:
- ❌ More complex setup
- ❌ Need to handle edge cases yourself
- ❌ Slightly more Python overhead

## 📊 Configuration

### Sweep Parameters
The sweep optimizes:
- **Buoyancy Mass**: Continuous range [0.18, 0.29]
- **Robot Morphology**: 40+ different USD files
- **Training Parameters**: Fixed (task, iterations, environments, seed)

### Optimization
- **Method**: Bayesian optimization
- **Metric**: Maximize `max_mean_reward`
- **Strategy**: Intelligent parameter exploration

## 🛠️ Usage Examples

### 1. Basic CLI Sweep
```bash
# Create sweep
python launch_sweep.py --project my-ballu-sweep --entity my-team

# Start single agent
wandb agent my-team/my-ballu-sweep/SWEEP_ID
```

### 2. Programmatic with Custom Settings
```bash
# Create and run with custom count
python launch_sweep.py \
    --start-agent \
    --programmatic \
    --count 10 \
    --project ballu-morphology-sweep
```

### 3. Long-Running Sequential Execution
```bash
# Run many experiments sequentially
python launch_sweep.py \
    --start-agent \
    --programmatic \
    --count 20 \
    --project ballu-optimization-v2
```

## 🔧 Environment Variables

The system uses environment variables for morphology selection:
- `BALLU_MORPHOLOGY_USD_PATH`: Full path to robot USD file
- Set automatically by the sweep agent based on W&B config

## 📈 Monitoring

### W&B Dashboard
All runs are logged to W&B with:
- Real-time training metrics
- Morphology information
- Parameter values
- Training duration
- Success/failure status

### Programmatic Monitoring
The programmatic approach provides additional monitoring:
- Console progress updates
- Agent status tracking
- Completion summaries
- Error reporting

## 🐛 Debugging

### Common Issues

**Import Errors**:
```bash
# Check if all dependencies are installed
pip install wandb tensorboard
```

**USD File Not Found**:
```bash
# Validate USD paths
python -c "from sweep_utils import get_robot_usd_full_path; print(get_robot_usd_full_path('original/original.usd'))"
```

**Training Failures**:
```bash
# Run single experiment for debugging
python ../rsl_rl/train.py --task Isaac-Vel-BALLU-v0 --seed 42 --max_iterations 100
```

### Debug Mode
For debugging, use single experiment execution:
```bash
python launch_sweep.py --start-agent --programmatic --count 1
```

## 📝 Customization

### Adding New Morphologies
1. Add USD files to `robots/` directory
2. Update `sweep_config.yaml` with new paths
3. Validate with `launch_sweep.py` (it checks file existence)

### Modifying Parameters
Edit `sweep_config.yaml`:
```yaml
parameters:
  buoyancy_mass:
    distribution: uniform
    min: 0.15  # Adjust range
    max: 0.35
  robot_morphology:
    values: ["path/to/new/morphology.usd"]  # Add new files
```

### Custom Metrics
Modify `sweep_agent.py` to log additional metrics:
```python
wandb.log({
    'custom_metric': value,
    'max_mean_reward': max_reward  # Keep this for optimization
})
```

## 🎯 Expected Results

A typical sweep will:
- Test 40+ morphologies × buoyancy mass range sequentially
- Use Bayesian optimization for efficient exploration
- Achieve convergence in 50-200 experiments
- Identify optimal morphology-buoyancy combinations
- Provide reproducible results via W&B logging
- Complete experiments one at a time (optimized for single GPU)

## 🔗 Integration

The sweep system integrates with:
- **Isaac Lab**: Environment simulation
- **RSL-RL**: Training algorithm
- **BALLU Assets**: Robot morphologies
- **W&B**: Experiment tracking and optimization

## 📞 Support

For issues:
1. Check the console output for error messages
2. Verify USD file paths in the robots directory
3. Ensure W&B credentials are configured
4. Check Isaac Lab environment setup

---

🤖 **Happy optimizing!** The sweep will find the best BALLU configuration for your task.