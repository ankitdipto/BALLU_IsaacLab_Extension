# Multi-Run Training Script

This script (`multi_run_training.py`) automates multiple RSL-RL training experiments with different parameters, enabling systematic hyperparameter sweeps, statistical significance testing, and comparative studies.

## Requirements

- **TensorBoard** (recommended): For rich training metrics extraction
  ```bash
  pip install tensorboard
  ```
- **Fallback**: Script works without TensorBoard but with limited metrics from text files

## Features

- **Automated Multi-Seed Training**: Run multiple training sessions with different random seeds for statistical reliability
- **Subprocess Isolation**: Each training runs in a separate Python process to avoid simulator conflicts
- **TensorBoard Integration**: Extracts 7 specific training metrics from TensorBoard event files (reward, entropy, learning rate, mirror symmetry, surrogate loss, value function loss, noise std)
- **Comprehensive Result Analysis**: Multi-source data extraction from TensorBoard events and RSL-RL log files
- **Advanced Visualizations**: Generates 6-panel analysis including loss evolution, convergence curves, and performance correlations
- **Robust Error Handling**: Continues running even if individual experiments fail with graceful fallbacks
- **Intermediate Saving**: Saves partial results after each experiment to prevent data loss

## Usage

### Basic Multi-Seed Training
Run the same task with multiple seeds for statistical analysis:
```bash
python multi_run_training.py --task Isc-Vel-BALLU-encoder --seeds 42 123 456 789 999 --max_iterations 2000
```

### Quick Testing with Short Training
```bash
python multi_run_training.py --task Isc-Vel-BALLU-encoder --seeds 42 123 --max_iterations 500 --num_envs 2048
```

### Hyperparameter Comparison
Compare different reward frame configurations:
```bash
# Base frame training
python multi_run_training.py --task Isaac-Ballu-Indirect-Act-v0 --seeds 42 123 456 --common_folder "base_frame_study" --max_iterations 1000

# World frame training  
python multi_run_training.py --task Isaac-Ballu-Indirect-Act-v0 --seeds 42 123 456 --common_folder "world_frame_study" --max_iterations 1000 --world
```

### Custom Configuration
```bash
python multi_run_training.py \
    --task Isaac-Ballu-Indirect-Act-v0 \
    --seeds 1 2 3 4 5 \
    --max_iterations 3000 \
    --num_envs 8192 \
    --common_folder "ballu_performance_study" \
    --output_dir "experiments/ballu_2024" \
    --additional_args "--video" "--video_interval" "1000"
```

## Arguments

### Core Training Parameters
- `--task`: Task name for training (default: `Isc-Vel-BALLU-encoder`)
- `--seeds`: List of random seeds to test (default: `[42, 123, 456, 789, 999]`)
- `--max_iterations`: Maximum training iterations per experiment (default: `1000`)
- `--num_envs`: Number of environments for training (default: `4096`)

### Organization & Output
- `--output_dir`: Output directory for results (default: timestamped folder)
- `--common_folder`: Common folder name for all training runs (default: timestamped)

### Training Configuration  
- `--headless`: Run training in headless mode (default: `True`)
- `--world`: Use world frame for velocity tracking reward (default: `False`)
- `--additional_args`: Additional arguments to pass to `train.py`

## Output Files

The script generates comprehensive results in the specified output directory:

### 1. **training_results.json**
Raw experiment data including:
- Experiment metadata (seeds, parameters, timing)
- Training metrics (final rewards, convergence statistics)
- Error logs for failed experiments

### 2. **training_analysis.png**
Six-panel comprehensive visualization:
- **Final Reward by Seed**: Bar chart comparing final performance across seeds
- **Training Convergence Curves**: Mean reward progression over training iterations
- **Entropy Evolution**: Loss/entropy progression during training
- **Surrogate Loss Evolution**: Loss/surrogate convergence curves
- **Value Function Loss Evolution**: Loss/value_function tracking
- **Learning Rate Evolution**: Loss/learning_rate schedule visualization

### 3. **partial_results.json**
Intermediate results saved after each experiment (for monitoring progress)

## Training Metrics Extracted

The script automatically extracts comprehensive training data from multiple sources:

### Primary: TensorBoard Event Files
- **Train/mean_reward**: Reward progression and final values
- **Loss/entropy**: Entropy evolution during training
- **Loss/learning_rate**: Learning rate schedule tracking
- **Loss/mirror_symmetry**: Mirror symmetry loss progression
- **Loss/surrogate**: Surrogate loss evolution
- **Loss/value_function**: Value function loss tracking
- **Policy/mean_noise_std**: Policy noise standard deviation

### Fallback: RSL-RL Text Log Files
- **Basic Performance**: Mean/std reward and episode length at training end  
- **Simple Progression**: Reward and episode length curves from summaries.txt
- **Training Metadata**: Duration, iterations completed, basic convergence status

### Advanced Analysis
- **Convergence Stability**: Statistical analysis of final 20% of reward training
- **Improvement Rates**: Learning trajectory analysis for mean reward
- **Loss Evolution**: Surrogate, value function, entropy, and learning rate tracking
- **Policy Analysis**: Noise standard deviation and mirror symmetry monitoring
- **Multi-Seed Statistics**: Performance variance and reliability metrics

## Common Use Cases

### 1. Statistical Significance Testing
```bash
# Run 10 seeds for robust statistics
python multi_run_training.py --seeds 1 2 3 4 5 6 7 8 9 10 --max_iterations 2000
```

### 2. Hyperparameter Sweeps
```bash
# Compare different environment counts
python multi_run_training.py --num_envs 2048 --common_folder "env_2k" --seeds 42 123 456
python multi_run_training.py --num_envs 4096 --common_folder "env_4k" --seeds 42 123 456  
python multi_run_training.py --num_envs 8192 --common_folder "env_8k" --seeds 42 123 456
```

### 3. Algorithm Comparison
```bash
# Base configuration
python multi_run_training.py --common_folder "baseline" --seeds 1 2 3 4 5

# Modified reward function (using world frame)
python multi_run_training.py --common_folder "world_frame" --world --seeds 1 2 3 4 5
```

### 4. Convergence Studies
```bash
# Short training
python multi_run_training.py --max_iterations 500 --common_folder "short_train"

# Long training  
python multi_run_training.py --max_iterations 5000 --common_folder "long_train"
```

## Tips for Effective Usage

### 1. **Resource Management**
- Each training subprocess can use significant GPU memory
- Monitor GPU usage if running multiple experiments
- Consider reducing `--num_envs` for memory-constrained systems

### 2. **Time Planning**
- Training duration depends on `--max_iterations` and `--num_envs`
- 1000 iterations with 4096 envs typically takes 15-30 minutes
- Plan accordingly for multi-seed studies

### 3. **Result Organization**
- Use descriptive `--common_folder` names for easy identification
- Include key parameters in folder names (e.g., "ballu_world_frame_2k_iter")
- Save `--output_dir` to meaningful locations for long-term storage

### 4. **Monitoring Progress**
```bash
# Watch partial results during execution
tail -f multi_training_*/partial_results.json

# Monitor log files
tail -f logs/rsl_rl/*/common_folder_name/seed_*/summaries.txt
```

## Example Workflow

Here's a complete workflow for comparing two different reward configurations:

```bash
# 1. Run baseline configuration
python multi_run_training.py \
    --task Isaac-Ballu-Indirect-Act-v0 \
    --seeds 42 123 456 789 999 \
    --max_iterations 2000 \
    --common_folder "baseline_base_frame" \
    --output_dir "results/reward_comparison"

# 2. Run world frame configuration  
python multi_run_training.py \
    --task Isaac-Ballu-Indirect-Act-v0 \
    --seeds 42 123 456 789 999 \
    --max_iterations 2000 \
    --common_folder "modified_world_frame" \
    --output_dir "results/reward_comparison" \
    --world

# 3. Compare results
# The training_analysis.png files will show the differences
# The training_results.json files contain raw data for further analysis
```

## Troubleshooting

### Common Issues

1. **"No log directory found"**: The training may have failed before creating logs
   - Check the subprocess error messages
   - Verify task name and parameters

2. **GPU Memory Issues**: Multiple subprocess may exhaust GPU memory
   - Reduce `--num_envs`
   - Ensure previous training processes are terminated

3. **Long Training Times**: Training taking longer than expected
   - Monitor GPU utilization
   - Consider reducing `--max_iterations` for testing

### Debug Mode
Add debugging arguments to see more detailed output:
```bash
python multi_run_training.py --additional_args "--device" "cuda:0" --seeds 42
```

This multi-run training script enables systematic and reproducible RL research by automating the tedious process of running multiple training experiments and analyzing their results. 