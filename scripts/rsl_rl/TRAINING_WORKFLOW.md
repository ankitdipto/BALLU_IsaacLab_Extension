# BALLU Training Workflow Documentation

**Complete guide to the training pipeline architecture and code flow**

---

## 🎯 Quick Start

Train a BALLU policy with a single command:

```bash
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-imu-tibia --num_envs 4096 --seed 42 --max_iterations 2000
```

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Complete Code Flow](#complete-code-flow)
3. [Component Details](#component-details)
4. [BALLU-Specific Physics](#ballu-specific-physics)
5. [Configuration System](#configuration-system)
6. [Training Loop Details](#training-loop-details)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The BALLU training system uses a **manager-based architecture** built on Isaac Lab. The workflow involves:

- **Task Registration**: Map task IDs to environment + agent configurations
- **Environment Creation**: Instantiate parallel simulation environments
- **Manager Initialization**: Load specialized managers for MDP components
- **Training Loop**: Collect rollouts, compute advantages, update policy
- **BALLU Physics**: Apply custom buoyancy, drag, and indirect actuation

### Key Files

| File | Purpose |
|------|---------|
| `train.py` | Entry point and orchestration |
| `tasks/ballu_locomotion/__init__.py` | Task registration |
| `tasks/ballu_locomotion/*_env_cfg.py` | Environment configurations |
| `tasks/ballu_locomotion/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| `isaac_lab/envs/manager_based_rl_env.py` | Core RL environment |
| `rsl_rl/runners/on_policy_runner.py` | PPO training runner |

---

## Complete Code Flow

### Step 1: Script Invocation (`train.py`)

```python
# Command line
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-imu-tibia --num_envs 4096

# What happens (train.py lines 19-57):
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--GCR", type=float, default=0.84)  # Gravity Compensation Ratio
parser.add_argument("--spcf", type=float, default=0.005)  # Spring coefficient

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

**CLI Arguments:**
- `--task`: Task ID that maps to env + agent config
- `--num_envs`: Number of parallel environments (default: from config)
- `--seed`: Random seed for reproducibility
- `--max_iterations`: Training iterations (default: 1500)
- `--GCR`: Gravity Compensation Ratio (buoyancy, default: 0.84)
- `--GCR_range`: Randomize GCR per environment
- `--spcf`: Spring coefficient for joints
- `--world`: Use world-frame velocity tracking (vs base-frame)
- `--common_folder`: Organize multi-seed runs

---

### Step 2: Task Registration (`__init__.py`)

```python
# File: tasks/ballu_locomotion/__init__.py

gym.register(
    id="Isaac-Vel-BALLU-imu-tibia",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tibia_imu_env_cfg:BALLU_TibiaIMU_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)
```

**Task ID Structure:**
```
Isaac-Vel-BALLU-<sensor>-<terrain>
  │    │    │      │        │
  │    │    │      │        └─ Terrain type (flat/rough/obstacle)
  │    │    │      └────────── Sensor suite (priv/imu/mocap)
  │    │    └─────────────────── Robot name
  │    └──────────────────────── Task type (Vel=velocity tracking)
  └───────────────────────────── Framework
```

**Key Insight:** One task ID bundles:
- Environment configuration (robot, sensors, rewards, observations)
- Agent configuration (PPO hyperparameters, network architecture)

---

### Step 3: Configuration Loading (`train.py` line 96)

```python
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Hydra loads both configs automatically"""
```

**What Hydra Does:**
1. Looks up task ID in gymnasium registry
2. Loads `env_cfg_entry_point` → environment configuration class
3. Loads `rsl_rl_cfg_entry_point` → agent configuration class
4. Passes both as arguments to `main()`

---

### Step 4: Environment Creation (`train.py` line 151)

```python
env = gym.make(
    args_cli.task,
    cfg=env_cfg,
    render_mode="rgb_array" if args_cli.video else None,
    GCR=args_cli.GCR,              # BALLU-specific: buoyancy
    GCR_range=args_cli.GCR_range,
    spcf=args_cli.spcf,            # BALLU-specific: spring coef
    spcf_range=args_cli.spcf_range
)
```

This instantiates `ManagerBasedRLEnv` → triggers full initialization.

---

### Step 5: Environment Initialization (`manager_based_rl_env.py`)

```python
class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode, **kwargs):
        # Store BALLU-specific parameters
        self.GCR = kwargs.get("GCR", 0.84)
        self.GCR_range = kwargs.get("GCR_range", None)
        
        # Initialize base environment (loads scene, physics)
        super().__init__(cfg=cfg)
        
        # Calculate buoyancy per environment
        robot = self.scene["robot"]
        self.robot_total_mass = robot.data.default_mass.sum(dim=1)
        
        if self.GCR_range is not None:
            # Randomize buoyancy across environments
            GCR_tensor = torch.rand(num_envs, 1) * (GCR_range[1] - GCR_range[0]) + GCR_range[0]
            self.balloon_buoyancy_mass_t = GCR_tensor * self.robot_total_mass.mean()
        else:
            balloon_buoyancy_mass = self.GCR * self.robot_total_mass.mean()
            self.balloon_buoyancy_mass_t = torch.full((num_envs, 1), balloon_buoyancy_mass)
        
        # Initialize episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
```

**Key Initialization Steps:**
1. Store BALLU physics parameters (GCR, spring coefficients)
2. Load scene (robot, terrain, sensors)
3. Calculate per-environment buoyancy forces
4. Initialize episode buffers

---

### Step 6: Manager Loading (`manager_based_rl_env.py` line 163)

```python
def load_managers(self):
    # Load managers in specific order (dependencies matter!)
    
    # 1. Command Manager - generates target velocities
    self.command_manager = CommandManager(self.cfg.commands, self)
    
    # 2. Action Manager - processes raw actions to joint commands
    self.action_manager = ActionManager(self.cfg.actions, self)
    
    # 3. Observation Manager - collects sensor data
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    
    # 4. Termination Manager - checks episode end conditions
    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    
    # 5. Reward Manager - computes reward signals
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    
    # 6. Curriculum Manager - adjusts difficulty
    self.curriculum_manager = CurriculumManager(self.cfg.curriculums, self)
```

**Manager Responsibilities:**

| Manager | Purpose | Example |
|---------|---------|---------|
| **CommandManager** | Generate task commands | Target velocity: `lin_vel_x ∈ [0.3, 0.5] m/s` |
| **ActionManager** | Process RL actions | Map 2D actions → joint targets |
| **ObservationManager** | Build observation vectors | `[cmd, joint_pos, joint_vel, IMU, ...]` |
| **RewardManager** | Compute reward terms | `track_velocity + alive - energy` |
| **TerminationManager** | Check termination | `base_height < 0.2m → terminate` |
| **CurriculumManager** | Adjust difficulty | Increase obstacle height over time |

---

### Step 7: Wrapper & Runner Creation (`train.py` lines 177-181)

```python
# Wrap environment for RSL-RL compatibility
env = RslRlVecEnvWrapper(env)

# Create PPO runner
runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=log_dir,
    device=agent_cfg.device
)
```

---

### Step 8: Runner Initialization (`on_policy_runner.py`)

```python
class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir, device):
        # Get observation dimensions
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]  # e.g., 16 for [cmd(2) + joint_pos(7) + joint_vel(7)]
        
        # Check for privileged observations (asymmetric actor-critic)
        if "critic" in extras["observations"]:
            num_privileged_obs = extras["observations"]["critic"].shape[1]
        
        # Create policy network
        policy = ActorCritic(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_actions=self.env.num_actions,  # 2 for BALLU (MOTOR_LEFT, MOTOR_RIGHT)
            actor_hidden_dims=[128, 64, 32],
            critic_hidden_dims=[128, 64, 32],
            activation="elu"
        ).to(self.device)
        
        # Create PPO algorithm
        self.alg = PPO(
            policy,
            device=self.device,
            value_loss_coef=1.0,
            clip_param=0.2,
            entropy_coef=0.01,
            learning_rate=1e-4,
            gamma=0.99,
            lam=0.95
        )
        
        # Initialize observation normalizers
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs])
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs])
```

**Network Architecture:**
```
Actor:  obs[16] → FC[128] → ELU → FC[64] → ELU → FC[32] → ELU → actions[2]
Critic: priv_obs[N] → FC[128] → ELU → FC[64] → ELU → FC[32] → ELU → value[1]
```

---

### Step 9: Training Loop (`on_policy_runner.py` line 174)

```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # Initialize observations
    obs, extras = self.env.get_observations()
    privileged_obs = extras["observations"].get("critic", obs)
    
    # Randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, 
            high=int(self.env.max_episode_length)
        )
    
    # Training loop
    for it in range(num_learning_iterations):
        # ========== ROLLOUT PHASE ==========
        with torch.inference_mode():
            for step in range(self.num_steps_per_env):  # Default: 20 steps
                # 1. Sample actions from policy
                actions = self.alg.act(obs, privileged_obs)
                
                # 2. Step environment
                obs, rewards, dones, infos = self.env.step(actions)
                
                # 3. Normalize observations
                obs = self.obs_normalizer(obs)
                privileged_obs = self.privileged_obs_normalizer(infos["observations"]["critic"])
                
                # 4. Store rollout data
                self.alg.process_env_step(rewards, dones, infos)
                
                # 5. Track episode statistics
                cur_reward_sum += rewards
                cur_episode_length += 1
                
                # Reset completed episodes
                new_ids = (dones > 0).nonzero()
                rewbuffer.extend(cur_reward_sum[new_ids])
                cur_reward_sum[new_ids] = 0
        
        # 6. Compute advantages using GAE
        self.alg.compute_returns(privileged_obs)
        
        # ========== UPDATE PHASE ==========
        loss_dict = self.alg.update()  # PPO update
        
        # ========== LOGGING & CHECKPOINTING ==========
        if it % self.save_interval == 0:
            self.save(f"model_{it}.pt")
            self.log(locals())
```

**Rollout Statistics:**
- **Batch size per iteration**: `num_envs × num_steps_per_env`
  - Example: 4096 envs × 20 steps = **81,920 transitions** per iteration
- **Updates per iteration**: 5 epochs × 4 mini-batches = 20 gradient updates
- **Effective samples per update**: 81,920 / 4 = 20,480 samples per mini-batch

---

### Step 10: Environment Step (`manager_based_rl_env.py` line 207)

```python
def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
    """Execute one control step (with physics decimation)"""
    
    # 1. Process actions through action manager
    self.action_manager.process_action(actions)
    processed_actions = self.action_manager._terms["joint_pos"].processed_actions
    
    # 2. Physics loop with decimation
    for _ in range(self.cfg.decimation):  # Default: 4 sub-steps per control step
        # Apply BALLU-specific forces
        self._apply_ballu_physics()
        self._apply_indirect_actuation(processed_actions)
        
        # Simulate one physics step
        self.sim.step(render=False)
    
    # 3. Update counters
    self.episode_length_buf += 1
    self.common_step_counter += 1
    
    # 4. Compute rewards
    self.reward_manager.compute(dt=self.step_dt)
    
    # 5. Check terminations
    self.termination_manager.compute(dt=self.step_dt)
    
    # 6. Reset terminated environments
    reset_env_ids = self.termination_manager.terminated.nonzero().squeeze(-1)
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
    
    # 7. Update curriculum
    self.curriculum_manager.compute(dt=self.step_dt)
    
    # 8. Compute observations
    self.observation_manager.compute()
    
    # 9. Return MDP tuple
    return (
        self.observation_manager.compute_group("policy"),
        self.reward_manager.get_sum(),
        self.termination_manager.terminated,
        self.observation_manager.compute_group_extras("policy")
    )
```

**Timing:**
- **Control frequency**: `1 / (decimation × sim_dt)` = `1 / (4 × 0.005)` = **50 Hz**
- **Physics frequency**: `1 / sim_dt` = `1 / 0.005` = **200 Hz**

---

## BALLU-Specific Physics

### Buoyancy Force Application

```python
# Compute buoyancy force in world frame
GRAVITY = torch.tensor([0.0, 0.0, 9.81], device=self.device)
buoyancy_force_w = GRAVITY * self.balloon_buoyancy_mass_t  # Shape: (num_envs, 3)

# Convert to local frame
balloons_quat_w = robot.data.body_link_quat_w[:, balloon_body_id, :]
buoyancy_force_l = quat_rotate_inverse(balloons_quat_w, buoyancy_force_w)

# Compute torque from offset application point
distance_from_neck_l = torch.tensor([0.0, -0.38, 0.0], device=self.device)
buoyancy_torque_l = torch.cross(distance_from_neck_l, buoyancy_force_l)

# Apply to simulation
robot.set_external_force_and_torque(
    forces=buoyancy_force_l,
    torques=buoyancy_torque_l,
    body_ids=balloon_ids,
    is_global=False
)
```

### Drag Force Application

```python
# Compute drag force (simplified model)
DRAG_COEFFICIENT = 0.0  # Can be tuned
robot_balloon_lin_vel_w = robot.data.body_lin_vel_w[:, balloon_body_id, :]
drag_force_w = -torch.sign(robot_balloon_lin_vel_w) * DRAG_COEFFICIENT * (robot_balloon_lin_vel_w ** 2)

# Convert to local frame and apply
drag_force_l = quat_rotate_inverse(balloons_quat_w, drag_force_w)
drag_torque_l = torch.cross(distance_from_neck_l, drag_force_l)
```

### Indirect Actuation (Four-Bar Linkage Simulation)

```python
# BALLU's knee joints are driven indirectly through motor arms
# RL policy outputs: [MOTOR_LEFT, MOTOR_RIGHT] ∈ [0, π]
# These map linearly to knee positions: [KNEE_LEFT, KNEE_RIGHT] ∈ [0, 1.745 rad]

motor_left_action = processed_actions[:, 0]
motor_right_action = processed_actions[:, 1]

# Linear mapping
knee_min, knee_max = 0.0, 1.74532925  # 100 degrees in radians
motor_min, motor_max = 0.0, math.pi

target_knee_left = knee_min + (motor_left_action - motor_min) / (motor_max - motor_min) * (knee_max - knee_min)
target_knee_right = knee_min + (motor_right_action - motor_min) / (motor_max - motor_min) * (knee_max - knee_min)

# Set targets (PhysX handles PD control)
robot.set_joint_position_target(target_knee_left, joint_ids=knee_left_jidx)
robot.set_joint_position_target(target_knee_right, joint_ids=knee_right_jidx)
```

**Why Indirect Actuation?**
- Simulates the real BALLU's four-bar linkage mechanism
- Motor rotation drives knee angle through mechanical coupling
- Improves sim-to-real transfer

---

## Configuration System

### Environment Configuration Structure

```python
@configclass
class BalluFlatEnvCfg(ManagerBasedRLEnvCfg):
    # Simulation parameters
    sim: SimulationCfg = SimulationCfg(dt=1/200, substeps=1)
    
    # Scene configuration
    scene: InteractiveSceneCfg = BALLUSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # MDP configuration
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculums: CurriculumCfg = CurriculumCfg()
    
    # RL parameters
    decimation = 4
    episode_length_s = 20.0
    viewer: ViewerCfg = ViewerCfg(eye=(7.5, 7.5, 7.5))
```

### Observation Configuration Example

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
```

**Result:** Observation vector = `[cmd(2), joint_pos(7), joint_vel(7)]` = 16D

### Reward Configuration Example

```python
@configclass
class RewardsCfg:
    # Tracking rewards
    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # Alive bonus
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # Penalties
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
```

---

## Training Loop Details

### PPO Update Process

After collecting a rollout of `num_steps_per_env` × `num_envs` transitions:

```python
# 1. Compute advantages using GAE
advantages = rewards + gamma * next_values - values
returns = advantages + values

# 2. For each epoch (default: 5)
for epoch in range(num_learning_epochs):
    # 3. Shuffle and split into mini-batches (default: 4)
    for mini_batch in mini_batches:
        # a) Recompute action log probs with current policy
        new_action_log_probs, entropy = policy.evaluate_actions(obs, actions)
        
        # b) Compute policy ratio
        ratio = torch.exp(new_action_log_probs - old_action_log_probs)
        
        # c) Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # d) Value loss (clipped)
        value_pred = policy.evaluate_value(privileged_obs)
        value_loss = (returns - value_pred).pow(2).mean()
        
        # e) Mirror symmetry loss (optional)
        if mirror_symmetry_enabled:
            mirror_loss = compute_mirror_symmetry_loss(obs, actions)
        
        # f) Total loss
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + mirror_weight * mirror_loss
        
        # g) Gradient step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
```

### Adaptive Learning Rate Schedule

```python
if schedule == "adaptive":
    # Compute KL divergence between old and new policy
    kl_div = torch.mean(old_log_probs - new_log_probs)
    
    if kl_div > desired_kl * 2.0:
        learning_rate *= 0.5  # Decrease LR
    elif kl_div < desired_kl / 2.0:
        learning_rate *= 2.0  # Increase LR
```

---

## Common Patterns

### Adding a New Task

1. **Create environment configuration** (`tasks/ballu_locomotion/my_task_env_cfg.py`):
```python
@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096)
    observations: MyObsCfg = MyObsCfg()
    actions: MyActionsCfg = MyActionsCfg()
    rewards: MyRewardsCfg = MyRewardsCfg()
    # ... etc
```

2. **Register task** (`tasks/ballu_locomotion/__init__.py`):
```python
gym.register(
    id="Isaac-MyTask-BALLU",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_task_env_cfg:MyTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)
```

3. **Train:**
```bash
python scripts/rsl_rl/train.py --task Isaac-MyTask-BALLU --num_envs 4096
```

### Adding Custom Reward Terms

1. **Define reward function** (`tasks/ballu_locomotion/mdp/rewards.py`):
```python
def my_custom_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Custom reward computation."""
    robot = env.scene["robot"]
    # Compute reward logic
    return reward_tensor  # Shape: (num_envs,)
```

2. **Add to reward config** (in your `*_env_cfg.py`):
```python
@configclass
class RewardsCfg:
    my_custom_reward = RewTerm(
        func=mdp.my_custom_reward,
        weight=1.0,
        params={"param1": value1}
    )
```

### Multi-Seed Training

```bash
python scripts/rsl_rl/multi_run_training.py \
    --task Isaac-Vel-BALLU-imu-tibia \
    --seeds 42 123 456 789 999 \
    --max_iterations 2000 \
    --common_folder "imu_tibia_study"
```

**Output:**
```
logs/rsl_rl/lab_12.10.2025/
└── imu_tibia_study/
    ├── seed_42/
    ├── seed_123/
    ├── seed_456/
    ├── seed_789/
    └── seed_999/
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce parallel environments
```bash
python train.py --task Isaac-Vel-BALLU-imu-tibia --num_envs 2048  # Instead of 4096
```

### Issue: Training Unstable / NaN Values

**Checklist:**
- [ ] Check observation normalization is enabled (`empirical_normalization=True`)
- [ ] Verify reward scales are reasonable (not too large)
- [ ] Check action clipping is working
- [ ] Reduce learning rate (`--learning_rate 5e-5`)
- [ ] Increase `desired_kl` for more conservative updates

### Issue: Environment Not Found

**Error:** `gymnasium.error.UnregisteredEnv: Environment Isaac-Vel-BALLU-xyz doesn't exist`

**Solution:** Ensure the task is registered in `__init__.py` and the import is present in `train.py`:
```python
import ballu_isaac_extension.tasks  # noqa: F401
```

### Issue: Slow Training

**Optimizations:**
- Increase `num_envs` to maximize GPU utilization (4096 or 8192)
- Reduce `num_steps_per_env` if memory-constrained (10-20 is typical)
- Disable video recording during training (`--video` flag)
- Use headless mode (default in train.py)

### Issue: Policy Not Learning

**Debug steps:**
1. Check reward signals: `tail -f logs/.../summaries.txt`
2. Visualize with TensorBoard: `tensorboard --logdir logs/`
3. Test with privileged observations first (easier task)
4. Verify action space is correct (2D for BALLU)
5. Check termination conditions aren't too strict

---

## Performance Metrics

### Typical Training Performance

| Metric | Value |
|--------|-------|
| **FPS** | 10,000 - 20,000 steps/s (on A100) |
| **Iteration time** | 4-8 seconds |
| **Time per 1000 iterations** | 1-2 hours |
| **Episodes per iteration** | 200-400 (depends on episode length) |
| **GPU memory** | 8-16 GB (for 4096 envs) |

### Convergence Expectations

| Task | Iterations to Good Performance |
|------|-------------------------------|
| Flat terrain walking | 500-1000 |
| Rough terrain | 1500-2500 |
| Obstacle navigation | 2000-3000 |
| Morphology generalization | 3000-5000 |

---

## Additional Resources

- **Isaac Lab Docs**: https://isaac-sim.github.io/IsaacLab
- **RSL-RL Paper**: [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978)
- **PPO Paper**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **Multi-Run Training**: See `scripts/rsl_rl/README_multi_run_training.md`
- **Morphology System**: See `source/.../morphology/README.md`

---

## Glossary

- **Decimation**: Number of physics steps per control step
- **GAE**: Generalized Advantage Estimation - method for computing advantages
- **GCR**: Gravity Compensation Ratio - buoyancy force as fraction of robot weight
- **Privileged observations**: Information available in simulation but not on real robot
- **Asymmetric actor-critic**: Actor and critic see different observations
- **Mirror symmetry loss**: Regularization encouraging symmetric gaits
- **Curriculum learning**: Gradually increasing task difficulty during training

---

**Last Updated:** December 2025  
**Maintainer:** BALLU Research Team  
**Questions?** Check existing issues or create a new one in the repository.
