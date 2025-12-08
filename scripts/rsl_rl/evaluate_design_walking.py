"""Script to evaluate BALLU morphology with universal controller for walking tasks."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate BALLU morphology with universal walking controller.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isc-BALLU-fast-walk-hetero", help="Name of the task.")
parser.add_argument("--other_dirs", type=str, default=None, help="Other directories to append to the run directory.")
parser.add_argument("--GCR", type=float, default=0.84, 
                   help="Gravity compensation ratio")
parser.add_argument("--spcf", type=float, default=0.005, help="Spring Coefficient")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# clear out sys.argv for Hydra after launching the app
# This ensures that any arguments added by AppLauncher are not passed to Hydra
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401
from tqdm import tqdm

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate morphology with universal walking controller."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_path = os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint_path: {checkpoint_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None,
                   GCR=args_cli.GCR, spcf=args_cli.spcf)
    
    isaac_env = env.unwrapped
    print("Isaac environment: ", isaac_env)
    
    EP_LENGTH = isaac_env.max_episode_length
    print(f"Env episode length: {EP_LENGTH}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Track performance metrics
    velocities_x = []
    velocities_y = []
    episode_rewards = []
    current_episode_reward = torch.zeros(env_cfg.scene.num_envs, device=isaac_env.device)
    
    # simulate environment
    with tqdm(total=EP_LENGTH, desc="Evaluating") as pbar:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                
                # env stepping
                obs, rew, dones, _ = env.step(actions)
                
                # Track velocity (base linear velocity in x direction)
                base_lin_vel = isaac_env.scene["robot"].data.root_lin_vel_w[:, 0]  # x-component
                base_lin_vel_y = isaac_env.scene["robot"].data.root_lin_vel_w[:, 1]  # y-component
                velocities_x.append(base_lin_vel.cpu().numpy())
                velocities_y.append(base_lin_vel_y.cpu().numpy())
                
                # Track rewards
                current_episode_reward += rew
                
                # Check for episode terminations
                if torch.any(dones):
                    # Store completed episode rewards
                    done_indices = torch.where(dones)[0]
                    for idx in done_indices:
                        episode_rewards.append(current_episode_reward[idx].item())
                        current_episode_reward[idx] = 0.0
                
                timestep += 1
            
            if timestep == EP_LENGTH:
                break
            pbar.update(1)

    # Calculate performance metrics
    velocities_x = np.concatenate(velocities_x)
    velocities_y = np.concatenate(velocities_y)
    
    mean_velocity_x = np.mean(velocities_x)
    std_velocity_x = np.std(velocities_x)
    mean_velocity_y = np.mean(np.abs(velocities_y))  # Mean absolute lateral deviation
    
    if len(episode_rewards) > 0:
        mean_episode_reward = np.mean(episode_rewards)
    else:
        mean_episode_reward = 0.0
    
    # Performance metric: prioritize forward velocity with penalty for lateral drift
    # Higher is better
    performance_metric = mean_velocity_x # - 0.5 * mean_velocity_y
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mean Forward Velocity (x): {mean_velocity_x:.4f} m/s")
    print(f"Std Forward Velocity (x): {std_velocity_x:.4f} m/s")
    print(f"Mean Abs Lateral Velocity (y): {mean_velocity_y:.4f} m/s")
    print(f"Mean Episode Reward: {mean_episode_reward:.4f}")
    print(f"Performance Metric: {performance_metric:.4f}")
    print(f"{'='*80}\n")
    
    # Output for parsing by optimization script
    print(f"PERFORMANCE_METRIC: {performance_metric}")

    # close the simulator
    env.close()
    

if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        raise e
    finally:
        # close sim app
        simulation_app.close()

