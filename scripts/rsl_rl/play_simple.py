"""Script to play a checkpoint of an RL agent from RSL-RL without plotting or data saving."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import datetime

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (minimal version).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=399, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--balloon_buoyancy_mass", type=float, default=0.24, 
                   help="Buoyancy mass of the balloon")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent (minimal version)."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_path = os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.join(log_root_path, agent_cfg.load_run)
    print(f"[INFO] checkpoint_path: {checkpoint_path}")
    print(f"[INFO] log_dir: {log_dir}")

    timestamp = datetime.datetime.now().strftime('%b%d_%H_%M_%S') # Format: Apr08_19_25_38
    # Create the play folder for this run
    play_folder = os.path.join(log_dir, "play", timestamp, f"{agent_cfg.load_checkpoint[:-3]}")
    os.makedirs(play_folder, exist_ok=True)
    print(f"[INFO] The play folder for this run: {play_folder}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None,
                   balloon_buoyancy_mass=args_cli.balloon_buoyancy_mass)
    
    if args_cli.video:
        video_kwargs = {
            "name_prefix": 'ballu',  
            "video_folder": play_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # resume training
    runner.load(checkpoint_path)
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    cumulative_rewards = torch.zeros(env.num_envs, device=env.device)

    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, _, _ = env.step(actions)
            
                # accumulate rewards
                cumulative_rewards += rewards
            
                timestep += 1
                
                if args_cli.video and timestep == args_cli.video_length:
                    break
    except Exception as e:
        print(f"[INFO] Exception received: {e}")
    finally:
        # close the simulator
        env.close()

    # print cumulative rewards
    print(f"[INFO] Episode completed after {timestep} steps")
    # print(f"[INFO] Cumulative rewards: {cumulative_rewards}")
    print(f"[INFO] Mean cumulative reward: {cumulative_rewards.mean().item():.4f}")
    if env.num_envs > 1:
        print(f"[INFO] Std cumulative reward: {cumulative_rewards.std().item():.4f}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
