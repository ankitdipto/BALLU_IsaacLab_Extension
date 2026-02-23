# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--reward_std", type=float, default=None, 
                   help="Standard deviation for velocity tracking reward (default: sqrt(0.25))")
parser.add_argument("--common_folder", type=str, default=None, 
                   help="Common folder name for all seeded runs (overrides timestamp-based folders)")
parser.add_argument("--world", action="store_true", default=False, 
                   help="Use world frame for velocity tracking reward")
parser.add_argument("--GCR", type=float, default=0.84, 
                   help="Gravity compensation ratio")
parser.add_argument("--GCR_range", type=float, nargs=2, default=None, 
                   help="Range of gravity compensation ratio (min max)")
parser.add_argument("--GCR_samples_file", type=str, default=None,
                   help="Path to a .npy file with per-env GCR values (shape: num_envs,). "
                        "Takes priority over --GCR_range and --GCR.")
parser.add_argument("--spcf", type=float, default=0.005, 
                   help="Spring coefficient")
parser.add_argument("--spcf_range", type=float, nargs=2, default=None, 
                   help="Range of spring coefficient (min max)")
parser.add_argument("--spcf_samples_file", type=str, default=None,
                   help="Path to a .npy file with per-env spcf values (shape: num_envs,). "
                        "Takes priority over --spcf_range and --spcf.")
parser.add_argument("--dl", type=int, default=None, 
                   help="Difficulty level of the obstacle (default: 0)")
parser.add_argument("--resume_path", type=str, default=None,
                   help="Full absolute path to a checkpoint to warm-start from. "
                        "Takes priority over agent_cfg.resume / --load_run / --checkpoint.")
# parser.add_argument("--fl_ratio", type=float, default=0.5, 
#                    help="Ratio of femur length to total leg length")

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
import numpy as np
from datetime import datetime
import math
import traceback

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    if args_cli.common_folder:
        # If common_folder is provided, use it instead of timestamp
        if agent_cfg.run_name:
            # Include run_name if provided
            log_dir = os.path.join(log_root_path, args_cli.common_folder, f"{agent_cfg.run_name}")
        else:
            # Just use seed if no run_name
            log_dir = os.path.join(log_root_path, args_cli.common_folder, f"seed_{agent_cfg.seed}")
        print(f"[INFO] Using common folder: {args_cli.common_folder}")
    else:
        # Original timestamp-based directory
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.world:
        # Zero out the base frame reward
        env_cfg.rewards.track_lin_vel_xy_base_exp.weight = 0.0
        # Set the world frame reward to 1.0
        env_cfg.rewards.track_lin_vel_xy_world_exp.weight = 1.0

    # else:
    #     # Zero out the world frame reward
    #     env_cfg.rewards.track_lin_vel_xy_world_exp.weight = 0.0
    #     # Set the base frame reward to 1.0
    #     env_cfg.rewards.track_lin_vel_xy_base_exp.weight = 1.0

    if args_cli.reward_std is not None:
        raise ValueError("Reward standard deviation is not supported for this task.")
        # set the reward standard deviation for velocity tracking
        env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(args_cli.reward_std)
    
    # Load per-env sample files if provided (PEC expert training mode).
    # These take priority over --GCR_range/--GCR and --spcf_range/--spcf respectively.
    gcr_values = None
    if args_cli.GCR_samples_file is not None:
        gcr_values = np.load(args_cli.GCR_samples_file).tolist()
        print(f"[INFO] Loaded {len(gcr_values)} GCR samples from: {args_cli.GCR_samples_file}")

    spcf_values = None
    if args_cli.spcf_samples_file is not None:
        spcf_values = np.load(args_cli.spcf_samples_file).tolist()
        print(f"[INFO] Loaded {len(spcf_values)} spcf samples from: {args_cli.spcf_samples_file}")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
        GCR=args_cli.GCR, GCR_range=args_cli.GCR_range, GCR_values=gcr_values,
        spcf=args_cli.spcf, spcf_range=args_cli.spcf_range, spcf_values=spcf_values,
    )

    if args_cli.dl is not None:
        isaac_env = env.unwrapped
        # print("Isaac environment: ", isaac_env)
        inter_obstacle_spacing_y = -2.0
        isaac_env.scene._default_env_origins = isaac_env.scene._default_env_origins + \
            torch.tensor([0.0, inter_obstacle_spacing_y, 0.0], device=isaac_env.device) * args_cli.dl

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Warm-start: --resume_path (full path, PEC) takes priority over agent_cfg.resume
    if args_cli.resume_path is not None:
        print(f"[INFO]: Warm-starting from checkpoint: {args_cli.resume_path}")
        runner.load(args_cli.resume_path)
    elif agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # dump CLI args (argparse.Namespace) into log-directory
    # note: dump_yaml expects dict or config-class; argparse.Namespace needs explicit conversion
    args_cli_cfg = vars(args_cli).copy()
    dump_yaml(os.path.join(log_dir, "params", "cli.yaml"), args_cli_cfg)

    try:
        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except Exception as e:
        print(f"This run failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        # close the simulator
        env.close()

    # print the log directory
    print(f"EXP_DIR: {log_dir}")


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        print(f"This run failed with error: {e}")
    finally:
        # close sim app
        simulation_app.close()
