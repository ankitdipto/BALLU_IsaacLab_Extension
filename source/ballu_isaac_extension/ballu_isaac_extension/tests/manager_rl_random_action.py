# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ballu_isaac_extension.tasks.ballu_locomotion.basic_vel_env_cfg import BALLUEnvCfg

def main():
    """Main function."""
    # create environment configuration
    env_cfg = BALLUEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # print environment information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max episode length: {env.max_episode_length}")
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            # sample random actions from the normal distribution
            # joint_pos_targets = torch.randn_like(env.action_manager.action)

            # Sample random actions from the uniform distribution
            # Convert 0 to 99 degrees to radians (uniform distribution)
            min_angle_rad = -100000
            max_angle_rad = 100000 * torch.pi / 180.0  # Convert 99 degrees to radians
            joint_pos_targets = torch.rand_like(env.action_manager.action) * (max_angle_rad - min_angle_rad) + min_angle_rad
            #joint_pos_targets = torch.zeros_like(env.action_manager.action)
            #joint_pos_targets = max_angle_rad * torch.ones_like(env.action_manager.action)
            #print("[INFO]: Random action: ", joint_pos_targets)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_pos_targets)
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()