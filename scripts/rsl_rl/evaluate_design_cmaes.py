"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate the design of BALLU with universal controller.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isc-BALLU-hetero-general", help="Name of the task.")
parser.add_argument("--other_dirs", type=str, default=None, help="Other directories to append to the run directory.")
parser.add_argument("--GCR", type=float, default=0.84, 
                   help="Gravity compensation ratio")
parser.add_argument("-dl", "--difficulty_level", type=int, default=-1, help="Difficulty level of the obstacle.")
parser.add_argument("--spcf", type=float, default=0.005, help="Spring Coefficient")
parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes to simulate.")

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
from ballu_isaac_extension.tasks.ballu_locomotion.mdp.geometry_utils import get_femur_dimensions, get_tibia_dimensions, get_pelvis_dimensions
from tqdm import tqdm

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # if args_cli.spcf is not None:
    #     env_cfg.scene.robot.actuators["knee_effort_actuators"].spring_coeff = args_cli.spcf

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_path = os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint_path: {checkpoint_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None,
                   GCR=args_cli.GCR, spcf=args_cli.spcf)
    
    # Shifting the env origins as per the difficulty level
    isaac_env = env.unwrapped
    print("Isaac environment: ", isaac_env)
    inter_obstacle_spacing_y = -2.0

    if args_cli.difficulty_level == -1 and agent_cfg.load_checkpoint == "model_best.pt":
        ckpt_dict = torch.load(checkpoint_path)
        args_cli.difficulty_level = ckpt_dict["best_crclm_level"]
    
    print(f"[INFO] Difficulty level: {args_cli.difficulty_level}")
    isaac_env.scene._default_env_origins = isaac_env.scene._default_env_origins + \
        torch.tensor([0.0, inter_obstacle_spacing_y, 0.0], device=isaac_env.device) * args_cli.difficulty_level
    
    # IMPORTANT: Set rsl_rl_iteration to bypass warmup period in curriculum
    # The curriculum function obstacle_height_levels_same_row has a warmup_period of 100 iterations
    # During evaluation, if this is not set, the curriculum manager won't compute updates
    isaac_env.rsl_rl_iteration = 1000  # Set to a value > warmup_period (default 100)
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

    cum_rewards = 0
    curr_crclm_state = -1
    # simulate environment
    with tqdm(total=args_cli.num_episodes * EP_LENGTH, desc="Processing") as pbar:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                #print(f"[INFO] Obstacle height levels: {isaac_env.curriculum_manager._curriculum_state['obstacle_height_levels_custom']}")
                curr_crclm_state = isaac_env.curriculum_manager._curriculum_state['obstacle_height_levels_custom']
                # env stepping
                obs, rew, _, _ = env.step(actions)
                # Store rewards for plotting
                #all_rewards.append(rewards.mean().item())  # Store the mean reward across environments
                timestep += 1
                cum_rewards += rew
            
            if timestep == args_cli.num_episodes * EP_LENGTH:
                break
            pbar.update(1)

    # close the simulator (Very very important)
    FL = get_femur_dimensions(0, side="RIGHT").height.item()
    TL = get_tibia_dimensions(0, side="RIGHT").height.item()

    env.close()
    print(f"BEST_CRCLM_LEVEL: {curr_crclm_state}")
    print(f"FL: {FL}")
    print(f"TL: {TL}")

if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        raise e
    finally:
        # close sim app
        simulation_app.close()
