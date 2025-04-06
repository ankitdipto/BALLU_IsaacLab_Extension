"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import numpy as np

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    #all_rewards = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # Store rewards for plotting
            #all_rewards.append(rewards.mean().item())  # Store the mean reward across environments
            #timestep += 1

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        #if timestep == env.max_episode_length:
        #    break

    #print("Printing root linear velocity history:")
    # The wrapper might be hiding the actual environment, so we need to access the unwrapped version
    #root_vel_history = env.unwrapped.root_lin_vel_history
    #print(f"Number of history entries: {len(root_vel_history)}")

    # close the simulator (Very very important)
    env.close()

    # Plot the velocity history
    #import matplotlib.pyplot as plt

    # Stack all tensors in the history
    #stacked_history = torch.stack(root_vel_history)
    # Calculate average velocity for each environment
    #avg_velocity = stacked_history.mean(dim=0)
    # print("Average velocity for each environment:")
    # for i in range(avg_velocity.shape[0]):
    #     vx, vy, vz = avg_velocity[i].cpu().tolist()
    #     magnitude = torch.norm(avg_velocity[i]).item()
    #     print(f"Environment {i}: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}, magnitude={magnitude:.4f}")
    # Convert to numpy for plotting
    #history_np = stacked_history.cpu().numpy()

    # Create plot for each environment
    #plt.figure(figsize=(12, 8))
    # Plot for first environment (you can loop for all if needed)
    #env_idx = 0  # Choose which environment to plot

    # Plot x, y, z components of velocity
    # plt.subplot(3, 1, 1)
    # plt.plot(history_np[:, env_idx, 0])
    # plt.title(f'X Linear Velocity - Env {env_idx}')
    # plt.grid(True)

    # plt.subplot(3, 1, 2)
    # plt.plot(history_np[:, env_idx, 1])
    # plt.title(f'Y Linear Velocity - Env {env_idx}')
    # plt.grid(True)

    # plt.subplot(3, 1, 3)
    # plt.plot(history_np[:, env_idx, 2])
    # plt.title(f'Z Linear Velocity - Env {env_idx}')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(f"debug/root_linear_velocity_env_{env_idx}_3_maj_chg.png")
    # print(f"Saved velocity plot to debug/root_linear_velocity_env_{env_idx}_3_maj_chg.png")

    # import matplotlib.pyplot as plt
                
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(all_rewards)), all_rewards)
    # plt.xlabel("Timestep")
    # plt.ylabel("Reward")
    # plt.title("Rewards per Timestep during Playing")
    # plt.grid(True)
    # plt.savefig("debug/rewards_plot_curr_best_policy.png")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
