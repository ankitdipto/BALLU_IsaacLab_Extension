"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime

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
parser.add_argument("--other_dirs", type=str, default=None, help="Other directories to append to the run directory.")

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
import matplotlib.pyplot as plt
#from isaacsim.core.utils.transformations import transform_points
#from isaacsim.debug_draw import _debug_draw as debug_draw

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401


def plot_joint_data(joint_pos_hist_tch, joint_vel_hist_tch, joint_names, num_envs, save_dir):
    """Plot joint positions and velocities for each environment.
    
    Args:
        joint_pos_hist_tch: Tensor of joint positions history [timesteps, envs, joints]
        joint_vel_hist_tch: Tensor of joint velocities history [timesteps, envs, joints]
        joint_names: List of joint names
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    for env_idx in range(num_envs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Position plot
        for joint_idx in range(joint_pos_hist_tch.shape[2]):
            ax1.plot(joint_pos_hist_tch[:, env_idx, joint_idx], label=f'{joint_names[joint_idx]}')
        ax1.set_title(f'Joint Positions Over Time (Env {env_idx})')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Position (rad)')
        ax1.legend()
        ax1.grid(True)
        
        # Velocity plot
        for joint_idx in range(joint_vel_hist_tch.shape[2]):
            ax2.plot(joint_vel_hist_tch[:, env_idx, joint_idx], label=f'{joint_names[joint_idx]}')
        ax2.set_title(f'Joint Velocities Over Time (Env {env_idx})')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"joint_plots_env_{env_idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Saved joint plots for environment {env_idx} to {plot_path}")


def plot_root_com_xy(root_com_xyz_hist_tch, num_envs, save_dir):
    """Plot root COM X,Y positions for all environments in a single figure.
    
    Args:
        root_com_xyz_hist_tch: Tensor of root COM XYZ history [timesteps, envs, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Calculate figure size based on number of environments
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    fig_size = max(8, grid_size * 4)  # Minimum 8 inches, scale with env count
    
    plt.figure(figsize=(fig_size, fig_size))
        
    # Extract all X,Y coordinates
    all_xy = np.array([[pos[env_idx][:2] for pos in root_com_xyz_hist_tch] 
                      for env_idx in range(num_envs)])
        
    # Plot each environment's trajectory (swap axes, invert x for Y-left, X-up)
    for env_idx in range(num_envs):
        plt.plot(all_xy[env_idx][:, 1], all_xy[env_idx][:, 0], 
                label=f'Env {env_idx}', alpha=0.7)
    plt.gca().invert_xaxis()  # Invert X so positive Y points left
    
    plt.title(f'Root COM (Y,X) Positions (X↑, Y←) (All {num_envs} Environments)')
    plt.xlabel('Y Position')  # Now horizontal, but inverted
    plt.ylabel('X Position')  # Now vertical
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
        
    plot_path = os.path.join(save_dir, "root_com_xy_positions_all.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved combined root COM XY plot to {plot_path}")


def plot_feet_heights(left_foot_pos_history, right_foot_pos_history, num_envs, save_dir):
    """Plot feet heights (Z) over time for all environments as subplots in a single figure.
    
    Args:
        left_foot_pos_history: List of left foot position tensors
        right_foot_pos_history: List of right foot position tensors
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Skip if foot position history is empty or contains None values
    if not left_foot_pos_history or not right_foot_pos_history or None in left_foot_pos_history or None in right_foot_pos_history:
        print("[INFO] Skipping feet height plots due to missing data")
        return
        
    left_foot_np = torch.cat(left_foot_pos_history, dim=0).cpu().numpy()  # shape: (timesteps, envs, 3)
    right_foot_np = torch.cat(right_foot_pos_history, dim=0).cpu().numpy()
    timesteps = left_foot_np.shape[0]

    # Dynamic figure size: width=5 per env, height=3 per env (up to a reasonable max)
    fig_width = max(8, min(5 * num_envs, 30))
    fig_height = max(3, min(2 * num_envs, 20))
    fig, axs = plt.subplots(num_envs, 1, figsize=(fig_width, fig_height), squeeze=False)
    
    for env_idx in range(num_envs):
        ax = axs[env_idx, 0]
        ax.plot(range(timesteps), left_foot_np[:, env_idx, 2], label='Left Foot', color='blue', alpha=0.7)
        ax.plot(range(timesteps), right_foot_np[:, env_idx, 2], label='Right Foot', color='red', alpha=0.7)
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Env {env_idx}')
        ax.grid(True)
        ax.legend()
    
    axs[-1, 0].set_xlabel('Timestep')
    plt.suptitle('Feet Heights (Z) Over Time (All Environments)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    feet_plot_path = os.path.join(save_dir, 'feet_heights_z_all_envs.png')
    plt.savefig(feet_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved feet heights Z plot to {feet_plot_path}")


def plot_base_velocity(base_vel_hist_tch, num_envs, save_dir):
    """Plot base velocity over time for all environments as subplots in a single figure.
    
    Args:
        base_vel_hist_tch: Tensor of base velocity history [timesteps, envs, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Convert to numpy for plotting
    base_vel_np = base_vel_hist_tch.cpu().numpy()  # shape: (timesteps, envs, 3)
    timesteps = base_vel_np.shape[0]
    
    # Component labels
    component_labels = ['X (Forward)', 'Y (Lateral)', 'Z (Vertical)']
    component_colors = ['#4878CF', '#6ACC65', '#D65F5F']  # Muted blue, green, red
    
    # Calculate grid dimensions for a balanced layout
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))
    
    # Dynamic figure size: based on grid dimensions
    fig_width = max(4 * n_cols, 8)  # At least 4 inches per column, minimum 8 inches
    fig_height = max(3 * n_rows, 6)  # At least 3 inches per row, minimum 6 inches
    
    # Create figure with subplots in a grid layout
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Plot each environment in a separate subplot
    env_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if env_idx < num_envs:
                ax = axs[row, col]
                
                # Plot each velocity component
                for comp_idx in range(3):
                    ax.plot(
                        range(timesteps), 
                        base_vel_np[:, env_idx, comp_idx], 
                        label=component_labels[comp_idx], 
                        color=component_colors[comp_idx], 
                        alpha=0.7,  # Slightly reduce opacity
                        linewidth=1.5  # Slightly thicker lines for better visibility with muted colors
                    )
                
                ax.set_ylabel('Velocity (m/s)')
                ax.set_xlabel('Timestep')
                ax.set_title(f'Env {env_idx}')
                ax.grid(True)
                
                # Only add legend to the first subplot to save space
                if row == 0 and col == 0:
                    ax.legend(loc='upper right')
                
                env_idx += 1
            else:
                # Hide unused subplots
                axs[row, col].set_visible(False)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3)
    
    # Remove the individual legend from the first subplot since we have a common one
    if axs[0, 0].get_legend():
        axs[0, 0].get_legend().remove()
    
    # Add overall title with more space
    plt.suptitle('Base Velocity Over Time (All Environments)', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Increase top margin for title and legend
    
    # Save the figure
    vel_plot_path = os.path.join(save_dir, 'base_velocity_all_envs.png')
    plt.savefig(vel_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved base velocity plot to {vel_plot_path}")


def plot_knee_phase_portraits(joint_pos_hist_tch, joint_vel_hist_tch, joint_names, num_envs, save_dir):
    """Create phase portraits (position vs. velocity) for knee joints across all environments.
    
    Args:
        joint_pos_hist_tch: Tensor of joint positions history [timesteps, envs, num_joints]
        joint_vel_hist_tch: Tensor of joint velocities history [timesteps, envs, num_joints]
        joint_names: List of joint names
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Convert to numpy for plotting
    joint_pos_np = joint_pos_hist_tch.cpu().numpy()  # shape: (timesteps, envs, num_joints)
    joint_vel_np = joint_vel_hist_tch.cpu().numpy()  # shape: (timesteps, envs, num_joints)
    
    # Find indices for knee joints
    knee_indices = []
    knee_names = []
    for i, name in enumerate(joint_names):
        if "KNEE_LEFT" in name or "KNEE_RIGHT" in name:
            knee_indices.append(i)
            knee_names.append(name)
    
    if not knee_indices:
        print("[INFO] No knee joints found, skipping phase portrait")
        return
    
    # We need one row per environment and two columns (left and right knee)
    n_rows = num_envs
    n_cols = len(knee_indices)  # Should be 2 for left and right knees
    
    # Create figure
    fig_width = max(12, 6 * n_cols)
    fig_height = max(10, 3 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Colors for different environments - use a different color for each environment
    env_colors = plt.cm.viridis(np.linspace(0, 1, num_envs))
    
    # Plot phase portraits for each environment and knee joint
    for env_idx in range(num_envs):
        for col, (joint_idx, joint_name) in enumerate(zip(knee_indices, knee_names)):
            # Get position and velocity data for this joint and environment
            pos = joint_pos_np[:, env_idx, joint_idx]
            vel = joint_vel_np[:, env_idx, joint_idx]
            
            # Plot phase portrait
            axs[env_idx, col].plot(pos, vel, color=env_colors[env_idx], alpha=0.8)
            axs[env_idx, col].set_xlabel('Joint Position (rad)')
            axs[env_idx, col].set_ylabel('Joint Velocity (rad/s)')
            axs[env_idx, col].set_title(f'{joint_name} - Env {env_idx}')
            axs[env_idx, col].grid(True)
            
            # Add arrows to show direction of trajectory
            n_arrows = 10
            step = max(1, len(pos) // n_arrows)
            for i in range(0, len(pos) - 1, step):
                if i + 1 < len(pos):  # Ensure we don't go out of bounds
                    axs[env_idx, col].annotate('', 
                        xytext=(pos[i], vel[i]),
                        xy=(pos[i+1], vel[i+1]), 
                        arrowprops=dict(arrowstyle='->', color=env_colors[env_idx], lw=1.5),
                        annotation_clip=True)
    
    plt.suptitle('Knee Joint Phase Portraits (Position vs. Velocity)', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
    
    # Save the figure
    phase_plot_path = os.path.join(save_dir, 'knee_phase_portraits.png')
    plt.savefig(phase_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved knee joint phase portraits to {phase_plot_path}")


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
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, other_dirs=args_cli.other_dirs)
    log_dir = os.path.dirname(resume_path)

    timestamp = datetime.datetime.now().strftime('%b%d_%H_%M_%S') # Format: Apr08_19_25_38

    # create debug directory for this run
    play_folder = os.path.join(log_dir, "play", timestamp, f"{agent_cfg.load_checkpoint[:-3]}")
    os.makedirs(play_folder, exist_ok=True)
    print(f"[INFO] Saving plots to: {play_folder}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    
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

    # note: export policy to onnx/jit is disabled as of now
    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    joint_pos_history = []
    joint_vel_history = []
    root_com_xyz_history = []
    left_foot_pos_history = []
    right_foot_pos_history = []
    base_vel_history = []
    #all_rewards = []

    # Get robots_data and tibia indices once before simulation loop
    robots = env.unwrapped.scene["robot"]
    left_tibia_indices = robots.find_bodies("TIBIA_LEFT")
    right_tibia_indices = robots.find_bodies("TIBIA_RIGHT")
    left_tibia_idx = left_tibia_indices[0] if len(left_tibia_indices) > 0 else None
    right_tibia_idx = right_tibia_indices[0] if len(right_tibia_indices) > 0 else None
    left_foot_pos = None
    right_foot_pos = None

    # Acquire DebugDraw interface
    # debug_draw_instance = debug_draw.acquire_debug_draw_interface()

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
            timestep += 1
            # Extract robot's joint positions and joint velocities
            robots_data = env.unwrapped.scene["robot"].data
            joint_pos = robots_data.joint_pos.clone().detach().cpu()
            joint_vel = robots_data.joint_vel.clone().detach().cpu()
            root_com_xyz = robots_data.root_com_state_w.detach().cpu()[..., :3]
            body_states = robots_data.body_link_state_w.clone().detach().cpu()
            base_vel = robots_data.root_lin_vel_b.clone().detach().cpu()
            # Print body names once for debugging
            if timestep == 1:
                print("Available body names:", robots_data.body_names)

            # Extract tibia positions and calculate foot positions
            # if left_tibia_idx is not None:
            #     left_tibia_pos = body_states[:, left_tibia_idx, :3]
            #     left_tibia_quat = body_states[:, left_tibia_idx, 3:7]
            #     foot_offset = torch.tensor([0.0, 0.38485, 0.0], device=left_tibia_pos.device).unsqueeze(0).expand(left_tibia_pos.shape)
            #     #pose = torch.cat([left_tibia_pos, left_tibia_quat], dim=-1)
            #     left_foot_pos = transform_points(
            #         left_tibia_pos, 
            #         left_tibia_quat,
            #         foot_offset)
            # if right_tibia_idx is not None:
            #     right_tibia_pos = body_states[:, right_tibia_idx, :3]
            #     right_tibia_quat = body_states[:, right_tibia_idx, 3:7]
            #     foot_offset = torch.tensor([0.0, 0.38485, 0.0], device=right_tibia_pos.device).unsqueeze(0).expand(right_tibia_pos.shape)
            #     #pose = torch.cat([right_tibia_pos, right_tibia_quat], dim=-1)
            #     right_foot_pos = transform_points(
            #         right_tibia_pos, 
            #         right_tibia_quat,
            #         foot_offset)

            # --- DebugDraw visualization for feet ---
            # if left_foot_pos is not None:
            #     for env_idx in range(left_foot_pos.shape[0]):
            #         pos = left_foot_pos[env_idx].cpu().numpy().tolist()
            #         debug_draw_instance.draw_sphere(
            #             position=pos,
            #             color=[1.0, 0.5, 0.0, 1.0],  # Orange RGBA
            #             radius=0.025
            #         )
            # if right_foot_pos is not None:
            #     for env_idx in range(right_foot_pos.shape[0]):
            #         pos = right_foot_pos[env_idx].cpu().numpy().tolist()
            #         debug_draw_instance.draw_sphere(
            #             position=pos,
            #             color=[0.0, 0.8, 0.0, 1.0],  # Green RGBA
            #             radius=0.025
            #         )
            # Store positions
            left_foot_pos_history.append(left_foot_pos)
            right_foot_pos_history.append(right_foot_pos)
            # Store joint positions and velocities
            joint_pos_history.append(joint_pos)
            joint_vel_history.append(joint_vel)
            root_com_xyz_history.append(root_com_xyz)
            base_vel_history.append(base_vel)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator (Very very important)
    env.close()

    # Process data for plotting
    joint_pos_hist_tch = torch.stack(joint_pos_history)
    joint_vel_hist_tch = torch.stack(joint_vel_history)
    root_com_xyz_hist_tch = torch.stack(root_com_xyz_history)
    base_vel_hist_tch = torch.stack(base_vel_history)
    
    # Generate plots
    plot_joint_data(joint_pos_hist_tch, joint_vel_hist_tch, robots_data.joint_names, env.num_envs, play_folder)
    plot_root_com_xy(root_com_xyz_hist_tch, env.num_envs, play_folder)
    plot_feet_heights(left_foot_pos_history, right_foot_pos_history, env.num_envs, play_folder)
    plot_base_velocity(base_vel_hist_tch, env.num_envs, play_folder)
    plot_knee_phase_portraits(joint_pos_hist_tch, joint_vel_hist_tch, robots_data.joint_names, env.num_envs, play_folder)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
