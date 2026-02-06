"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=399, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isc-BALLU-hetero-general", help="Name of the task.")
parser.add_argument("--other_dirs", type=str, default=None, help="Other directories to append to the run directory.")
parser.add_argument("--GCR", type=float, default=0.84, 
                   help="Gravity compensation ratio")
parser.add_argument("--spcf", type=float, default=None, help="Spring Coefficient")
parser.add_argument("-dl", "--difficulty_level", type=int, default=-1, help="Difficulty level of the obstacle.")
parser.add_argument("--cmdir", type=str, required=True, help="Name of the common directory.")

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
import pandas as pd

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import MoEActorCritic

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from plotters import (
    plot_joint_data, 
    plot_root_com_xy, 
    plot_feet_heights, 
    plot_base_velocity, 
    plot_knee_phase_portraits, 
    plot_toe_heights,
    plot_local_positions_scatter
)
from evals.evaluate_obstacle_stepping_task import threshold_based_verification
#from ..scratchpad.action_generators import stepper
#from isaacsim.core.utils.transformations import transform_points
#from isaacsim.debug_draw import _debug_draw as debug_draw

# Import extensions to set up environment tasks
import ballu_isaac_extension.tasks  # noqa: F401
from ballu_isaac_extension.tasks.ballu_locomotion.mdp.geometry_utils import get_femur_dimensions, get_tibia_dimensions, get_pelvis_dimensions

import isaaclab.utils.math as math_utils
from tqdm import tqdm

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_path = os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.join(log_root_path, agent_cfg.load_run, args_cli.cmdir)
    print(f"[INFO] checkpoint_path: {checkpoint_path}")
    print(f"[INFO] log_dir: {log_dir}")


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None,
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

    spcf = isaac_env.scene["robot"].actuators["knee_effort_actuators"].spring_coeff[0][0].item()
    GCR = isaac_env.GCR
    timestamp = datetime.datetime.now().strftime('%b%d_%H_%M_%S') # Format: Apr08_19_25_38
    FL = get_femur_dimensions(0, side="RIGHT").height.item()
    TL = get_tibia_dimensions(0, side="RIGHT").height.item()
    HL = get_pelvis_dimensions(0).height.item()
    # create debug directory for this run
    eval_folder = os.path.join(log_dir, f"{timestamp}_fl{FL:.3f}_tl{TL:.3f}_hl{HL:.3f}_gcr{GCR:.3f}_spcf{spcf:.4f}_Ht{args_cli.difficulty_level}")
    os.makedirs(eval_folder, exist_ok=True)
    print(f"[INFO] Saving plots to: {eval_folder}")

    EYE = (
        1.0 - 5.5/1.414, 
        5.5/1.414 + inter_obstacle_spacing_y * args_cli.difficulty_level, 
        1.8
    )
    LOOKAT = (
        1.0,
        inter_obstacle_spacing_y * args_cli.difficulty_level,
        1.0
    )

    isaac_env.viewport_camera_controller.update_view_location(eye=EYE, lookat=LOOKAT)
    # wrap for video recording
    
    if args_cli.video:
        video_kwargs = {
            "name_prefix": 'ballu',  
            "video_folder": eval_folder,
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

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    # Check if using MoE policy
    is_moe_policy = isinstance(ppo_runner.alg.policy, MoEActorCritic)
    if is_moe_policy:
        print("[INFO] MoE policy detected - will log expert indices")
        moe_policy = ppo_runner.alg.policy
        # Initialize expert tracking for the test environments
        moe_policy.init_expert_tracking(env.num_envs, env.unwrapped.device)

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
    toe_endpoints_world_history = []  # list of tensors (num_envs, 2, 3) per step
    base_vel_history = []
    actions_history = []
    comp_torq_history = []
    applied_torq_history = []
    contact_forces_tibia_history = []
    expert_indices_history = []  # For MoE policy
    expert_gate_probs_history = []
    #all_rewards = []

    # Get robots_data and tibia indices once before simulation loop
    robots = env.unwrapped.scene["robot"]
    left_tibia_indices = robots.find_bodies("ELECTRONICS_LEFT")
    right_tibia_indices = robots.find_bodies("ELECTRONICS_RIGHT")
    left_tibia_idx = left_tibia_indices[0] if len(left_tibia_indices) > 0 else None
    right_tibia_idx = right_tibia_indices[0] if len(right_tibia_indices) > 0 else None
    left_foot_pos = None
    right_foot_pos = None

    # Acquire DebugDraw interface
    # debug_draw_instance = debug_draw.acquire_debug_draw_interface()

    cum_rewards = 0
    robot_jnt_names = []
    # simulate environment
    with tqdm(total=args_cli.video_length, desc="Processing") as pbar:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                
                # Capture expert indices for MoE policy (before env step, after policy forward)
                if is_moe_policy and timestep == 0:
                    # Expert indices are selected on first step and remain fixed per episode
                    expert_indices_history.append(moe_policy.current_expert_indices.clone().detach().cpu())
                    expert_gate_probs_history.append(moe_policy._last_gate_probs.clone().detach().cpu())
                
                # actions = stepper(timestep,
                #                   period=40,
                #                   num_envs=env.num_envs)
                # env stepping
                obs, rew, _, _ = env.step(actions)
                # Store rewards for plotting
                #all_rewards.append(rewards.mean().item())  # Store the mean reward across environments
                timestep += 1
                cum_rewards += rew
                # Extract robot's joint positions and joint velocities
                robots_data = env.unwrapped.scene["robot"].data
                joint_pos = robots_data.joint_pos.clone().detach().cpu()
                joint_vel = robots_data.joint_vel.clone().detach().cpu()
                root_com_xyz = robots_data.root_com_state_w.detach().cpu()[..., :3]
                # body_states = robots_data.body_link_state_w.clone().detach().cpu()
                base_vel = robots_data.root_lin_vel_b.clone().detach().cpu()
                comp_torq = robots_data.computed_torque.clone().detach().cpu()
                applied_torq = robots_data.applied_torque.clone().detach().cpu()
                contact_forces_tibia = isaac_env.scene["contact_forces_tibia"].data.net_forces_w.clone().detach().cpu()
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                # print("Torques from play.py")
                # print(f"Computed torque: {comp_torq.cpu().numpy()}")
                # print(f"Applied torque: {applied_torq.cpu().numpy()}")
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                # Print body names once for debugging
                if timestep == 1:
                    print("Available body names:", robots_data.body_names)
                    robot_jnt_names = robots_data.joint_names
                # Extract toe endpoints (world) from tibia link poses if indices available
                if left_tibia_idx is not None and right_tibia_idx is not None:
                    link_pos_w = robots.data.body_link_pos_w  # (num_envs, num_bodies, 3)
                    link_quat_w = robots.data.body_link_quat_w  # (num_envs, num_bodies, 4) wxyz
                    tibia_pos_w = torch.stack([
                        link_pos_w[:, left_tibia_idx, :],
                        link_pos_w[:, right_tibia_idx, :]
                    ], dim=1)  # (num_envs, 2, 3)
                    tibia_quat_w = torch.stack([
                        link_quat_w[:, left_tibia_idx, :],
                        link_quat_w[:, right_tibia_idx, :]
                    ], dim=1)  # (num_envs, 2, 4)
                    foot_offset_b = torch.tensor([0.0, 0.06 + 0.004, 0.0], device=tibia_pos_w.device, dtype=tibia_pos_w.dtype)
                    foot_offset_b = foot_offset_b.unsqueeze(0).unsqueeze(0).expand(tibia_pos_w.shape)
                    rot_offset_w = math_utils.quat_apply(tibia_quat_w.reshape(-1, 4), foot_offset_b.reshape(-1, 3)).reshape_as(tibia_pos_w)
                    toe_endpoints_w = tibia_pos_w + rot_offset_w  # (num_envs, 2, 3)
                    toe_endpoints_world_history.append(toe_endpoints_w.detach().cpu())
                    # Keep backward-compatible single foot positions for any other plots if needed
                    left_foot_pos = toe_endpoints_w[:, 0, :]
                    right_foot_pos = toe_endpoints_w[:, 1, :]

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
                # Store positions (ensure shape [1, envs, 3] per step for plotting)
                if left_foot_pos is not None and right_foot_pos is not None:
                    left_foot_pos_history.append(left_foot_pos.unsqueeze(0))
                    right_foot_pos_history.append(right_foot_pos.unsqueeze(0))
                else:
                    left_foot_pos_history.append(None)
                    right_foot_pos_history.append(None)
                # Store joint positions and velocities
                joint_pos_history.append(joint_pos)
                joint_vel_history.append(joint_vel)
                root_com_xyz_history.append(root_com_xyz)
                base_vel_history.append(base_vel)
                actions_history.append(actions)
                comp_torq_history.append(comp_torq)
                applied_torq_history.append(applied_torq)
                contact_forces_tibia_history.append(contact_forces_tibia)
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            # else:
            #     if timestep == env.unwrapped.max_episode_length:
            #         break
            #if timestep == 400: # TODO: Remove this
            #    break
            pbar.update(1)

    # Before closing the simulator evaluate the performance
    successes, local_positions = threshold_based_verification(env.unwrapped, threshold_x = 0.6)
    success_rate = successes.float().mean().item()
    # close the simulator (Very very important)
    env.close()

    # Process data for plotting
    joint_pos_hist_tch = torch.stack(joint_pos_history)
    joint_vel_hist_tch = torch.stack(joint_vel_history)
    root_com_xyz_hist_tch = torch.stack(root_com_xyz_history)
    base_vel_hist_tch = torch.stack(base_vel_history)
    actions_hist_tch = torch.stack(actions_history)
    comp_torq_hist_tch = torch.stack(comp_torq_history)
    applied_torq_hist_tch = torch.stack(applied_torq_history)
    # Toe endpoints history if available
    toe_endpoints_world_hist_tch = torch.stack(toe_endpoints_world_history)
    contact_forces_tibia_hist_tch = torch.stack(contact_forces_tibia_history)
    # tibia_endpoints_hist_tch = None
    #if len(toe_endpoints_world_history) > 0:

    # Generate plots
    plot_joint_data(joint_pos_hist_tch, joint_vel_hist_tch, robots_data.joint_names, env.num_envs, eval_folder)
    plot_root_com_xy(root_com_xyz_hist_tch, env.num_envs, eval_folder)
    # plot_feet_heights(left_foot_pos_history, right_foot_pos_history, env.num_envs, play_folder)
    plot_base_velocity(base_vel_hist_tch, env.num_envs, eval_folder)
    # plot_knee_phase_portraits(joint_pos_hist_tch, joint_vel_hist_tch, robots_data.joint_names, env.num_envs, eval_folder)
    plot_toe_heights(toe_endpoints_world_hist_tch, env.num_envs, eval_folder)
    plot_local_positions_scatter(local_positions, eval_folder, threshold_x=1.0, success_rate=success_rate)
    print("Plotting complete.")
    # Save data to CSV file for env_idx=0
    env_idx = 0
    
    # Extract data for env_idx=0
    actions_data = actions_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 2)
    joint_pos_data = joint_pos_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 7)
    root_com_data = root_com_xyz_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 3)
    base_vel_data = base_vel_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 3)
    comp_torq_data = comp_torq_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 7)
    applied_torq_data = applied_torq_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 7)
    contact_forces_tibia_data = contact_forces_tibia_hist_tch[:, env_idx, :].cpu().numpy()  # Shape: (num_timesteps, 2, 3)
    
    left_knee_idx = robot_jnt_names.index("KNEE_LEFT")
    right_knee_idx = robot_jnt_names.index("KNEE_RIGHT")

    # Create DataFrame with specified headers
    csv_data = {
        'ACT_LEFT': actions_data[:, 0],
        'ACT_RIGHT': actions_data[:, 1],
        'HIP_LEFT': joint_pos_data[:, 0],
        'HIP_RIGHT': joint_pos_data[:, 1],
        'NECK': joint_pos_data[:, 2],
        'KNEE_LEFT': joint_pos_data[:, 3],
        'KNEE_RIGHT': joint_pos_data[:, 4],
        'MOTOR_LEFT': joint_pos_data[:, 5],
        'MOTOR_RIGHT': joint_pos_data[:, 6],
        'POS_X': root_com_data[:, 0],
        'POS_Y': root_com_data[:, 1],
        'POS_Z': root_com_data[:, 2],
        'VEL_X': base_vel_data[:, 0],
        'VEL_Y': base_vel_data[:, 1],
        'VEL_Z': base_vel_data[:, 2],
        'COMP_TORQ_LEFT_KNEE': comp_torq_data[:, left_knee_idx],
        'COMP_TORQ_RIGHT_KNEE': comp_torq_data[:, right_knee_idx],
        'APPLIED_TORQ_LEFT_KNEE': applied_torq_data[:, left_knee_idx],
        'APPLIED_TORQ_RIGHT_KNEE': applied_torq_data[:, right_knee_idx],
        'CONTACT_FORCE_LEFT_X': contact_forces_tibia_data[:, 0, 0],
        'CONTACT_FORCE_LEFT_Y': contact_forces_tibia_data[:, 0, 1],
        'CONTACT_FORCE_LEFT_Z': contact_forces_tibia_data[:, 0, 2],
        'CONTACT_FORCE_RIGHT_X': contact_forces_tibia_data[:, 1, 0],
        'CONTACT_FORCE_RIGHT_Y': contact_forces_tibia_data[:, 1, 1],
        'CONTACT_FORCE_RIGHT_Z': contact_forces_tibia_data[:, 1, 2],
    }
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(eval_folder, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved data to CSV file: {csv_path}")
    
    # Save MoE expert indices to text file
    if is_moe_policy and len(expert_indices_history) > 0:
        expert_indices = expert_indices_history[0]  # Shape: (num_envs,)
        expert_gate_probs = expert_gate_probs_history[0]  # Shape: (num_envs, num_experts)
        expert_indices_path = os.path.join(eval_folder, "expert_indices.txt")
        with open(expert_indices_path, 'w') as f:
            f.write("# MoE Expert Indices for Test Environments\n")
            f.write(f"# Number of environments: {len(expert_indices)}\n")
            f.write(f"# Number of experts: {moe_policy.num_experts}\n")
            f.write(f"# Gumbel temperature at inference: {moe_policy.tau:.4f}\n")
            f.write("#\n")
            f.write("# env_idx, expert_idx\n")
            for env_idx, exp_idx in enumerate(expert_indices.numpy()):
                f.write(f"{env_idx}, {exp_idx}\n")
            f.write("\n# === Expert Gate Probs ===\n")
            for env_idx, gate_probs in enumerate(expert_gate_probs.numpy()):
                f.write(f"# Environment {env_idx}: {gate_probs}\n")
            # Summary statistics
            f.write("\n# === Summary ===\n")
            for exp_id in range(moe_policy.num_experts):
                count = (expert_indices == exp_id).sum().item()
                pct = 100 * count / len(expert_indices)
                f.write(f"# Expert {exp_id}: {count} envs ({pct:.1f}%)\n")
        
        print(f"[INFO] Saved MoE expert indices to: {expert_indices_path}")
    
    base_vel_mean = base_vel_hist_tch.mean(dim=0)
    base_vel_std = base_vel_hist_tch.std(dim=0)
    print("base_vel_mean of RL policy: ", base_vel_mean)
    print("base_vel_std of RL policy: ", base_vel_std)
    print("cumulative rewards of RL policy: ", cum_rewards)
    print("success rate of RL policy: ", success_rate)
    print(f"SUCCESS_RATE: {success_rate:.6f}")  # Parseable format for automated scripts
    
    # Print MoE expert utilization summary
    if is_moe_policy and len(expert_indices_history) > 0:
        expert_indices = expert_indices_history[0]
        print("\n=== MoE Expert Utilization ===")
        for exp_id in range(moe_policy.num_experts):
            count = (expert_indices == exp_id).sum().item()
            pct = 100 * count / len(expert_indices)
            print(f"  Expert {exp_id}: {count} envs ({pct:.1f}%)")

if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        raise e
    finally:
        # close sim app
        simulation_app.close()
