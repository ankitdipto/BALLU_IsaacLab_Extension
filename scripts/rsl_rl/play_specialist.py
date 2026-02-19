"""Script to evaluate SpecialistActorCritic policies."""

import argparse
import datetime
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play specialist policy (Mixture of Specialists).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=399, help="Length of recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isc-BALLU-hetero-general-specialist", help="Task name.")
parser.add_argument("--GCR", type=float, default=0.84, help="Gravity compensation ratio.")
parser.add_argument("--GCR_range", type=float, nargs=2, default=None, help="Range of gravity compensation ratio.")
parser.add_argument("--spcf", type=float, default=0.005, help="Spring coefficient.")
parser.add_argument("--spcf_range", type=float, nargs=2, default=None, help="Range of spring coefficient.")
parser.add_argument("-dl", "--difficulty_level", type=int, default=-1, help="Obstacle difficulty level.")
parser.add_argument("--cmdir", type=str, default="tests_specialist", help="Subdirectory for outputs.")
parser.add_argument("--require_base_checkpoint", action="store_true", default=False)
parser.add_argument("--specialist_ckpt_by0spc0", type=str, default=None)
parser.add_argument("--specialist_ckpt_by0spc1", type=str, default=None)
parser.add_argument("--specialist_ckpt_by1spc0", type=str, default=None)
parser.add_argument("--specialist_ckpt_by1spc1", type=str, default=None)
parser.add_argument("--debug_reset_stats", action="store_true", default=False)
parser.add_argument("--debug_reset_interval", type=int, default=50)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + hydra_args


import gymnasium as gym
import torch
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import SpecialistActorCritic

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from evals.evaluate_obstacle_stepping_task import threshold_based_verification

import ballu_isaac_extension.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Specialist checkpoint overrides consumed by SpecialistActorCritic.__init__()
    ckpt_env_map = {
        "BALLU_SPECIALIST_CKPT_BY0SPC0": args_cli.specialist_ckpt_by0spc0,
        "BALLU_SPECIALIST_CKPT_BY0SPC1": args_cli.specialist_ckpt_by0spc1,
        "BALLU_SPECIALIST_CKPT_BY1SPC0": args_cli.specialist_ckpt_by1spc0,
        "BALLU_SPECIALIST_CKPT_BY1SPC1": args_cli.specialist_ckpt_by1spc1,
    }
    for k, v in ckpt_env_map.items():
        if v:
            os.environ[k] = v
            print(f"[INFO] Specialist checkpoint override set: {k}={v}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = agent_cfg.load_run if agent_cfg.load_run not in (None, "", ".*") else "specialist_eval"
    log_dir = os.path.join(log_root_path, run_name, args_cli.cmdir)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] log_dir: {log_dir}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
        GCR=args_cli.GCR,
        GCR_range=args_cli.GCR_range,
        spcf=args_cli.spcf,
        spcf_range=args_cli.spcf_range,
    )
    isaac_env = env.unwrapped

    inter_obstacle_spacing_y = -2.0
    if args_cli.difficulty_level == -1:
        args_cli.difficulty_level = 0
    isaac_env.scene._default_env_origins = isaac_env.scene._default_env_origins + torch.tensor(
        [0.0, inter_obstacle_spacing_y, 0.0], device=isaac_env.device
    ) * args_cli.difficulty_level

    timestamp = datetime.datetime.now().strftime("%b%d_%H_%M_%S")
    eval_folder = os.path.join(log_dir, f"{timestamp}_Ht{args_cli.difficulty_level}")
    os.makedirs(eval_folder, exist_ok=True)

    EYE = (
        1.0 - 5.5 / 1.414,
        5.5 / 1.414 + inter_obstacle_spacing_y * args_cli.difficulty_level,
        1.8,
    )
    LOOKAT = (
        1.0,
        inter_obstacle_spacing_y * args_cli.difficulty_level,
        1.0,
    )
    isaac_env.viewport_camera_controller.update_view_location(eye=EYE, lookat=LOOKAT)

    if args_cli.video:
        video_kwargs = {
            "name_prefix": "ballu",
            "video_folder": eval_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    is_specialist_policy = isinstance(ppo_runner.alg.policy, SpecialistActorCritic)
    if not is_specialist_policy:
        raise RuntimeError(
            f"Expected SpecialistActorCritic but got {type(ppo_runner.alg.policy).__name__}. "
            "Use specialist task config."
        )
    if args_cli.require_base_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Loading base checkpoint: {checkpoint_path}")
        ppo_runner.load(checkpoint_path)
    else:
        print("[INFO] Skipping base checkpoint load (specialist-only mode).")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    specialist_policy: SpecialistActorCritic = ppo_runner.alg.policy

    def get_cluster_ids() -> torch.Tensor:
        env_num_envs = env.num_envs
        gcr_t = getattr(isaac_env, "gcr_t", None)
        spcf_t = getattr(isaac_env, "spcf_t", None)
        # print(f"[DEBUG] gcr_t: {gcr_t}")
        # print(f"[DEBUG] spcf_t: {spcf_t}")

        if args_cli.GCR_range is None and args_cli.GCR is not None:
            gcr_t = torch.full((env_num_envs,), float(args_cli.GCR), device=env.unwrapped.device, dtype=torch.float32)
        elif gcr_t is None:
            gcr_t = torch.full((env_num_envs,), float(getattr(isaac_env, "GCR", 0.84)), device=env.unwrapped.device, dtype=torch.float32)
        else:
            gcr_t = gcr_t.to(env.unwrapped.device).view(-1).float()

        if args_cli.spcf_range is None and args_cli.spcf is not None:
            spcf_t = torch.full((env_num_envs,), float(args_cli.spcf), device=env.unwrapped.device, dtype=torch.float32)
        elif spcf_t is None:
            spcf_t = torch.full((env_num_envs,), float(getattr(isaac_env, "spcf", 0.005)), device=env.unwrapped.device, dtype=torch.float32)
        else:
            spcf_t = spcf_t.to(env.unwrapped.device).view(-1).float()

        return ((gcr_t >= 0.82).long() << 1) | ((spcf_t >= 0.006).long())

    obs, _ = env.get_observations()
    cluster_ids = get_cluster_ids()
    specialist_policy.forced_expert_indices = cluster_ids.clone().detach()
    # Print the histogram of cluster_ids (specialist routing histogram)
    num_specialists = int(cluster_ids.max().item()) + 1
    cluster_hist = torch.bincount(cluster_ids, minlength=num_specialists)
    print("[INFO] Specialist routing histogram:")
    for i in range(num_specialists):
        print(f"  Specialist {i}: {int(cluster_hist[i].item())} envs")
    
    print(
        f"[INFO] Specialist routing inputs (env0): GCR={float(args_cli.GCR):.4f}, "
        f"SPCF={float(args_cli.spcf):.4f}, cluster_id={int(cluster_ids[0].item())}"
    )

    total_done, total_timeout, total_terminated = 0, 0, 0
    timestep = 0
    with tqdm(total=args_cli.video_length, desc="Processing") as pbar:
        while simulation_app.is_running():
            with torch.inference_mode():
                cluster_ids = get_cluster_ids()
                specialist_policy.forced_expert_indices = cluster_ids.clone().detach()
                actions = policy(obs)
                obs, rew, dones, infos = env.step(actions)
                timestep += 1

            if timestep >= args_cli.video_length:
                break
            pbar.update(1)

    successes, _ = threshold_based_verification(env.unwrapped, threshold_x=0.6)
    success_rate = successes.float().mean().item()
    env.close()

    expert_ids = specialist_policy.current_expert_indices.detach().cpu() if specialist_policy.current_expert_indices is not None else None
    if expert_ids is not None:
        expert_path = os.path.join(eval_folder, "specialist_indices.txt")
        with open(expert_path, "w") as f:
            f.write("# Mapping: 0=by0spc0, 1=by0spc1, 2=by1spc0, 3=by1spc1\n")
            for env_idx, exp_idx in enumerate(expert_ids.numpy()):
                f.write(f"{env_idx}, {exp_idx}\n")
        print(f"[INFO] Saved specialist indices to: {expert_path}")
        print("\n=== Specialist Utilization ===")
        for sp_id in range(4):
            count = int((expert_ids == sp_id).sum().item())
            pct = 100.0 * count / len(expert_ids)
            print(f"  Specialist {sp_id}: {count} envs ({pct:.1f}%)")

    print(f"SUCCESS_RATE: {success_rate:.6f}")
    if args_cli.debug_reset_stats:
        print(
            f"[RESET_DEBUG_SUMMARY] total_done={total_done} "
            f"total_timeout={total_timeout} total_non_timeout={total_terminated}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
