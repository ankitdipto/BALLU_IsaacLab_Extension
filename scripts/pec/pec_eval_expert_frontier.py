"""
PEC Expert Frontier Evaluation Script
======================================
Progressive Expert Coverage — batched frontier evaluation for a single expert.

Loads one expert checkpoint and evaluates it across ALL frontier candidate
designs in a single Isaac Sim process.  Each candidate design gets its own
parallel simulation environment (num_envs = number of candidates).

The curriculum runs independently per environment: after N episodes each
environment's Y-origin encodes the highest obstacle difficulty the expert
could clear for that design.  We track the running-maximum level index per
environment throughout the evaluation so that curriculum oscillations do not
hide the true performance ceiling.

Output
------
A JSON file at ``--output`` containing per-design results, plus a parseable
summary line printed to stdout:

    FRONTIER_RESULTS: <json string>

This script is called once per expert by ``pec_evaluate_frontier.py`` (Step 2
orchestrator).  The orchestrator collects all K result files and assembles the
scores matrix fed to ``pec_refit_gaussians.py`` (Step 3).

Usage (run from ballu_isclb_extension/)
-----------------------------------------
    python scripts/pec/pec_eval_expert_frontier.py \\
        --checkpoint_path  /abs/path/to/expert_0/model_best.pt \\
        --frontier_file    logs/pec/my_run/frontier_evals/iter_1_candidates.json \\
        --output           logs/pec/my_run/frontier_evals/iter_1_expert_0_results.json \\
        --num_episodes     30 \\
        --start_difficulty 15 \\
        --task             Isc-BALLU-hetero-general \\
        --headless
"""

"""Launch Isaac Sim first."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

import cli_args

# ── Argument parsing (must happen before AppLauncher) ─────────────────────────
parser = argparse.ArgumentParser(
    description="PEC: evaluate one expert across a batch of frontier designs."
)

# PEC-specific args
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="Absolute path to the expert's model_best.pt checkpoint.")
parser.add_argument("--frontier_file", type=str, required=True,
                    help="JSON file with frontier candidates: "
                         "[{id, GCR, spcf}, ...]")
parser.add_argument("--output", type=str, required=True,
                    help="Path to write the per-design results JSON.")
parser.add_argument("--num_episodes", type=int, default=30,
                    help="Episodes to run per design (curriculum discovers ceiling). "
                         "Default: 30.")
parser.add_argument("--start_difficulty", type=int, default=15,
                    help="Obstacle level index to start from (0=flat, 15≈15cm). "
                         "Default: 15.")
parser.add_argument("--task", type=str, default="Isc-BALLU-hetero-general",
                    help="Isaac Lab task name. Default: Isc-BALLU-hetero-general.")

# RSL-RL args (needed so @hydra_task_config can load the agent config)
cli_args.add_rsl_rl_args(parser)
# AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# ── Launch Isaac Sim ──────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import torch
import gymnasium as gym
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnvCfg,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import ballu_isaac_extension.tasks  # noqa: F401


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    # ── Load frontier candidates ──────────────────────────────────────────────
    with open(args_cli.frontier_file) as f:
        candidates = json.load(f)   # [{id, GCR, spcf}, ...]

    n_frontier = len(candidates)
    if n_frontier == 0:
        print("[ERROR] Frontier file contains no candidates.")
        sys.exit(1)

    gcr_values  = [c["GCR"]  for c in candidates]
    spcf_values = [c["spcf"] for c in candidates]

    print(f"\n{'='*70}")
    print(f"  PEC Frontier Eval — expert checkpoint:")
    print(f"    {args_cli.checkpoint_path}")
    print(f"  Frontier: {args_cli.frontier_file}")
    print(f"  Candidates : {n_frontier}")
    print(f"  Episodes   : {args_cli.num_episodes}")
    print(f"  Start level: {args_cli.start_difficulty}")
    print(f"{'='*70}\n")

    # ── Configure environment ─────────────────────────────────────────────────
    # One parallel env per frontier design.
    env_cfg.scene.num_envs = n_frontier
    env_cfg.sim.device = args_cli.device if args_cli.device is not None \
                         else env_cfg.sim.device

    # Disable curriculum warmup guard during eval by setting a high iteration.
    # (The curriculum function checks env.rsl_rl_iteration < warmup_period=100.)
    # We set it after env creation below.

    # Create the environment, passing per-env GCR and spcf values.
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode=None,
        GCR_values=gcr_values,
        spcf_values=spcf_values,
    )

    isaac_env = env.unwrapped

    # Bypass curriculum warmup.
    isaac_env.rsl_rl_iteration = 1000

    # Shift all env origins to the starting difficulty level.
    # env_origins[:, 1] = -2.0 * level_idx  (obstacle spacing = 2 m along -Y)
    inter_obstacle_spacing_y = 2.0
    isaac_env.scene._default_env_origins[:, 1] -= (
        inter_obstacle_spacing_y * args_cli.start_difficulty
    )

    # ── Wrap and load checkpoint ──────────────────────────────────────────────
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(args_cli.checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=isaac_env.device)

    print(f"[INFO] Checkpoint loaded: {args_cli.checkpoint_path}")

    # ── Prepare level tracking ────────────────────────────────────────────────
    obstacle_height_list = isaac_env.obstacle_height_list   # list[float], len=75
    max_level = len(obstacle_height_list) - 1

    EP_LENGTH = isaac_env.max_episode_length
    total_steps = args_cli.num_episodes * EP_LENGTH
    print(f"[INFO] EP_LENGTH={EP_LENGTH}  total_steps={total_steps}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    obs, _ = env.get_observations()

    with tqdm(total=total_steps, desc="Frontier eval") as pbar:
        for _ in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _rew, _done, _info = env.step(actions)
            pbar.update(1)

    # Read the final curriculum level per environment from env origins.
    # The curriculum updates _default_env_origins at each episode reset, so
    # the value after the last step is the level the robot settled at.
    final_origins_y = isaac_env.scene._default_env_origins[:, 1]  # (F,)
    final_level_idx = (
        (-final_origins_y / inter_obstacle_spacing_y)
        .long()
        .clamp(0, max_level)
    )
    final_level_idx_cpu = final_level_idx.cpu().tolist()

    # ── Assemble results ──────────────────────────────────────────────────────
    results = []
    for i, cand in enumerate(candidates):
        lvl = int(final_level_idx_cpu[i])
        results.append({
            "id":             cand["id"],
            "GCR":            cand["GCR"],
            "spcf":           cand["spcf"],
            "best_level_idx": lvl,
            "best_height_m":  float(obstacle_height_list[lvl]),
        })

    output_payload = {
        "checkpoint": args_cli.checkpoint_path,
        "frontier_file": args_cli.frontier_file,
        "num_episodes": args_cli.num_episodes,
        "start_difficulty": args_cli.start_difficulty,
        "results": results,
    }

    # ── Save and print ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    with open(args_cli.output, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[INFO] Results saved to: {args_cli.output}")

    # Parseable summary line for the orchestrator.
    print(f"FRONTIER_RESULTS: {json.dumps(results)}")

    # Human-readable per-design summary.
    print(f"\n{'='*70}")
    print(f"  {'ID':>4}  {'GCR':>6}  {'spcf':>8}  {'LevelIdx':>9}  {'Height(m)':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*10}")
    for r in results:
        print(f"  {r['id']:>4}  {r['GCR']:>6.4f}  {r['spcf']:>8.5f}"
              f"  {r['best_level_idx']:>9}  {r['best_height_m']:>10.4f}")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
