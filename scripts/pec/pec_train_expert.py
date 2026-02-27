"""
PEC Expert Training Script
==========================
Progressive Expert Coverage — Step 1: Train one expert on its Gaussian subspace.

Reads the current PEC state, samples ``num_envs`` (GCR, spcf) points from
expert i's Gaussian N(mu_i, Sigma_i), saves them as .npy sample files, then
launches ``train.py`` as a subprocess with ``--GCR_samples_file`` /
``--spcf_samples_file`` so that each of the parallel simulation environments
uses a distinct design drawn from that Gaussian.

After training completes the script updates ``pec_state.json`` with the
checkpoint path and sets ``trained = true`` for the expert.

Usage
-----
    # From ballu_isclb_extension/ (the working directory for train.py)
    python scripts/pec/pec_train_expert.py \\
        --run_name   my_pec_run \\
        --expert_id  0 \\
        --task       Isc-BALLU-hetero-general \\
        --num_envs   4096 \\
        --max_iterations 2000 \\
        --seed       42 \\
        --headless

Key paths
---------
    State file  : logs/pec/<run_name>/pec_state.json
    Sample files: logs/pec/<run_name>/expert_<i>/samples/iter_<N>/gcr.npy
                                                                   spcf.npy
    Checkpoint  : logs/rsl_rl/pec_<run_name>/expert_<i>/iter_<N>/model_best.pt
                  (relative to cwd = ballu_isclb_extension/)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# This script lives at scripts/pec/pec_train_expert.py.
# The project root (ballu_isclb_extension/) is two levels up.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # ballu_isclb_extension/
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl", "train.py")


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian sampling helper
# ──────────────────────────────────────────────────────────────────────────────

def sample_from_gaussian(mu, sigma_2x2, n: int,
                         gcr_lo: float, gcr_hi: float,
                         spcf_lo: float, spcf_hi: float,
                         rng: np.random.Generator):
    """
    Draw n samples from N(mu, Sigma) using the diagonal of sigma_2x2.
    Values are clamped to the design-space bounds.

    Returns two arrays: gcr_samples (n,), spcf_samples (n,)
    """
    std_gcr  = math.sqrt(sigma_2x2[0][0])
    std_spcf = math.sqrt(sigma_2x2[1][1])

    gcr_samples  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo,  gcr_hi)
    spcf_samples = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)

    return gcr_samples.astype(np.float32), spcf_samples.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Step 1 — train one expert on its Gaussian design subspace."
    )
    # PEC state
    parser.add_argument("--run_name",   type=str, required=True,
                        help="PEC run name (matches the one used in pec_init.py).")
    parser.add_argument("--expert_id",  type=int, required=True,
                        help="Index of the expert to train (0-based).")
    parser.add_argument("--dl", type=int, required=True, 
                   help="Difficulty level of the obstacle (default: 0)")
    parser.add_argument("--log_root",   type=str, default="logs/pec",
                        help="Root directory for PEC logs (default: logs/pec).")

    # Training hyper-parameters forwarded to train.py
    parser.add_argument("--task",       type=str,
                        default="Isc-BALLU-hetero-general",
                        help="Isaac Lab task name (default: Isc-BALLU-hetero-general).")
    parser.add_argument("--num_envs",   type=int, default=4096,
                        help="Number of parallel simulation environments (default: 4096).")
    parser.add_argument("--max_iterations", type=int, default=2000,
                        help="PPO training iterations (default: 2000).")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for both sample generation and training (default: 42).")
    parser.add_argument("--device",     type=str, default="cuda:0",
                        help="Torch device (default: cuda:0).")
    parser.add_argument("--headless",   action="store_true",
                        help="Run Isaac Sim in headless mode.")
    parser.add_argument("--timeout_h",  type=float, default=8.0,
                        help="Subprocess timeout in hours (default: 8).")

    # Sample generation
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Number of (GCR, spcf) samples to draw from the Gaussian. "
                             "Defaults to --num_envs.")

    args = parser.parse_args()

    n_samples = args.n_samples if args.n_samples is not None else args.num_envs
    rng = np.random.default_rng(args.seed)

    # ── Load PEC state ────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        print("        Run pec_init.py first.")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    # Locate the requested expert
    expert = next((e for e in state["experts"] if e["id"] == args.expert_id), None)
    if expert is None:
        print(f"[ERROR] Expert id={args.expert_id} not found in state file.")
        sys.exit(1)

    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    iteration        = state["iteration"]
    usd_rel_path     = state.get("usd_rel_path")

    print(f"\n{'='*70}")
    print(f"  PEC Train Expert — run: {args.run_name}  |  expert: {args.expert_id}"
          f"  |  iter: {iteration}")
    print(f"{'='*70}")
    print(f"  usd_rel_path : {usd_rel_path or '(not set)'}")
    print(f"  mu  = (GCR={expert['mu'][0]:.4f}, spcf={expert['mu'][1]:.5f})")
    sig_g = math.sqrt(expert["sigma"][0][0])
    sig_s = math.sqrt(expert["sigma"][1][1])
    print(f"  std = (GCR={sig_g:.4f},  spcf={sig_s:.5f})")
    print(f"  n_samples    : {n_samples}")
    print(f"  num_envs     : {args.num_envs}")
    print(f"  max_iters    : {args.max_iterations}")
    print(f"  seed         : {args.seed}")
    print(f"  task         : {args.task}")

    # ── Sample from Gaussian ──────────────────────────────────────────────────
    gcr_samples, spcf_samples = sample_from_gaussian(
        mu=expert["mu"], sigma_2x2=expert["sigma"],
        n=n_samples,
        gcr_lo=gcr_lo, gcr_hi=gcr_hi,
        spcf_lo=spcf_lo, spcf_hi=spcf_hi,
        rng=rng,
    )

    print(f"\n  GCR  samples — mean={gcr_samples.mean():.4f}  "
          f"std={gcr_samples.std():.4f}  "
          f"range=[{gcr_samples.min():.4f}, {gcr_samples.max():.4f}]")
    print(f"  spcf samples — mean={spcf_samples.mean():.5f}  "
          f"std={spcf_samples.std():.5f}  "
          f"range=[{spcf_samples.min():.5f}, {spcf_samples.max():.5f}]")

    # ── Save sample files ─────────────────────────────────────────────────────
    samples_dir = os.path.join(run_dir, f"expert_{args.expert_id}",
                               "samples", f"iter_{iteration}")
    os.makedirs(samples_dir, exist_ok=True)

    gcr_file  = os.path.join(samples_dir, "gcr.npy")
    spcf_file = os.path.join(samples_dir, "spcf.npy")
    np.save(gcr_file,  gcr_samples)
    np.save(spcf_file, spcf_samples)
    print(f"\n  Sample files saved:")
    print(f"    {gcr_file}")
    print(f"    {spcf_file}")

    # ── Build log directory name ──────────────────────────────────────────────
    # train.py produces:
    #   logs/rsl_rl/<experiment_name>/<common_folder>/<run_name>/model_best.pt
    #
    # experiment_name comes from the PPO config (e.g. "lab_02.24.2026") and is
    # NOT overridden here so that all experiments stay in the same lab folder.
    #
    # We set:
    #   common_folder = expert_<id>
    #   run_name      = iter_<iteration>
    #
    # The exact log_dir is parsed from the "EXP_DIR: ..." line printed by train.py.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    common_folder = f"{args.run_name}/expert_{args.expert_id}"
    run_name_tag  = f"iter_{iteration}"

    # ── Warm-start detection ──────────────────────────────────────────────────
    # If the expert has already been trained at least once, load its previous
    # checkpoint so the new iteration fine-tunes rather than trains from scratch.
    prev_checkpoint = expert.get("checkpoint")
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        print(f"\n  Warm-starting from previous checkpoint:")
        print(f"    {prev_checkpoint}")
    elif prev_checkpoint:
        print(f"\n  [WARNING] Previous checkpoint not found at: {prev_checkpoint}")
        print(f"             Training from scratch for this iteration.")
        prev_checkpoint = None
    else:
        print(f"\n  No previous checkpoint — training from scratch.")

    print(f"\n  Checkpoint will be parsed from EXP_DIR printed by train.py.")

    # ── Build train.py command ────────────────────────────────────────────────
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--task",               args.task,
        "--num_envs",           str(args.num_envs),
        "--max_iterations",     str(args.max_iterations),
        "--seed",               str(args.seed),
        "--device",             args.device,
        "--GCR_samples_file",   gcr_file,
        "--spcf_samples_file",  spcf_file,
        "--common_folder",      common_folder,
        "--run_name",           run_name_tag,
        "--dl",                 str(args.dl),
    ]
    if prev_checkpoint:
        cmd += ["--resume_path", prev_checkpoint]
    if args.headless:
        cmd.append("--headless")

    print(f"\n  Command:\n    {' '.join(cmd)}\n")
    print(f"{'='*70}")

    # ── Launch subprocess ─────────────────────────────────────────────────────
    # Inherit the current environment and inject BALLU_USD_REL_PATH if set.
    subprocess_env = os.environ.copy()
    if usd_rel_path:
        subprocess_env["BALLU_USD_REL_PATH"] = usd_rel_path
        print(f"  BALLU_USD_REL_PATH injected into subprocess env.")

    t_start = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=PROJECT_ROOT,
        env=subprocess_env,
    )

    captured_lines = []
    exp_dir = None

    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            captured_lines.append(line)
            # Parse the log directory printed by train.py
            if line.startswith("EXP_DIR:"):
                exp_dir = line.split("EXP_DIR:", 1)[1].strip()

        process.wait(timeout=args.timeout_h * 3600)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"\n[ERROR] Training subprocess timed out after {args.timeout_h:.1f} h.")
        sys.exit(1)

    elapsed = (time.time() - t_start) / 60.0

    if process.returncode != 0:
        print(f"\n[ERROR] Training subprocess exited with code {process.returncode}.")
        sys.exit(process.returncode)

    print(f"\n{'='*70}")
    print(f"  Training completed in {elapsed:.1f} min.")

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if exp_dir is None:
        print("[ERROR] Could not parse EXP_DIR from train.py output.")
        print("        Check the training log and update pec_state.json manually.")
        sys.exit(1)

    ckpt_path = os.path.join(exp_dir, "model_best.pt")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"[WARNING] model_best.pt not found at: {ckpt_path}")
        print("          The state file will record this path anyway.")
    else:
        print(f"  Checkpoint found : {ckpt_path}")

    # ── Update PEC state ──────────────────────────────────────────────────────
    expert["checkpoint"] = ckpt_path
    expert["trained"]    = True

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"  State updated    : {state_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
