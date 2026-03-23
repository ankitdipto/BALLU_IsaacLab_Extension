"""
PEC Expert Training Script
==========================
Progressive Expert Coverage — Step 1: Train one expert on its Gaussian subspace.

Supports both 2D (GCR, spcf) and 3D (GCR, spcf, leg) modes based on the
design_space in pec_state.json.

3D mode additions:
  - Samples leg_length from the expert's 3D Gaussian.
  - Calls pec_generate_usds.py to generate USD files for unique leg_lengths.
  - Writes usd_order_file.json listing the per-env USD path.
  - Sets BALLU_USD_ORDER_FILE env var so the env config assigns env i → usd_paths[i].

Usage
-----
    # From ballu_isclb_extension/
    python scripts/pec/pec_train_expert.py \\
        --run_name   my_pec_run \\
        --expert_id  0 \\
        --task       Isc-BALLU-hetero-pretrain-ramp \\
        --num_envs   4096 \\
        --max_iterations 2000 \\
        --seed       42 \\
        --headless
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


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_SCRIPT         = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl", "train.py")
GEN_USDS_SCRIPT      = os.path.join(SCRIPT_DIR, "pec_generate_usds.py")


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def sample_from_gaussian_2d(mu, sigma_2x2, n, gcr_lo, gcr_hi, spcf_lo, spcf_hi, rng):
    std_gcr  = math.sqrt(sigma_2x2[0][0])
    std_spcf = math.sqrt(sigma_2x2[1][1])
    gcr_s  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo, gcr_hi)
    spcf_s = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)
    return gcr_s.astype(np.float32), spcf_s.astype(np.float32)


def sample_from_gaussian_3d(mu, sigma_3x3, n,
                             gcr_lo, gcr_hi,
                             spcf_lo, spcf_hi,
                             leg_lo, leg_hi, rng):
    std_gcr  = math.sqrt(sigma_3x3[0][0])
    std_spcf = math.sqrt(sigma_3x3[1][1])
    std_leg  = math.sqrt(sigma_3x3[2][2])
    gcr_s  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo, gcr_hi)
    spcf_s = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)
    leg_s  = rng.normal(mu[2], std_leg,  size=n).clip(leg_lo, leg_hi)
    return gcr_s.astype(np.float32), spcf_s.astype(np.float32), leg_s.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# USD generation helper (3D mode)
# ──────────────────────────────────────────────────────────────────────────────

def generate_usds_for_legs(leg_samples: np.ndarray, output_dir: str,
                            leg_precision: int = 4) -> dict:
    """
    Call pec_generate_usds.py to generate USDs for unique leg_lengths.

    Returns a dict mapping leg key (e.g. '0.3000') to abs USD path.
    Exits the process if generation fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write designs file: [[0.0, 0.0, leg], ...] (GCR/spcf are irrelevant here)
    designs = [[0.0, 0.0, float(l)] for l in leg_samples]
    designs_file = os.path.join(output_dir, "tmp_leg_designs.json")
    with open(designs_file, "w") as f:
        json.dump(designs, f)

    cmd = [
        sys.executable, GEN_USDS_SCRIPT,
        "--output_dir",   output_dir,
        "--designs_file", designs_file,
        "--leg_precision", str(leg_precision),
        "--skip_existing",
    ]
    print(f"\n  [USD Gen] Running pec_generate_usds.py ...")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] pec_generate_usds.py failed (exit {result.returncode}).")
        sys.exit(1)

    # Parse USD_LEG_MAP from stdout — re-run with capture to get the map.
    result2 = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    leg_map: dict[str, str] = {}
    for line in result2.stdout.splitlines():
        if line.startswith("USD_LEG_MAP:"):
            payload = line.split("USD_LEG_MAP:", 1)[1].strip()
            try:
                leg_map = json.loads(payload)
            except json.JSONDecodeError:
                pass
            break

    if not leg_map:
        # Fallback: read from registry.
        registry_path = os.path.join(output_dir, "morphology_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry = json.load(f)
            leg_map = {k: v["usd_path"] for k, v in registry.items()}

    return leg_map


def leg_key(leg: float, precision: int = 4) -> str:
    return f"{round(leg, precision):.{precision}f}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Step 1 — train one expert on its Gaussian design subspace."
    )
    parser.add_argument("--run_name",   type=str, required=True)
    parser.add_argument("--expert_id",  type=int, required=True)
    parser.add_argument("--dl", type=int, required=True,
                        help="Difficulty level of the obstacle.")
    parser.add_argument("--log_root",   type=str, default="logs/pec")
    parser.add_argument("--task",       type=str,
                        default="Isc-BALLU-hetero-pretrain-ramp")
    parser.add_argument("--num_envs",   type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--timeout_h",  type=float, default=8.0)
    parser.add_argument("--n_samples",  type=int, default=None)
    parser.add_argument("--leg_precision", type=int, default=4,
                        help="Decimal places for leg_length rounding (default: 4).")

    args = parser.parse_args()
    n_samples = args.n_samples if args.n_samples is not None else args.num_envs
    rng = np.random.default_rng(args.seed)

    # ── Load PEC state ────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    expert = next((e for e in state["experts"] if e["id"] == args.expert_id), None)
    if expert is None:
        print(f"[ERROR] Expert id={args.expert_id} not found.")
        sys.exit(1)

    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    iteration        = state["iteration"]
    usd_rel_path     = state.get("usd_rel_path")
    is_3d            = "leg" in state["design_space"]

    if is_3d:
        leg_lo, leg_hi = state["design_space"]["leg"]

    print(f"\n{'='*70}")
    print(f"  PEC Train Expert — run: {args.run_name}  |  expert: {args.expert_id}"
          f"  |  iter: {iteration}  |  mode: {'3D' if is_3d else '2D'}")
    print(f"{'='*70}")
    if not is_3d:
        print(f"  usd_rel_path : {usd_rel_path or '(not set)'}")
    mu_str = f"GCR={expert['mu'][0]:.4f}, spcf={expert['mu'][1]:.5f}"
    if is_3d:
        mu_str += f", leg={expert['mu'][2]:.4f}"
    print(f"  mu  = ({mu_str})")
    sig_g = math.sqrt(expert["sigma"][0][0])
    sig_s = math.sqrt(expert["sigma"][1][1])
    std_str = f"GCR={sig_g:.4f},  spcf={sig_s:.5f}"
    if is_3d:
        sig_l = math.sqrt(expert["sigma"][2][2])
        std_str += f",  leg={sig_l:.4f}"
    print(f"  std = ({std_str})")
    print(f"  n_samples    : {n_samples}")
    print(f"  num_envs     : {args.num_envs}")
    print(f"  max_iters    : {args.max_iterations}")
    print(f"  seed         : {args.seed}")
    print(f"  task         : {args.task}")

    # ── Sample from Gaussian ──────────────────────────────────────────────────
    samples_dir = os.path.join(run_dir, f"expert_{args.expert_id}",
                               "samples", f"iter_{iteration}")
    os.makedirs(samples_dir, exist_ok=True)

    if is_3d:
        # Kinematic designs (USD files) are expensive to generate — cap at N_init.
        # GCR / spcf are cheap runtime parameters — sample freely across num_envs.
        N_init = state.get("N_init", n_samples)
        n_kin  = min(N_init, n_samples)   # number of unique leg_lengths to generate

        _, _, leg_designs = sample_from_gaussian_3d(
            mu=expert["mu"], sigma_3x3=expert["sigma"],
            n=n_kin,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            leg_lo=leg_lo, leg_hi=leg_hi,
            rng=rng,
        )

        gcr_samples, spcf_samples = sample_from_gaussian_2d(
            mu=expert["mu"], sigma_2x2=expert["sigma"],
            n=n_samples,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            rng=rng,
        )

        # Tile the n_kin kinematic designs across all num_envs environments.
        leg_samples = np.array(
            [leg_designs[i % n_kin] for i in range(n_samples)], dtype=np.float32
        )

        print(f"\n  Kinematic designs: {n_kin}  (N_init={N_init})"
              f"  tiled across {n_samples} envs")
        print(f"  leg  designs — mean={leg_designs.mean():.4f}  "
              f"std={leg_designs.std():.4f}  "
              f"range=[{leg_designs.min():.4f}, {leg_designs.max():.4f}]")
    else:
        gcr_samples, spcf_samples = sample_from_gaussian_2d(
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
    gcr_file  = os.path.join(samples_dir, "gcr.npy")
    spcf_file = os.path.join(samples_dir, "spcf.npy")
    np.save(gcr_file,  gcr_samples)
    np.save(spcf_file, spcf_samples)
    print(f"\n  Sample files saved:")
    print(f"    {gcr_file}")
    print(f"    {spcf_file}")

    usd_order_file = None
    if is_3d:
        leg_file = os.path.join(samples_dir, "leg.npy")
        np.save(leg_file, leg_samples)
        print(f"    {leg_file}")

        # ── Generate USD files for unique leg_lengths ─────────────────────────
        usd_lib_dir = os.path.join(run_dir, f"expert_{args.expert_id}", "usds")
        leg_map = generate_usds_for_legs(leg_designs, usd_lib_dir, args.leg_precision)

        if not leg_map:
            print("[ERROR] USD generation returned empty leg_map.")
            sys.exit(1)

        # Build ordered USD list: usd_order[i] = USD for env i.
        usd_order = []
        for l in leg_samples:
            key = leg_key(float(l), args.leg_precision)
            if key not in leg_map:
                print(f"[ERROR] No USD found for leg key '{key}'.")
                sys.exit(1)
            usd_order.append(leg_map[key])

        usd_order_file = os.path.join(samples_dir, "usd_order.json")
        with open(usd_order_file, "w") as f:
            json.dump(usd_order, f)
        print(f"    {usd_order_file}  ({len(usd_order)} entries, "
              f"{len(set(usd_order))} unique USDs)")

        # Store morphology_library_path in state (first time only).
        if expert.get("morphology_library_path") is None:
            expert["morphology_library_path"] = os.path.abspath(usd_lib_dir)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    # ── Warm-start detection ──────────────────────────────────────────────────
    prev_checkpoint = expert.get("checkpoint")
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        print(f"\n  Warm-starting from: {prev_checkpoint}")
    elif prev_checkpoint:
        print(f"\n  [WARNING] Previous checkpoint not found: {prev_checkpoint}")
        prev_checkpoint = None
    else:
        print(f"\n  No previous checkpoint — training from scratch.")

    # ── Build train.py command ────────────────────────────────────────────────
    common_folder = f"{args.run_name}/expert_{args.expert_id}"
    run_name_tag  = f"iter_{iteration}"

    cmd = [
        sys.executable, TRAIN_SCRIPT,
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
    subprocess_env = os.environ.copy()
    if usd_order_file:
        subprocess_env["BALLU_USD_ORDER_FILE"] = os.path.abspath(usd_order_file)
        print(f"  BALLU_USD_ORDER_FILE = {subprocess_env['BALLU_USD_ORDER_FILE']}")
    elif usd_rel_path:
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

    exp_dir = None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("EXP_DIR:"):
                exp_dir = line.split("EXP_DIR:", 1)[1].strip()
        process.wait(timeout=args.timeout_h * 3600)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"\n[ERROR] Training subprocess timed out after {args.timeout_h:.1f} h.")
        sys.exit(1)

    elapsed = (time.time() - t_start) / 60.0
    crashed = process.returncode != 0

    if crashed:
        print(f"\n[WARN] Training subprocess crashed (exit {process.returncode}).")
    else:
        print(f"\n{'='*70}")
        print(f"  Training completed in {elapsed:.1f} min.")

    if exp_dir is None:
        if crashed:
            print("[ERROR] EXP_DIR was never printed — training crashed before logging.")
            print("        The orchestrator will fall back to the previous checkpoint.")
        else:
            print("[ERROR] Could not parse EXP_DIR from train.py output.")
        sys.exit(1)

    ckpt_path = os.path.join(exp_dir, "model_best.pt")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        if crashed:
            print(f"[ERROR] Crash and model_best.pt not found: {ckpt_path}")
            sys.exit(process.returncode)
        else:
            print(f"[WARNING] model_best.pt not found: {ckpt_path}")
    else:
        if crashed:
            print(f"  [WARN] Checkpoint salvaged despite crash: {ckpt_path}")
        else:
            print(f"  Checkpoint found : {ckpt_path}")

    # ── Update PEC state ──────────────────────────────────────────────────────
    with open(state_path) as f:
        state = json.load(f)
    expert = next(e for e in state["experts"] if e["id"] == args.expert_id)

    expert["checkpoint"]            = ckpt_path
    expert["trained"]               = True
    expert["last_trained_pec_iter"] = iteration

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"  State updated    : {state_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
