"""
PEC Frontier Evaluation Orchestrator
=====================================
Progressive Expert Coverage — Step 2: Evaluate frontier candidates across
all K experts.

Supports 2D (GCR, spcf) and 3D (GCR, spcf, leg) modes.

In 3D mode:
  - Candidates include a `leg` field and a `usd_path` field.
  - pec_generate_usds.py is called once to generate USD files for unique
    leg_lengths before any expert evaluation subprocess.
  - BALLU_USD_ORDER_FILE is set in each expert's subprocess env so the
    env config assigns env i → usd_paths[i].

Key files
---------
    State     : logs/pec/<run_name>/pec_state.json
    Output    : logs/pec/<run_name>/frontier_evals/iter_<N>/scores.json
                logs/pec/<run_name>/frontier_evals/iter_<N>/candidates.json
                logs/pec/<run_name>/frontier_evals/iter_<N>/expert_<k>_results.json
                logs/pec/<run_name>/frontier_evals/iter_<N>/usds/  (3D only)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np


SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EVAL_SCRIPT     = os.path.join(SCRIPT_DIR, "pec_eval_expert_frontier.py")
GEN_USDS_SCRIPT = os.path.join(SCRIPT_DIR, "pec_generate_usds.py")


# ──────────────────────────────────────────────────────────────────────────────
# 2D frontier sampling
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_log_density(theta_gcr, theta_spcf, expert):
    mu    = expert["mu"]
    sigma = expert["sigma"]
    d_g   = (theta_gcr  - mu[0]) ** 2 / (2.0 * sigma[0][0])
    d_s   = (theta_spcf - mu[1]) ** 2 / (2.0 * sigma[1][1])
    return -(d_g + d_s)


def _max_log_densities_2d(experts, gcr_pool, spcf_pool):
    max_ld = np.full(len(gcr_pool), -np.inf)
    for ex in experts:
        mu    = ex["mu"]
        sigma = ex["sigma"]
        d_g = (gcr_pool  - mu[0]) ** 2 / (2.0 * sigma[0][0])
        d_s = (spcf_pool - mu[1]) ** 2 / (2.0 * sigma[1][1])
        max_ld = np.maximum(max_ld, -(d_g + d_s))
    return max_ld


def sample_frontier_candidates(experts, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                F, pool_factor, rng):
    n_pool = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)
    max_ld = _max_log_densities_2d(experts, gcr_pool, spcf_pool)
    idx = np.argsort(max_ld)[:F]
    return [{"id": rank, "GCR": float(gcr_pool[i]), "spcf": float(spcf_pool[i])}
            for rank, i in enumerate(idx)]


def sample_border_candidates(experts, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                              F, pool_factor, border_inner_ld, border_outer_ld, rng):
    n_pool = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)
    max_ld = _max_log_densities_2d(experts, gcr_pool, spcf_pool)

    in_band  = (max_ld > border_outer_ld) & (max_ld < border_inner_ld)
    in_front = ~in_band
    border_idx   = np.where(in_band)[0][np.argsort(max_ld[in_band])]
    frontier_idx = np.where(in_front)[0][np.argsort(max_ld[in_front])]

    n_border_take  = min(len(border_idx), F)
    n_frontier_take = F - n_border_take
    if n_frontier_take > 0:
        print(f"  [INFO] Border band {len(border_idx)}/{F} — "
              f"supplementing {n_frontier_take} from frontier.")

    selected = np.concatenate([border_idx[:n_border_take], frontier_idx[:n_frontier_take]])
    candidates = [{"id": rank, "GCR": float(gcr_pool[i]), "spcf": float(spcf_pool[i])}
                  for rank, i in enumerate(selected)]
    return candidates, n_border_take


# ──────────────────────────────────────────────────────────────────────────────
# 3D frontier sampling
# ──────────────────────────────────────────────────────────────────────────────

def _max_log_densities_3d(experts, gcr_pool, spcf_pool, leg_pool):
    max_ld = np.full(len(gcr_pool), -np.inf)
    for ex in experts:
        mu    = ex["mu"]
        sigma = ex["sigma"]
        d_g = (gcr_pool  - mu[0]) ** 2 / (2.0 * sigma[0][0])
        d_s = (spcf_pool - mu[1]) ** 2 / (2.0 * sigma[1][1])
        d_l = (leg_pool  - mu[2]) ** 2 / (2.0 * sigma[2][2])
        max_ld = np.maximum(max_ld, -(d_g + d_s + d_l))
    return max_ld


def sample_frontier_candidates_3d(experts,
                                   gcr_lo, gcr_hi,
                                   spcf_lo, spcf_hi,
                                   leg_lo, leg_hi,
                                   F, pool_factor, rng):
    n_pool    = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)
    leg_pool  = rng.uniform(leg_lo,  leg_hi,  size=n_pool)
    max_ld = _max_log_densities_3d(experts, gcr_pool, spcf_pool, leg_pool)
    idx = np.argsort(max_ld)[:F]
    return [{"id": rank,
             "GCR":  float(gcr_pool[i]),
             "spcf": float(spcf_pool[i]),
             "leg":  float(leg_pool[i])}
            for rank, i in enumerate(idx)]


def sample_border_candidates_3d(experts,
                                 gcr_lo, gcr_hi,
                                 spcf_lo, spcf_hi,
                                 leg_lo, leg_hi,
                                 F, pool_factor,
                                 border_inner_ld, border_outer_ld, rng):
    n_pool    = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)
    leg_pool  = rng.uniform(leg_lo,  leg_hi,  size=n_pool)
    max_ld = _max_log_densities_3d(experts, gcr_pool, spcf_pool, leg_pool)

    in_band  = (max_ld > border_outer_ld) & (max_ld < border_inner_ld)
    in_front = ~in_band
    border_idx   = np.where(in_band)[0][np.argsort(max_ld[in_band])]
    frontier_idx = np.where(in_front)[0][np.argsort(max_ld[in_front])]

    n_border_take   = min(len(border_idx), F)
    n_frontier_take = F - n_border_take
    if n_frontier_take > 0:
        print(f"  [INFO] Border band {len(border_idx)}/{F} — "
              f"supplementing {n_frontier_take} from frontier.")

    selected = np.concatenate([border_idx[:n_border_take], frontier_idx[:n_frontier_take]])
    candidates = [{"id": rank,
                   "GCR":  float(gcr_pool[i]),
                   "spcf": float(spcf_pool[i]),
                   "leg":  float(leg_pool[i])}
                  for rank, i in enumerate(selected)]
    return candidates, n_border_take


# ──────────────────────────────────────────────────────────────────────────────
# USD generation for frontier candidates (3D mode)
# ──────────────────────────────────────────────────────────────────────────────

def leg_key(leg: float, precision: int = 4) -> str:
    return f"{round(leg, precision):.{precision}f}"


def generate_usds_for_candidates(candidates: list, output_dir: str,
                                  leg_precision: int = 4) -> dict:
    """
    Generate USD files for the unique leg_lengths in `candidates`.
    Returns leg_map: {key -> abs_usd_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    designs = [[0.0, 0.0, float(c["leg"])] for c in candidates]
    designs_file = os.path.join(output_dir, "tmp_frontier_designs.json")
    with open(designs_file, "w") as f:
        json.dump(designs, f)

    cmd = [
        sys.executable, GEN_USDS_SCRIPT,
        "--output_dir",    output_dir,
        "--designs_file",  designs_file,
        "--leg_precision", str(leg_precision),
        "--skip_existing",
    ]
    print(f"\n  [USD Gen] Generating USDs for {len(candidates)} frontier candidates ...")
    # Run with output visible
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)

    # Re-run to capture USD_LEG_MAP
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
        registry_path = os.path.join(output_dir, "morphology_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                reg = json.load(f)
            leg_map = {k: v["usd_path"] for k, v in reg.items()}

    return leg_map


# ──────────────────────────────────────────────────────────────────────────────
# Expert evaluation subprocess launcher
# ──────────────────────────────────────────────────────────────────────────────

def run_expert_eval(expert, candidates_file, output_file, args,
                    usd_rel_path=None, usd_order_file=None):
    """
    Launch pec_eval_expert_frontier.py for one expert and block until done.
    Returns the list of per-design result dicts on success, None on failure.
    """
    checkpoint = expert.get("checkpoint")
    if not checkpoint:
        print(f"  [WARNING] Expert {expert['id']} has no checkpoint — skipping.")
        return None
    if not os.path.exists(checkpoint):
        print(f"  [WARNING] Checkpoint not found: {checkpoint} — skipping.")
        return None

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint_path",  checkpoint,
        "--frontier_file",    candidates_file,
        "--output",           output_file,
        "--num_episodes",     str(args.num_episodes),
        "--start_difficulty", str(args.start_difficulty),
        "--task",             args.task,
        "--device",           args.device,
    ]
    if args.headless:
        cmd.append("--headless")

    print(f"\n  {'─'*66}")
    print(f"  Expert {expert['id']}  checkpoint: {os.path.basename(checkpoint)}")
    print(f"  {'─'*66}")

    subprocess_env = os.environ.copy()
    if usd_order_file:
        subprocess_env["BALLU_USD_ORDER_FILE"] = os.path.abspath(usd_order_file)
        print(f"  BALLU_USD_ORDER_FILE = {subprocess_env['BALLU_USD_ORDER_FILE']}")
    elif usd_rel_path:
        subprocess_env["BALLU_USD_REL_PATH"] = usd_rel_path

    t0 = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=PROJECT_ROOT,
        env=subprocess_env,
    )

    results = None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("FRONTIER_RESULTS:"):
                payload = line.split("FRONTIER_RESULTS:", 1)[1].strip()
                try:
                    results = json.loads(payload)
                except json.JSONDecodeError:
                    print(f"  [WARNING] Could not parse FRONTIER_RESULTS line.")
        process.wait(timeout=args.timeout_h * 3600)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"\n  [ERROR] Expert {expert['id']} eval timed out.")
        return None

    elapsed = (time.time() - t0) / 60.0
    print(f"\n  Expert {expert['id']} eval finished in {elapsed:.1f} min "
          f"(exit={process.returncode})")

    if process.returncode != 0:
        print(f"  [ERROR] Subprocess exited with code {process.returncode}.")
        return None

    if results is None and os.path.exists(output_file):
        with open(output_file) as f:
            payload = json.load(f)
        results = payload.get("results")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Step 2 — sample frontier designs and evaluate all K experts."
    )
    parser.add_argument("--run_name",   type=str, required=True)
    parser.add_argument("--log_root",   type=str, default="logs/pec")
    parser.add_argument("--F",          type=int, default=50)
    parser.add_argument("--pool_factor", type=int, default=20)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--use_iter_seed", action="store_true")
    parser.add_argument("--sampling_mode", type=str, default="frontier",
                        choices=["frontier", "border", "auto"])
    parser.add_argument("--border_inner_ld", type=float, default=-0.5)
    parser.add_argument("--border_outer_ld", type=float, default=-3.0)
    parser.add_argument("--auto_switch_iter", type=int, default=4)
    parser.add_argument("--num_episodes",    type=int, default=30)
    parser.add_argument("--start_difficulty", type=int, default=15)
    parser.add_argument("--task",       type=str,
                        default="Isc-BALLU-hetero-pretrain-ramp")
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--timeout_h",  type=float, default=4.0)
    parser.add_argument("--leg_precision", type=int, default=4)

    args = parser.parse_args()

    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    experts   = state["experts"]
    iteration = state["iteration"]
    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    usd_rel_path = state.get("usd_rel_path")
    is_3d = "leg" in state["design_space"]
    K = len(experts)

    if is_3d:
        leg_lo, leg_hi = state["design_space"]["leg"]

    seed = iteration if args.use_iter_seed else args.seed
    rng  = np.random.default_rng(seed)

    if args.sampling_mode == "auto":
        effective_mode = "border" if iteration < args.auto_switch_iter else "frontier"
        print(f"  [auto] iter={iteration}  switch_at={args.auto_switch_iter}"
              f"  → using '{effective_mode}'")
    else:
        effective_mode = args.sampling_mode

    print(f"\n{'='*70}")
    print(f"  PEC Evaluate Frontier — run: {args.run_name}  |  iter: {iteration}"
          f"  |  mode: {'3D' if is_3d else '2D'}")
    print(f"{'='*70}")
    print(f"  K experts        : {K}")
    print(f"  F candidates     : {args.F}")
    print(f"  Pool factor      : {args.pool_factor}")
    print(f"  Sampling mode    : {effective_mode}")
    print(f"  Seed             : {seed}")

    untrained = [ex["id"] for ex in experts if not ex.get("checkpoint")]
    if untrained:
        print(f"\n[WARNING] Experts without checkpoints: {untrained} — will be skipped.")

    # ── Sample candidates ─────────────────────────────────────────────────────
    n_border = 0
    if is_3d:
        if effective_mode == "border":
            candidates, n_border = sample_border_candidates_3d(
                experts=experts,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                F=args.F, pool_factor=args.pool_factor,
                border_inner_ld=args.border_inner_ld,
                border_outer_ld=args.border_outer_ld,
                rng=rng,
            )
        else:
            candidates = sample_frontier_candidates_3d(
                experts=experts,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                F=args.F, pool_factor=args.pool_factor, rng=rng,
            )
    else:
        if effective_mode == "border":
            candidates, n_border = sample_border_candidates(
                experts=experts,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                F=args.F, pool_factor=args.pool_factor,
                border_inner_ld=args.border_inner_ld,
                border_outer_ld=args.border_outer_ld,
                rng=rng,
            )
        else:
            candidates = sample_frontier_candidates(
                experts=experts,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                F=args.F, pool_factor=args.pool_factor, rng=rng,
            )
    print(f"  Sampled {len(candidates)} candidates.")

    eval_dir = os.path.join(run_dir, "frontier_evals", f"iter_{iteration}")
    os.makedirs(eval_dir, exist_ok=True)

    # ── 3D: generate USDs and annotate candidates with usd_path ───────────────
    usd_order_file = None
    if is_3d:
        usd_dir = os.path.join(eval_dir, "usds")
        leg_map = generate_usds_for_candidates(candidates, usd_dir, args.leg_precision)

        if not leg_map:
            print("[ERROR] USD generation returned empty leg_map.")
            sys.exit(1)

        # Annotate candidates with their usd_path.
        for c in candidates:
            key = leg_key(c["leg"], args.leg_precision)
            if key not in leg_map:
                print(f"[ERROR] No USD for leg key '{key}'.")
                sys.exit(1)
            c["usd_path"] = leg_map[key]

        # Write a single ordered USD file for the eval subprocesses.
        usd_order = [c["usd_path"] for c in candidates]
        usd_order_file = os.path.join(eval_dir, "usd_order.json")
        with open(usd_order_file, "w") as f:
            json.dump(usd_order, f)
        print(f"  USD order file   : {usd_order_file}  ({len(usd_order)} entries)")

    candidates_file = os.path.join(eval_dir, "candidates.json")
    with open(candidates_file, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"  Candidates saved : {candidates_file}")

    # Print preview.
    header = f"  {'ID':>4}  {'GCR':>7}  {'spcf':>9}"
    if is_3d:
        header += f"  {'leg':>7}"
    print(f"\n{header}")
    for c in candidates[:10]:
        row = f"  {c['id']:>4}  {c['GCR']:>7.4f}  {c['spcf']:>9.5f}"
        if is_3d:
            row += f"  {c['leg']:>7.4f}"
        print(row)
    if len(candidates) > 10:
        print(f"  ... ({len(candidates) - 10} more)")

    # ── Run one Isaac Sim subprocess per expert ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Running {K} expert eval subprocesses (sequential)...")
    print(f"{'='*70}")

    scores_matrix  = {}
    expert_results = {}

    t_total_start = time.time()
    for ex in experts:
        kid = ex["id"]
        output_file = os.path.join(eval_dir, f"expert_{kid}_results.json")

        results = run_expert_eval(
            expert=ex,
            candidates_file=candidates_file,
            output_file=output_file,
            args=args,
            usd_rel_path=usd_rel_path,
            usd_order_file=usd_order_file,
        )

        if results is None:
            scores_matrix[kid]  = None
            expert_results[kid] = None
        else:
            results_sorted = sorted(results, key=lambda r: r["id"])
            scores_matrix[kid]  = [r["best_level_idx"] for r in results_sorted]
            expert_results[kid] = results_sorted

    t_total = (time.time() - t_total_start) / 60.0
    print(f"\n{'='*70}")
    print(f"  All expert evals complete — total time: {t_total:.1f} min")

    # ── Assemble and save scores ──────────────────────────────────────────────
    scores_payload = {
        "run_name":         args.run_name,
        "iteration":        iteration,
        "F":                args.F,
        "K":                K,
        "num_episodes":     args.num_episodes,
        "start_difficulty": args.start_difficulty,
        "sampling_mode":    effective_mode,
        "n_border":         n_border,
        "candidates":       candidates,
        "scores_matrix":    scores_matrix,
        "expert_results":   expert_results,
    }

    scores_file = os.path.join(eval_dir, "scores.json")
    with open(scores_file, "w") as f:
        json.dump(scores_payload, f, indent=2)
    print(f"  Scores saved     : {scores_file}")

    print(f"\n  Scores matrix summary:")
    for kid in range(K):
        s = scores_matrix.get(kid)
        if s is None:
            print(f"    Expert {kid}: skipped (no checkpoint)")
        else:
            print(f"    Expert {kid}: mean={sum(s)/len(s):.2f}  min={min(s)}  max={max(s)}")

    print(f"\n  Next step: run pec_refit_gaussians.py --run_name {args.run_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
