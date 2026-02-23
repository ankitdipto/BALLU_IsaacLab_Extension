"""
PEC Frontier Evaluation Orchestrator
=====================================
Progressive Expert Coverage — Step 2: Evaluate frontier candidates across
all K experts.

Algorithm
---------
1. Read pec_state.json to get the current Gaussian parameters for each expert.
2. Sample F candidate designs using one of three strategies (--sampling_mode):

   frontier (default)
       Keep the F pool points with the *lowest* max_k log p(θ|G_k) — designs
       farthest from every Gaussian.  Best for mature runs when coverage gaps
       are large.

   border
       Keep pool points whose max log-density falls in the band
           (border_outer_ld, border_inner_ld)
       i.e. just outside the Gaussian cores but not in the wilderness.
       Sorted ascending by max log-density so we expand toward the outer edge
       of the band first.  Falls back to frontier for any shortfall.
       Recommended for early iterations when experts are not yet capable of
       controlling fully uncharted designs.

   auto
       Uses 'border' for iterations < auto_switch_iter, then 'frontier'.
       This produces a gradual expansion: ring-by-ring at first, then gap-fill.

3. For each of the K experts (sequentially): launch pec_eval_expert_frontier.py
   as a subprocess; that script runs the expert checkpoint across all F frontier
   environments in parallel and writes a per-design results file.
4. Aggregate into a scores matrix S (K × F) where S[k][f] = best curriculum
   level index achieved by expert k on frontier design f.
5. Save candidates + scores to
       logs/pec/<run_name>/frontier_evals/iter_<N>/scores.json

Density band for border mode
-----------------------------
For any design θ we compute max_k log p(θ|G_k)  (unnormalised diagonal
Gaussian, value ≤ 0):
  - close to a Gaussian center  → value near 0        (e.g. 0.0)
  - on the 1-sigma ring         → value ≈ -0.5 to -1.0
  - on the 2-sigma ring         → value ≈ -2.0 to -4.0
  - far from all Gaussians      → value << -4.0

border_inner_ld (default -0.5) is the upper cut: designs with max log-density
ABOVE this are already well-covered and excluded.
border_outer_ld (default -3.0) is the lower cut: designs BELOW this are in the
frontier wilderness and excluded in border mode (but used by frontier mode).

Usage (run from ballu_isclb_extension/)
-----------------------------------------
    # Early phase — border expansion:
    python scripts/pec/pec_evaluate_frontier.py \\
        --run_name my_pec_run --F 50 --sampling_mode border \\
        --headless

    # Later phase — frontier gap-filling:
    python scripts/pec/pec_evaluate_frontier.py \\
        --run_name my_pec_run --F 50 --sampling_mode frontier \\
        --headless

    # Automatic switching:
    python scripts/pec/pec_evaluate_frontier.py \\
        --run_name my_pec_run --F 50 --sampling_mode auto \\
        --auto_switch_iter 4 --headless

Key files
---------
    State     : logs/pec/<run_name>/pec_state.json
    Output    : logs/pec/<run_name>/frontier_evals/iter_<N>/scores.json
                logs/pec/<run_name>/frontier_evals/iter_<N>/candidates.json
                logs/pec/<run_name>/frontier_evals/iter_<N>/expert_<k>_results.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # ballu_isclb_extension/
EVAL_SCRIPT  = os.path.join(SCRIPT_DIR, "pec_eval_expert_frontier.py")


# ──────────────────────────────────────────────────────────────────────────────
# Frontier sampling
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_log_density(theta_gcr: float, theta_spcf: float, expert: dict) -> float:
    """Unnormalised log-density of a point under one expert's diagonal Gaussian."""
    mu    = expert["mu"]
    sigma = expert["sigma"]
    d_g   = (theta_gcr  - mu[0]) ** 2 / (2.0 * sigma[0][0])
    d_s   = (theta_spcf - mu[1]) ** 2 / (2.0 * sigma[1][1])
    return -(d_g + d_s)


def sample_frontier_candidates(
    experts: list,
    gcr_lo: float, gcr_hi: float,
    spcf_lo: float, spcf_hi: float,
    F: int,
    pool_factor: int,
    rng: np.random.Generator,
) -> list:
    """
    Sample F frontier candidates from low-density regions.

    Strategy: generate pool_factor * F uniform random points, compute the
    maximum (unnormalised) Gaussian density across all experts at each point,
    and return the F points with the *lowest* max-density (i.e., the designs
    that are farthest from every existing expert).

    Returns
    -------
    list of dicts [{id, GCR, spcf}, ...], length F
    """
    n_pool = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)

    # For each pool point compute max log-density across experts.
    max_log_densities = np.full(n_pool, -np.inf)
    for ex in experts:
        mu    = ex["mu"]
        sigma = ex["sigma"]
        d_g   = (gcr_pool  - mu[0]) ** 2 / (2.0 * sigma[0][0])
        d_s   = (spcf_pool - mu[1]) ** 2 / (2.0 * sigma[1][1])
        log_d = -(d_g + d_s)
        max_log_densities = np.maximum(max_log_densities, log_d)

    # Rank ascending: smallest max-density → least covered (frontier).
    frontier_idx = np.argsort(max_log_densities)[:F]

    candidates = []
    for rank, idx in enumerate(frontier_idx):
        candidates.append({
            "id":   rank,
            "GCR":  float(gcr_pool[idx]),
            "spcf": float(spcf_pool[idx]),
        })
    return candidates


def sample_border_candidates(
    experts: list,
    gcr_lo: float, gcr_hi: float,
    spcf_lo: float, spcf_hi: float,
    F: int,
    pool_factor: int,
    border_inner_ld: float,
    border_outer_ld: float,
    rng: np.random.Generator,
) -> tuple[list, int]:
    """
    Sample F candidates from the annular border band around the Gaussians.

    A pool point θ is in the border band when:
        border_outer_ld  <  max_k log p(θ|G_k)  <  border_inner_ld

    Within the band, points are sorted ascending by max log-density so the
    least-covered border designs (closest to the outer edge of the band) come
    first.

    If the band contains fewer than F points the shortfall is filled by
    frontier-mode points (globally lowest density, outside the band), with a
    warning printed to stdout.

    Returns
    -------
    candidates : list of dicts [{id, GCR, spcf}, ...], length == F
    n_border   : how many of the F came from the border band (rest: frontier)
    """
    n_pool = pool_factor * F
    gcr_pool  = rng.uniform(gcr_lo,  gcr_hi,  size=n_pool)
    spcf_pool = rng.uniform(spcf_lo, spcf_hi, size=n_pool)

    # Compute max log-density across all experts for every pool point.
    max_log_densities = np.full(n_pool, -np.inf)
    for ex in experts:
        mu    = ex["mu"]
        sigma = ex["sigma"]
        d_g   = (gcr_pool  - mu[0]) ** 2 / (2.0 * sigma[0][0])
        d_s   = (spcf_pool - mu[1]) ** 2 / (2.0 * sigma[1][1])
        max_log_densities = np.maximum(max_log_densities, -(d_g + d_s))

    # Split pool into border and frontier sets.
    in_band  = (max_log_densities > border_outer_ld) & (max_log_densities < border_inner_ld)
    in_front = ~in_band   # everything outside the band (including very low density)

    border_idx   = np.where(in_band)[0]
    frontier_idx = np.where(in_front)[0]

    # Sort border ascending (lowest max-density = outermost edge of the band first).
    border_idx   = border_idx[np.argsort(max_log_densities[border_idx])]
    # Sort frontier ascending (lowest max-density = farthest from all Gaussians first).
    frontier_idx = frontier_idx[np.argsort(max_log_densities[frontier_idx])]

    n_border_avail = len(border_idx)
    n_border_take  = min(n_border_avail, F)
    n_frontier_take = F - n_border_take

    if n_frontier_take > 0:
        print(f"  [INFO] Border band contains {n_border_avail} / {F} needed points "
              f"— supplementing {n_frontier_take} from frontier pool.")

    selected_idx = np.concatenate([
        border_idx[:n_border_take],
        frontier_idx[:n_frontier_take],
    ])

    candidates = []
    for rank, idx in enumerate(selected_idx):
        candidates.append({
            "id":   rank,
            "GCR":  float(gcr_pool[idx]),
            "spcf": float(spcf_pool[idx]),
        })

    return candidates, n_border_take


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess launcher
# ──────────────────────────────────────────────────────────────────────────────

def run_expert_eval(
    expert: dict,
    candidates_file: str,
    output_file: str,
    args,
    usd_rel_path: str | None = None,
) -> list | None:
    """
    Launch pec_eval_expert_frontier.py for one expert and block until done.

    Returns the list of per-design result dicts on success, None on failure.
    """
    checkpoint = expert.get("checkpoint")
    if not checkpoint:
        print(f"  [WARNING] Expert {expert['id']} has no checkpoint — skipping.")
        return None
    if not os.path.exists(checkpoint):
        print(f"  [WARNING] Checkpoint not found: {checkpoint} — skipping expert {expert['id']}.")
        return None

    cmd = [
        sys.executable,
        EVAL_SCRIPT,
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
    print(f"  Command: {' '.join(cmd)}")
    print(f"  {'─'*66}")

    # Inherit current environment and inject BALLU_USD_REL_PATH if set.
    subprocess_env = os.environ.copy()
    if usd_rel_path:
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
            # Parse the inline results summary from the eval script.
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
        print(f"\n  [ERROR] Expert {expert['id']} eval timed out after {args.timeout_h:.1f} h.")
        return None

    elapsed = (time.time() - t0) / 60.0
    print(f"\n  Expert {expert['id']} eval finished in {elapsed:.1f} min  "
          f"(exit={process.returncode})")

    if process.returncode != 0:
        print(f"  [ERROR] Subprocess exited with code {process.returncode}.")
        return None

    # Fallback: if stdout parsing failed, read the output file directly.
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
    # PEC state
    parser.add_argument("--run_name",   type=str, required=True,
                        help="PEC run name (must match pec_init.py run).")
    parser.add_argument("--log_root",   type=str, default="logs/pec",
                        help="Root directory for PEC logs (default: logs/pec).")

    # Frontier / border sampling
    parser.add_argument("--F",          type=int, default=50,
                        help="Number of candidate designs to evaluate (default: 50).")
    parser.add_argument("--pool_factor", type=int, default=20,
                        help="Pool size = pool_factor * F uniform random points "
                             "from which F are selected (default: 20).")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for candidate sampling. "
                             "Ignored when --use_iter_seed is set (default: 0).")
    parser.add_argument("--use_iter_seed", action="store_true",
                        help="Use the current PEC iteration as the random seed "
                             "(ensures different candidates each iteration).")
    parser.add_argument("--sampling_mode", type=str, default="frontier",
                        choices=["frontier", "border", "auto"],
                        help="Candidate sampling strategy.\n"
                             "  frontier — lowest max log-density (farthest from all Gaussians).\n"
                             "  border   — annular band just outside the Gaussian cores.\n"
                             "  auto     — border for iter < auto_switch_iter, then frontier.\n"
                             "(default: frontier)")
    parser.add_argument("--border_inner_ld", type=float, default=-0.5,
                        help="Upper max log-density cut for border mode. "
                             "Designs with max log-density ABOVE this are already well-covered "
                             "and excluded. (default: -0.5, roughly inside the 1-sigma core)")
    parser.add_argument("--border_outer_ld", type=float, default=-3.0,
                        help="Lower max log-density cut for border mode. "
                             "Designs BELOW this are in the frontier wilderness and excluded "
                             "in border mode (but used by frontier mode). "
                             "(default: -3.0, roughly beyond 2-sigma)")
    parser.add_argument("--auto_switch_iter", type=int, default=4,
                        help="In 'auto' mode, switch from border to frontier sampling "
                             "after this many PEC iterations (default: 4).")

    # Eval subprocess args forwarded to pec_eval_expert_frontier.py
    parser.add_argument("--num_episodes",    type=int, default=30,
                        help="Episodes per frontier design during eval (default: 30).")
    parser.add_argument("--start_difficulty", type=int, default=15,
                        help="Starting obstacle level index (default: 15).")
    parser.add_argument("--task",       type=str, default="Isc-BALLU-hetero-general",
                        help="Isaac Lab task name (default: Isc-BALLU-hetero-general).")
    parser.add_argument("--device",     type=str, default="cuda:0",
                        help="Torch device (default: cuda:0).")
    parser.add_argument("--headless",   action="store_true",
                        help="Run Isaac Sim in headless mode.")
    parser.add_argument("--timeout_h",  type=float, default=4.0,
                        help="Per-expert eval subprocess timeout in hours (default: 4).")

    args = parser.parse_args()

    # ── Load PEC state ────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        print("        Run pec_init.py first.")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    experts      = state["experts"]
    iteration    = state["iteration"]
    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    usd_rel_path = state.get("usd_rel_path")
    K = len(experts)

    seed = iteration if args.use_iter_seed else args.seed
    rng  = np.random.default_rng(seed)

    # Resolve effective sampling mode for auto.
    if args.sampling_mode == "auto":
        effective_mode = "border" if iteration < args.auto_switch_iter else "frontier"
        print(f"  [auto] iteration={iteration}  auto_switch_iter={args.auto_switch_iter}"
              f"  → using '{effective_mode}' sampling.")
    else:
        effective_mode = args.sampling_mode

    print(f"\n{'='*70}")
    print(f"  PEC Evaluate Frontier — run: {args.run_name}  |  iter: {iteration}")
    print(f"{'='*70}")
    print(f"  K experts        : {K}")
    print(f"  F candidates     : {args.F}")
    print(f"  Pool factor      : {args.pool_factor}  ({args.pool_factor * args.F} pool points)")
    print(f"  usd_rel_path     : {usd_rel_path or '(not set)'}")
    print(f"  Sampling mode    : {effective_mode}  (requested: {args.sampling_mode})")
    if effective_mode == "border":
        print(f"  Border band      : ({args.border_outer_ld:.2f}, {args.border_inner_ld:.2f})"
              f"  [outer_ld, inner_ld]")
    print(f"  Candidate seed   : {seed}")
    print(f"  Episodes / design: {args.num_episodes}")
    print(f"  Start difficulty : {args.start_difficulty}")
    print(f"  Timeout / expert : {args.timeout_h:.1f} h")

    # ── Check that all experts have checkpoints ───────────────────────────────
    untrained = [ex["id"] for ex in experts if not ex.get("checkpoint")]
    if untrained:
        print(f"\n[WARNING] Experts without checkpoints: {untrained}")
        print("          These will be skipped during evaluation.")
        print("          Run pec_train_expert.py for each before evaluating frontiers.")

    # ── Sample candidates ─────────────────────────────────────────────────────
    print(f"\n  Sampling candidates (mode='{effective_mode}')...")
    n_border = 0
    if effective_mode == "border":
        candidates, n_border = sample_border_candidates(
            experts=experts,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            F=args.F,
            pool_factor=args.pool_factor,
            border_inner_ld=args.border_inner_ld,
            border_outer_ld=args.border_outer_ld,
            rng=rng,
        )
        n_frontier_supp = len(candidates) - n_border
        print(f"  Sampled {len(candidates)} designs "
              f"({n_border} from border band, {n_frontier_supp} frontier supplement).")
    else:
        candidates = sample_frontier_candidates(
            experts=experts,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            F=args.F,
            pool_factor=args.pool_factor,
            rng=rng,
        )
        print(f"  Sampled {len(candidates)} frontier designs.")

    # ── Create output directory ───────────────────────────────────────────────
    eval_dir = os.path.join(run_dir, "frontier_evals", f"iter_{iteration}")
    os.makedirs(eval_dir, exist_ok=True)

    candidates_file = os.path.join(eval_dir, "candidates.json")
    with open(candidates_file, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"  Candidates saved : {candidates_file}")

    # Print candidate summary.
    print(f"\n  {'ID':>4}  {'GCR':>7}  {'spcf':>9}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*9}")
    for c in candidates[:10]:
        print(f"  {c['id']:>4}  {c['GCR']:>7.4f}  {c['spcf']:>9.5f}")
    if len(candidates) > 10:
        print(f"  ... ({len(candidates) - 10} more)")

    # ── Run one Isaac Sim subprocess per expert ───────────────────────────────
    # Isaac Sim can only run one process at a time on a single GPU, so we
    # iterate sequentially.
    print(f"\n{'='*70}")
    print(f"  Running {K} expert eval subprocesses (sequential, one per expert)...")
    print(f"{'='*70}")

    # scores_matrix[k] = list of best_level_idx for each of the F candidates.
    # None means the expert was skipped (no checkpoint).
    scores_matrix = {}       # expert_id -> list[int] | None
    expert_results = {}      # expert_id -> list[dict] | None

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
        )

        if results is None:
            scores_matrix[kid] = None
            expert_results[kid] = None
        else:
            # results is [{id, GCR, spcf, best_level_idx, best_height_m}, ...]
            # Sort by candidate id to guarantee consistent ordering.
            results_sorted = sorted(results, key=lambda r: r["id"])
            scores_matrix[kid] = [r["best_level_idx"] for r in results_sorted]
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
        "scores_matrix":    scores_matrix,   # {str(expert_id): [int, ...] | null}
        "expert_results":   expert_results,  # full per-design detail per expert
    }

    scores_file = os.path.join(eval_dir, "scores.json")
    with open(scores_file, "w") as f:
        json.dump(scores_payload, f, indent=2)

    print(f"  Scores saved     : {scores_file}")

    # ── Print scores matrix summary ───────────────────────────────────────────
    print(f"\n  Scores matrix (best curriculum level per design):")
    header = f"  {'Design':>8}" + "".join(f"  Expert{k:>2}" for k in range(K))
    print(header)
    print("  " + "-" * (8 + K * 10))
    for f_idx, cand in enumerate(candidates[:20]):   # show first 20
        row = f"  {f_idx:>8}"
        for kid in range(K):
            s = scores_matrix.get(kid)
            val = s[f_idx] if s is not None else "  skip"
            row += f"  {val:>8}" if isinstance(val, int) else f"  {val:>8}"
        print(row)
    if len(candidates) > 20:
        print(f"  ... ({len(candidates) - 20} more rows)")

    # Mean score per expert across valid designs.
    print(f"\n  Mean best-level per expert:")
    for kid in range(K):
        s = scores_matrix.get(kid)
        if s is None:
            print(f"    Expert {kid}: skipped (no checkpoint)")
        else:
            print(f"    Expert {kid}: mean={sum(s)/len(s):.2f}  "
                  f"min={min(s)}  max={max(s)}")

    print(f"\n  Next step: run pec_refit_gaussians.py --run_name {args.run_name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
