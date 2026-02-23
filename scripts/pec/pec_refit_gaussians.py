"""
PEC Gaussian Re-fitting Script
================================
Progressive Expert Coverage — Step 3: Assign frontier designs and re-fit
expert Gaussians.

Algorithm
---------
1. Read pec_state.json (current experts: mu, sigma, designs, checkpoints).
2. Read the scores matrix from
       logs/pec/<run_name>/frontier_evals/iter_<N>/scores.json
   This matrix has shape (K, F): scores_matrix[k][f] = best curriculum level
   achieved by expert k on frontier design f.
3. For each frontier design f, find the winning expert:
       k*(f) = argmax_k  scores_matrix[k][f]
   Ties are broken by expert index (lowest wins).
   Designs where *all* experts were skipped (None scores) are discarded.
4. Append each frontier design to its winning expert's `designs` list.
5. Re-fit each expert's Gaussian from its full design history:
       mu_new    = mean of all assigned [GCR, spcf] designs
       sigma_new = diagonal covariance; variance floored at min_var to prevent
                   collapse in low-diversity clusters.
6. Save the updated state back to pec_state.json and increment `iteration`.
7. Report coverage statistics.

Usage (run from ballu_isclb_extension/)
-----------------------------------------
    python scripts/pec/pec_refit_gaussians.py \\
        --run_name   my_pec_run \\
        --min_var_scale 0.01

Key files
---------
    Input  : logs/pec/<run_name>/pec_state.json
             logs/pec/<run_name>/frontier_evals/iter_<N>/scores.json
    Output : logs/pec/<run_name>/pec_state.json  (updated in-place)
             logs/pec/<run_name>/frontier_evals/iter_<N>/assignments.json
"""

import argparse
import json
import math
import os
import sys
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian fitting helpers (shared with pec_init.py)
# ──────────────────────────────────────────────────────────────────────────────

def fit_gaussian(designs: list,
                 gcr_lo: float, gcr_hi: float,
                 spcf_lo: float, spcf_hi: float,
                 min_var_scale: float) -> tuple[list, list]:
    """
    Fit a diagonal Gaussian to a list of [GCR, spcf] design points.

    Returns
    -------
    mu    : [mu_gcr, mu_spcf]
    sigma : [[var_gcr, 0.0], [0.0, var_spcf]]

    The variance for each dimension is floored at
        (min_var_scale * range_d) ** 2
    to prevent degenerate near-zero covariances when a cluster is very tight.
    """
    arr = np.array(designs, dtype=np.float64)   # (N, 2)

    mu_gcr  = float(arr[:, 0].mean())
    mu_spcf = float(arr[:, 1].mean())

    # Sample variance (ddof=1); use ddof=0 when only one point.
    ddof = 1 if len(arr) > 1 else 0
    var_gcr  = float(arr[:, 0].var(ddof=ddof))
    var_spcf = float(arr[:, 1].var(ddof=ddof))

    # Variance floor.
    floor_gcr  = (min_var_scale * (gcr_hi  - gcr_lo))  ** 2
    floor_spcf = (min_var_scale * (spcf_hi - spcf_lo)) ** 2
    var_gcr  = max(var_gcr,  floor_gcr)
    var_spcf = max(var_spcf, floor_spcf)

    mu    = [mu_gcr, mu_spcf]
    sigma = [[var_gcr, 0.0], [0.0, var_spcf]]
    return mu, sigma


def coverage_estimate(experts: list,
                      gcr_lo: float, gcr_hi: float,
                      spcf_lo: float, spcf_hi: float,
                      threshold: float, n_mc: int = 10_000,
                      rng: np.random.Generator = None) -> float:
    """
    Monte-Carlo estimate of the fraction of the design space covered by the
    union of K Gaussian regions.  A point θ is 'covered' if
        max_k exp(log_density(θ, G_k)) > threshold.
    """
    if rng is None:
        rng = np.random.default_rng()

    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)

    covered = 0
    for gcr_v, spcf_v in zip(gcr_pts, spcf_pts):
        for ex in experts:
            mu    = ex["mu"]
            sigma = ex["sigma"]
            d_g   = (gcr_v  - mu[0]) ** 2 / (2.0 * sigma[0][0])
            d_s   = (spcf_v - mu[1]) ** 2 / (2.0 * sigma[1][1])
            if math.exp(-(d_g + d_s)) > threshold:
                covered += 1
                break   # point is covered; no need to check other experts

    return covered / n_mc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Step 3 — assign frontier designs and re-fit Gaussians."
    )
    parser.add_argument("--run_name",       type=str, required=True,
                        help="PEC run name (must match pec_init.py run).")
    parser.add_argument("--log_root",       type=str, default="logs/pec",
                        help="Root directory for PEC logs (default: logs/pec).")
    parser.add_argument("--min_var_scale",  type=float, default=0.01,
                        help="Minimum std as a fraction of each dimension's range, "
                             "applied as a variance floor during Gaussian re-fitting. "
                             "(default: 0.01)")
    parser.add_argument("--min_designs",    type=int, default=2,
                        help="Minimum number of designs required to re-fit a "
                             "Gaussian; experts with fewer designs keep their "
                             "current mu/sigma (default: 2).")
    parser.add_argument("--coverage_threshold", type=float, default=None,
                        help="Unnormalised density threshold for the MC coverage "
                             "estimate. Defaults to exp(-2) ≈ 0.135 (~2-sigma).")
    parser.add_argument("--seed",           type=int, default=0,
                        help="Random seed for the MC coverage estimate (default: 0).")

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

    experts   = state["experts"]
    iteration = state["iteration"]
    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    K = len(experts)

    print(f"\n{'='*70}")
    print(f"  PEC Refit Gaussians — run: {args.run_name}  |  iter: {iteration}")
    print(f"{'='*70}")

    # ── Load scores ───────────────────────────────────────────────────────────
    scores_file = os.path.join(run_dir, "frontier_evals", f"iter_{iteration}", "scores.json")
    if not os.path.exists(scores_file):
        print(f"[ERROR] Scores file not found: {scores_file}")
        print("        Run pec_evaluate_frontier.py first.")
        sys.exit(1)

    with open(scores_file) as f:
        scores_payload = json.load(f)

    candidates     = scores_payload["candidates"]          # [{id, GCR, spcf}, ...]
    # Keys in JSON are strings; convert to int.
    scores_matrix  = {int(k): v for k, v in scores_payload["scores_matrix"].items()}
    F = len(candidates)

    print(f"  Scores loaded from : {scores_file}")
    print(f"  F candidates       : {F}")
    print(f"  K experts          : {K}")

    # Identify which experts have valid scores.
    valid_experts = [kid for kid, s in scores_matrix.items() if s is not None]
    skipped       = [kid for kid, s in scores_matrix.items() if s is None]
    if skipped:
        print(f"  Skipped experts    : {skipped}  (no checkpoint at eval time)")
    if not valid_experts:
        print("\n[ERROR] All experts were skipped — cannot assign designs.")
        sys.exit(1)

    # ── argmax assignment ─────────────────────────────────────────────────────
    # For each frontier design f, find the expert with the highest score.
    # Designs where all evaluating experts scored 0 are still assigned (to the
    # expert with the lowest index, by the argmax tie-break).
    assignments = {}   # expert_id -> list of [GCR, spcf]
    for ex in experts:
        assignments[ex["id"]] = []

    assignment_log = []   # [{design_id, GCR, spcf, winner_id, scores}, ...]

    for f_idx, cand in enumerate(candidates):
        best_score = -1
        best_kid   = None

        for kid in valid_experts:
            score = scores_matrix[kid][f_idx]
            if best_kid is None or score > best_score:
                best_score = score
                best_kid   = kid

        if best_kid is None:
            continue   # all experts skipped — discard this design

        assignments[best_kid].append([cand["GCR"], cand["spcf"]])
        assignment_log.append({
            "design_id": cand["id"],
            "GCR":       cand["GCR"],
            "spcf":      cand["spcf"],
            "winner_id": best_kid,
            "scores":    {kid: scores_matrix[kid][f_idx]
                          for kid in valid_experts},
        })

    # Save assignment log.
    assignments_file = os.path.join(
        run_dir, "frontier_evals", f"iter_{iteration}", "assignments.json"
    )
    with open(assignments_file, "w") as f:
        json.dump(assignment_log, f, indent=2)
    print(f"  Assignments saved  : {assignments_file}")

    # Print assignment summary.
    print(f"\n  Assignment summary:")
    for kid in range(K):
        n_new = len(assignments[kid])
        n_old = len(experts[kid]["designs"])
        print(f"    Expert {kid}: {n_new:>3} new designs  "
              f"(total after merge: {n_old + n_new})")

    # ── Snapshot current state into history before overwriting ───────────────
    # Record mu, sigma, checkpoint, and design count AT this iteration so that
    # pec_visualize.py --itr N can reconstruct the Gaussians that were active
    # when frontier evaluation iter N was run.
    if "history" not in state:
        state["history"] = []

    snapshot = {
        "iteration": iteration,
        "experts_snapshot": [
            {
                "id":         ex["id"],
                "mu":         ex["mu"][:],
                "sigma":      [row[:] for row in ex["sigma"]],
                "checkpoint": ex.get("checkpoint"),
                "n_designs":  len(ex["designs"]),
            }
            for ex in experts
        ],
    }
    state["history"].append(snapshot)
    print(f"\n  Snapshot saved to history for iteration {iteration}.")

    # ── Append designs and re-fit Gaussians ───────────────────────────────────
    print(f"\n  Re-fitting Gaussians (min_var_scale={args.min_var_scale}, "
          f"min_designs={args.min_designs}):")

    for expert in experts:
        kid = expert["id"]
        new_designs = assignments[kid]

        # Merge new designs into the expert's history.
        expert["designs"].extend(new_designs)
        all_designs = expert["designs"]

        if len(all_designs) < args.min_designs:
            print(f"    Expert {kid}: only {len(all_designs)} design(s) — "
                  f"keeping current mu/sigma.")
            continue

        old_mu    = expert["mu"][:]
        old_std_g = math.sqrt(expert["sigma"][0][0])
        old_std_s = math.sqrt(expert["sigma"][1][1])

        new_mu, new_sigma = fit_gaussian(
            designs=all_designs,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            min_var_scale=args.min_var_scale,
        )

        expert["mu"]    = new_mu
        expert["sigma"] = new_sigma

        new_std_g = math.sqrt(new_sigma[0][0])
        new_std_s = math.sqrt(new_sigma[1][1])

        print(f"    Expert {kid}:")
        print(f"      GCR  : mu {old_mu[0]:.4f} → {new_mu[0]:.4f}   "
              f"std {old_std_g:.4f} → {new_std_g:.4f}")
        print(f"      spcf : mu {old_mu[1]:.5f} → {new_mu[1]:.5f}   "
              f"std {old_std_s:.5f} → {new_std_s:.5f}")
        print(f"      Designs: {len(all_designs)} total  "
              f"({len(new_designs)} new from frontier)")

    # ── Increment iteration ───────────────────────────────────────────────────
    state["iteration"] = iteration + 1

    # ── Save updated state ────────────────────────────────────────────────────
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n  State saved  : {state_path}  (iteration {iteration} → {iteration + 1})")

    # ── Coverage estimate ─────────────────────────────────────────────────────
    threshold = args.coverage_threshold if args.coverage_threshold is not None \
                else math.exp(-2.0)
    rng = np.random.default_rng(args.seed)
    cov = coverage_estimate(
        experts=experts,
        gcr_lo=gcr_lo, gcr_hi=gcr_hi,
        spcf_lo=spcf_lo, spcf_hi=spcf_hi,
        threshold=threshold,
        n_mc=10_000,
        rng=rng,
    )

    print(f"\n  MC coverage estimate (threshold={threshold:.4f}): {cov*100:.1f}%")
    print(f"\n  Updated expert Gaussians:")
    for ex in experts:
        std_g = math.sqrt(ex["sigma"][0][0])
        std_s = math.sqrt(ex["sigma"][1][1])
        print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {std_g:.4f}   "
              f"spcf={ex['mu'][1]:.5f} ± {std_s:.5f}   "
              f"({len(ex['designs'])} designs)")

    print(f"\n  Next step: re-train each expert with pec_train_expert.py "
          f"(iteration is now {iteration + 1}).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
