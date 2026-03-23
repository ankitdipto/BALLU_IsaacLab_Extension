"""
PEC Gaussian Re-fitting Script
================================
Progressive Expert Coverage — Step 3: Assign frontier designs and re-fit
expert Gaussians.

Supports 2D (GCR, spcf) and 3D (GCR, spcf, leg) modes based on the
design_space in pec_state.json.

3D changes:
  - Designs stored as [gcr, spcf, leg] triplets.
  - fit_gaussian_3d: diagonal 3×3 MLE covariance with per-axis variance floor.
  - coverage_estimate_3d: 3D MC with exp(-2) threshold.

Usage (run from ballu_isclb_extension/)
-----------------------------------------
    python scripts/pec/pec_refit_gaussians.py --run_name my_pec_run
"""

import argparse
import json
import math
import os
import sys
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 2D Gaussian fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_gaussian(designs: list,
                 gcr_lo: float, gcr_hi: float,
                 spcf_lo: float, spcf_hi: float,
                 min_var_scale: float) -> tuple[list, list]:
    """Fit a 2D diagonal Gaussian to [GCR, spcf] design points."""
    arr = np.array(designs, dtype=np.float64)
    mu_gcr  = float(arr[:, 0].mean())
    mu_spcf = float(arr[:, 1].mean())
    ddof = 1 if len(arr) > 1 else 0
    var_gcr  = float(arr[:, 0].var(ddof=ddof))
    var_spcf = float(arr[:, 1].var(ddof=ddof))
    floor_gcr  = (min_var_scale * (gcr_hi  - gcr_lo))  ** 2
    floor_spcf = (min_var_scale * (spcf_hi - spcf_lo)) ** 2
    var_gcr  = max(var_gcr,  floor_gcr)
    var_spcf = max(var_spcf, floor_spcf)
    return [mu_gcr, mu_spcf], [[var_gcr, 0.0], [0.0, var_spcf]]


def coverage_estimate(experts, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                      threshold, n_mc=10_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)
    covered = 0
    for gcr_v, spcf_v in zip(gcr_pts, spcf_pts):
        for ex in experts:
            mu = ex["mu"]; sigma = ex["sigma"]
            d_g = (gcr_v - mu[0]) ** 2 / (2.0 * sigma[0][0])
            d_s = (spcf_v - mu[1]) ** 2 / (2.0 * sigma[1][1])
            if math.exp(-(d_g + d_s)) > threshold:
                covered += 1
                break
    return covered / n_mc


# ──────────────────────────────────────────────────────────────────────────────
# 3D Gaussian fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_gaussian_3d(designs: list,
                    gcr_lo: float, gcr_hi: float,
                    spcf_lo: float, spcf_hi: float,
                    leg_lo: float, leg_hi: float,
                    min_var_scale: float) -> tuple[list, list]:
    """
    Fit a 3D diagonal Gaussian to [gcr, spcf, leg] design points.

    Returns
    -------
    mu    : [mu_gcr, mu_spcf, mu_leg]
    sigma : 3×3 list with variances on the diagonal, zeros off-diagonal
    """
    arr = np.array(designs, dtype=np.float64)   # (N, 3)
    ddof = 1 if len(arr) > 1 else 0

    mu_gcr  = float(arr[:, 0].mean())
    mu_spcf = float(arr[:, 1].mean())
    mu_leg  = float(arr[:, 2].mean())

    var_gcr  = float(arr[:, 0].var(ddof=ddof))
    var_spcf = float(arr[:, 1].var(ddof=ddof))
    var_leg  = float(arr[:, 2].var(ddof=ddof))

    floor_gcr  = (min_var_scale * (gcr_hi  - gcr_lo))  ** 2
    floor_spcf = (min_var_scale * (spcf_hi - spcf_lo)) ** 2
    floor_leg  = (min_var_scale * (leg_hi  - leg_lo))  ** 2

    var_gcr  = max(var_gcr,  floor_gcr)
    var_spcf = max(var_spcf, floor_spcf)
    var_leg  = max(var_leg,  floor_leg)

    mu    = [mu_gcr, mu_spcf, mu_leg]
    sigma = [[var_gcr, 0.0, 0.0],
             [0.0, var_spcf, 0.0],
             [0.0, 0.0, var_leg]]
    return mu, sigma


def coverage_estimate_3d(experts, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                          leg_lo, leg_hi, threshold, n_mc=10_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)
    leg_pts  = rng.uniform(leg_lo,  leg_hi,  size=n_mc)
    covered = 0
    for gcr_v, spcf_v, leg_v in zip(gcr_pts, spcf_pts, leg_pts):
        for ex in experts:
            mu = ex["mu"]; sigma = ex["sigma"]
            d_g = (gcr_v  - mu[0]) ** 2 / (2.0 * sigma[0][0])
            d_s = (spcf_v - mu[1]) ** 2 / (2.0 * sigma[1][1])
            d_l = (leg_v  - mu[2]) ** 2 / (2.0 * sigma[2][2])
            if math.exp(-(d_g + d_s + d_l)) > threshold:
                covered += 1
                break
    return covered / n_mc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Step 3 — assign frontier designs and re-fit Gaussians."
    )
    parser.add_argument("--run_name",       type=str, required=True)
    parser.add_argument("--log_root",       type=str, default="logs/pec")
    parser.add_argument("--min_var_scale",  type=float, default=0.01)
    parser.add_argument("--min_designs",    type=int, default=2)
    parser.add_argument("--coverage_threshold", type=float, default=None)
    parser.add_argument("--seed",           type=int, default=0)

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
    is_3d = "leg" in state["design_space"]
    K = len(experts)

    if is_3d:
        leg_lo, leg_hi = state["design_space"]["leg"]

    print(f"\n{'='*70}")
    print(f"  PEC Refit Gaussians — run: {args.run_name}  |  iter: {iteration}"
          f"  |  mode: {'3D' if is_3d else '2D'}")
    print(f"{'='*70}")

    # ── Load scores ───────────────────────────────────────────────────────────
    scores_file = os.path.join(run_dir, "frontier_evals", f"iter_{iteration}", "scores.json")
    if not os.path.exists(scores_file):
        print(f"[ERROR] Scores file not found: {scores_file}")
        sys.exit(1)

    with open(scores_file) as f:
        scores_payload = json.load(f)

    candidates    = scores_payload["candidates"]
    scores_matrix = {int(k): v for k, v in scores_payload["scores_matrix"].items()}
    F = len(candidates)

    print(f"  Scores loaded : {scores_file}")
    print(f"  F candidates  : {F}  |  K experts: {K}")

    valid_experts = [kid for kid, s in scores_matrix.items() if s is not None]
    skipped       = [kid for kid, s in scores_matrix.items() if s is None]
    if skipped:
        print(f"  Skipped experts: {skipped}")
    if not valid_experts:
        print("\n[ERROR] All experts were skipped — cannot assign designs.")
        sys.exit(1)

    # ── argmax assignment ─────────────────────────────────────────────────────
    assignments   = {ex["id"]: [] for ex in experts}
    assignment_log = []

    for f_idx, cand in enumerate(candidates):
        best_score, best_kid = -1, None
        for kid in valid_experts:
            score = scores_matrix[kid][f_idx]
            if best_kid is None or score > best_score:
                best_score, best_kid = score, kid
        if best_kid is None:
            continue

        if is_3d:
            design = [cand["GCR"], cand["spcf"], cand["leg"]]
        else:
            design = [cand["GCR"], cand["spcf"]]
        assignments[best_kid].append(design)
        assignment_log.append({
            "design_id": cand["id"],
            "GCR":       cand["GCR"],
            "spcf":      cand["spcf"],
            **({"leg": cand["leg"]} if is_3d else {}),
            "winner_id": best_kid,
            "scores":    {kid: scores_matrix[kid][f_idx] for kid in valid_experts},
        })

    assignments_file = os.path.join(
        run_dir, "frontier_evals", f"iter_{iteration}", "assignments.json"
    )
    with open(assignments_file, "w") as f:
        json.dump(assignment_log, f, indent=2)
    print(f"  Assignments   : {assignments_file}")

    print(f"\n  Assignment summary:")
    for kid in range(K):
        n_new = len(assignments[kid])
        n_old = len(experts[kid]["designs"])
        print(f"    Expert {kid}: {n_new:>3} new  (total: {n_old + n_new})")

    # ── Snapshot before overwriting ───────────────────────────────────────────
    if "history" not in state:
        state["history"] = []

    if is_3d:
        snapshot_sigma = [[row[:] for row in ex["sigma"]] for ex in experts]
    else:
        snapshot_sigma = [[row[:] for row in ex["sigma"]] for ex in experts]

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
    print(f"\n  Re-fitting (min_var_scale={args.min_var_scale}, "
          f"min_designs={args.min_designs}):")

    for expert in experts:
        kid = expert["id"]
        expert["designs"].extend(assignments[kid])
        all_designs = expert["designs"]

        if len(all_designs) < args.min_designs:
            print(f"    Expert {kid}: only {len(all_designs)} design(s) — keeping current.")
            continue

        old_mu = expert["mu"][:]

        if is_3d:
            new_mu, new_sigma = fit_gaussian_3d(
                designs=all_designs,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                min_var_scale=args.min_var_scale,
            )
            old_std_g = math.sqrt(expert["sigma"][0][0])
            old_std_s = math.sqrt(expert["sigma"][1][1])
            old_std_l = math.sqrt(expert["sigma"][2][2])
            new_std_g = math.sqrt(new_sigma[0][0])
            new_std_s = math.sqrt(new_sigma[1][1])
            new_std_l = math.sqrt(new_sigma[2][2])
            print(f"    Expert {kid}:")
            print(f"      GCR  : mu {old_mu[0]:.4f} → {new_mu[0]:.4f}   "
                  f"std {old_std_g:.4f} → {new_std_g:.4f}")
            print(f"      spcf : mu {old_mu[1]:.5f} → {new_mu[1]:.5f}   "
                  f"std {old_std_s:.5f} → {new_std_s:.5f}")
            print(f"      leg  : mu {old_mu[2]:.4f} → {new_mu[2]:.4f}   "
                  f"std {old_std_l:.4f} → {new_std_l:.4f}")
            print(f"      Designs: {len(all_designs)} total")
        else:
            new_mu, new_sigma = fit_gaussian(
                designs=all_designs,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                min_var_scale=args.min_var_scale,
            )
            old_std_g = math.sqrt(expert["sigma"][0][0])
            old_std_s = math.sqrt(expert["sigma"][1][1])
            new_std_g = math.sqrt(new_sigma[0][0])
            new_std_s = math.sqrt(new_sigma[1][1])
            print(f"    Expert {kid}:")
            print(f"      GCR  : mu {old_mu[0]:.4f} → {new_mu[0]:.4f}   "
                  f"std {old_std_g:.4f} → {new_std_g:.4f}")
            print(f"      spcf : mu {old_mu[1]:.5f} → {new_mu[1]:.5f}   "
                  f"std {old_std_s:.5f} → {new_std_s:.5f}")
            print(f"      Designs: {len(all_designs)} total")

        expert["mu"]    = new_mu
        expert["sigma"] = new_sigma

    # ── Increment iteration ───────────────────────────────────────────────────
    state["iteration"] = iteration + 1

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n  State saved  : {state_path}  (iter {iteration} → {iteration + 1})")

    # ── Coverage estimate ─────────────────────────────────────────────────────
    threshold = args.coverage_threshold if args.coverage_threshold is not None \
                else math.exp(-2.0)
    rng = np.random.default_rng(args.seed)

    if is_3d:
        cov = coverage_estimate_3d(
            experts=experts,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            leg_lo=leg_lo, leg_hi=leg_hi,
            threshold=threshold, n_mc=10_000, rng=rng,
        )
    else:
        cov = coverage_estimate(
            experts=experts,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            threshold=threshold, n_mc=10_000, rng=rng,
        )

    print(f"\n  MC coverage estimate: {cov*100:.1f}%  (threshold={threshold:.4f})")
    print(f"\n  Updated expert Gaussians:")
    for ex in experts:
        std_g = math.sqrt(ex["sigma"][0][0])
        std_s = math.sqrt(ex["sigma"][1][1])
        if is_3d:
            std_l = math.sqrt(ex["sigma"][2][2])
            print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {std_g:.4f}   "
                  f"spcf={ex['mu'][1]:.5f} ± {std_s:.5f}   "
                  f"leg={ex['mu'][2]:.4f} ± {std_l:.4f}   "
                  f"({len(ex['designs'])} designs)")
        else:
            print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {std_g:.4f}   "
                  f"spcf={ex['mu'][1]:.5f} ± {std_s:.5f}   "
                  f"({len(ex['designs'])} designs)")

    print(f"\n  Next: re-train each expert (iteration is now {iteration + 1}).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
