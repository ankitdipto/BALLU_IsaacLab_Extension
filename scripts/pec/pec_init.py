"""
PEC Initialization Script
=========================
Progressive Expert Coverage — Step 0: Bootstrap

Creates the initial PEC state file with K experts, each assigned a Gaussian
region in the 2D (GCR, spcf) design space. Experts can be seeded either on the
legacy grid or with a boundary-aware stochastic farthest-point initializer that
spreads centers across the design space in a seed-controlled way.

For each expert the script also draws N_init design points from its Gaussian
and records them as the initial training set.

Output
------
    logs/pec/<run_name>/pec_state.json   (created / overwritten)
"""

import argparse
import json
import math
import os
import sys
import numpy as np

DEFAULT_COVERAGE_SEED = 1729


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def grid_centers(K: int, gcr_lo: float, gcr_hi: float,
                 spcf_lo: float, spcf_hi: float):
    """
    Place K Gaussian centers on the closest square(-ish) grid inside the
    design space.  For K that is a perfect square the grid is exactly square;
    for other values it falls back to the factorisation that minimises the
    aspect-ratio difference.

    Returns
    -------
    list of (gcr, spcf) tuples, length K
    """
    # Find the grid dimensions n_gcr × n_spcf ≥ K that minimise |n_gcr - n_spcf|
    best = None
    for n1 in range(1, K + 1):
        n2 = math.ceil(K / n1)
        if best is None or abs(n1 - n2) < abs(best[0] - best[1]):
            best = (n1, n2)
    n_gcr, n_spcf = best

    # Evenly-spaced centres (cell centres, not edges)
    gcr_centers  = np.linspace(gcr_lo  + (gcr_hi  - gcr_lo)  / (2 * n_gcr),
                               gcr_hi  - (gcr_hi  - gcr_lo)  / (2 * n_gcr),
                               n_gcr)
    spcf_centers = np.linspace(spcf_lo + (spcf_hi - spcf_lo) / (2 * n_spcf),
                               spcf_hi - (spcf_hi - spcf_lo) / (2 * n_spcf),
                               n_spcf)

    centers = []
    for g in gcr_centers:
        for s in spcf_centers:
            centers.append((float(g), float(s)))

    # Keep only the first K (in case n_gcr * n_spcf > K)
    return centers[:K]


def stochastic_fps_centers(K: int,
                           gcr_lo: float, gcr_hi: float,
                           spcf_lo: float, spcf_hi: float,
                           pool_size: int,
                           anchor_region: str | None,
                           jitter_scale: float,
                           rng: np.random.Generator):
    """
    Sample K well-spread centers with boundary-aware stochastic farthest-point
    sampling.

    The algorithm works in normalized [0, 1]^2 space:
    1. Draw a uniform candidate pool.
    2. Optionally anchor the first center to a user-preferred region.
    3. Otherwise prefer an interior first center.
    4. Repeatedly pick the candidate with the best spread vs boundary-clearance score.
    5. Optionally add a small seed-controlled jitter, then clip to bounds.

    This keeps the spread-seeking behavior of FPS while discouraging corner/
    edge placements that tend to appear for small K in a rectangular box.
    """
    if pool_size < K:
        raise ValueError(f"pool_size must be >= K, got pool_size={pool_size}, K={K}")

    pool = rng.uniform(0.0, 1.0, size=(pool_size, 2))
    chosen = np.zeros(pool_size, dtype=bool)

    # Boundary clearance in normalized space: 0 on the boundary, 0.5 at center.
    # The selection score later uses a maximin criterion against both already
    # chosen centers and the box boundary, which yields much more symmetric
    # placements than plain FPS in a rectangle.
    boundary_clearance = np.min(np.stack([
        pool[:, 0],
        1.0 - pool[:, 0],
        pool[:, 1],
        1.0 - pool[:, 1],
    ], axis=1), axis=1)
    boundary_gain = 2.0

    if anchor_region == "top_right":
        # Anchor to a valuable interior point near the top-right, not the literal
        # corner. This preserves room for the other experts to spread out while
        # ensuring the initialization always covers that region.
        anchor_target = np.array([0.88, 0.88], dtype=np.float64)
        anchor_dist_sq = np.sum((pool - anchor_target) ** 2, axis=1)
        anchor_score = -anchor_dist_sq + 0.10 * boundary_clearance + 1e-6 * rng.random(pool_size)
        first_idx = int(np.argmax(anchor_score))
    else:
        first_idx = int(np.argmax(boundary_clearance + 1e-6 * rng.random(pool_size)))
    chosen[first_idx] = True
    selected = [pool[first_idx].copy()]

    min_dist_sq = np.sum((pool - pool[first_idx]) ** 2, axis=1)
    min_dist_sq[chosen] = -np.inf

    for _ in range(1, K):
        nearest_dist = np.sqrt(np.maximum(min_dist_sq, 0.0))
        score = np.minimum(nearest_dist, boundary_gain * boundary_clearance)
        score[chosen] = -np.inf
        idx = int(np.argmax(score))
        chosen[idx] = True
        selected.append(pool[idx].copy())
        dist_sq = np.sum((pool - pool[idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        min_dist_sq[chosen] = -np.inf

    centers_unit = np.array(selected, dtype=np.float64)
    if jitter_scale > 0.0:
        jitter_std = jitter_scale / math.sqrt(max(K, 1))
        centers_unit += rng.normal(loc=0.0, scale=jitter_std, size=centers_unit.shape)
        centers_unit = np.clip(centers_unit, 0.0, 1.0)

    centers = []
    for u_gcr, u_spcf in centers_unit:
        centers.append((
            float(gcr_lo + u_gcr * (gcr_hi - gcr_lo)),
            float(spcf_lo + u_spcf * (spcf_hi - spcf_lo)),
        ))
    return centers


def initial_sigma(K: int, gcr_lo: float, gcr_hi: float,
                  spcf_lo: float, spcf_hi: float,
                  sigma_scale: float):
    """
    Compute a diagonal covariance matrix (stored as 2×2 list) whose standard
    deviations are sigma_scale × (range / sqrt(K)).  The sqrt(K) factor means
    that each expert's 1-sigma ellipse covers roughly 1/K of the range.
    """
    std_gcr  = sigma_scale * (gcr_hi  - gcr_lo)  / math.sqrt(K)
    std_spcf = sigma_scale * (spcf_hi - spcf_lo) / math.sqrt(K)
    return [[std_gcr ** 2, 0.0],
            [0.0,          std_spcf ** 2]]


def experts_from_centers(centers: list, sigma: list) -> list:
    """Minimal expert dicts for coverage computations."""
    return [{"mu": [g, s], "sigma": sigma} for g, s in centers]


def sample_from_gaussian(mu, sigma_2x2, n: int,
                         gcr_lo: float, gcr_hi: float,
                         spcf_lo: float, spcf_hi: float,
                         rng: np.random.Generator):
    """
    Draw n samples from N(mu, Sigma), clamp to the design-space bounds.
    sigma_2x2 is a 2×2 list (we only use diagonal entries for now).

    Returns list of [gcr, spcf] pairs.
    """
    std_gcr  = math.sqrt(sigma_2x2[0][0])
    std_spcf = math.sqrt(sigma_2x2[1][1])

    gcr_samples  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo,  gcr_hi)
    spcf_samples = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)

    return [[float(g), float(s)] for g, s in zip(gcr_samples, spcf_samples)]


def coverage_estimate_from_points(experts: list,
                                  gcr_pts: np.ndarray,
                                  spcf_pts: np.ndarray,
                                  threshold: float) -> float:
    """Coverage estimate on a fixed Monte-Carlo point set."""
    covered = 0
    for gcr_v, spcf_v in zip(gcr_pts, spcf_pts):
        max_density = 0.0
        for ex in experts:
            mu = ex["mu"]
            sigma = ex["sigma"]
            var_g = sigma[0][0]
            var_s = sigma[1][1]
            d_g = (gcr_v - mu[0]) ** 2 / (2 * var_g)
            d_s = (spcf_v - mu[1]) ** 2 / (2 * var_s)
            density = math.exp(-(d_g + d_s))
            max_density = max(max_density, density)
        if max_density > threshold:
            covered += 1
    return covered / len(gcr_pts)


def coverage_estimate(experts: list, gcr_lo: float, gcr_hi: float,
                      spcf_lo: float, spcf_hi: float,
                      threshold: float, n_mc: int = 5000,
                      rng: np.random.Generator = None) -> float:
    """
    Monte-Carlo estimate of the fraction of the design space covered.
    A point θ is 'covered' if max_i p(θ | G_i) > threshold.

    Uses the Gaussian PDF (not log) evaluated with each expert's diagonal cov.
    """
    if rng is None:
        rng = np.random.default_rng()

    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)

    return coverage_estimate_from_points(experts, gcr_pts, spcf_pts, threshold)


def calibrate_sigma_scale(K: int,
                          centers: list,
                          gcr_lo: float, gcr_hi: float,
                          spcf_lo: float, spcf_hi: float,
                          target_coverage: float,
                          threshold: float,
                          gcr_pts: np.ndarray,
                          spcf_pts: np.ndarray,
                          initial_guess: float,
                          tol: float = 1e-3,
                          max_iter: int = 40) -> tuple[float, float]:
    """
    Binary-search sigma_scale so the MC coverage matches target_coverage.
    """
    if not (0.0 < target_coverage < 1.0):
        raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")

    def coverage_at(scale: float) -> float:
        sigma = initial_sigma(K, gcr_lo, gcr_hi, spcf_lo, spcf_hi, scale)
        experts = experts_from_centers(centers, sigma)
        return coverage_estimate_from_points(experts, gcr_pts, spcf_pts, threshold)

    lo = 1e-6
    hi = max(initial_guess, 1e-3)
    cov_lo = coverage_at(lo)
    cov_hi = coverage_at(hi)

    while cov_hi < target_coverage and hi < 64.0:
        lo = hi
        cov_lo = cov_hi
        hi *= 2.0
        cov_hi = coverage_at(hi)

    if cov_hi < target_coverage - tol:
        raise RuntimeError(
            "Could not reach the requested initial coverage. "
            f"Last attempt: sigma_scale={hi:.6f}, coverage={cov_hi:.4f}, "
            f"target={target_coverage:.4f}"
        )

    if cov_lo >= target_coverage:
        return lo, cov_lo

    best_scale = hi
    best_cov = cov_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov_mid = coverage_at(mid)
        if abs(cov_mid - target_coverage) < abs(best_cov - target_coverage):
            best_scale = mid
            best_cov = cov_mid
        if abs(cov_mid - target_coverage) <= tol:
            return mid, cov_mid
        if cov_mid < target_coverage:
            lo = mid
        else:
            hi = mid

    return best_scale, best_cov


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC Init — create the initial K-expert state file."
    )
    parser.add_argument("--run_name",   type=str, required=True,
                        help="Name for this PEC run (used as subdirectory).")
    parser.add_argument("--K",          type=int, default=4,
                        help="Number of experts (default: 4).")
    parser.add_argument("--GCR_range",  type=float, nargs=2, required=True,
                        metavar=("LO", "HI"),
                        help="Full GCR design-space bounds, e.g. 0.70 0.95.")
    parser.add_argument("--spcf_range", type=float, nargs=2, required=True,
                        metavar=("LO", "HI"),
                        help="Full spcf design-space bounds, e.g. 0.001 0.010.")
    parser.add_argument("--N_init",     type=int, default=16,
                        help="Initial designs sampled per expert (default: 16).")
    parser.add_argument("--sigma_scale", type=float, default=0.3,
                        help="Controls initial Gaussian width. "
                             "std_d = sigma_scale × range_d / sqrt(K). "
                             "(default: 0.3)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Legacy global initialization seed (default: 42).")
    parser.add_argument("--init_seed",  type=int, default=None,
                        help="Preferred global seed for stochastic center placement and "
                             "initial sampling. Defaults to --seed when omitted.")
    parser.add_argument("--init_anchor_region", type=str, default=None,
                        choices=["top_right"],
                        help="Optional region to force into the initial stochastic_fps "
                             "layout. Useful when one area of the design space is known "
                             "to be valuable. (default: none)")
    parser.add_argument("--log_root",   type=str, default="logs/pec",
                        help="Root directory for PEC logs (default: logs/pec).")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Overwrite an existing state file if present.")
    parser.add_argument("--usd_rel_path", type=str, default=None,
                        help="Relative path to the robot USD file, stored as BALLU_USD_REL_PATH "
                             "in the state file and injected into every train/eval subprocess. "
                             "Example: morphologies/hetero_library_hvyBloon_lab01.20.26/"
                             "hetero_0000_fl0.368_tl0.368/hetero_0000_fl0.368_tl0.368.usd")
    parser.add_argument("--centers",   type=float, nargs="+", default=None,
                        metavar="VAL",
                        help="Manually specify Gaussian centers as a flat list: "
                             "GCR_0 spcf_0  GCR_1 spcf_1  ...  (must supply exactly K pairs). "
                             "Overrides the automatic grid placement.")
    parser.add_argument("--init_strategy", type=str, default="grid",
                        choices=["grid", "stochastic_fps"],
                        help="Automatic center initializer to use when --centers is not "
                             "supplied. (default: grid)")
    parser.add_argument("--target_init_coverage", type=float, default=None,
                        help="Optional MC coverage target in (0, 1). When using "
                             "stochastic_fps and omitted, the target defaults to the "
                             "coverage produced by the legacy grid at the given K and "
                             "sigma_scale.")
    parser.add_argument("--init_pool_size", type=int, default=2048,
                        help="Candidate pool size for stochastic_fps center placement. "
                             "(default: 2048)")
    parser.add_argument("--init_jitter_scale", type=float, default=0.0,
                        help="Optional jitter in normalized design-space units for "
                             "stochastic_fps centers. Applied as jitter_scale / sqrt(K). "
                             "(default: 0.0)")
    parser.add_argument("--coverage_n_mc", type=int, default=10_000,
                        help="Monte-Carlo sample count for initialization coverage "
                             "estimation/calibration. (default: 10000)")
    parser.add_argument("--coverage_seed", type=int, default=DEFAULT_COVERAGE_SEED,
                        help="Fixed seed used for initialization coverage estimation and "
                             "sigma calibration. (default: 1729)")
    parser.add_argument("--coverage_tol", type=float, default=1e-3,
                        help="Absolute tolerance for sigma calibration against the target "
                             "initial coverage. (default: 1e-3)")
    args = parser.parse_args()

    init_seed = args.init_seed if args.init_seed is not None else args.seed
    center_rng = np.random.default_rng(init_seed)
    sample_rng = np.random.default_rng(init_seed + 1)
    coverage_rng = np.random.default_rng(args.coverage_seed)

    gcr_lo,  gcr_hi  = args.GCR_range
    spcf_lo, spcf_hi = args.spcf_range

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if os.path.exists(state_path) and not args.overwrite:
        print(f"[ERROR] State file already exists: {state_path}")
        print("        Use --overwrite to replace it.")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)

    if args.init_pool_size < args.K:
        print(f"[ERROR] --init_pool_size must be >= K (got {args.init_pool_size} < {args.K}).")
        sys.exit(1)

    threshold = math.exp(-2.0)
    coverage_gcr_pts = coverage_rng.uniform(gcr_lo, gcr_hi, size=args.coverage_n_mc)
    coverage_spcf_pts = coverage_rng.uniform(spcf_lo, spcf_hi, size=args.coverage_n_mc)

    # ── Determine Gaussian centres ────────────────────────────────────────────
    effective_strategy = args.init_strategy
    centers_source = None
    if args.centers is not None:
        if len(args.centers) != 2 * args.K:
            print(f"[ERROR] --centers expects exactly {2 * args.K} values "
                  f"({args.K} GCR-spcf pairs) but got {len(args.centers)}.")
            sys.exit(1)
        centers = [
            (args.centers[2 * i], args.centers[2 * i + 1])
            for i in range(args.K)
        ]
        # Warn if any center lies outside the design-space bounds.
        for i, (cg, cs) in enumerate(centers):
            if not (gcr_lo <= cg <= gcr_hi):
                print(f"  [WARNING] Expert {i}: GCR center {cg:.4f} is outside "
                      f"[{gcr_lo:.4f}, {gcr_hi:.4f}].")
            if not (spcf_lo <= cs <= spcf_hi):
                print(f"  [WARNING] Expert {i}: spcf center {cs:.5f} is outside "
                      f"[{spcf_lo:.5f}, {spcf_hi:.5f}].")
        effective_strategy = "manual"
        centers_source = "manual (--centers)"
    elif args.init_strategy == "stochastic_fps":
        centers = stochastic_fps_centers(
            K=args.K,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            pool_size=args.init_pool_size,
            anchor_region=args.init_anchor_region,
            jitter_scale=args.init_jitter_scale,
            rng=center_rng,
        )
        if args.init_anchor_region is not None:
            centers_source = f"automatic stochastic_fps + anchor({args.init_anchor_region})"
        else:
            centers_source = "automatic stochastic_fps"
    else:
        centers = grid_centers(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi)
        centers_source = "automatic grid"

    target_coverage = args.target_init_coverage
    target_source = "explicit" if target_coverage is not None else None
    if target_coverage is None and effective_strategy == "stochastic_fps":
        ref_sigma = initial_sigma(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi, args.sigma_scale)
        ref_experts = experts_from_centers(
            grid_centers(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi),
            ref_sigma,
        )
        target_coverage = coverage_estimate_from_points(
            ref_experts, coverage_gcr_pts, coverage_spcf_pts, threshold
        )
        target_source = "grid_reference"

    sigma_scale_used = args.sigma_scale
    calibrated_coverage = None
    sigma_calibrated = target_coverage is not None
    if sigma_calibrated:
        sigma_scale_used, calibrated_coverage = calibrate_sigma_scale(
            K=args.K,
            centers=centers,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            target_coverage=target_coverage,
            threshold=threshold,
            gcr_pts=coverage_gcr_pts,
            spcf_pts=coverage_spcf_pts,
            initial_guess=args.sigma_scale,
            tol=args.coverage_tol,
        )

    sigma = initial_sigma(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi, sigma_scale_used)

    # ── Build expert list ─────────────────────────────────────────────────────
    experts = []
    for i, (mu_gcr, mu_spcf) in enumerate(centers):
        designs = sample_from_gaussian(
            mu=[mu_gcr, mu_spcf],
            sigma_2x2=sigma,
            n=args.N_init,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            rng=sample_rng,
        )
        experts.append({
            "id":         i,
            "mu":         [mu_gcr, mu_spcf],
            "sigma":      sigma,
            "designs":    designs,
            "checkpoint": None,
            "trained":    False,
        })

    # ── Build and save state ──────────────────────────────────────────────────
    state = {
        "run_name":     args.run_name,
        "iteration":    0,
        "usd_rel_path": args.usd_rel_path,   # None if not specified
        "design_space": {
            "GCR":  [gcr_lo,  gcr_hi],
            "spcf": [spcf_lo, spcf_hi],
        },
        "experts": experts,
        "init_strategy": effective_strategy,
        "init_seed": init_seed,
        "init_anchor_region": args.init_anchor_region,
        "init_sigma_scale": sigma_scale_used,
        "init_target_coverage": target_coverage,
        "init_target_coverage_source": target_source,
        "init_coverage_threshold": threshold,
        "init_coverage_n_mc": args.coverage_n_mc,
        "init_coverage_seed": args.coverage_seed,
        "init_sigma_calibrated": sigma_calibrated,
    }
    if effective_strategy == "stochastic_fps":
        state["init_pool_size"] = args.init_pool_size
        state["init_jitter_scale"] = args.init_jitter_scale

    cov = coverage_estimate_from_points(experts, coverage_gcr_pts, coverage_spcf_pts, threshold)
    state["init_realized_coverage"] = cov

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PEC Initialization — run: {args.run_name}")
    print(f"{'='*70}")
    print(f"  K experts        : {args.K}")
    print(f"  GCR  range       : [{gcr_lo:.4f},  {gcr_hi:.4f}]")
    print(f"  spcf range       : [{spcf_lo:.5f}, {spcf_hi:.5f}]")
    print(f"  usd_rel_path     : {args.usd_rel_path or '(not set — uses env var at runtime)'}")
    print(f"  init_strategy    : {effective_strategy}")
    print(f"  init_seed        : {init_seed}")
    if args.init_anchor_region is not None:
        print(f"  init_anchor      : {args.init_anchor_region}")
    print(f"  Centers source   : {centers_source}")
    print(f"  N_init / expert  : {args.N_init}")
    print(f"  sigma_scale      : {sigma_scale_used:.6f}")
    if sigma_calibrated:
        print(f"  target coverage  : {target_coverage * 100:.2f}%  ({target_source})")
        print(f"  coverage tol     : {args.coverage_tol * 100:.2f}%")
    if effective_strategy == "stochastic_fps":
        print(f"  init_pool_size   : {args.init_pool_size}")
        print(f"  init_jitter      : {args.init_jitter_scale}")
    print(f"  State saved to   : {state_path}")
    print(f"\n  Expert centres (mu_GCR, mu_spcf):")
    for ex in experts:
        sig_g  = math.sqrt(ex["sigma"][0][0])
        sig_s  = math.sqrt(ex["sigma"][1][1])
        print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {sig_g:.4f}   "
              f"spcf={ex['mu'][1]:.5f} ± {sig_s:.5f}")

    if calibrated_coverage is not None:
        print(f"\n  Calibrated MC coverage (fixed points): {calibrated_coverage*100:.2f}%")
    print(f"  Initial MC coverage (threshold=exp(-2)): {cov*100:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
