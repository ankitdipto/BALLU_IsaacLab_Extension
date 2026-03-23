"""
PEC Initialization Script
=========================
Progressive Expert Coverage — Step 0: Bootstrap

Creates the initial PEC state file with K experts, each assigned a Gaussian
region in the design space.  Supports two modes:

  2D  (legacy):  design_space = {GCR, spcf}
  3D  (default): design_space = {GCR, spcf, leg}  when --leg_range is given

Experts are seeded on a grid or with boundary-aware stochastic farthest-point
sampling, and optionally calibrated to a target initial MC coverage.

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
# 2-D helpers (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def grid_centers(K: int, gcr_lo: float, gcr_hi: float,
                 spcf_lo: float, spcf_hi: float):
    """Place K centers on a square-ish grid in 2D GCR×spcf space."""
    best = None
    for n1 in range(1, K + 1):
        n2 = math.ceil(K / n1)
        if best is None or abs(n1 - n2) < abs(best[0] - best[1]):
            best = (n1, n2)
    n_gcr, n_spcf = best

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
    return centers[:K]


def stochastic_fps_centers(K: int,
                           gcr_lo: float, gcr_hi: float,
                           spcf_lo: float, spcf_hi: float,
                           pool_size: int,
                           anchor_region: str | None,
                           jitter_scale: float,
                           rng: np.random.Generator):
    """Boundary-aware stochastic FPS in normalized 2D [0,1]² space."""
    if pool_size < K:
        raise ValueError(f"pool_size must be >= K, got pool_size={pool_size}, K={K}")

    pool = rng.uniform(0.0, 1.0, size=(pool_size, 2))
    chosen = np.zeros(pool_size, dtype=bool)

    boundary_clearance = np.min(np.stack([
        pool[:, 0], 1.0 - pool[:, 0],
        pool[:, 1], 1.0 - pool[:, 1],
    ], axis=1), axis=1)
    boundary_gain = 2.0

    if anchor_region == "top_right":
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
    """2D diagonal covariance matrix for legacy mode."""
    std_gcr  = sigma_scale * (gcr_hi  - gcr_lo)  / math.sqrt(K)
    std_spcf = sigma_scale * (spcf_hi - spcf_lo) / math.sqrt(K)
    return [[std_gcr ** 2, 0.0],
            [0.0,          std_spcf ** 2]]


def experts_from_centers(centers: list, sigma: list) -> list:
    return [{"mu": [g, s], "sigma": sigma} for g, s in centers]


def sample_from_gaussian(mu, sigma_2x2, n: int,
                         gcr_lo: float, gcr_hi: float,
                         spcf_lo: float, spcf_hi: float,
                         rng: np.random.Generator):
    std_gcr  = math.sqrt(sigma_2x2[0][0])
    std_spcf = math.sqrt(sigma_2x2[1][1])
    gcr_samples  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo,  gcr_hi)
    spcf_samples = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)
    return [[float(g), float(s)] for g, s in zip(gcr_samples, spcf_samples)]


def coverage_estimate_from_points(experts: list,
                                  gcr_pts: np.ndarray,
                                  spcf_pts: np.ndarray,
                                  threshold: float) -> float:
    """2D coverage on a fixed MC point set."""
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
    if rng is None:
        rng = np.random.default_rng()
    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)
    return coverage_estimate_from_points(experts, gcr_pts, spcf_pts, threshold)


def calibrate_sigma_scale(K: int, centers: list,
                          gcr_lo: float, gcr_hi: float,
                          spcf_lo: float, spcf_hi: float,
                          target_coverage: float, threshold: float,
                          gcr_pts: np.ndarray, spcf_pts: np.ndarray,
                          initial_guess: float,
                          tol: float = 1e-3, max_iter: int = 40) -> tuple[float, float]:
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
        lo, cov_lo = hi, cov_hi
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

    best_scale, best_cov = hi, cov_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov_mid = coverage_at(mid)
        if abs(cov_mid - target_coverage) < abs(best_cov - target_coverage):
            best_scale, best_cov = mid, cov_mid
        if abs(cov_mid - target_coverage) <= tol:
            return mid, cov_mid
        if cov_mid < target_coverage:
            lo = mid
        else:
            hi = mid

    return best_scale, best_cov


# ──────────────────────────────────────────────────────────────────────────────
# 3-D helpers
# ──────────────────────────────────────────────────────────────────────────────

def grid_centers_3d(K: int,
                    gcr_lo: float, gcr_hi: float,
                    spcf_lo: float, spcf_hi: float,
                    leg_lo: float,  leg_hi: float):
    """Place K centers on a cube-ish grid in 3D GCR×spcf×leg space."""
    # Find n such that n³ is the smallest cube >= K, minimizing grid points.
    n = max(1, math.ceil(K ** (1.0 / 3.0)))
    while n ** 3 < K:
        n += 1

    def axis_centers(lo, hi, n_pts):
        if n_pts == 1:
            return [(lo + hi) / 2.0]
        return list(np.linspace(lo + (hi - lo) / (2 * n_pts),
                                hi - (hi - lo) / (2 * n_pts),
                                n_pts))

    gcr_cs  = axis_centers(gcr_lo,  gcr_hi,  n)
    spcf_cs = axis_centers(spcf_lo, spcf_hi, n)
    leg_cs  = axis_centers(leg_lo,  leg_hi,  n)

    centers = []
    for g in gcr_cs:
        for s in spcf_cs:
            for l in leg_cs:
                centers.append((float(g), float(s), float(l)))
                if len(centers) == K:
                    return centers
    return centers[:K]


def stochastic_fps_centers_3d(K: int,
                               gcr_lo: float, gcr_hi: float,
                               spcf_lo: float, spcf_hi: float,
                               leg_lo: float,  leg_hi: float,
                               pool_size: int,
                               jitter_scale: float,
                               rng: np.random.Generator):
    """Boundary-aware stochastic FPS in normalized 3D [0,1]³ space."""
    if pool_size < K:
        raise ValueError(f"pool_size must be >= K, got pool_size={pool_size}, K={K}")

    pool = rng.uniform(0.0, 1.0, size=(pool_size, 3))
    chosen = np.zeros(pool_size, dtype=bool)

    # Boundary clearance: min distance to any face of the [0,1]³ cube.
    boundary_clearance = np.min(np.stack([
        pool[:, 0], 1.0 - pool[:, 0],
        pool[:, 1], 1.0 - pool[:, 1],
        pool[:, 2], 1.0 - pool[:, 2],
    ], axis=1), axis=1)
    boundary_gain = 2.0

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
    for u_g, u_s, u_l in centers_unit:
        centers.append((
            float(gcr_lo  + u_g * (gcr_hi  - gcr_lo)),
            float(spcf_lo + u_s * (spcf_hi - spcf_lo)),
            float(leg_lo  + u_l * (leg_hi  - leg_lo)),
        ))
    return centers


def initial_sigma_3d(K: int,
                     gcr_lo: float, gcr_hi: float,
                     spcf_lo: float, spcf_hi: float,
                     leg_lo: float,  leg_hi: float,
                     sigma_scale: float):
    """3D diagonal covariance matrix.  std_d = sigma_scale × range_d / K^(1/3)."""
    k_factor = K ** (1.0 / 3.0)
    std_gcr  = sigma_scale * (gcr_hi  - gcr_lo)  / k_factor
    std_spcf = sigma_scale * (spcf_hi - spcf_lo) / k_factor
    std_leg  = sigma_scale * (leg_hi  - leg_lo)  / k_factor
    return [[std_gcr ** 2, 0.0, 0.0],
            [0.0, std_spcf ** 2, 0.0],
            [0.0, 0.0, std_leg ** 2]]


def experts_from_centers_3d(centers: list, sigma: list) -> list:
    return [{"mu": [g, s, l], "sigma": sigma} for g, s, l in centers]


def sample_from_gaussian_3d(mu, sigma_3x3, n: int,
                             gcr_lo: float, gcr_hi: float,
                             spcf_lo: float, spcf_hi: float,
                             leg_lo: float,  leg_hi: float,
                             rng: np.random.Generator):
    std_gcr  = math.sqrt(sigma_3x3[0][0])
    std_spcf = math.sqrt(sigma_3x3[1][1])
    std_leg  = math.sqrt(sigma_3x3[2][2])
    gcr_s  = rng.normal(mu[0], std_gcr,  size=n).clip(gcr_lo,  gcr_hi)
    spcf_s = rng.normal(mu[1], std_spcf, size=n).clip(spcf_lo, spcf_hi)
    leg_s  = rng.normal(mu[2], std_leg,  size=n).clip(leg_lo,  leg_hi)
    return [[float(g), float(s), float(l)]
            for g, s, l in zip(gcr_s, spcf_s, leg_s)]


def coverage_estimate_3d_from_points(experts: list,
                                     gcr_pts: np.ndarray,
                                     spcf_pts: np.ndarray,
                                     leg_pts: np.ndarray,
                                     threshold: float) -> float:
    """3D coverage on a fixed MC point set using the same exp(-2) criterion."""
    covered = 0
    for gcr_v, spcf_v, leg_v in zip(gcr_pts, spcf_pts, leg_pts):
        max_density = 0.0
        for ex in experts:
            mu    = ex["mu"]
            sigma = ex["sigma"]
            d_g = (gcr_v  - mu[0]) ** 2 / (2 * sigma[0][0])
            d_s = (spcf_v - mu[1]) ** 2 / (2 * sigma[1][1])
            d_l = (leg_v  - mu[2]) ** 2 / (2 * sigma[2][2])
            density = math.exp(-(d_g + d_s + d_l))
            max_density = max(max_density, density)
        if max_density > threshold:
            covered += 1
    return covered / len(gcr_pts)


def coverage_estimate_3d(experts: list,
                         gcr_lo: float, gcr_hi: float,
                         spcf_lo: float, spcf_hi: float,
                         leg_lo: float, leg_hi: float,
                         threshold: float, n_mc: int = 5000,
                         rng: np.random.Generator = None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    gcr_pts  = rng.uniform(gcr_lo,  gcr_hi,  size=n_mc)
    spcf_pts = rng.uniform(spcf_lo, spcf_hi, size=n_mc)
    leg_pts  = rng.uniform(leg_lo,  leg_hi,  size=n_mc)
    return coverage_estimate_3d_from_points(experts, gcr_pts, spcf_pts, leg_pts, threshold)


def calibrate_sigma_scale_3d(K: int, centers: list,
                              gcr_lo: float, gcr_hi: float,
                              spcf_lo: float, spcf_hi: float,
                              leg_lo: float, leg_hi: float,
                              target_coverage: float, threshold: float,
                              gcr_pts: np.ndarray, spcf_pts: np.ndarray,
                              leg_pts: np.ndarray,
                              initial_guess: float,
                              tol: float = 1e-3,
                              max_iter: int = 40) -> tuple[float, float]:
    if not (0.0 < target_coverage < 1.0):
        raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")

    def coverage_at(scale: float) -> float:
        sigma = initial_sigma_3d(K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                  leg_lo, leg_hi, scale)
        experts = experts_from_centers_3d(centers, sigma)
        return coverage_estimate_3d_from_points(experts, gcr_pts, spcf_pts,
                                                leg_pts, threshold)

    lo = 1e-6
    hi = max(initial_guess, 1e-3)
    cov_lo = coverage_at(lo)
    cov_hi = coverage_at(hi)

    while cov_hi < target_coverage and hi < 64.0:
        lo, cov_lo = hi, cov_hi
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

    best_scale, best_cov = hi, cov_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cov_mid = coverage_at(mid)
        if abs(cov_mid - target_coverage) < abs(best_cov - target_coverage):
            best_scale, best_cov = mid, cov_mid
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
    parser.add_argument("--run_name",   type=str, required=True)
    parser.add_argument("--K",          type=int, default=4)
    parser.add_argument("--GCR_range",  type=float, nargs=2, required=True,
                        metavar=("LO", "HI"))
    parser.add_argument("--spcf_range", type=float, nargs=2, required=True,
                        metavar=("LO", "HI"))
    parser.add_argument("--leg_range",  type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Leg length design-space bounds, e.g. 0.20 0.50. "
                             "When provided, activates 3D mode (GCR×spcf×leg).")
    parser.add_argument("--N_init",     type=int, default=16)
    parser.add_argument("--sigma_scale", type=float, default=0.3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--init_seed",  type=int, default=None)
    parser.add_argument("--init_anchor_region", type=str, default=None,
                        choices=["top_right"],
                        help="Anchor region for stochastic_fps in 2D mode only.")
    parser.add_argument("--log_root",   type=str, default="logs/pec")
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--usd_rel_path", type=str, default=None,
                        help="Relative path to the robot USD file (2D mode / legacy).")
    parser.add_argument("--centers",   type=float, nargs="+", default=None,
                        metavar="VAL",
                        help="Manual Gaussian centers as flat list. "
                             "2D: GCR_0 spcf_0 GCR_1 spcf_1 ... (2×K values). "
                             "3D: GCR_0 spcf_0 leg_0 GCR_1 spcf_1 leg_1 ... (3×K values).")
    parser.add_argument("--init_strategy", type=str, default="grid",
                        choices=["grid", "stochastic_fps"])
    parser.add_argument("--target_init_coverage", type=float, default=None)
    parser.add_argument("--init_pool_size", type=int, default=2048)
    parser.add_argument("--init_jitter_scale", type=float, default=0.0)
    parser.add_argument("--coverage_n_mc", type=int, default=10_000)
    parser.add_argument("--coverage_seed", type=int, default=DEFAULT_COVERAGE_SEED)
    parser.add_argument("--coverage_tol", type=float, default=1e-3)
    args = parser.parse_args()

    init_seed  = args.init_seed if args.init_seed is not None else args.seed
    center_rng = np.random.default_rng(init_seed)
    sample_rng = np.random.default_rng(init_seed + 1)
    coverage_rng = np.random.default_rng(args.coverage_seed)

    gcr_lo,  gcr_hi  = args.GCR_range
    spcf_lo, spcf_hi = args.spcf_range
    is_3d = args.leg_range is not None
    if is_3d:
        leg_lo, leg_hi = args.leg_range

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if os.path.exists(state_path) and not args.overwrite:
        print(f"[ERROR] State file already exists: {state_path}")
        print("        Use --overwrite to replace it.")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)

    if args.init_pool_size < args.K:
        print(f"[ERROR] --init_pool_size must be >= K.")
        sys.exit(1)

    threshold = math.exp(-2.0)

    # ── MC points for coverage / calibration ──────────────────────────────────
    if is_3d:
        coverage_gcr_pts  = coverage_rng.uniform(gcr_lo,  gcr_hi,  size=args.coverage_n_mc)
        coverage_spcf_pts = coverage_rng.uniform(spcf_lo, spcf_hi, size=args.coverage_n_mc)
        coverage_leg_pts  = coverage_rng.uniform(leg_lo,  leg_hi,  size=args.coverage_n_mc)
    else:
        coverage_gcr_pts  = coverage_rng.uniform(gcr_lo,  gcr_hi,  size=args.coverage_n_mc)
        coverage_spcf_pts = coverage_rng.uniform(spcf_lo, spcf_hi, size=args.coverage_n_mc)

    # ── Determine Gaussian centres ────────────────────────────────────────────
    effective_strategy = args.init_strategy
    centers_source = None

    n_vals_per_center = 3 if is_3d else 2

    if args.centers is not None:
        expected = n_vals_per_center * args.K
        if len(args.centers) != expected:
            print(f"[ERROR] --centers expects exactly {expected} values "
                  f"({args.K} {'3D' if is_3d else '2D'} centers) "
                  f"but got {len(args.centers)}.")
            sys.exit(1)
        if is_3d:
            centers = [
                (args.centers[3*i], args.centers[3*i+1], args.centers[3*i+2])
                for i in range(args.K)
            ]
        else:
            centers = [
                (args.centers[2*i], args.centers[2*i+1])
                for i in range(args.K)
            ]
        effective_strategy = "manual"
        centers_source = "manual (--centers)"

    elif args.init_strategy == "stochastic_fps":
        if is_3d:
            centers = stochastic_fps_centers_3d(
                K=args.K,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                pool_size=args.init_pool_size,
                jitter_scale=args.init_jitter_scale,
                rng=center_rng,
            )
        else:
            centers = stochastic_fps_centers(
                K=args.K,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                pool_size=args.init_pool_size,
                anchor_region=args.init_anchor_region,
                jitter_scale=args.init_jitter_scale,
                rng=center_rng,
            )
        centers_source = "automatic stochastic_fps"
        if not is_3d and args.init_anchor_region is not None:
            centers_source += f" + anchor({args.init_anchor_region})"
    else:
        if is_3d:
            centers = grid_centers_3d(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                      leg_lo, leg_hi)
        else:
            centers = grid_centers(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi)
        centers_source = "automatic grid"

    # ── Sigma calibration ─────────────────────────────────────────────────────
    target_coverage = args.target_init_coverage
    target_source   = "explicit" if target_coverage is not None else None

    if target_coverage is None and effective_strategy == "stochastic_fps":
        # Default target: match legacy grid coverage at this K and sigma_scale.
        if is_3d:
            ref_sigma   = initial_sigma_3d(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                           leg_lo, leg_hi, args.sigma_scale)
            ref_centers = grid_centers_3d(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                          leg_lo, leg_hi)
            ref_experts = experts_from_centers_3d(ref_centers, ref_sigma)
            target_coverage = coverage_estimate_3d_from_points(
                ref_experts, coverage_gcr_pts, coverage_spcf_pts,
                coverage_leg_pts, threshold
            )
        else:
            ref_sigma   = initial_sigma(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                        args.sigma_scale)
            ref_experts = experts_from_centers(
                grid_centers(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi), ref_sigma
            )
            target_coverage = coverage_estimate_from_points(
                ref_experts, coverage_gcr_pts, coverage_spcf_pts, threshold
            )
        target_source = "grid_reference"

    sigma_scale_used    = args.sigma_scale
    calibrated_coverage = None
    sigma_calibrated    = target_coverage is not None

    if sigma_calibrated:
        if is_3d:
            sigma_scale_used, calibrated_coverage = calibrate_sigma_scale_3d(
                K=args.K, centers=centers,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                target_coverage=target_coverage,
                threshold=threshold,
                gcr_pts=coverage_gcr_pts, spcf_pts=coverage_spcf_pts,
                leg_pts=coverage_leg_pts,
                initial_guess=args.sigma_scale,
                tol=args.coverage_tol,
            )
        else:
            sigma_scale_used, calibrated_coverage = calibrate_sigma_scale(
                K=args.K, centers=centers,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                target_coverage=target_coverage, threshold=threshold,
                gcr_pts=coverage_gcr_pts, spcf_pts=coverage_spcf_pts,
                initial_guess=args.sigma_scale, tol=args.coverage_tol,
            )

    if is_3d:
        sigma = initial_sigma_3d(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                                  leg_lo, leg_hi, sigma_scale_used)
    else:
        sigma = initial_sigma(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                              sigma_scale_used)

    # ── Build expert list ─────────────────────────────────────────────────────
    experts = []
    for i, center in enumerate(centers):
        if is_3d:
            mu_gcr, mu_spcf, mu_leg = center
            designs = sample_from_gaussian_3d(
                mu=[mu_gcr, mu_spcf, mu_leg], sigma_3x3=sigma,
                n=args.N_init,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                leg_lo=leg_lo, leg_hi=leg_hi,
                rng=sample_rng,
            )
            mu_list = [mu_gcr, mu_spcf, mu_leg]
        else:
            mu_gcr, mu_spcf = center
            designs = sample_from_gaussian(
                mu=[mu_gcr, mu_spcf], sigma_2x2=sigma,
                n=args.N_init,
                gcr_lo=gcr_lo, gcr_hi=gcr_hi,
                spcf_lo=spcf_lo, spcf_hi=spcf_hi,
                rng=sample_rng,
            )
            mu_list = [mu_gcr, mu_spcf]

        expert_dict = {
            "id":         i,
            "mu":         mu_list,
            "sigma":      sigma,
            "designs":    designs,
            "checkpoint": None,
            "trained":    False,
        }
        if is_3d:
            # Per-expert USD library path (populated lazily when first trained).
            expert_dict["morphology_library_path"] = None
        experts.append(expert_dict)

    # ── Coverage estimate on the built experts ────────────────────────────────
    if is_3d:
        cov = coverage_estimate_3d_from_points(
            experts, coverage_gcr_pts, coverage_spcf_pts, coverage_leg_pts, threshold
        )
    else:
        cov = coverage_estimate_from_points(
            experts, coverage_gcr_pts, coverage_spcf_pts, threshold
        )

    # ── Build and save state ──────────────────────────────────────────────────
    design_space = {"GCR": [gcr_lo, gcr_hi], "spcf": [spcf_lo, spcf_hi]}
    if is_3d:
        design_space["leg"] = [leg_lo, leg_hi]

    state = {
        "run_name":     args.run_name,
        "iteration":    0,
        "N_init":       args.N_init,
        "usd_rel_path": args.usd_rel_path,
        "design_space": design_space,
        "experts":      experts,
        "init_strategy":                effective_strategy,
        "init_seed":                    init_seed,
        "init_anchor_region":           args.init_anchor_region if not is_3d else None,
        "init_sigma_scale":             sigma_scale_used,
        "init_target_coverage":         target_coverage,
        "init_target_coverage_source":  target_source,
        "init_coverage_threshold":      threshold,
        "init_coverage_n_mc":           args.coverage_n_mc,
        "init_coverage_seed":           args.coverage_seed,
        "init_sigma_calibrated":        sigma_calibrated,
        "init_realized_coverage":       cov,
    }
    if is_3d:
        state["pec_mode"] = "3d"
    if effective_strategy == "stochastic_fps":
        state["init_pool_size"]    = args.init_pool_size
        state["init_jitter_scale"] = args.init_jitter_scale

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    mode_label = "3D (GCR × spcf × leg)" if is_3d else "2D (GCR × spcf)"
    print(f"\n{'='*70}")
    print(f"  PEC Initialization — run: {args.run_name}")
    print(f"{'='*70}")
    print(f"  Mode             : {mode_label}")
    print(f"  K experts        : {args.K}")
    print(f"  GCR  range       : [{gcr_lo:.4f},  {gcr_hi:.4f}]")
    print(f"  spcf range       : [{spcf_lo:.5f}, {spcf_hi:.5f}]")
    if is_3d:
        print(f"  leg  range       : [{leg_lo:.4f},  {leg_hi:.4f}]")
    print(f"  usd_rel_path     : {args.usd_rel_path or '(not set)'}")
    print(f"  init_strategy    : {effective_strategy}")
    print(f"  init_seed        : {init_seed}")
    print(f"  Centers source   : {centers_source}")
    print(f"  N_init / expert  : {args.N_init}")
    print(f"  sigma_scale      : {sigma_scale_used:.6f}")
    if sigma_calibrated:
        print(f"  target coverage  : {target_coverage * 100:.2f}%  ({target_source})")
        print(f"  coverage tol     : {args.coverage_tol * 100:.2f}%")
    print(f"  State saved to   : {state_path}")

    if is_3d:
        print(f"\n  Expert centres (mu_GCR, mu_spcf, mu_leg):")
        for ex in experts:
            sig_g = math.sqrt(ex["sigma"][0][0])
            sig_s = math.sqrt(ex["sigma"][1][1])
            sig_l = math.sqrt(ex["sigma"][2][2])
            print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {sig_g:.4f}   "
                  f"spcf={ex['mu'][1]:.5f} ± {sig_s:.5f}   "
                  f"leg={ex['mu'][2]:.4f} ± {sig_l:.4f}")
    else:
        print(f"\n  Expert centres (mu_GCR, mu_spcf):")
        for ex in experts:
            sig_g = math.sqrt(ex["sigma"][0][0])
            sig_s = math.sqrt(ex["sigma"][1][1])
            print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {sig_g:.4f}   "
                  f"spcf={ex['mu'][1]:.5f} ± {sig_s:.5f}")

    if calibrated_coverage is not None:
        print(f"\n  Calibrated MC coverage (fixed points): {calibrated_coverage*100:.2f}%")
    print(f"  Initial MC coverage (threshold=exp(-2)): {cov*100:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
