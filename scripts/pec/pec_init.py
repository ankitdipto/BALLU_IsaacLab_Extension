"""
PEC Initialization Script
=========================
Progressive Expert Coverage — Step 0: Bootstrap

Creates the initial PEC state file with K experts, each assigned a Gaussian
region in the 2D (GCR, spcf) design space.  Gaussians are seeded on a uniform
grid so they collectively cover the full design space from the start.

For each expert the script also draws N_init design points from its Gaussian
and records them as the initial training set.

Usage
-----
    python pec_init.py \\
        --run_name   my_pec_run \\
        --K          4 \\
        --GCR_range  0.70 0.95 \\
        --spcf_range 0.001 0.010 \\
        --N_init     16 \\
        --sigma_scale 0.3

Output
------
    logs/pec/<run_name>/pec_state.json   (created / overwritten)

State file schema
-----------------
{
  "run_name":      str,
  "iteration":     int,          # 0 at init
  "design_space": {
      "GCR":  [lo, hi],
      "spcf": [lo, hi]
  },
  "experts": [
    {
      "id":          int,
      "mu":          [gcr_center, spcf_center],
      "sigma":       [[var_gcr, 0.0], [0.0, var_spcf]],   # diagonal cov
      "designs":     [[gcr, spcf], ...],                   # N_init samples
      "checkpoint":  null,          # filled in after training
      "trained":     false
    },
    ...
  ]
}
"""

import argparse
import json
import math
import os
import sys
import numpy as np


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

    covered = 0
    for gcr_v, spcf_v in zip(gcr_pts, spcf_pts):
        max_density = 0.0
        for ex in experts:
            mu    = ex["mu"]
            sigma = ex["sigma"]
            var_g = sigma[0][0]
            var_s = sigma[1][1]
            d_g   = (gcr_v  - mu[0]) ** 2 / (2 * var_g)
            d_s   = (spcf_v - mu[1]) ** 2 / (2 * var_s)
            density = math.exp(-(d_g + d_s))   # unnormalised — sufficient for argmax
            max_density = max(max_density, density)
        if max_density > threshold:
            covered += 1

    return covered / n_mc


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
                        help="Random seed (default: 42).")
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
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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

    # ── Determine Gaussian centres ────────────────────────────────────────────
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
        print(f"  Centers source   : manual (--centers)")
    else:
        centers = grid_centers(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi)
        print(f"  Centers source   : automatic grid")

    sigma = initial_sigma(args.K, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                          args.sigma_scale)

    # ── Build expert list ─────────────────────────────────────────────────────
    experts = []
    for i, (mu_gcr, mu_spcf) in enumerate(centers):
        designs = sample_from_gaussian(
            mu=[mu_gcr, mu_spcf],
            sigma_2x2=sigma,
            n=args.N_init,
            gcr_lo=gcr_lo, gcr_hi=gcr_hi,
            spcf_lo=spcf_lo, spcf_hi=spcf_hi,
            rng=rng,
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
    }

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
    print(f"  N_init / expert  : {args.N_init}")
    print(f"  sigma_scale      : {args.sigma_scale}")
    print(f"  seed             : {args.seed}")
    print(f"  State saved to   : {state_path}")
    print(f"\n  Expert centres (mu_GCR, mu_spcf):")
    for ex in experts:
        sig_g  = math.sqrt(ex["sigma"][0][0])
        sig_s  = math.sqrt(ex["sigma"][1][1])
        print(f"    Expert {ex['id']}: GCR={ex['mu'][0]:.4f} ± {sig_g:.4f}   "
              f"spcf={ex['mu'][1]:.5f} ± {sig_s:.5f}")

    # Monte-Carlo coverage with threshold = exp(-2) ≈ 0.135 (≈ 2σ radius)
    threshold = math.exp(-2.0)
    cov = coverage_estimate(experts, gcr_lo, gcr_hi, spcf_lo, spcf_hi,
                            threshold=threshold, n_mc=10_000, rng=rng)
    print(f"\n  Initial MC coverage (threshold=exp(-2)): {cov*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
