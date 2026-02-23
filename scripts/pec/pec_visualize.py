"""
PEC Gaussian Mixture Visualizer
=================================
Plots the K expert Gaussians in the 2D (spcf, GCR) design space.

Layout
------
- Background heatmap: dominant expert at each grid point (argmax of
  unnormalized Gaussian density), fading to white in uncovered regions.
- 1σ and 2σ iso-density ellipses for every expert.
- Training design points accumulated by each expert (circles).
- If --itr is given and frontier_evals/iter_<N>/scores.json exists:
    - Frontier candidate designs (stars), color = winning expert,
      gray = all experts tied at 0.
    - Score labels (best curriculum level) next to each candidate.

Axes: X = spcf,  Y = GCR  (as requested).

Usage (run from ballu_isclb_extension/)
-----------------------------------------
    # Current state (uses pec_state.json directly)
    python scripts/pec/pec_visualize.py --run_name dry_run_k2

    # Overlay frontier evaluation results from iteration 0
    python scripts/pec/pec_visualize.py --run_name dry_run_k2 --itr 0

    # Save to file instead of showing interactively
    python scripts/pec/pec_visualize.py --run_name dry_run_k2 --itr 0 \\
        --output plots/pec_iter0.png
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette — one distinct colour per expert (up to 8)
# ──────────────────────────────────────────────────────────────────────────────

EXPERT_COLORS = [
    "#e41a1c",   # red
    "#377eb8",   # blue
    "#4daf4a",   # green
    "#ff7f00",   # orange
    "#984ea3",   # purple
    "#a65628",   # brown
    "#f781bf",   # pink
    "#999999",   # grey
]


def _expert_color(kid: int) -> str:
    return EXPERT_COLORS[kid % len(EXPERT_COLORS)]


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_density_grid(mu, sigma, spcf_grid, gcr_grid):
    """Unnormalised log-density at every point of a 2D meshgrid."""
    d_gcr  = (gcr_grid  - mu[0]) ** 2 / (2.0 * sigma[0][0])
    d_spcf = (spcf_grid - mu[1]) ** 2 / (2.0 * sigma[1][1])
    return -(d_gcr + d_spcf)


def _ellipse_patch(mu, sigma, n_sigma: float, color, linestyle="-", lw=1.5,
                   label=None):
    """Return a matplotlib Ellipse patch for the n-sigma iso-density contour."""
    std_gcr  = math.sqrt(sigma[0][0])
    std_spcf = math.sqrt(sigma[1][1])
    # Width along X (spcf), height along Y (GCR)
    return Ellipse(
        xy=(mu[1], mu[0]),          # (spcf, GCR)
        width=2 * n_sigma * std_spcf,
        height=2 * n_sigma * std_gcr,
        angle=0,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        linewidth=lw,
        label=label,
        zorder=4,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PEC Gaussian mixture in the (spcf, GCR) plane."
    )
    parser.add_argument("--run_name",  type=str, required=True,
                        help="PEC run name (matches pec_init.py).")
    parser.add_argument("--log_root",  type=str, default="logs/pec",
                        help="Root PEC log directory (default: logs/pec).")
    parser.add_argument("--itr",       type=int, default=None,
                        help="PEC iteration whose frontier evaluation results "
                             "to overlay (optional).")
    parser.add_argument("--output",    type=str, default=None,
                        help="If set, save the figure to this path instead of "
                             "showing it interactively (e.g. plot.png).")
    parser.add_argument("--grid_res",  type=int, default=400,
                        help="Grid resolution for the background heatmap "
                             "(default: 400).")
    parser.add_argument("--coverage_threshold", type=float,
                        default=None,
                        help="Unnormalised log-density threshold below which a "
                             "grid cell is shown as uncovered (default: exp(-2) "
                             "≈ -2 in log space).")
    parser.add_argument(
        "--no_frontier", action="store_true",
        help="When --itr is given, plot only the historical Gaussian state "
             "without overlaying frontier candidates from that iteration.")
    parser.add_argument(
        "--no_2sigma", action="store_true",
        help="Omit the dashed 2-sigma ellipse for every expert.")
    args = parser.parse_args()

    # ── Load state ────────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    cur_iter  = state["iteration"]

    log_thresh = args.coverage_threshold if args.coverage_threshold is not None \
                 else -2.0    # exp(-2) ≈ 0.135

    # ── Resolve which Gaussian parameters to plot ─────────────────────────────
    # If --itr is given and a history snapshot exists for that iteration, use
    # the Gaussians that were active at that time.  Otherwise use current state.
    if args.itr is not None:
        history = state.get("history", [])
        snap = next((h for h in history if h["iteration"] == args.itr), None)
        if snap is not None:
            # Reconstruct expert dicts from snapshot (add designs from current
            # state so training-design dots still render).
            current_by_id = {ex["id"]: ex for ex in state["experts"]}
            experts = []
            for s in snap["experts_snapshot"]:
                ex = dict(s)
                # Show only the designs that existed at this iteration
                # (n_designs is the count at snapshot time).
                cur_designs = current_by_id.get(s["id"], {}).get("designs", [])
                ex["designs"] = cur_designs[: s["n_designs"]]
                experts.append(ex)
            print(f"[INFO] Using Gaussian snapshot from history at iteration {args.itr}.")
        else:
            experts = state["experts"]
            if history:
                print(f"[WARNING] No history snapshot found for iter {args.itr}. "
                      f"Available: {[h['iteration'] for h in history]}. "
                      f"Falling back to current Gaussians.")
            else:
                print(f"[INFO] No history in state file — showing current Gaussians. "
                      f"Run pec_refit_gaussians.py to start recording history.")
    else:
        experts = state["experts"]

    K = len(experts)

    # ── Load frontier data for --itr ──────────────────────────────────────────
    frontier_data = None
    if args.itr is not None and not args.no_frontier:
        scores_file = os.path.join(
            run_dir, "frontier_evals", f"iter_{args.itr}", "scores.json"
        )
        if not os.path.exists(scores_file):
            print(f"[WARNING] scores.json not found for iter {args.itr}: "
                  f"{scores_file}")
        else:
            with open(scores_file) as f:
                frontier_data = json.load(f)
    elif args.itr is not None and args.no_frontier:
        print(f"[INFO] --no_frontier: Gaussian state at iter "
              f"{args.itr} shown without frontier overlay.")

    # ── Build background heatmap ──────────────────────────────────────────────
    spcf_vals = np.linspace(spcf_lo, spcf_hi, args.grid_res)
    gcr_vals  = np.linspace(gcr_lo,  gcr_hi,  args.grid_res)
    SPCF, GCR = np.meshgrid(spcf_vals, gcr_vals)   # (R, R) each

    # Per-expert log-density at every grid point.
    log_dens = np.stack([
        _log_density_grid(ex["mu"], ex["sigma"], SPCF, GCR)
        for ex in experts
    ], axis=0)   # (K, R, R)

    dominant = np.argmax(log_dens, axis=0)          # (R, R)  expert index
    max_log  = np.max(log_dens, axis=0)             # (R, R)

    # Coverage mask: True where at least one expert has density > threshold.
    covered = max_log > log_thresh                   # (R, R)

    # Build RGBA image: each pixel = expert colour, alpha ∝ coverage strength.
    # In uncovered regions, alpha = 0 (transparent → white background).
    img = np.ones((args.grid_res, args.grid_res, 4))   # white RGBA
    for kid, ex in enumerate(experts):
        r, g, b = matplotlib.colors.to_rgb(_expert_color(kid))
        mask = (dominant == kid) & covered
        # Alpha: linearly maps log-density [log_thresh, 0] → [0.15, 0.55]
        alpha = np.clip(
            0.15 + 0.40 * (max_log - log_thresh) / (-log_thresh),
            0.0, 0.55
        )
        img[mask, 0] = r
        img[mask, 1] = g
        img[mask, 2] = b
        img[mask, 3] = alpha[mask]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("white")

    # Background heatmap
    ax.imshow(
        img,
        extent=[spcf_lo, spcf_hi, gcr_lo, gcr_hi],
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )

    # Design-space boundary
    ax.add_patch(mpatches.Rectangle(
        (spcf_lo, gcr_lo),
        spcf_hi - spcf_lo, gcr_hi - gcr_lo,
        linewidth=1.5, edgecolor="black", facecolor="none", zorder=1,
    ))

    legend_handles = []

    for ex in experts:
        kid   = ex["id"]
        color = _expert_color(kid)
        mu_gcr, mu_spcf = ex["mu"][0], ex["mu"][1]

        # 1σ ellipse (solid)
        ax.add_patch(_ellipse_patch(ex["mu"], ex["sigma"], 1.0, color,
                                    linestyle="-",  lw=2.0))
        # 2σ ellipse (dashed) — skipped when --no_2sigma is set
        if not args.no_2sigma:
            ax.add_patch(_ellipse_patch(ex["mu"], ex["sigma"], 2.0, color,
                                        linestyle="--", lw=1.2))

        # Center cross
        ax.plot(mu_spcf, mu_gcr, "+", color=color, ms=12, mew=2.5, zorder=5)

        # Training design points
        designs = ex.get("designs", [])
        if designs:
            d_arr = np.array(designs)   # (N, 2): col0=GCR, col1=spcf
            ax.scatter(d_arr[:, 1], d_arr[:, 0],
                       s=18, color=color, alpha=0.6,
                       edgecolors="white", linewidths=0.4,
                       zorder=3, marker="o")

        legend_handles.append(Line2D(
            [], [], color=color, marker="o", linestyle="-",
            markersize=7, label=f"Expert {kid}  μ=({mu_gcr:.3f}, {mu_spcf:.5f})"
        ))

    # ── Frontier overlay ──────────────────────────────────────────────────────
    if frontier_data is not None:
        candidates   = frontier_data["candidates"]
        scores_mat   = {int(k): v for k, v in
                        frontier_data["scores_matrix"].items()}
        valid_experts = [kid for kid, s in scores_mat.items() if s is not None]

        for cand in candidates:
            fid   = cand["id"]
            g_val = cand["GCR"]
            s_val = cand["spcf"]

            # Find winner
            best_score = -1
            winner     = None
            for kid in valid_experts:
                sc = scores_mat[kid][fid]
                if sc > best_score:
                    best_score = sc
                    winner     = kid

            color = _expert_color(winner) if winner is not None and best_score > 0 \
                    else "#aaaaaa"

            ax.scatter(s_val, g_val, s=80, color=color,
                       marker="*", edgecolors="black", linewidths=0.5,
                       zorder=6, alpha=0.9)

            if best_score > 0:
                ax.annotate(
                    str(best_score),
                    xy=(s_val, g_val),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=color, zorder=7,
                )

        legend_handles += [
            Line2D([], [], color="#555555", marker="*", linestyle="none",
                   markersize=9, markeredgecolor="black",
                   label=f"Frontier iter {args.itr}  (★ = winner, ✦ = tied at 0)"),
        ]

    # ── Labels / cosmetics ────────────────────────────────────────────────────
    if args.itr is not None and not args.no_frontier:
        title_itr = f"  |  iter {args.itr} frontier overlay"
    elif args.itr is not None:
        title_itr = f"  |  Gaussian state at iter {args.itr}"
    else:
        title_itr = f"  |  current state (iter {cur_iter})"
    ax.set_title(f"PEC Gaussian Mixture — {args.run_name}{title_itr}",
                 fontsize=11, pad=10)
    ax.set_xlabel("Spring Coefficient  (spcf)", fontsize=11)
    ax.set_ylabel("Gravity Compensation Ratio  (GCR)", fontsize=11)

    ax.set_xlim(spcf_lo, spcf_hi)
    ax.set_ylim(gcr_lo,  gcr_hi)

    # Format spcf x-tick labels with 4 decimal places
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.tick_params(axis="x", rotation=30)

    legend_handles.append(
        Line2D([], [], color="black", linestyle="-", lw=2.0, label="1σ ellipse")
    )
    if not args.no_2sigma:
        legend_handles.append(
            Line2D([], [], color="black", linestyle="--", lw=1.2, label="2σ ellipse")
        )
    legend_handles.append(
        Line2D([], [], color="black", marker="o", linestyle="none",
               markersize=5, label="Training designs")
    )
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8.5, framealpha=0.85)

    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    # Use non-interactive backend when no display is available.
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    main()
