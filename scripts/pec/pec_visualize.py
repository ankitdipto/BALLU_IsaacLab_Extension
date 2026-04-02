"""
PEC Gaussian Mixture Visualizer
=================================
Plots the K expert Gaussians in the design space.

2D mode (GCR × spcf)
---------------------
- Single plot: background heatmap + 1σ/2σ ellipses + training designs.
- If --itr is given and frontier scores exist: frontier overlay.
- Axes: X = spcf,  Y = GCR.

3D mode (GCR × spcf × leg)
----------------------------
- Single 3D axes: wireframe ellipsoids (1σ/2σ) + 3D design scatter.
- Axes: X = spcf,  Y = leg,  Z = GCR.
- If --itr is given and frontier scores exist: 3D frontier star overlay.

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
    """Unnormalised 2D log-density for the original (GCR, spcf) plane."""
    d_gcr  = (gcr_grid  - mu[0]) ** 2 / (2.0 * sigma[0][0])
    d_spcf = (spcf_grid - mu[1]) ** 2 / (2.0 * sigma[1][1])
    return -(d_gcr + d_spcf)


def _log_density_grid_marginal(mu_x, var_x, mu_y, var_y, X_grid, Y_grid):
    """Unnormalised 2D log-density for any marginal pair (diagonal covariance)."""
    d_x = (X_grid - mu_x) ** 2 / (2.0 * var_x)
    d_y = (Y_grid - mu_y) ** 2 / (2.0 * var_y)
    return -(d_x + d_y)


def _ellipse_patch(mu, sigma, n_sigma: float, color, linestyle="-", lw=1.5,
                   label=None):
    """Return a matplotlib Ellipse patch for the 2D (GCR, spcf) plane."""
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


def _ellipse_patch_marginal(mu_x, std_x, mu_y, std_y, n_sigma: float,
                             color, linestyle="-", lw=1.5):
    """Return a matplotlib Ellipse patch for a generic 2D marginal plane."""
    return Ellipse(
        xy=(mu_x, mu_y),
        width=2 * n_sigma * std_x,
        height=2 * n_sigma * std_y,
        angle=0,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        linewidth=lw,
        zorder=4,
    )


def _ellipsoid_wireframe(ax3d, mu, sigma, n_sigma: float, color,
                          linestyle="-", alpha=0.28, lw=0.6,
                          rstride=3, cstride=3):
    """Draw an axis-aligned ellipsoid wireframe on a 3D axes.

    Design-space convention: mu = [GCR, spcf, leg].
    Plot axis mapping: X = spcf (dim 1), Y = leg (dim 2), Z = GCR (dim 0).
    """
    std_gcr  = math.sqrt(sigma[0][0])
    std_spcf = math.sqrt(sigma[1][1])
    std_leg  = math.sqrt(sigma[2][2])

    # Parametric unit sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    # Scale each axis by the expert's std and shift to its mean.
    xs = mu[1] + n_sigma * std_spcf * sx  # X = spcf
    ys = mu[2] + n_sigma * std_leg  * sy  # Y = leg
    zs = mu[0] + n_sigma * std_gcr  * sz  # Z = GCR

    ax3d.plot_wireframe(
        xs, ys, zs,
        color=color, linestyle=linestyle, alpha=alpha,
        linewidth=lw, rstride=rstride, cstride=cstride,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2D subplot renderer (used by 3D mode for each marginal pair)
# ──────────────────────────────────────────────────────────────────────────────

def _render_marginal_subplot(
    ax,
    experts,
    ix: int, iy: int,             # design-space dimension indices for X and Y axes
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    x_label: str, y_label: str,
    grid_res: int,
    log_thresh: float,
    no_2sigma: bool,
    frontier_data,                # loaded scores JSON or None
    itr,                          # for annotation title
):
    """
    Render one 2D marginal subplot.

    ix / iy are indices into the 3-element design vector [GCR, spcf, leg]:
      0 = GCR, 1 = spcf, 2 = leg
    X axis = dimension ix,  Y axis = dimension iy.
    """
    x_vals = np.linspace(x_lo, x_hi, grid_res)
    y_vals = np.linspace(y_lo, y_hi, grid_res)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Per-expert 2D marginal log-density.
    log_dens = np.stack([
        _log_density_grid_marginal(
            ex["mu"][ix], ex["sigma"][ix][ix],
            ex["mu"][iy], ex["sigma"][iy][iy],
            X_grid, Y_grid,
        )
        for ex in experts
    ], axis=0)   # (K, R, R)

    dominant = np.argmax(log_dens, axis=0)
    max_log  = np.max(log_dens, axis=0)
    covered  = max_log > log_thresh

    img = np.ones((grid_res, grid_res, 4))
    for kid, ex in enumerate(experts):
        r, g, b = matplotlib.colors.to_rgb(_expert_color(kid))
        mask  = (dominant == kid) & covered
        alpha = np.clip(
            0.15 + 0.40 * (max_log - log_thresh) / (-log_thresh), 0.0, 0.55
        )
        img[mask, 0] = r
        img[mask, 1] = g
        img[mask, 2] = b
        img[mask, 3] = alpha[mask]

    ax.set_facecolor("white")
    ax.imshow(
        img,
        extent=[x_lo, x_hi, y_lo, y_hi],
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(mpatches.Rectangle(
        (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
        linewidth=1.5, edgecolor="black", facecolor="none", zorder=1,
    ))

    for ex in experts:
        kid   = ex["id"]
        color = _expert_color(kid)
        mu_x  = ex["mu"][ix]
        mu_y  = ex["mu"][iy]
        std_x = math.sqrt(ex["sigma"][ix][ix])
        std_y = math.sqrt(ex["sigma"][iy][iy])

        # 1σ ellipse
        ax.add_patch(_ellipse_patch_marginal(mu_x, std_x, mu_y, std_y,
                                             1.0, color, "-", 2.0))
        # 2σ ellipse
        if not no_2sigma:
            ax.add_patch(_ellipse_patch_marginal(mu_x, std_x, mu_y, std_y,
                                                 2.0, color, "--", 1.2))
        # Center cross
        ax.plot(mu_x, mu_y, "+", color=color, ms=12, mew=2.5, zorder=5)

        # Training design points projected to (ix, iy) axes.
        designs = ex.get("designs", [])
        if designs:
            d_arr = np.array(designs)
            ax.scatter(d_arr[:, ix], d_arr[:, iy],
                       s=60, color=color, alpha=0.7,
                       edgecolors="white", linewidths=0.5,
                       zorder=3, marker="o")

    # Frontier overlay
    if frontier_data is not None:
        candidates   = frontier_data["candidates"]
        scores_mat   = {int(k): v for k, v in
                        frontier_data["scores_matrix"].items()}
        valid_experts = [kid for kid, s in scores_mat.items() if s is not None]

        for cand in candidates:
            fid    = cand["id"]
            coords = [cand["GCR"], cand["spcf"], cand.get("leg", 0.0)]
            x_val  = coords[ix]
            y_val  = coords[iy]

            best_score = -1
            winner     = None
            for kid in valid_experts:
                sc = scores_mat[kid][fid]
                if sc > best_score:
                    best_score = sc
                    winner     = kid

            color = _expert_color(winner) if winner is not None and best_score > 0 \
                    else "#aaaaaa"

            ax.scatter(x_val, y_val, s=110, color=color,
                       marker="*", edgecolors="black", linewidths=0.5,
                       zorder=6, alpha=0.9)

            if best_score > 0:
                ax.annotate(
                    str(best_score),
                    xy=(x_val, y_val),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=color, zorder=7,
                )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)

    # Format x-tick labels with enough decimal places for spcf.
    if x_hi - x_lo < 0.5:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="x", rotation=30)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PEC Gaussian mixture in the design space."
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
    is_3d    = "leg" in state["design_space"]
    cur_iter = state["iteration"]

    if is_3d:
        leg_lo, leg_hi = state["design_space"]["leg"]

    log_thresh = args.coverage_threshold if args.coverage_threshold is not None \
                 else -2.0    # exp(-2) ≈ 0.135

    # ── Resolve which Gaussian parameters to plot ─────────────────────────────
    if args.itr is not None:
        history = state.get("history", [])
        snap = next((h for h in history if h["iteration"] == args.itr), None)
        if snap is not None:
            current_by_id = {ex["id"]: ex for ex in state["experts"]}
            experts = []
            for s in snap["experts_snapshot"]:
                ex = dict(s)
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

    # ── Title suffix ──────────────────────────────────────────────────────────
    if args.itr is not None and not args.no_frontier:
        title_itr = f"  |  iter {args.itr} frontier overlay"
    elif args.itr is not None:
        title_itr = f"  |  Gaussian state at iter {args.itr}"
    else:
        title_itr = f"  |  current state (iter {cur_iter})"

    # ── Shared legend handles (built once, attached to last subplot or fig) ───
    def _build_legend_handles(with_frontier: bool) -> list:
        handles = []
        for ex in experts:
            kid   = ex["id"]
            color = _expert_color(kid)
            mu    = ex["mu"]
            if is_3d:
                label = (f"Expert {kid}  "
                         f"μ=({mu[0]:.3f}, {mu[1]:.5f}, {mu[2]:.3f})")
            else:
                label = f"Expert {kid}  μ=({mu[0]:.3f}, {mu[1]:.5f})"
            handles.append(Line2D(
                [], [], color=color, marker="o", linestyle="-",
                markersize=7, label=label,
            ))
        handles.append(Line2D([], [], color="black", linestyle="-",
                              lw=2.0, label="1σ ellipse"))
        if not args.no_2sigma:
            handles.append(Line2D([], [], color="black", linestyle="--",
                                  lw=1.2, label="2σ ellipse"))
        handles.append(Line2D([], [], color="black", marker="o",
                              linestyle="none", markersize=5,
                              label="Training designs"))
        if with_frontier:
            handles.append(Line2D(
                [], [], color="#555555", marker="*", linestyle="none",
                markersize=9, markeredgecolor="black",
                label=f"Frontier iter {args.itr}  (★ = winner, ✦ = tied at 0)"),
            )
        return handles

    # ═════════════════════════════════════════════════════════════════════════
    # 2D mode — single plot (preserved exactly from original implementation)
    # ═════════════════════════════════════════════════════════════════════════
    if not is_3d:
        spcf_vals = np.linspace(spcf_lo, spcf_hi, args.grid_res)
        gcr_vals  = np.linspace(gcr_lo,  gcr_hi,  args.grid_res)
        SPCF, GCR = np.meshgrid(spcf_vals, gcr_vals)

        log_dens = np.stack([
            _log_density_grid(ex["mu"], ex["sigma"], SPCF, GCR)
            for ex in experts
        ], axis=0)

        dominant = np.argmax(log_dens, axis=0)
        max_log  = np.max(log_dens, axis=0)
        covered  = max_log > log_thresh

        img = np.ones((args.grid_res, args.grid_res, 4))
        for kid, ex in enumerate(experts):
            r, g, b = matplotlib.colors.to_rgb(_expert_color(kid))
            mask  = (dominant == kid) & covered
            alpha = np.clip(
                0.15 + 0.40 * (max_log - log_thresh) / (-log_thresh), 0.0, 0.55
            )
            img[mask, 0] = r
            img[mask, 1] = g
            img[mask, 2] = b
            img[mask, 3] = alpha[mask]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_facecolor("white")

        ax.imshow(
            img,
            extent=[spcf_lo, spcf_hi, gcr_lo, gcr_hi],
            aspect="auto",
            origin="lower",
            interpolation="bilinear",
            zorder=0,
        )
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

            ax.add_patch(_ellipse_patch(ex["mu"], ex["sigma"], 1.0, color,
                                        linestyle="-",  lw=2.0))
            if not args.no_2sigma:
                ax.add_patch(_ellipse_patch(ex["mu"], ex["sigma"], 2.0, color,
                                            linestyle="--", lw=1.2))

            ax.plot(mu_spcf, mu_gcr, "+", color=color, ms=12, mew=2.5, zorder=5)

            designs = ex.get("designs", [])
            if designs:
                d_arr = np.array(designs)
                ax.scatter(d_arr[:, 1], d_arr[:, 0],
                           s=18, color=color, alpha=0.6,
                           edgecolors="white", linewidths=0.4,
                           zorder=3, marker="o")

            legend_handles.append(Line2D(
                [], [], color=color, marker="o", linestyle="-",
                markersize=7, label=f"Expert {kid}  μ=({mu_gcr:.3f}, {mu_spcf:.5f})"
            ))

        # Frontier overlay
        if frontier_data is not None:
            candidates   = frontier_data["candidates"]
            scores_mat   = {int(k): v for k, v in
                            frontier_data["scores_matrix"].items()}
            valid_experts = [kid for kid, s in scores_mat.items() if s is not None]

            for cand in candidates:
                fid   = cand["id"]
                g_val = cand["GCR"]
                s_val = cand["spcf"]

                best_score = -1
                winner     = None
                for kid in valid_experts:
                    sc = scores_mat[kid][fid]
                    if sc > best_score:
                        best_score = sc
                        winner     = kid

                color = _expert_color(winner) if winner is not None and best_score > 0 \
                        else "#aaaaaa"

                ax.scatter(s_val, g_val, s=110, color=color,
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

        ax.set_title(f"PEC Gaussian Mixture — {args.run_name}{title_itr}",
                     fontsize=11, pad=10)
        ax.set_xlabel("Spring Coefficient  (spcf)", fontsize=11)
        ax.set_ylabel("Gravity Compensation Ratio  (GCR)", fontsize=11)
        ax.set_xlim(spcf_lo, spcf_hi)
        ax.set_ylim(gcr_lo,  gcr_hi)
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

    # ═════════════════════════════════════════════════════════════════════════
    # 3D mode — single true-3D axes with wireframe ellipsoids
    # Axis mapping: X = spcf,  Y = leg,  Z = GCR
    # ═════════════════════════════════════════════════════════════════════════
    else:
        fig = plt.figure(figsize=(13, 9))
        ax3d = fig.add_subplot(111, projection="3d")

        for ex in experts:
            kid   = ex["id"]
            color = _expert_color(kid)
            mu    = ex["mu"]   # [GCR, spcf, leg]

            # 1σ solid wireframe
            _ellipsoid_wireframe(ax3d, mu, ex["sigma"], 1.0, color,
                                 linestyle="-", alpha=0.30, lw=0.8)
            # 2σ dashed wireframe
            if not args.no_2sigma:
                _ellipsoid_wireframe(ax3d, mu, ex["sigma"], 2.0, color,
                                     linestyle="--", alpha=0.12, lw=0.5)

            # Gaussian centre — filled plus marker (X=spcf, Y=leg, Z=GCR)
            ax3d.scatter([mu[1]], [mu[2]], [mu[0]],
                         color=color, s=140, marker="P",
                         edgecolors="black", linewidths=0.8,
                         zorder=10, depthshade=False)

            # Training design points
            designs = ex.get("designs", [])
            if designs:
                d_arr = np.array(designs)   # (N, 3): [GCR, spcf, leg]
                ax3d.scatter(d_arr[:, 1], d_arr[:, 2], d_arr[:, 0],
                             s=60, color=color, alpha=0.75,
                             edgecolors="white", linewidths=0.4,
                             marker="o", depthshade=True)

        # Frontier overlay
        if frontier_data is not None:
            candidates    = frontier_data["candidates"]
            scores_mat    = {int(k): v for k, v in
                             frontier_data["scores_matrix"].items()}
            valid_experts = [kid for kid, s in scores_mat.items()
                             if s is not None]
            leg_mid = (leg_lo + leg_hi) / 2.0

            for cand in candidates:
                fid   = cand["id"]
                g_val = cand["GCR"]
                s_val = cand["spcf"]
                l_val = cand.get("leg", leg_mid)

                best_score = -1
                winner = None
                for kid in valid_experts:
                    sc = scores_mat[kid][fid]
                    if sc > best_score:
                        best_score = sc
                        winner = kid

                color = _expert_color(winner) \
                        if winner is not None and best_score > 0 else "#aaaaaa"

                ax3d.scatter([s_val], [l_val], [g_val],
                             s=110, color=color, marker="*",
                             edgecolors="black", linewidths=0.5,
                             alpha=0.9, depthshade=False)

                if best_score > 0:
                    ax3d.text(s_val, l_val, g_val, f"  {best_score}",
                              fontsize=6.5, color=color)

        # Axis limits and labels
        ax3d.set_xlim(spcf_lo, spcf_hi)
        ax3d.set_ylim(leg_lo,  leg_hi)
        ax3d.set_zlim(gcr_lo,  gcr_hi)
        ax3d.set_xlabel("Spring Coefficient (spcf)", labelpad=10)
        ax3d.set_ylabel("Leg Length (leg)",           labelpad=10)
        ax3d.set_zlabel("GCR",                        labelpad=10)
        ax3d.xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.4f"))
        ax3d.tick_params(axis="x", rotation=30)

        ax3d.set_title(
            f"PEC 3D Gaussian Mixture — {args.run_name}{title_itr}",
            fontsize=11, pad=15,
        )

        legend_handles = _build_legend_handles(
            with_frontier=frontier_data is not None)
        ax3d.legend(handles=legend_handles, loc="upper left",
                    fontsize=8, framealpha=0.85)

        plt.tight_layout()

    # ── Save or show ──────────────────────────────────────────────────────────
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
