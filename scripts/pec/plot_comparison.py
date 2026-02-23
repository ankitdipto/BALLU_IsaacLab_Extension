"""
PEC vs Baseline Comparison Plots
=================================
Visualises the head-to-head performance of
  - Baseline universal controller
  - Expert 0 (on its 190 designs)
  - Expert 1 (on its 210 designs)
  - PEC oracle: max(Expert 0, Expert 1)
across all 400 evaluated designs.

Usage (from ballu_isclb_extension/):
    python scripts/pec/plot_comparison.py \
        --baseline   logs/pec/dbg_fullrun_1/baseline_eval_results.json \
        --expert0    logs/pec/dbg_fullrun_1/expert0_iter3_md3300_results.json \
        --expert1    logs/pec/dbg_fullrun_1/expert1_iter3_md3584_results.json \
        --split      190 \
        --output_dir logs/pec/dbg_fullrun_1/plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--baseline",   type=str,
                    default="logs/pec/dbg_fullrun_1/baseline_eval_results.json")
parser.add_argument("--expert0",    type=str,
                    default="logs/pec/dbg_fullrun_1/expert0_iter3_md3300_results.json")
parser.add_argument("--expert1",    type=str,
                    default="logs/pec/dbg_fullrun_1/expert1_iter3_md3584_results.json")
parser.add_argument("--split",      type=int, default=190,
                    help="Index splitting Expert-0 designs from Expert-1 designs.")
parser.add_argument("--output_dir", type=str,
                    default="logs/pec/dbg_fullrun_1/plots")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(args.baseline) as f:  bl_data = json.load(f)["results"]
with open(args.expert0)  as f:  e0_data = json.load(f)["results"]
with open(args.expert1)  as f:  e1_data = json.load(f)["results"]

S = args.split  # 190

bl_all  = np.array([r["best_level_idx"] for r in bl_data])
e0_all  = np.array([r["best_level_idx"] for r in e0_data])
e1_all  = np.array([r["best_level_idx"] for r in e1_data])
pec_all = np.maximum(e0_all, e1_all)

gcr_all  = np.array([r["GCR"]  for r in bl_data])
spcf_all = np.array([r["spcf"] for r in bl_data])

# Per-expert slices
bl_e0,  e0_e0  = bl_all[:S],  e0_all[:S]   # Expert-0 designs
bl_e1,  e1_e1  = bl_all[S:],  e1_all[S:]   # Expert-1 designs

# Diffs
diff_e0  = e0_e0  - bl_e0          # Expert-0 region
diff_e1  = e1_e1  - bl_e1          # Expert-1 region
diff_pec = pec_all - bl_all         # All designs, PEC oracle

COLORS = {"baseline": "#555555", "expert0": "orange",
          "expert1": "#1f77b4",  "pec": "#2ca02c"}
BIN_EDGES = np.arange(-0.5, 35.5, 1)   # integer levels 0-35

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Score distributions (all 400 designs)
# ─────────────────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig = plt.figure(figsize=(8, 6))
fig.suptitle("Score distributions across all 400 evaluated designs", fontsize=13)

# Left: overlapping histograms
# ax = axes[0]
kw = dict(bins=BIN_EDGES, alpha=0.55, edgecolor="white", linewidth=0.4)
plt.hist(bl_all,  label=f"Baseline  (μ={bl_all.mean():.1f})",  color=COLORS["baseline"], **kw)
plt.hist(pec_all, label=f"max(E0, E1) (μ={pec_all.mean():.1f})", color=COLORS["pec"],      **kw)
plt.xlabel("Obstacle Height (cm)")
plt.ylabel("Number of designs")
plt.title("Baseline vs max(E0, E1)")
plt.legend(fontsize=9)
plt.grid(axis="y", alpha=0.3)

# Right: per-expert vs baseline in their own region
# ax = axes[1]
# kw2 = dict(bins=BIN_EDGES, alpha=0.55, edgecolor="white", linewidth=0.4)
# ax.hist(bl_e0,  label=f"Baseline on E0 designs (μ={bl_e0.mean():.1f})",  color=COLORS["baseline"], **kw2)
# ax.hist(e0_e0,  label=f"Expert 0  on E0 designs (μ={e0_e0.mean():.1f})", color=COLORS["expert0"],  **kw2)
# ax.hist(bl_e1,  label=f"Baseline on E1 designs (μ={bl_e1.mean():.1f})",  color="#aaaaaa",          **kw2)
# ax.hist(e1_e1,  label=f"Expert 1  on E1 designs (μ={e1_e1.mean():.1f})", color=COLORS["expert1"],  **kw2)
# ax.set_xlabel("Curriculum level reached")
# ax.set_ylabel("Number of designs")
# ax.set_title("Per-expert specialist vs Baseline on own territory")
# ax.legend(fontsize=8)
# ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out1 = os.path.join(args.output_dir, "comparison_distributions.png")
fig.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Difference histograms
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle("Score improvement over Baseline (specialist − baseline)", fontsize=13)

diff_bins = np.arange(-16.5, 17.5, 1)

def plot_diff(ax, diffs, color, title, xlabel):
    wins   = (diffs > 0).sum()
    losses = (diffs < 0).sum()
    ties   = (diffs == 0).sum()
    n      = len(diffs)
    bars = ax.hist(diffs, bins=diff_bins, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(diffs.mean(), color="red", linewidth=1.4,
               linestyle="-", label=f"mean={diffs.mean():+.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Designs")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Annotate win/loss/tie
    ax.text(0.97, 0.97,
            f"wins={wins} ({100*wins/n:.0f}%)\n"
            f"ties={ties} ({100*ties/n:.0f}%)\n"
            f"losses={losses} ({100*losses/n:.0f}%)",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plot_diff(axes[0], diff_e0,  COLORS["expert0"], f"Expert 0's territory (190 designs)", "Level improvement (Expert 0 − Baseline)")
plot_diff(axes[1], diff_e1,  COLORS["expert1"], f"Expert 1's territory (210 designs)", "Level improvement (Expert 1 − Baseline)")
plot_diff(axes[2], diff_pec, COLORS["pec"],     f"max(E0, E1) on all 400 designs", "Level improvement (max(E0, E1) − Baseline)")

plt.tight_layout()
out2 = os.path.join(args.output_dir, "comparison_diff_histograms.png")
fig.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Spatial scatter: improvement coloured by (PEC - Baseline)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Spatial view: design location vs performance", fontsize=13)

vmax = int(max(abs(diff_pec).max(), 1))
cmap_div = plt.cm.RdYlGn

# Left: absolute PEC score
ax = axes[0]
sc = ax.scatter(spcf_all, gcr_all, c=pec_all, cmap="viridis",
                s=40, alpha=0.85, edgecolors="none",
                vmin=bl_all.min(), vmax=bl_all.max())
plt.colorbar(sc, ax=ax, label="Curriculum level (PEC)")
# Mark expert centers from pec_state
centers = [(0.7907, 0.00382), (0.8467, 0.00702)]
for i, (gcr_c, spcf_c) in enumerate(centers):
    ax.plot(spcf_c, gcr_c, marker="+", ms=14, mew=2.5,
            color=["#d62728","#1f77b4"][i], label=f"E{i} center")
ax.set_xlabel("Spring Coefficient (spcf)")
ax.set_ylabel("Gravity Compensation Ratio (GCR)")
ax.set_title("PEC oracle score per design")
ax.legend(fontsize=9)

# Right: improvement (PEC - Baseline), diverging colormap
ax = axes[1]
sc2 = ax.scatter(spcf_all, gcr_all, c=diff_pec, cmap=cmap_div,
                 s=40, alpha=0.85, edgecolors="none",
                 vmin=-vmax, vmax=vmax)
plt.colorbar(sc2, ax=ax, label="PEC − Baseline (levels)")
for i, (gcr_c, spcf_c) in enumerate(centers):
    ax.plot(spcf_c, gcr_c, marker="+", ms=14, mew=2.5,
            color=["#d62728","#1f77b4"][i], label=f"E{i} center")
ax.set_xlabel("Spring Coefficient (spcf)")
ax.set_ylabel("Gravity Compensation Ratio (GCR)")
ax.set_title("Improvement: PEC oracle − Baseline")
ax.legend(fontsize=9)

plt.tight_layout()
out3 = os.path.join(args.output_dir, "comparison_spatial.png")
fig.savefig(out3, dpi=150)
print(f"Saved: {out3}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Win / Tie / Loss breakdown (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Win / Tie / Loss breakdown vs Baseline", fontsize=13)

labels    = [f"Expert 0's territory ({S} designs)",
             f"Expert 1's territory ({len(bl_e1)} designs)",
             f"max(E0, E1) on all {len(bl_all)} designs"]
diffs_all = [diff_e0, diff_e1, diff_pec]
colors_all = [COLORS["expert0"], COLORS["expert1"], COLORS["pec"]]

x = np.arange(len(labels))
bar_w = 0.5

win_labels = ["Expert 0 wins", "Expert 1 wins", "max(E0, E1) wins"]
for i, (diffs, color, win_label) in enumerate(zip(diffs_all, colors_all, win_labels)):
    n = len(diffs)
    wins   = 100 * (diffs > 0).sum() / n
    ties   = 100 * (diffs == 0).sum() / n
    losses = 100 * (diffs < 0).sum() / n
    ax.bar(x[i], wins,             bar_w, color=color,    label=win_label)
    ax.bar(x[i], ties,             bar_w, bottom=wins,    color="#cccccc", label="Tied"          if i==0 else "")
    ax.bar(x[i], losses,           bar_w, bottom=wins+ties, color="#e07070", label="Baseline wins" if i==0 else "")
    ax.text(x[i], wins / 2,                    f"{wins:.0f}%",   ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(x[i], wins + ties / 2,             f"{ties:.0f}%",   ha="center", va="center", fontsize=10)
    ax.text(x[i], wins + ties + losses / 2,    f"{losses:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("% of designs")
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out4 = os.path.join(args.output_dir, "comparison_win_breakdown.png")
fig.savefig(out4, dpi=150)
print(f"Saved: {out4}")
plt.close(fig)

print("\nAll plots saved.")
