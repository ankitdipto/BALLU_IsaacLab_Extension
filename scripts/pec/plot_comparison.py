"""
PEC vs Baseline Comparison — generic K-expert version
======================================================
Compares max(Expert_0 .. Expert_K) against a baseline universal controller
across a shared set of evaluated designs.

Usage (from ballu_isclb_extension/):
    python scripts/pec/plot_comparison.py \\
        --baseline   logs/pec/triple_specialists/baseline_univctrl_results.json \\
        --experts    logs/pec/triple_specialists/expert0_iter2_mdbest_results.json \\
                     logs/pec/triple_specialists/expert1_iter2_md2997_results.json \\
                     logs/pec/triple_specialists/expert2_iter2_md3397_results.json \\
        --pec_state  logs/pec/triple_specialists/pec_state.json \\
        --run_name   triple_specialists \\
        --output_dir logs/pec/triple_specialists/plots
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Colour palette (expert index → colour) ───────────────────────────────────
EXPERT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f00",
                 "#984ea3", "#a65628", "#f781bf", "#999999"]

def expert_color(kid):
    return EXPERT_COLORS[kid % len(EXPERT_COLORS)]

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--baseline",   type=str, required=True)
parser.add_argument("--experts",    type=str, nargs="+", required=True,
                    help="Result JSON files for each expert, in order E0 E1 ...")
parser.add_argument("--pec_state",  type=str, default=None,
                    help="pec_state.json (optional, used for expert centres on map).")
parser.add_argument("--run_name",   type=str, default="pec_run")
parser.add_argument("--output_dir", type=str, default="logs/pec/plots")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(args.baseline) as f:
    bl_raw = json.load(f)
bl_results = bl_raw.get("results", bl_raw) if isinstance(bl_raw, dict) else bl_raw

expert_results = []
for path in args.experts:
    with open(path) as f:
        d = json.load(f)
    expert_results.append(d.get("results", d) if isinstance(d, dict) else d)

K  = len(expert_results)
N  = len(bl_results)

bl_scores    = np.array([r["best_level_idx"] for r in bl_results])
expert_scores = np.array([[r["best_level_idx"] for r in er] for er in expert_results])  # (K, N)
pec_scores   = expert_scores.max(axis=0)   # (N,)
pec_winner   = expert_scores.argmax(axis=0)  # (K,) index of best expert per design

gcr_all  = np.array([r["GCR"]  for r in bl_results])
spcf_all = np.array([r["spcf"] for r in bl_results])

diff_pec = pec_scores - bl_scores

# Expert centres from pec_state (optional)
centers = []
if args.pec_state and os.path.exists(args.pec_state):
    with open(args.pec_state) as f:
        state = json.load(f)
    centers = [(ex["mu"][0], ex["mu"][1]) for ex in state["experts"]]

# ── Print summary ─────────────────────────────────────────────────────────────
pec_better = (diff_pec > 0).sum()
bl_better  = (diff_pec < 0).sum()
tied       = (diff_pec == 0).sum()

print(f"\n=== {args.run_name}: GES oracle (K={K}) vs Baseline ({N} designs) ===")
print(f"{'Metric':<28} {'Baseline':>10} {'GES (max)':>10} {'Delta':>8}")
print("-" * 58)
print(f"{'Mean level':<28} {bl_scores.mean():>10.2f} {pec_scores.mean():>10.2f} {pec_scores.mean()-bl_scores.mean():>+8.2f}")
print(f"{'Median level':<28} {np.median(bl_scores):>10.1f} {np.median(pec_scores):>10.1f} {np.median(pec_scores)-np.median(bl_scores):>+8.1f}")
print(f"{'Min level':<28} {bl_scores.min():>10d} {pec_scores.min():>10d} {int(pec_scores.min()-bl_scores.min()):>+8d}")
print(f"{'Max level':<28} {bl_scores.max():>10d} {pec_scores.max():>10d} {int(pec_scores.max()-bl_scores.max()):>+8d}")
print(f"{'GES wins':<28} {'—':>10} {pec_better:>10d} ({100*pec_better/N:.1f}%)")
print(f"{'Baseline wins':<28} {bl_better:>10d} {'—':>10} ({100*bl_better/N:.1f}%)")
print(f"{'Tied':<28} {tied:>10d} {'':>10} ({100*tied/N:.1f}%)")
print(f"\nMean improvement   (GES - Baseline): {diff_pec.mean():+.2f} levels")
print(f"Median improvement (GES - Baseline): {np.median(diff_pec):+.1f} levels")

# Winner distribution among GES victories
wins_mask = diff_pec > 0
winner_counts = Counter(pec_winner[wins_mask])
print(f"\nWinning expert breakdown (on {pec_better} GES-wins):")
for kid in range(K):
    cnt = winner_counts.get(kid, 0)
    print(f"  Expert {kid}: {cnt:4d} wins  ({100*cnt/max(pec_better,1):.1f}%)")

BIN_EDGES = np.arange(-0.5, max(pec_scores.max(), bl_scores.max()) + 1.5, 1)
DIFF_BINS = np.arange(diff_pec.min() - 1.5, diff_pec.max() + 2.5, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Score distributions
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{args.run_name} — Score distributions ({N} designs, K={K} experts)", fontsize=13)

# Left: Baseline vs GES oracle
ax = axes[0]
kw = dict(bins=BIN_EDGES, alpha=0.6, edgecolor="white", linewidth=0.4)
ax.hist(bl_scores,  color="#555555", label=f"Baseline (μ={bl_scores.mean():.1f})", **kw)
ax.hist(pec_scores, color="#2ca02c", label=f"GES max  (μ={pec_scores.mean():.1f})", **kw)
ax.set_xlabel("Curriculum level reached")
ax.set_ylabel("Designs")
ax.set_title("Baseline vs GES oracle")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Right: per-expert distributions
ax = axes[1]
ax.hist(bl_scores, color="#555555", alpha=0.35, bins=BIN_EDGES,
        edgecolor="white", linewidth=0.4, label=f"Baseline (μ={bl_scores.mean():.1f})")
for kid in range(K):
    sc = expert_scores[kid]
    ax.hist(sc, bins=BIN_EDGES, alpha=0.45, edgecolor="white", linewidth=0.4,
            color=expert_color(kid), label=f"Expert {kid} (μ={sc.mean():.1f})")
ax.set_xlabel("Curriculum level reached")
ax.set_ylabel("Designs")
ax.set_title("Per-expert score distribution (all designs)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out1 = os.path.join(args.output_dir, "comparison_distributions.png")
fig.savefig(out1, dpi=150)
print(f"\nSaved: {out1}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Difference histogram (GES - Baseline)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(f"{args.run_name} — GES oracle improvement over Baseline", fontsize=13)

bucket_size = 2
buckets = Counter(int(d // bucket_size) * bucket_size for d in diff_pec)
ax.hist(diff_pec, bins=DIFF_BINS, color="#2ca02c", alpha=0.75,
        edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.axvline(diff_pec.mean(), color="red", linewidth=1.5, linestyle="-",
           label=f"mean = {diff_pec.mean():+.2f}")
ax.set_xlabel("Level improvement (GES − Baseline)")
ax.set_ylabel("Designs")
ax.set_title(f"GES oracle on all {N} designs")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.text(0.97, 0.97,
        f"wins={pec_better} ({100*pec_better/N:.0f}%)\n"
        f"ties={tied} ({100*tied/N:.0f}%)\n"
        f"losses={bl_better} ({100*bl_better/N:.0f}%)",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

plt.tight_layout()
out2 = os.path.join(args.output_dir, "comparison_diff_histogram.png")
fig.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Win/Tie/Loss bar + Expert win-share pie (side by side)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"{args.run_name} — Win / Tie / Loss & Expert win attribution", fontsize=13)

# Left: stacked bar
ax = axes[0]
win_pct  = 100 * pec_better / N
tie_pct  = 100 * tied        / N
loss_pct = 100 * bl_better   / N

ax.bar(0, win_pct,  0.45, color="#2ca02c",  label="GES wins")
ax.bar(0, tie_pct,  0.45, bottom=win_pct,   color="#cccccc",  label="Tied")
ax.bar(0, loss_pct, 0.45, bottom=win_pct+tie_pct, color="#e07070", label="Baseline wins")
ax.text(0, win_pct / 2,             f"{win_pct:.0f}%",  ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.text(0, win_pct + tie_pct / 2,  f"{tie_pct:.0f}%",  ha="center", va="center",
        fontsize=12)
ax.text(0, win_pct + tie_pct + loss_pct / 2, f"{loss_pct:.0f}%", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.set_xticks([0])
ax.set_xticklabels([f"GES oracle\n(K={K}, n={N})"], fontsize=11)
ax.set_ylabel("% of designs")
ax.set_ylim(0, 100)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Right: pie chart — which expert wins when GES wins
ax = axes[1]
pie_labels = [f"Expert {kid}\n({winner_counts.get(kid,0)} wins, "
              f"{100*winner_counts.get(kid,0)/max(pec_better,1):.1f}%)"
              for kid in range(K)]
pie_vals   = [winner_counts.get(kid, 0) for kid in range(K)]
pie_colors = [expert_color(kid) for kid in range(K)]
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    textprops=dict(fontsize=9),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax.set_title(f"Which expert wins when GES beats baseline\n"
             f"({pec_better} GES-win designs)", fontsize=10)

plt.tight_layout()
out3 = os.path.join(args.output_dir, "comparison_win_breakdown.png")
fig.savefig(out3, dpi=150)
print(f"Saved: {out3}")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Spatial scatter: improvement over design space
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"{args.run_name} — Spatial view over (spcf, GCR) design space", fontsize=13)

vmax = int(max(abs(diff_pec).max(), 1))

# Left: winning expert for each design (colour) + GES score (size)
ax = axes[0]
norm_size = 10 + 25 * (pec_scores - pec_scores.min()) / max(pec_scores.ptp(), 1)
for kid in range(K):
    mask = (pec_winner == kid)
    ax.scatter(spcf_all[mask], gcr_all[mask], c=expert_color(kid),
               s=norm_size[mask], alpha=0.7, edgecolors="none",
               label=f"Expert {kid} wins ({mask.sum()})")
for kid, (gcr_c, spcf_c) in enumerate(centers):
    ax.plot(spcf_c, gcr_c, "+", ms=14, mew=2.5,
            color=expert_color(kid), zorder=5)
ax.set_xlabel("Spring Coefficient (spcf)")
ax.set_ylabel("Gravity Compensation Ratio (GCR)")
ax.set_title("Winning expert per design\n(dot size ∝ GES score)")
ax.legend(fontsize=8, loc="upper left")

# Right: improvement map (diverging colormap)
ax = axes[1]
sc = ax.scatter(spcf_all, gcr_all, c=diff_pec, cmap="RdYlGn",
                s=20, alpha=0.85, edgecolors="none",
                vmin=-vmax, vmax=vmax)
plt.colorbar(sc, ax=ax, label="GES − Baseline (levels)")
for kid, (gcr_c, spcf_c) in enumerate(centers):
    ax.plot(spcf_c, gcr_c, "+", ms=14, mew=2.5,
            color=expert_color(kid), label=f"E{kid} centre", zorder=5)
ax.set_xlabel("Spring Coefficient (spcf)")
ax.set_ylabel("Gravity Compensation Ratio (GCR)")
ax.set_title("Improvement: GES oracle − Baseline")
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
out4 = os.path.join(args.output_dir, "comparison_spatial.png")
fig.savefig(out4, dpi=150)
print(f"Saved: {out4}")
plt.close(fig)

print("\nAll plots saved.")
