#!/usr/bin/env python3
"""Static publication-style plots for MoE gate_probs.jsonl time series.

When softmax weights stay near uniform, line plots of six ~0.17 curves are hard to read.
This script adds views that amplify small changes:

  1) Stacked area — composition over time (always sums to 1).
  2) Heatmap — iteration × expert, color = probability.
  3) Residual heatmap — color = p_k - 1/K (diverging scale around zero).
  4) Summary curves — entropy, max prob, L2 distance from uniform (optional).

Run from ballu_isclb_extension with BALLU_env0 (needs numpy, matplotlib):

  python scripts/analysis/plot_gate_probs_timelapse.py \\
    --jsonl logs/rsl_rl/lab_04.14.2026/2026-04-09_13-04-38_soft_moe_univctrl_3d_heldout1000/gate_probs.jsonl \\
    --output-dir logs/rsl_rl/.../plots_gate_timelapse
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=str, required=True, help="Path to gate_probs.jsonl")
    p.add_argument("--output-dir", type=str, required=True, help="Directory for PNG outputs")
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every Nth sample along iteration axis (1 = no downsampling)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="If >1, rolling-mean window size along iteration (applied after downsample)",
    )
    return p.parse_args()


def load_series(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Returns (header, iterations (T,), probs (num_envs, T, K))."""
    header = None
    iters: list[int] = []
    rows_probs: list[list[list[float]]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "header":
                header = o
            elif o.get("type") == "data":
                iters.append(int(o["iteration"]))
                n_env = int(header["num_envs_logged"])
                K = int(header["num_experts"])
                env_block = []
                for e in range(n_env):
                    env_block.append(o[f"env_{e}_gate_probs"])
                rows_probs.append(env_block)

    if header is None or not iters:
        raise SystemExit("No header or data rows found.")

    T = len(iters)
    n_env = int(header["num_envs_logged"])
    K = int(header["num_experts"])
    P = np.zeros((n_env, T, K), dtype=np.float64)
    for t, env_block in enumerate(rows_probs):
        for e in range(n_env):
            P[e, t, :] = np.array(env_block[e], dtype=np.float64)

    return header, np.array(iters, dtype=np.int64), P


def maybe_downsample_smooth(
    iters: np.ndarray, P: np.ndarray, downsample: int, smooth: int
) -> tuple[np.ndarray, np.ndarray]:
    if downsample > 1:
        iters = iters[::downsample]
        P = P[:, ::downsample, :]
    if smooth > 1:
        pad = smooth // 2
        # pad along time with edge values for same-length convolution
        def roll_mean(x: np.ndarray) -> np.ndarray:
            # x: (T, K)
            xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
            ker = np.ones(smooth, dtype=np.float64) / smooth
            out = np.zeros_like(x)
            for k in range(x.shape[1]):
                out[:, k] = np.convolve(xp[:, k], ker, mode="valid")
            return out

        T = P.shape[1]
        out = np.zeros_like(P)
        for e in range(P.shape[0]):
            out[e] = roll_mean(P[e])
        P = out
    return iters, P


def entropy_rows(P: np.ndarray) -> np.ndarray:
    """P: (T, K) -> (T,) Shannon entropy nats."""
    p = np.clip(P, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def plot_env(
    env_idx: int,
    iters: np.ndarray,
    P: np.ndarray,
    K: int,
    uniform: float,
    out_dir: Path,
    routing: str,
) -> None:
    """P shape (T, K) for this env."""
    colors = plt.cm.tab10(np.linspace(0, 0.9, K))
    H = entropy_rows(P)
    H_u = math.log(K)
    maxp = np.max(P, axis=-1)

    # --- Stacked gate mixture only ---
    fig, ax = plt.subplots(figsize=(11, 4.2))
    layers = [P[:, k] for k in range(K)]
    ax.stackplot(iters, layers, labels=[f"e{k}" for k in range(K)], colors=colors, alpha=0.92)
    ax.axhline(uniform, color="0.3", ls="--", lw=0.8, alpha=0.7, label="1/K")
    ax.set_ylabel("gate prob")
    ax.set_xlabel("PPO iteration")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncol=min(K, 6), fontsize=8, framealpha=0.9)
    ax.set_title(f"Env {env_idx}  |  stacked gate mixture  |  routing={routing}")
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_stacked_mixture.png", dpi=160)
    plt.close(fig)

    # --- Entropy only ---
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(iters, H, color="0.2", lw=1.0, label="H(π)")
    ax.axhline(H_u, color="C3", ls="--", lw=1.0, label=f"log({K}) uniform")
    ax.set_ylabel("entropy (nats)")
    ax.set_xlabel("PPO iteration")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Env {env_idx}  |  gate entropy  |  routing={routing}")
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_entropy.png", dpi=160)
    plt.close(fig)

    # --- Max prob only ---
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axhline(
        uniform,
        color="red",
        ls="--",
        lw=1.5,
        alpha=0.95,
        label="1/K",
        zorder=2,
    )
    ax.plot(iters, maxp, color="C0", lw=1.0, label="max π_k", zorder=3)
    ax.set_ylabel("max prob")
    ax.set_xlabel("PPO iteration")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Env {env_idx}  |  max gate probability  |  routing={routing}")
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_max_prob.png", dpi=160)
    plt.close(fig)

    # --- Heatmap probability ---
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(
        P.T,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=[iters[0], iters[-1], -0.5, K - 0.5],
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"expert {k}" for k in range(K)])
    ax.set_xlabel("PPO iteration")
    ax.set_title(f"Env {env_idx}  |  gate probability heatmap")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="π_k")
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_heatmap_probs.png", dpi=160)
    plt.close(fig)

    # --- Residual from uniform ---
    R = P - uniform
    vmax = max(0.02, float(np.nanmax(np.abs(R))) * 1.05)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(
        R.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        extent=[iters[0], iters[-1], -0.5, K - 0.5],
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"expert {k}" for k in range(K)])
    ax.set_xlabel("PPO iteration")
    ax.set_title(f"Env {env_idx}  |  π_k − 1/K  (symmetric color scale, max|res|={vmax:.4f})")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_heatmap_residual.png", dpi=160)
    plt.close(fig)

    # --- L2 distance from uniform ---
    u = np.full(K, uniform)
    d = np.linalg.norm(P - u, axis=-1)
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.plot(iters, d, color="C2", lw=1.2)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(r"$\|\pi - \mathbf{1}/K\|_2$")
    ax.set_title(f"Env {env_idx}  |  Euclidean distance from uniform mixture")
    fig.tight_layout()
    fig.savefig(out_dir / f"env{env_idx}_l2_from_uniform.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    header, iters, P = load_series(path)
    iters, P = maybe_downsample_smooth(iters, P, args.downsample, args.smooth)

    K = int(header["num_experts"])
    n_env = int(header["num_envs_logged"])
    routing = header.get("routing_type", "?")
    uniform = 1.0 / K

    meta = {
        "source": str(path),
        "num_envs": n_env,
        "num_experts": K,
        "routing_type": routing,
        "T": int(P.shape[1]),
        "downsample": args.downsample,
        "smooth": args.smooth,
    }
    (out_dir / "timelapse_meta.json").write_text(json.dumps(meta, indent=2))

    for e in range(n_env):
        plot_env(e, iters, P[e], K, uniform, out_dir, routing)

    # stacked, entropy, max_prob, heatmap_probs, heatmap_residual, l2
    per_env = 6
    print(f"Wrote {per_env * n_env} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
