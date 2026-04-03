"""
PEC Automated Run
=================
Progressive Expert Coverage — end-to-end orchestrator.

Chains pec_init → (pec_train_expert ×K → pec_evaluate_frontier →
pec_refit_gaussians → pec_visualize) × N iterations, driven by a
YAML configuration file.

Key features
------------
* **Automatic resume** — reads ``pec_state.json`` to determine where to
  continue; individual steps within a PEC iteration are also skipped if
  their output files already exist.
* **Per-expert ``--dl`` auto-scaling** — each expert's starting obstacle
  difficulty is set to ``max(0, floor(mean_eval_score) - dl_buffer)``
  based on that expert's mean score from the previous frontier evaluation.
* **Auto-visualization** — ``pec_visualize.py`` is called twice after each
  refit: once with ``--itr N`` (frontier overlay) and once without (current
  state). Visualisation failures are non-fatal.
* **Early stopping** — training ends when Monte Carlo coverage exceeds
  ``coverage_target``.

Usage (run from ballu_isclb_extension/)
----------------------------------------
    # Fresh run
    python scripts/pec/pec_run.py \\
        --run_name  my_run \\
        --config    scripts/pec/pec_config_template.yaml

    # Resume an interrupted run
    python scripts/pec/pec_run.py \\
        --run_name  my_run \\
        --config    scripts/pec/pec_config_template.yaml
    # (automatically resumes from where it left off)

    # Override number of iterations on the CLI
    python scripts/pec/pec_run.py \\
        --run_name  my_run \\
        --config    scripts/pec/pec_config_template.yaml \\
        --max_pec_iterations 5

    # Re-initialize an existing run (WARNING: overwrites state)
    python scripts/pec/pec_run.py \\
        --run_name  my_run \\
        --config    scripts/pec/pec_config_template.yaml \\
        --overwrite
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys

# PyYAML is available in the BALLU_env0 conda environment (Isaac Lab dependency).
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # ballu_isclb_extension/

_INIT_SCRIPT     = os.path.join(SCRIPT_DIR, "pec_init.py")
_TRAIN_SCRIPT    = os.path.join(SCRIPT_DIR, "pec_train_expert.py")
_EVAL_SCRIPT     = os.path.join(SCRIPT_DIR, "pec_evaluate_frontier.py")
_REFIT_SCRIPT    = os.path.join(SCRIPT_DIR, "pec_refit_gaussians.py")
_VIZ_SCRIPT      = os.path.join(SCRIPT_DIR, "pec_visualize.py")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — state I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_state(run_dir: str) -> dict:
    state_path = os.path.join(run_dir, "pec_state.json")
    with open(state_path) as f:
        return json.load(f)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — --dl auto-scaling
# ──────────────────────────────────────────────────────────────────────────────

def compute_expert_dl(expert_id: int, pec_iter: int,
                      run_dir: str, config: dict) -> int:
    """
    Compute the --dl (starting obstacle difficulty) for expert ``expert_id``
    at PEC iteration ``pec_iter``.

    * Iteration 0: returns ``config["dl_initial"]``.
    * Later iterations: reads the previous iteration's scores.json and sets
      ``dl = max(0, floor(mean_score) - dl_buffer)``.

    Falls back to ``dl_initial`` when scores are unavailable (e.g. the expert
    was skipped during the previous eval).
    """
    dl_initial = config.get("dl_initial", 0)
    dl_buffer  = config.get("dl_buffer",  2)

    if pec_iter == 0:
        return dl_initial

    prev_scores_file = os.path.join(
        run_dir, "frontier_evals", f"iter_{pec_iter - 1}", "scores.json"
    )
    if not os.path.exists(prev_scores_file):
        print(f"  [DL] Expert {expert_id}: scores.json for iter {pec_iter - 1} not found "
              f"— using dl_initial={dl_initial}")
        return dl_initial

    with open(prev_scores_file) as f:
        scores_payload = json.load(f)

    # JSON keys are strings; convert to int for lookup.
    scores_matrix = {int(k): v
                     for k, v in scores_payload.get("scores_matrix", {}).items()}
    expert_scores = scores_matrix.get(expert_id)

    if expert_scores is None:
        print(f"  [DL] Expert {expert_id}: no scores in iter {pec_iter - 1} eval "
              f"(skipped) — using dl_initial={dl_initial}")
        return dl_initial

    valid = [s for s in expert_scores if s is not None]
    if not valid:
        return dl_initial

    mean_score = sum(valid) / len(valid)
    dl = max(0, int(mean_score) - dl_buffer)
    print(f"  [DL] Expert {expert_id}: mean_score_prev={mean_score:.1f}  "
          f"dl_buffer={dl_buffer}  → dl={dl}")
    return dl


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — inline MC coverage (avoids importing numpy in the orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

def compute_coverage(state: dict, n_mc: int = 10_000) -> float:
    """
    Monte-Carlo estimate of the fraction of the design space covered by the
    current Gaussian mixture.  A point θ is covered if any expert's
    unnormalised density exceeds exp(-2) ≈ 0.135 (≈ 2-sigma radius).

    Mirrors the logic in ``pec_refit_gaussians.coverage_estimate``.
    Handles both 2D (GCR, spcf) and 3D (GCR, spcf, leg) design spaces.
    """
    experts  = state["experts"]
    gcr_lo,  gcr_hi  = state["design_space"]["GCR"]
    spcf_lo, spcf_hi = state["design_space"]["spcf"]
    is_3d = "leg" in state["design_space"]
    if is_3d:
        leg_lo, leg_hi = state["design_space"]["leg"]
    threshold = math.exp(-2.0)
    rng = random.Random(0)

    covered = 0
    for _ in range(n_mc):
        g = rng.uniform(gcr_lo, gcr_hi)
        s = rng.uniform(spcf_lo, spcf_hi)
        if is_3d:
            l = rng.uniform(leg_lo, leg_hi)
        for ex in experts:
            mu    = ex["mu"]
            sigma = ex["sigma"]
            d_g   = (g - mu[0]) ** 2 / (2.0 * sigma[0][0])
            d_s   = (s - mu[1]) ** 2 / (2.0 * sigma[1][1])
            if is_3d:
                d_l = (l - mu[2]) ** 2 / (2.0 * sigma[2][2])
                if math.exp(-(d_g + d_s + d_l)) > threshold:
                    covered += 1
                    break
            else:
                if math.exp(-(d_g + d_s)) > threshold:
                    covered += 1
                    break
    return covered / n_mc


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — subprocess runner
# ──────────────────────────────────────────────────────────────────────────────

def run_script(script_path: str, args_list: list, step_name: str) -> int:
    """
    Run a PEC sub-script as a subprocess, inheriting stdout/stderr so that
    all Isaac Sim and training output streams through in real time.

    Returns the exit code (0 = success).
    """
    cmd = [sys.executable, script_path] + [str(a) for a in args_list]
    print(f"\n{'─' * 70}")
    print(f"  {step_name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'─' * 70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


# ──────────────────────────────────────────────────────────────────────────────
# Step functions
# ──────────────────────────────────────────────────────────────────────────────

def step_init(run_name: str, config: dict, overwrite: bool) -> int:
    """Run pec_init.py to bootstrap the state file."""
    args = [
        "--run_name",    run_name,
        "--K",           config["K"],
        "--GCR_range",   *config["GCR_range"],
        "--spcf_range",  *config["spcf_range"],
        "--N_init",      config.get("N_init", 16),
        "--sigma_scale", config.get("sigma_scale", 0.3),
        "--seed",        config.get("seed", 42),
        "--log_root",    config.get("log_root", "logs/pec"),
    ]
    if config.get("leg_range") is not None:
        args += ["--leg_range"] + [str(v) for v in config["leg_range"]]
    if config.get("init_strategy") is not None:
        args += ["--init_strategy", config["init_strategy"]]
    if config.get("init_seed") is not None:
        args += ["--init_seed", config["init_seed"]]
    if config.get("init_anchor_region") is not None:
        args += ["--init_anchor_region", config["init_anchor_region"]]
    if config.get("target_init_coverage") is not None:
        args += ["--target_init_coverage", config["target_init_coverage"]]
    if config.get("lloyd_iterations") is not None:
        args += ["--lloyd_iterations", str(config["lloyd_iterations"])]
    if config.get("lloyd_tol") is not None:
        args += ["--lloyd_tol", str(config["lloyd_tol"])]
    if config.get("usd_rel_path"):
        args += ["--usd_rel_path", config["usd_rel_path"]]
    if config.get("centers"):
        # Flatten [[gcr0, spcf0], ...] → [gcr0, spcf0, gcr1, spcf1, ...]
        flat = [v for pair in config["centers"] for v in pair]
        args += ["--centers"] + flat
    if overwrite:
        args.append("--overwrite")
    return run_script(_INIT_SCRIPT, args, "INIT — pec_init.py")


def step_train_expert(run_name: str, expert_id: int, dl: int,
                      pec_iter: int, config: dict) -> int:
    """Run pec_train_expert.py for a single expert.

    Uses ``max_iterations_iter0`` for PEC iteration 0 (cold-start, needs more
    gradient steps) and ``max_iterations`` for all subsequent iterations
    (warm-started, fewer steps required).
    """
    if pec_iter == 0:
        max_iters = config.get("max_iterations_iter0",
                               config.get("max_iterations", 1500))
    else:
        max_iters = config.get("max_iterations", 500)

    args = [
        "--run_name",       run_name,
        "--expert_id",      expert_id,
        "--dl",             dl,
        "--log_root",       config.get("log_root", "logs/pec"),
        "--task",           config.get("task", "Isc-BALLU-hetero-general"),
        "--num_envs",       config.get("num_envs", 4096),
        "--max_iterations", max_iters,
        "--seed",           config.get("seed", 42),
        "--device",         config.get("device", "cuda:0"),
        "--timeout_h",      config.get("train_timeout_h", 8.0),
    ]
    if config.get("headless", True):
        args.append("--headless")
    return run_script(
        _TRAIN_SCRIPT, args,
        f"TRAIN — Expert {expert_id}  iter={pec_iter}  max_iters={max_iters}  dl={dl}"
    )


def step_evaluate_frontier(run_name: str, config: dict) -> int:
    """Run pec_evaluate_frontier.py (orchestrates K eval subprocesses)."""
    args = [
        "--run_name",         run_name,
        "--log_root",         config.get("log_root", "logs/pec"),
        "--F",                config.get("F", 100),
        "--pool_factor",      config.get("pool_factor", 20),
        "--sampling_mode",    config.get("sampling_mode", "auto"),
        "--auto_switch_iter", config.get("auto_switch_iter", 4),
        "--border_inner_ld",  config.get("border_inner_ld", -0.5),
        "--border_outer_ld",  config.get("border_outer_ld", -3.0),
        "--num_episodes",     config.get("num_episodes", 15),
        "--start_difficulty", config.get("start_difficulty", 21),
        "--task",             config.get("task", "Isc-BALLU-hetero-general"),
        "--device",           config.get("device", "cuda:0"),
        "--timeout_h",        config.get("eval_timeout_h", 4.0),
        "--use_iter_seed",    # different candidates each PEC iteration
    ]
    if config.get("headless", True):
        args.append("--headless")
    return run_script(_EVAL_SCRIPT, args, "EVAL — pec_evaluate_frontier.py")


def step_refit(run_name: str, config: dict) -> int:
    """Run pec_refit_gaussians.py."""
    args = [
        "--run_name",      run_name,
        "--log_root",      config.get("log_root", "logs/pec"),
        "--min_var_scale", config.get("min_var_scale", 0.01),
        "--min_designs",   config.get("min_designs", 2),
    ]
    return run_script(_REFIT_SCRIPT, args, "REFIT — pec_refit_gaussians.py")


def step_visualize_initial(run_name: str, config: dict) -> None:
    """
    Visualize the Gaussian layout immediately after pec_init.py.

    Produces ``plots/initial_state.png`` — current Gaussians, no frontier
    overlay.  Failure is non-fatal.
    """
    log_root  = config.get("log_root", "logs/pec")
    plots_dir = os.path.join(log_root, run_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out = os.path.join(plots_dir, "initial_state.png")
    rc = run_script(
        _VIZ_SCRIPT,
        ["--run_name", run_name, "--log_root", log_root, "--output", out],
        f"VISUALIZE — initial Gaussian state → {out}",
    )
    if rc != 0:
        print(f"  [WARN] pec_visualize.py (initial state) exited {rc} — continuing.")


def step_visualize(run_name: str, pec_iter: int, config: dict) -> None:
    """
    Run pec_visualize.py twice after a completed refit.

    Plot 1: ``--itr pec_iter`` — Gaussian state active *during* the frontier
    evaluation of iteration ``pec_iter``, with frontier candidate overlay.

    Plot 2: no ``--itr`` — current Gaussian state after the refit.

    Failures are silently absorbed (visualisation is non-critical).
    """
    log_root  = config.get("log_root", "logs/pec")
    plots_dir = os.path.join(log_root, run_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1 — frontier overlay
    out1 = os.path.join(plots_dir, f"iter_{pec_iter}_frontier.png")
    rc = run_script(
        _VIZ_SCRIPT,
        ["--run_name", run_name, "--log_root", log_root,
         "--itr", pec_iter, "--output", out1],
        f"VISUALIZE — iter {pec_iter} frontier overlay → {out1}",
    )
    if rc != 0:
        print(f"  [WARN] pec_visualize.py (frontier overlay) exited {rc} — continuing.")

    # Plot 2 — current state after refit
    out2 = os.path.join(plots_dir, f"after_iter_{pec_iter}_current.png")
    rc = run_script(
        _VIZ_SCRIPT,
        ["--run_name", run_name, "--log_root", log_root, "--output", out2],
        f"VISUALIZE — current state after iter {pec_iter} → {out2}",
    )
    if rc != 0:
        print(f"  [WARN] pec_visualize.py (current state) exited {rc} — continuing.")


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_iteration_summary(state: dict, pec_iter: int, coverage: float,
                            coverage_target: float) -> None:
    is_3d = "leg" in state["design_space"]
    print(f"\n{'=' * 70}")
    print(f"  PEC Iteration {pec_iter} complete  "
          f"(state iteration → {state['iteration']})")
    print(f"  MC coverage : {coverage * 100:.1f}%  "
          f"(target {coverage_target * 100:.0f}%)")
    if is_3d:
        print(f"\n  {'Expert':>7}  {'mu_GCR':>8}  {'mu_spcf':>9}  {'mu_leg':>7}  "
              f"{'σ_GCR':>7}  {'σ_spcf':>8}  {'σ_leg':>6}  {'n_designs':>9}  checkpoint")
        for ex in state["experts"]:
            std_g = math.sqrt(ex["sigma"][0][0])
            std_s = math.sqrt(ex["sigma"][1][1])
            std_l = math.sqrt(ex["sigma"][2][2])
            _ckpt = ex.get("checkpoint")
            ckpt  = os.path.basename(_ckpt) if _ckpt else "N/A"
            print(f"  {ex['id']:>7}  {ex['mu'][0]:>8.4f}  {ex['mu'][1]:>9.5f}  "
                  f"{ex['mu'][2]:>7.4f}  "
                  f"{std_g:>7.4f}  {std_s:>8.5f}  {std_l:>6.4f}  "
                  f"{len(ex['designs']):>9}  {ckpt}")
    else:
        print(f"\n  {'Expert':>7}  {'mu_GCR':>8}  {'mu_spcf':>9}  "
              f"{'σ_GCR':>7}  {'σ_spcf':>8}  {'n_designs':>9}  checkpoint")
        for ex in state["experts"]:
            std_g = math.sqrt(ex["sigma"][0][0])
            std_s = math.sqrt(ex["sigma"][1][1])
            _ckpt = ex.get("checkpoint")
            ckpt  = os.path.basename(_ckpt) if _ckpt else "N/A"
            print(f"  {ex['id']:>7}  {ex['mu'][0]:>8.4f}  {ex['mu'][1]:>9.5f}  "
                  f"{std_g:>7.4f}  {std_s:>8.5f}  {len(ex['designs']):>9}  {ckpt}")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Resumability helpers
# ──────────────────────────────────────────────────────────────────────────────

def expert_trained_this_iter(expert: dict, pec_iter: int) -> bool:
    """
    Return True if this expert has already been processed for PEC iteration
    ``pec_iter`` — either via successful training or via a graceful crash
    fallback that promoted the previous checkpoint.

    Uses the explicit ``last_trained_pec_iter`` field written by
    ``pec_train_expert.py`` (success) or by the fallback handler in ``main``
    (crash with a valid prior checkpoint).
    """
    ckpt = expert.get("checkpoint") or ""
    return (
        expert.get("last_trained_pec_iter") == pec_iter
        and bool(ckpt)
        and os.path.exists(ckpt)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC automated run — chains all PEC steps end-to-end."
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="PEC run name (subdirectory under logs/pec/).")
    parser.add_argument("--config",   type=str, required=True,
                        help="Path to YAML configuration file "
                             "(see pec_config_template.yaml).")
    parser.add_argument("--max_pec_iterations", type=int, default=None,
                        help="Override max_pec_iterations from the config file.")
    parser.add_argument("--K", type=int, default=None,
                        help="Override K (number of experts) from the config file.")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Override max_iterations (PEC iter >= 1) from the config file.")
    parser.add_argument("--max_iterations_iter0", type=int, default=None,
                        help="Override max_iterations_iter0 (PEC iter 0) from the config file.")
    parser.add_argument("--init_strategy", type=str, default=None,
                        choices=["grid", "stochastic_fps"],
                        help="Override init_strategy from the config file.")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="Override init_seed from the config file.")
    parser.add_argument("--init_anchor_region", type=str, default=None,
                        choices=["top_right"],
                        help="Override init_anchor_region from the config file.")
    parser.add_argument("--target_init_coverage", type=float, default=None,
                        help="Override target_init_coverage from the config file.")
    parser.add_argument("--leg_range", type=float, nargs=2, default=None,
                        metavar=("LEG_LO", "LEG_HI"),
                        help="Override leg_range [lo hi] from the config file "
                             "(enables 3D PEC over GCR × spcf × leg_length).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force re-initialisation even if pec_state.json "
                             "already exists (passes --overwrite to pec_init.py).")
    args = parser.parse_args()

    # ── Load config and apply CLI overrides ───────────────────────────────────
    config = load_config(args.config)
    if args.max_pec_iterations is not None:
        config["max_pec_iterations"] = args.max_pec_iterations
    if args.K is not None:
        config["K"] = args.K
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.max_iterations_iter0 is not None:
        config["max_iterations_iter0"] = args.max_iterations_iter0
    if args.init_strategy is not None:
        config["init_strategy"] = args.init_strategy
    if args.init_seed is not None:
        config["init_seed"] = args.init_seed
    if args.init_anchor_region is not None:
        config["init_anchor_region"] = args.init_anchor_region
    if args.target_init_coverage is not None:
        config["target_init_coverage"] = args.target_init_coverage
    if args.leg_range is not None:
        config["leg_range"] = args.leg_range

    log_root           = config.get("log_root", "logs/pec")
    run_dir            = os.path.join(log_root, args.run_name)
    max_pec_iterations = config.get("max_pec_iterations", 10)
    coverage_target    = config.get("coverage_target", 0.95)
    K                  = config["K"]

    state_path = os.path.join(run_dir, "pec_state.json")

    # ── Step 0: Initialise (or resume) ───────────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"  PEC Automated Run — {args.run_name}")
    print(f"  Config : {args.config}")
    print(f"  K={K}  max_iters={max_pec_iterations}  "
          f"coverage_target={coverage_target * 100:.0f}%")
    print(f"{'#' * 70}")

    # ── Persist effective config to run directory ─────────────────────────────
    # Written on every invocation so that CLI overrides are always captured.
    os.makedirs(run_dir, exist_ok=True)
    saved_config_path = os.path.join(run_dir, "pec_config.yaml")
    with open(saved_config_path, "w") as _f:
        yaml.dump(config, _f, default_flow_style=False, sort_keys=False)
    print(f"  Effective config saved to: {saved_config_path}")

    if not os.path.exists(state_path) or args.overwrite:
        action = "Overwriting existing run" if args.overwrite else "Starting fresh run"
        print(f"\n  {action} — running pec_init.py...")
        rc = step_init(args.run_name, config, overwrite=args.overwrite)
        if rc != 0:
            print(f"\n[ERROR] pec_init.py failed with exit code {rc}.")
            sys.exit(rc)
        step_visualize_initial(args.run_name, config)
    else:
        state = load_state(run_dir)
        print(f"\n  Resuming from PEC iteration {state['iteration']}.")

    # ── Main PEC Loop ─────────────────────────────────────────────────────────
    for pec_iter in range(max_pec_iterations):

        state = load_state(run_dir)

        # Skip fully completed iterations.
        if state["iteration"] > pec_iter:
            print(f"  [SKIP] PEC iteration {pec_iter} already complete.")
            continue

        print(f"\n{'#' * 70}")
        print(f"  PEC ITERATION {pec_iter}  /  {max_pec_iterations - 1}")
        print(f"{'#' * 70}")

        # ── Step 1: Train each expert ─────────────────────────────────────────
        for k in range(K):
            # Reload state: a previous expert may have updated the file.
            state = load_state(run_dir)
            expert = state["experts"][k]

            if expert_trained_this_iter(expert, pec_iter):
                print(f"  [SKIP] Expert {k} already trained for iter {pec_iter}.")
                continue

            dl = compute_expert_dl(k, pec_iter, run_dir, config)
            rc = step_train_expert(args.run_name, k, dl, pec_iter, config)
            if rc != 0:
                # Reload state: pec_train_expert.py did NOT update it on crash,
                # so expert["checkpoint"] still points to the previous iteration's
                # model_best.pt — exactly the fallback we want.
                state = load_state(run_dir)
                expert = state["experts"][k]
                fallback_ckpt = expert.get("checkpoint") or ""
                if fallback_ckpt and os.path.exists(fallback_ckpt):
                    print(f"\n[WARN] Training expert {k} crashed (exit {rc}). "
                          f"Falling back to previous checkpoint:")
                    print(f"       {fallback_ckpt}")
                    expert["last_trained_pec_iter"] = pec_iter
                    state_path_w = os.path.join(run_dir, "pec_state.json")
                    with open(state_path_w, "w") as _f:
                        json.dump(state, _f, indent=2)
                    print(f"       State updated — expert {k} marked done for iter {pec_iter}.")
                else:
                    print(f"\n[ERROR] Training expert {k} crashed (exit {rc}) "
                          f"and no fallback checkpoint is available. Aborting.")
                    sys.exit(rc)

        # ── Step 2: Evaluate frontier ─────────────────────────────────────────
        scores_file = os.path.join(
            run_dir, "frontier_evals", f"iter_{pec_iter}", "scores.json"
        )
        if os.path.exists(scores_file):
            print(f"  [SKIP] Frontier eval for iter {pec_iter} already complete.")
        else:
            rc = step_evaluate_frontier(args.run_name, config)
            if rc != 0:
                print(f"\n[ERROR] Frontier evaluation failed (exit code {rc}).")
                sys.exit(rc)

        # ── Step 3: Refit Gaussians ───────────────────────────────────────────
        # Always runs when state["iteration"] == pec_iter (refit increments it).
        rc = step_refit(args.run_name, config)
        if rc != 0:
            print(f"\n[ERROR] Gaussian refit failed (exit code {rc}).")
            sys.exit(rc)

        # ── Step 4: Visualize (non-fatal) ─────────────────────────────────────
        step_visualize(args.run_name, pec_iter, config)

        # ── Summary + early stopping ──────────────────────────────────────────
        state    = load_state(run_dir)
        coverage = compute_coverage(state)
        print_iteration_summary(state, pec_iter, coverage, coverage_target)

        if coverage >= coverage_target:
            print(f"  [DONE] Coverage target {coverage_target * 100:.0f}% reached "
                  f"after iteration {pec_iter}. Stopping.")
            break

    else:
        state = load_state(run_dir)
        print(f"\n  [DONE] Completed {max_pec_iterations} PEC iterations.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  Run name   : {args.run_name}")
    print(f"  State file : {state_path}")
    print(f"  Iterations : {state['iteration']}")
    print(f"\n  To visualise the final state:")
    print(f"    python scripts/pec/pec_visualize.py --run_name {args.run_name}")
    print(f"\n  To compare against a baseline:")
    print(f"    python scripts/pec/plot_comparison.py \\")
    print(f"        --baseline  <baseline_results.json> \\")
    print(f"        --experts   <expert0_results.json> ... \\")
    print(f"        --pec_state {state_path}")


if __name__ == "__main__":
    main()
