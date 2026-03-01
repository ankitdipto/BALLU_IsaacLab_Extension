"""
PEC Final Evaluation
====================
Evaluate all PEC experts on an unseen set of designs.

Reads ``pec_state.json`` to discover the latest checkpoint for each expert,
then calls ``pec_eval_expert_frontier.py`` once per expert (sequentially, one
GPU process at a time) using the provided designs file as ``--frontier_file``.

The designs file must be in the same JSON format produced by
``pec_evaluate_frontier.py``:
    [{\"id\": 0, \"GCR\": 0.82, \"spcf\": 0.005}, ...]

Outputs
-------
    logs/pec/<run_name>/final_eval/<designs_stem>/
        expert_<k>_results.json   — full per-design result payload
        summary.json              — aggregated scores matrix + metadata

Usage (run from ballu_isclb_extension/)
---------------------------------------
    python scripts/pec/pec_eval_final.py \\
        --run_name      my_pec_run \\
        --designs_file  path/to/test_designs.json \\
        --num_episodes  30 \\
        --start_difficulty 21 \\
        --headless
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # ballu_isclb_extension/
EVAL_SCRIPT  = os.path.join(SCRIPT_DIR, "pec_eval_expert_frontier.py")


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess launcher
# ──────────────────────────────────────────────────────────────────────────────

def run_expert_eval(
    expert_id: int,
    checkpoint: str,
    designs_file: str,
    output_file: str,
    num_episodes: int,
    start_difficulty: int,
    task: str,
    device: str,
    headless: bool,
    timeout_h: float,
    usd_rel_path: str | None,
) -> list | None:
    """
    Launch pec_eval_expert_frontier.py for one expert and block until done.

    Returns the list of per-design result dicts on success, None on failure.
    """
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--checkpoint_path",  checkpoint,
        "--frontier_file",    designs_file,
        "--output",           output_file,
        "--num_episodes",     str(num_episodes),
        "--start_difficulty", str(start_difficulty),
        "--task",             task,
        "--device",           device,
    ]
    if headless:
        cmd.append("--headless")

    print(f"\n  {'─' * 66}")
    print(f"  Expert {expert_id}  checkpoint: {os.path.basename(checkpoint)}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  {'─' * 66}")

    subprocess_env = os.environ.copy()
    if usd_rel_path:
        subprocess_env["BALLU_USD_REL_PATH"] = usd_rel_path

    t0 = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=PROJECT_ROOT,
        env=subprocess_env,
    )

    results = None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("FRONTIER_RESULTS:"):
                payload = line.split("FRONTIER_RESULTS:", 1)[1].strip()
                try:
                    results = json.loads(payload)
                except json.JSONDecodeError:
                    print("  [WARNING] Could not parse FRONTIER_RESULTS line.")

        process.wait(timeout=timeout_h * 3600)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"\n  [ERROR] Expert {expert_id} eval timed out after {timeout_h:.1f} h.")
        return None

    elapsed = (time.time() - t0) / 60.0
    print(f"\n  Expert {expert_id} eval finished in {elapsed:.1f} min "
          f"(exit={process.returncode})")

    if process.returncode != 0:
        print(f"  [ERROR] Subprocess exited with code {process.returncode}.")
        return None

    # Fallback: parse from output file if stdout line was missed.
    if results is None and os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f).get("results")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PEC final evaluation — run all experts on an unseen designs file."
    )
    # Required
    parser.add_argument("--run_name",     type=str, required=True,
                        help="PEC run name (must match an existing pec_state.json).")
    parser.add_argument("--designs_file", type=str, required=True,
                        help="JSON file of designs to evaluate: [{id, GCR, spcf}, ...]")

    # Optional PEC paths
    parser.add_argument("--log_root",     type=str, default="logs/pec",
                        help="Root directory for PEC logs (default: logs/pec).")

    # Eval parameters (forwarded to pec_eval_expert_frontier.py)
    parser.add_argument("--num_episodes",    type=int,   default=30,
                        help="Episodes per design (default: 30).")
    parser.add_argument("--start_difficulty", type=int,  default=21,
                        help="Starting obstacle level index (default: 21).")
    parser.add_argument("--task",   type=str, default="Isc-BALLU-hetero-general",
                        help="Isaac Lab task name (default: Isc-BALLU-hetero-general).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device (default: cuda:0).")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Sim in headless mode.")
    parser.add_argument("--timeout_h", type=float, default=4.0,
                        help="Per-expert eval subprocess timeout in hours (default: 4).")

    args = parser.parse_args()

    # ── Load PEC state ────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    experts      = state["experts"]
    pec_iter_done = state["iteration"]          # number of completed PEC iters
    last_iter     = pec_iter_done - 1            # 0-based index of last iter
    usd_rel_path  = state.get("usd_rel_path")
    K = len(experts)

    # ── Validate designs file ─────────────────────────────────────────────────
    designs_file = os.path.abspath(args.designs_file)
    if not os.path.exists(designs_file):
        print(f"[ERROR] Designs file not found: {designs_file}")
        sys.exit(1)

    with open(designs_file) as f:
        designs = json.load(f)
    F = len(designs)
    if F == 0:
        print("[ERROR] Designs file is empty.")
        sys.exit(1)

    # ── Create output directory ───────────────────────────────────────────────
    designs_stem = os.path.splitext(os.path.basename(designs_file))[0]
    out_dir = os.path.join(run_dir, "final_eval", designs_stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PEC Final Evaluation — run: {args.run_name}")
    print(f"{'=' * 70}")
    print(f"  State file       : {state_path}")
    print(f"  PEC iters done   : {pec_iter_done}  (evaluating checkpoints from iter {last_iter})")
    print(f"  K experts        : {K}")
    print(f"  Designs file     : {designs_file}  ({F} designs)")
    print(f"  Output dir       : {out_dir}")
    print(f"  usd_rel_path     : {usd_rel_path or '(not set)'}")
    print(f"  Episodes / design: {args.num_episodes}")
    print(f"  Start difficulty : {args.start_difficulty}")
    print(f"  Timeout / expert : {args.timeout_h:.1f} h")
    print(f"{'=' * 70}")

    # ── Verify checkpoints ────────────────────────────────────────────────────
    for ex in experts:
        kid  = ex["id"]
        ckpt = ex.get("checkpoint") or ""
        trained_at = ex.get("last_trained_pec_iter")

        if not ckpt:
            print(f"\n[WARNING] Expert {kid}: no checkpoint recorded — will be skipped.")
        elif not os.path.exists(ckpt):
            print(f"\n[WARNING] Expert {kid}: checkpoint not found on disk — will be skipped.")
            print(f"           {ckpt}")
        elif trained_at != last_iter:
            print(f"\n[WARNING] Expert {kid}: last trained at PEC iter {trained_at}, "
                  f"expected {last_iter}. Using checkpoint anyway:")
            print(f"           {ckpt}")

    # ── Evaluate each expert sequentially ─────────────────────────────────────
    print(f"\n  Running {K} expert eval subprocesses (sequential, single GPU)...")

    scores_matrix  = {}   # expert_id -> list[int] | None
    expert_results = {}   # expert_id -> list[dict] | None

    t_total_start = time.time()
    for ex in experts:
        kid  = ex["id"]
        ckpt = ex.get("checkpoint") or ""

        if not ckpt or not os.path.exists(ckpt):
            print(f"\n  [SKIP] Expert {kid}: no valid checkpoint.")
            scores_matrix[kid]  = None
            expert_results[kid] = None
            continue

        output_file = os.path.join(out_dir, f"expert_{kid}_results.json")

        results = run_expert_eval(
            expert_id=kid,
            checkpoint=ckpt,
            designs_file=designs_file,
            output_file=output_file,
            num_episodes=args.num_episodes,
            start_difficulty=args.start_difficulty,
            task=args.task,
            device=args.device,
            headless=args.headless,
            timeout_h=args.timeout_h,
            usd_rel_path=usd_rel_path,
        )

        if results is None:
            scores_matrix[kid]  = None
            expert_results[kid] = None
        else:
            results_sorted      = sorted(results, key=lambda r: r["id"])
            scores_matrix[kid]  = [r["best_level_idx"] for r in results_sorted]
            expert_results[kid] = results_sorted

    t_total = (time.time() - t_total_start) / 60.0

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "run_name":         args.run_name,
        "pec_iter_done":    pec_iter_done,
        "checkpoints_from_iter": last_iter,
        "designs_file":     designs_file,
        "F":                F,
        "K":                K,
        "num_episodes":     args.num_episodes,
        "start_difficulty": args.start_difficulty,
        "designs":          designs,
        "scores_matrix":    scores_matrix,
        "expert_results":   expert_results,
    }

    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Final Evaluation complete — {t_total:.1f} min total")
    print(f"  Summary saved: {summary_file}")

    print(f"\n  Scores (best curriculum level per design):")
    header = f"  {'Design':>8}" + "".join(f"  Expert{k:>2}" for k in range(K))
    print(header)
    print("  " + "-" * (8 + K * 10))
    for f_idx, des in enumerate(designs[:20]):
        row = f"  {f_idx:>8}"
        for kid in range(K):
            s = scores_matrix.get(kid)
            val = s[f_idx] if s is not None else "skip"
            row += f"  {val:>8}"
        print(row)
    if F > 20:
        print(f"  ... ({F - 20} more rows)")

    print(f"\n  Mean best-level per expert (PEC oracle = row-wise max):")
    for kid in range(K):
        s = scores_matrix.get(kid)
        if s is None:
            print(f"    Expert {kid}: skipped")
        else:
            print(f"    Expert {kid}: mean={sum(s)/len(s):.2f}  "
                  f"min={min(s)}  max={max(s)}")

    # PEC oracle: for each design, the best score any expert achieved.
    valid_matrices = [s for s in scores_matrix.values() if s is not None]
    if valid_matrices:
        oracle = [max(s[i] for s in valid_matrices) for i in range(F)]
        print(f"\n  PEC oracle    : mean={sum(oracle)/len(oracle):.2f}  "
              f"min={min(oracle)}  max={max(oracle)}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
