"""
PEC Final Evaluation
====================
Evaluate all PEC experts on an unseen set of designs.

Reads ``pec_state.json`` to discover the latest checkpoint for each expert,
then calls ``pec_eval_expert_frontier.py`` once per expert (sequentially, one
GPU process at a time) using the provided designs file as ``--frontier_file``.

The designs file format depends on the PEC mode:

  2D mode: [{\"id\": 0, \"GCR\": 0.82, \"spcf\": 0.005}, ...]
  3D mode: [{\"id\": 0, \"GCR\": 0.82, \"spcf\": 0.05, \"leg\": 0.35}, ...]

In 3D mode, USDs are generated for all unique leg_lengths found in the designs
file before evaluation begins.  ``BALLU_USD_ORDER_FILE`` is injected into each
expert eval subprocess so env i evaluates exactly design i.

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
import tempfile
import time


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT          = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # ballu_isclb_extension/
EVAL_SCRIPT           = os.path.join(SCRIPT_DIR, "pec_eval_expert_frontier.py")
GENERATE_USDS_SCRIPT  = os.path.join(SCRIPT_DIR, "pec_generate_usds.py")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — USD generation (3D mode)
# ──────────────────────────────────────────────────────────────────────────────

def _leg_key(leg: float, precision: int = 4) -> str:
    return f"{round(leg, precision):.{precision}f}"


def _generate_usds_for_designs(designs: list, output_dir: str,
                                leg_precision: int = 4) -> dict | None:
    """
    Call pec_generate_usds.py for all unique leg_lengths in *designs*.

    Returns a dict mapping leg_key → abs_usd_path, or None on failure.
    """
    designs_data = [[d["GCR"], d["spcf"], d["leg"]] for d in designs]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="pec_final_usds_"
    )
    json.dump(designs_data, tmp)
    tmp.close()

    cmd = [
        sys.executable,
        GENERATE_USDS_SCRIPT,
        "--output_dir",    output_dir,
        "--designs_file",  tmp.name,
        "--leg_precision", str(leg_precision),
        "--skip_existing",
    ]
    print(f"\n  Generating USDs for final eval designs...")
    print(f"  cmd: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,   # stream to stdout so user sees progress
            text=True,
            cwd=PROJECT_ROOT,
        )
    finally:
        os.unlink(tmp.name)

    if result.returncode != 0:
        print(f"  [ERROR] USD generation failed (exit {result.returncode}).")
        return None

    # pec_generate_usds.py prints "USD_LEG_MAP: <json>" — but since we used
    # capture_output=False, we must fall back to reading the registry file.
    registry_path = os.path.join(output_dir, "morphology_registry.json")
    if not os.path.exists(registry_path):
        print(f"  [ERROR] morphology_registry.json not found at {output_dir}")
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    leg_map = {entry["key"]: entry["usd_path"] for entry in registry}
    print(f"  [INFO] USD map loaded: {len(leg_map)} unique leg lengths.")
    return leg_map


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
    usd_order_file: str | None = None,
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
    if usd_order_file:
        subprocess_env["BALLU_USD_ORDER_FILE"] = os.path.abspath(usd_order_file)
    elif usd_rel_path:
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
                        help="JSON file of designs to evaluate: "
                             "[{id, GCR, spcf}, ...] for 2D or "
                             "[{id, GCR, spcf, leg}, ...] for 3D.")

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
    parser.add_argument("--leg_precision", type=int, default=4,
                        help="Decimal places for leg_length deduplication key (default: 4).")

    args = parser.parse_args()

    # ── Load PEC state ────────────────────────────────────────────────────────
    run_dir    = os.path.join(args.log_root, args.run_name)
    state_path = os.path.join(run_dir, "pec_state.json")

    if not os.path.exists(state_path):
        print(f"[ERROR] State file not found: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    experts       = state["experts"]
    pec_iter_done = state["iteration"]          # number of completed PEC iters
    last_iter     = pec_iter_done - 1            # 0-based index of last iter
    usd_rel_path  = state.get("usd_rel_path")
    is_3d         = "leg" in state["design_space"]
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

    # Validate 3D designs have leg field if in 3D mode.
    if is_3d:
        missing = [i for i, d in enumerate(designs) if "leg" not in d]
        if missing:
            print(f"[ERROR] 3D PEC mode but {len(missing)} designs lack 'leg' field "
                  f"(first missing idx: {missing[0]}).")
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
    print(f"  PEC mode         : {'3D (GCR, spcf, leg)' if is_3d else '2D (GCR, spcf)'}")
    print(f"  PEC iters done   : {pec_iter_done}  (evaluating checkpoints from iter {last_iter})")
    print(f"  K experts        : {K}")
    print(f"  Designs file     : {designs_file}  ({F} designs)")
    print(f"  Output dir       : {out_dir}")
    if not is_3d:
        print(f"  usd_rel_path     : {usd_rel_path or '(not set)'}")
    print(f"  Episodes / design: {args.num_episodes}")
    print(f"  Start difficulty : {args.start_difficulty}")
    print(f"  Timeout / expert : {args.timeout_h:.1f} h")
    print(f"{'=' * 70}")

    # ── 3D: Generate USDs for all unique leg_lengths ──────────────────────────
    usd_order_file = None
    if is_3d:
        usds_dir = os.path.join(out_dir, "usds")
        os.makedirs(usds_dir, exist_ok=True)

        leg_map = _generate_usds_for_designs(designs, usds_dir, args.leg_precision)
        if leg_map is None:
            print("[ERROR] USD generation failed — aborting final eval.")
            sys.exit(1)

        # Build ordered USD list: one path per design in order.
        usd_order = []
        for d in designs:
            key = _leg_key(d["leg"], args.leg_precision)
            usd_path = leg_map.get(key)
            if usd_path is None:
                print(f"[ERROR] No USD found for leg_key '{key}'. leg_map keys: {list(leg_map.keys())}")
                sys.exit(1)
            usd_order.append(usd_path)

        usd_order_file = os.path.join(out_dir, "usd_order.json")
        with open(usd_order_file, "w") as f:
            json.dump(usd_order, f, indent=2)
        print(f"  [INFO] USD order file written: {usd_order_file}  ({len(usd_order)} paths)")

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
            usd_order_file=usd_order_file,
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
        "pec_mode":         "3d" if is_3d else "2d",
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
