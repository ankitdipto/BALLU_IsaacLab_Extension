#!/usr/bin/env python3
"""
PEC USD Generation Script
=========================
Generate USD files for unique leg_lengths derived from 3D PEC designs.

Called by pec_train_expert.py and pec_evaluate_frontier.py before any
Isaac Sim training/evaluation subprocess.  Generates exactly one USD per
unique leg_length (rounded to --leg_precision decimal places) and maintains
a cumulative morphology_registry.json in the output directory.

Output directory layout:
    <output_dir>/
        leg_<key>/
            leg_<key>.usd
        urdf/                  (intermediate URDF files)
        morphology_registry.json

On completion prints:
    USD_LEG_MAP: {"0.3000": "/abs/path/leg_0.3000/leg_0.3000.usd", ...}

Usage
-----
    python scripts/pec/pec_generate_usds.py \\
        --output_dir  logs/pec/my_run/expert_0/usds \\
        --designs_file /tmp/designs.json   # [[gcr, spcf, leg], ...]
        [--leg_precision 4]
        [--skip_existing]
"""

import argparse
import json
import os
import shutil
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
# This script lives at scripts/pec/pec_generate_usds.py.
# PROJECT_ROOT = ballu_isclb_extension/ (two levels up from scripts/pec/).
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXT_DIR      = os.path.join(PROJECT_ROOT, "source", "ballu_isaac_extension",
                             "ballu_isaac_extension")

if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)

try:
    from morphology import BalluRobotGenerator, create_morphology_variant
except Exception as exc:
    print(f"[ERROR] Failed to import morphology tools: {exc}")
    print(f"[HINT]  Ensure PYTHONPATH includes: {EXT_DIR}")
    sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────────────────────

def leg_key(leg: float, precision: int) -> str:
    """Canonical string key for a rounded leg_length, e.g. '0.3000'."""
    return f"{round(leg, precision):.{precision}f}"


def registry_file(output_dir: str) -> str:
    return os.path.join(output_dir, "morphology_registry.json")


def load_registry(output_dir: str) -> dict:
    path = registry_file(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_registry(output_dir: str, registry: dict) -> None:
    with open(registry_file(output_dir), "w") as f:
        json.dump(registry, f, indent=2)


def expected_usd_path(output_dir: str, key: str) -> str:
    morph_id = f"leg_{key}"
    return os.path.join(os.path.abspath(output_dir), morph_id, f"{morph_id}.usd")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate USD files for unique leg_lengths in 3D PEC designs."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store generated USD files and registry.")
    parser.add_argument("--designs_file", type=str, required=True,
                        help="JSON file with list of [gcr, spcf, leg] triplets.")
    parser.add_argument("--leg_precision", type=int, default=4,
                        help="Decimal places for leg_length rounding (default: 4).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip USD generation if file already exists on disk.")
    args = parser.parse_args()

    # Load designs
    with open(args.designs_file) as f:
        designs = json.load(f)

    if not designs:
        print("[WARN] No designs in file — nothing to generate.")
        print("USD_LEG_MAP: {}")
        return

    # Extract unique leg_lengths (representative value per key).
    unique_keys: dict[str, float] = {}
    for d in designs:
        leg = float(d[2])
        key = leg_key(leg, args.leg_precision)
        if key not in unique_keys:
            unique_keys[key] = leg

    print(f"\n{'='*70}")
    print(f"  PEC USD Generation")
    print(f"{'='*70}")
    print(f"  Designs:         {len(designs)}")
    print(f"  Unique leg vals: {len(unique_keys)}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Leg precision:   {args.leg_precision}")
    print(f"  Skip existing:   {args.skip_existing}")
    print(f"{'='*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    registry = load_registry(args.output_dir)

    urdf_dir = os.path.join(args.output_dir, "urdf")
    os.makedirs(urdf_dir, exist_ok=True)

    # Copy mesh STL files so that package://urdf/meshes/*.STL references resolve.
    # Isaac Sim resolves package://urdf/ to the directory containing the URDF
    # (urdf_dir, which is literally named "urdf"), so meshes must live at
    # urdf_dir/meshes/.
    _mesh_src = os.path.join(PROJECT_ROOT, "source", "ballu_isaac_extension",
                              "ballu_isaac_extension", "ballu_assets", "old",
                              "urdf", "meshes")
    _mesh_dst = os.path.join(urdf_dir, "meshes")
    if os.path.isdir(_mesh_src) and not os.path.exists(_mesh_dst):
        shutil.copytree(_mesh_src, _mesh_dst)
        print(f"  [MESH] Copied mesh assets → {_mesh_dst}")

    leg_map: dict[str, str] = {}   # key -> abs usd path
    failed: list[str] = []

    for key, leg_val in sorted(unique_keys.items()):
        morph_id = f"leg_{key}"
        exp_usd  = expected_usd_path(args.output_dir, key)

        # Reuse from registry if it exists and the file is present.
        if args.skip_existing and os.path.exists(exp_usd):
            print(f"  [SKIP] {morph_id} — USD already exists.")
            leg_map[key] = exp_usd
            if key not in registry:
                registry[key] = {"leg": leg_val, "usd_path": exp_usd,
                                  "morph_id": morph_id}
            continue

        print(f"  [GEN ] {morph_id} (leg={leg_val:.{args.leg_precision}f}) ...",
              end=" ", flush=True)
        try:
            morph = create_morphology_variant(
                morphology_id=morph_id,
                femur_length=leg_val,
                tibia_length=leg_val,
            )
            generator = BalluRobotGenerator(
                morph,
                urdf_output_dir=urdf_dir,
                usd_output_dir=args.output_dir,
            )
            urdf_path = generator.generate_urdf()
            rc, usd_path = generator.generate_usd(urdf_path)

            if rc != 0 or not os.path.exists(usd_path):
                print(f"FAILED (rc={rc})")
                failed.append(key)
                continue

            abs_usd = os.path.abspath(usd_path)
            leg_map[key] = abs_usd
            registry[key] = {"leg": leg_val, "usd_path": abs_usd, "morph_id": morph_id}
            print("OK")

        except Exception as exc:
            print(f"ERROR: {exc}")
            failed.append(key)

    save_registry(args.output_dir, registry)

    if failed:
        print(f"\n[ERROR] USD generation failed for {len(failed)} key(s): {failed}")
        sys.exit(1)

    # Parseable output for calling script.
    print(f"\nUSD_LEG_MAP: {json.dumps(leg_map)}")
    print(f"\n  {len(leg_map)} USD(s) ready.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
