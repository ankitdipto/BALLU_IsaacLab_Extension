#!/usr/bin/env python3
"""
Morphology Library Management Utility

This script provides commands to manage BALLU morphology libraries:
- list: Show all morphologies in a library
- info: Display detailed information about the library
- validate: Validate all morphologies in the library
- add: Add a new morphology to the library
- remove: Remove a morphology from the library
- export: Export library metadata
- stats: Show statistics about the library

Usage:
    python manage_morphology_library.py list --library hetero_library
    python manage_morphology_library.py info --library hetero_library
    python manage_morphology_library.py validate --library hetero_library
    python manage_morphology_library.py stats --library hetero_library
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import shutil

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
ext_dir = os.path.join(project_dir, "source", "ballu_isaac_extension", "ballu_isaac_extension")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# --- Import Tools ---
try:
    from ballu_assets.morphology_loader import (
        load_morphology_library,
        find_morphology_registry,
        get_morphology_library_path
    )
except Exception as exc:
    print(f"[ERROR] Failed to import morphology loader: {exc}")
    sys.exit(1)


def cmd_list(args):
    """List all morphologies in the library."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    print(f"\n{'='*80}")
    print(f"MORPHOLOGY LIBRARY: {args.library}")
    print(f"Path: {library_path}")
    print(f"{'='*80}\n")
    
    morphologies = load_morphology_library(library_path)
    
    if not morphologies:
        print("No morphologies found in library.")
        return 0
    
    print(f"Found {len(morphologies)} morphologies:\n")
    
    # Print table header
    print(f"{'Index':<8} {'Morphology ID':<25} {'USD Path'}")
    print(f"{'-'*8} {'-'*25} {'-'*50}")
    
    for morph in morphologies:
        idx = morph.get("index", "?")
        morph_id = morph.get("morphology_id", "unknown")
        usd_path = morph.get("usd_path", "")
        
        # Shorten path for display
        if len(usd_path) > 50:
            usd_path = "..." + usd_path[-47:]
        
        print(f"{idx:<8} {morph_id:<25} {usd_path}")
    
    print(f"\n{'='*80}\n")
    return 0


def cmd_info(args):
    """Display detailed information about the library."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    registry_path = find_morphology_registry(library_path)
    
    if not registry_path:
        print(f"[WARNING] No registry found for library '{args.library}'")
        print(f"Library path: {library_path}")
        return 1
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"MORPHOLOGY LIBRARY INFO")
    print(f"{'='*80}")
    print(f"Library Name:      {args.library}")
    print(f"Library Path:      {library_path}")
    print(f"Registry Path:     {registry_path}")
    print(f"Version:           {registry.get('version', 'unknown')}")
    print(f"Generated:         {registry.get('generated_at', 'unknown')}")
    print(f"Num Morphologies:  {registry.get('num_morphologies', 0)}")
    print(f"Sampling Strategy: {registry.get('sampling_strategy', 'unknown')}")
    print(f"Seed:              {registry.get('seed', 'unknown')}")
    print(f"{'='*80}\n")
    
    if args.verbose:
        morphologies = registry.get("morphologies", [])
        if morphologies:
            print(f"Morphology Details:\n")
            for i, morph in enumerate(morphologies[:10]):  # Show first 10
                print(f"  [{i}] {morph.get('morphology_id', 'unknown')}")
                params = morph.get('parameters', {})
                for key, value in params.items():
                    print(f"      {key}: {value:.4f}")
                print()
            
            if len(morphologies) > 10:
                print(f"  ... and {len(morphologies) - 10} more\n")
    
    return 0


def cmd_validate(args):
    """Validate all morphologies in the library."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    print(f"\n{'='*80}")
    print(f"VALIDATING MORPHOLOGY LIBRARY: {args.library}")
    print(f"{'='*80}\n")
    
    morphologies = load_morphology_library(library_path)
    
    if not morphologies:
        print("No morphologies found in library.")
        return 0
    
    print(f"Checking {len(morphologies)} morphologies...\n")
    
    valid_count = 0
    invalid_count = 0
    missing_count = 0
    
    for morph in morphologies:
        morph_id = morph.get("morphology_id", "unknown")
        usd_path = morph.get("usd_path", "")
        
        # Check if USD file exists
        if not os.path.exists(usd_path):
            print(f"✗ {morph_id}: USD file missing")
            missing_count += 1
            continue
        
        # Check if URDF exists (if specified)
        urdf_path = morph.get("urdf_path", "")
        if urdf_path and not os.path.exists(urdf_path):
            print(f"⚠ {morph_id}: URDF file missing (USD exists)")
        
        valid_count += 1
        if args.verbose:
            print(f"✓ {morph_id}")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Valid:   {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Missing: {missing_count}")
    print(f"{'='*80}\n")
    
    return 0 if invalid_count == 0 and missing_count == 0 else 1


def cmd_stats(args):
    """Show statistics about the library."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    registry_path = find_morphology_registry(library_path)
    
    if not registry_path:
        print(f"[WARNING] No registry found for library '{args.library}'")
        return 1
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    morphologies = registry.get("morphologies", [])
    
    print(f"\n{'='*80}")
    print(f"MORPHOLOGY LIBRARY STATISTICS: {args.library}")
    print(f"{'='*80}\n")
    
    # Basic stats
    print(f"Total Morphologies: {len(morphologies)}")
    
    if not morphologies:
        print("\nNo morphologies in library.")
        return 0
    
    # Parameter statistics
    print(f"\nParameter Statistics:")
    print(f"{'-'*80}")
    
    # Collect all parameters
    all_params = {}
    for morph in morphologies:
        params = morph.get("parameters", {})
        for key, value in params.items():
            if key not in all_params:
                all_params[key] = []
            all_params[key].append(value)
    
    # Compute and display stats
    import numpy as np
    for param, values in sorted(all_params.items()):
        values_arr = np.array(values)
        print(f"\n{param}:")
        print(f"  Min:    {np.min(values_arr):.6f}")
        print(f"  Max:    {np.max(values_arr):.6f}")
        print(f"  Mean:   {np.mean(values_arr):.6f}")
        print(f"  Std:    {np.std(values_arr):.6f}")
        print(f"  Median: {np.median(values_arr):.6f}")
    
    # Derived property statistics
    print(f"\n{'-'*80}")
    print(f"Derived Property Statistics:")
    print(f"{'-'*80}")
    
    all_derived = {}
    for morph in morphologies:
        derived = morph.get("derived_properties", {})
        for key, value in derived.items():
            if key not in all_derived:
                all_derived[key] = []
            all_derived[key].append(value)
    
    for prop, values in sorted(all_derived.items()):
        values_arr = np.array(values)
        print(f"\n{prop}:")
        print(f"  Min:    {np.min(values_arr):.6f}")
        print(f"  Max:    {np.max(values_arr):.6f}")
        print(f"  Mean:   {np.mean(values_arr):.6f}")
        print(f"  Std:    {np.std(values_arr):.6f}")
        print(f"  Median: {np.median(values_arr):.6f}")
    
    print(f"\n{'='*80}\n")
    return 0


def cmd_export(args):
    """Export library metadata to a file."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    registry_path = find_morphology_registry(library_path)
    
    if not registry_path:
        print(f"[ERROR] No registry found for library '{args.library}'")
        return 1
    
    # Copy registry to output file
    output_path = args.output
    shutil.copy(registry_path, output_path)
    
    print(f"✓ Exported library metadata to: {output_path}")
    return 0


def cmd_remove(args):
    """Remove a morphology from the library."""
    library_path = get_morphology_library_path(args.library)
    
    if not library_path:
        print(f"[ERROR] Library '{args.library}' not found")
        return 1
    
    registry_path = find_morphology_registry(library_path)
    
    if not registry_path:
        print(f"[ERROR] No registry found for library '{args.library}'")
        return 1
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    morphologies = registry.get("morphologies", [])
    
    # Find morphology to remove
    to_remove = None
    for morph in morphologies:
        if morph.get("morphology_id") == args.morphology_id:
            to_remove = morph
            break
    
    if not to_remove:
        print(f"[ERROR] Morphology '{args.morphology_id}' not found in library")
        return 1
    
    # Confirm removal
    if not args.force:
        response = input(f"Remove morphology '{args.morphology_id}'? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Remove files
    usd_path = to_remove.get("usd_path", "")
    urdf_path = to_remove.get("urdf_path", "")
    
    if usd_path and os.path.exists(usd_path):
        # Remove entire morphology directory
        morph_dir = os.path.dirname(usd_path)
        shutil.rmtree(morph_dir)
        print(f"✓ Removed morphology directory: {morph_dir}")
    
    # Update registry
    morphologies.remove(to_remove)
    registry["morphologies"] = morphologies
    registry["num_morphologies"] = len(morphologies)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✓ Updated registry")
    print(f"✓ Removed morphology '{args.morphology_id}'")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Manage BALLU morphology libraries',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    parser_list = subparsers.add_parser('list', help='List all morphologies in library')
    parser_list.add_argument('--library', type=str, default='hetero_library', help='Library name')
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Display library information')
    parser_info.add_argument('--library', type=str, default='hetero_library', help='Library name')
    parser_info.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Validate command
    parser_validate = subparsers.add_parser('validate', help='Validate all morphologies')
    parser_validate.add_argument('--library', type=str, default='hetero_library', help='Library name')
    parser_validate.add_argument('--verbose', '-v', action='store_true', help='Show validation details')
    
    # Stats command
    parser_stats = subparsers.add_parser('stats', help='Show library statistics')
    parser_stats.add_argument('--library', type=str, default='hetero_library', help='Library name')
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export library metadata')
    parser_export.add_argument('--library', type=str, default='hetero_library', help='Library name')
    parser_export.add_argument('--output', '-o', type=str, required=True, help='Output file path')
    
    # Remove command
    parser_remove = subparsers.add_parser('remove', help='Remove a morphology from library')
    parser_remove.add_argument('--library', type=str, default='hetero_library', help='Library name')
    parser_remove.add_argument('--morphology_id', type=str, required=True, help='Morphology ID to remove')
    parser_remove.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'list': cmd_list,
        'info': cmd_info,
        'validate': cmd_validate,
        'stats': cmd_stats,
        'export': cmd_export,
        'remove': cmd_remove,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

