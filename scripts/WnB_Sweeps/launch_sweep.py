#!/usr/bin/env python3
"""
W&B Sweep Launcher for BALLU Morphology Optimization.

This script initializes and launches a W&B sweep to optimize BALLU robot morphologies
and buoyancy mass parameters for maximum cumulative reward.
"""

import argparse
import os
import sys
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any


def load_sweep_config(config_file: str) -> Dict[str, Any]:
    """
    Load sweep configuration from YAML file.
    
    Args:
        config_file: Path to the sweep configuration YAML file
        
    Returns:
        Dictionary containing the sweep configuration
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded sweep configuration from: {config_file}")
    print(f"   Method: {config.get('method', 'unknown')}")
    print(f"   Metric: {config.get('metric', {}).get('name', 'unknown')} ({config.get('metric', {}).get('goal', 'unknown')})")
    print(f"   Parameters: {len(config.get('parameters', {}))}")
    
    return config


def validate_sweep_config(config: Dict[str, Any]) -> bool:
    """
    Validate the sweep configuration.
    
    Args:
        config: Sweep configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = ['program', 'method', 'metric', 'parameters']
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required key in sweep config: {key}")
            return False
    
    # Validate metric configuration
    metric = config['metric']
    if 'name' not in metric or 'goal' not in metric:
        print("❌ Invalid metric configuration: must have 'name' and 'goal'")
        return False
    
    if metric['goal'] not in ['minimize', 'maximize']:
        print("❌ Invalid metric goal: must be 'minimize' or 'maximize'")
        return False
    
    # Validate parameters
    parameters = config['parameters']
    required_params = ['gravity_comp_ratio', 'femur_to_tibia_ratio', 'motor_limit', 'seed']
    
    for param in required_params:
        if param not in parameters:
            print(f"❌ Missing required parameter: {param}")
            return False
    
    # Validate USD file paths exist
    if 'robot_morphology' in parameters and 'values' in parameters['robot_morphology']:
        script_dir = Path(__file__).parent
        robots_dir = script_dir.parent.parent / "source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots"
        
        missing_files = []
        usd_files = parameters['robot_morphology']['values']
        
        for usd_file in usd_files:
            full_path = robots_dir / usd_file
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            print(f"❌ Missing USD files:")
            for missing_file in missing_files[:5]:  # Show first 5 missing files
                print(f"   {missing_file}")
            if len(missing_files) > 5:
                print(f"   ... and {len(missing_files) - 5} more files")
            return False
        else:
            print(f"✅ All {len(usd_files)} USD files found")
    
    print("✅ Sweep configuration is valid")
    return True


def setup_wandb_project(project_name: str, entity: str = None, 
                       description: str = None, tags: list = None) -> None:
    """
    Setup W&B project with proper configuration.
    
    Args:
        project_name: Name of the W&B project
        entity: W&B entity (username or team)
        description: Project description
        tags: List of tags for the project
    """
    try:
        # Initialize a temporary run to set up project metadata
        with wandb.init(project=project_name, entity=entity, job_type="setup") as run:
            # Set project description and tags
            if description:
                run.notes = description
            if tags:
                run.tags = tags
            
            # Log some metadata about the sweep setup
            run.log({
                'setup_status': 'complete',
                'sweep_type': 'morphology_optimization',
                'optimization_target': 'max_mean_reward'
            })
        
        print(f"✅ W&B project '{project_name}' setup complete")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not setup W&B project metadata: {e}")


def create_sweep(config: Dict[str, Any], project_name: str, 
                entity: str = None) -> str:
    """
    Create a W&B sweep with the given configuration.
    
    Args:
        config: Sweep configuration dictionary
        project_name: Name of the W&B project
        entity: W&B entity (username or team)
        
    Returns:
        Sweep ID string
    """
    try:
        sweep_id = wandb.sweep(sweep=config, project=project_name, entity=entity)
        print(f"✅ Created sweep with ID: {sweep_id}")
        return sweep_id
    except Exception as e:
        print(f"❌ Failed to create sweep: {e}")
        raise


def print_sweep_info(sweep_id: str, project_name: str, entity: str = None):
    """
    Print information about the created sweep.
    
    Args:
        sweep_id: W&B sweep ID
        project_name: Name of the W&B project
        entity: W&B entity (username or team)
    """
    entity_str = f"{entity}/" if entity else ""
    
    print(f"\n🎯 Sweep Created Successfully!")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   Project: {project_name}")
    if entity:
        print(f"   Entity: {entity}")
    
    print(f"\n🔗 Sweep URL: https://wandb.ai/{entity_str}{project_name}/sweeps/{sweep_id}")
    
    print(f"\n🚀 To start sweep agents, run:")
    print(f"   wandb agent {entity_str}{project_name}/{sweep_id}")
    
    print(f"\n📊 To start multiple agents in parallel:")
    print(f"   # Terminal 1:")
    print(f"   wandb agent {entity_str}{project_name}/{sweep_id}")
    print(f"   # Terminal 2:")
    print(f"   wandb agent {entity_str}{project_name}/{sweep_id}")
    print(f"   # ... (start as many as your hardware can handle)")


def main():
    """Main function for launching W&B sweep."""
    parser = argparse.ArgumentParser(description="Launch W&B Sweep for BALLU Morphology Optimization")
    parser.add_argument("--config", type=str, default="sweep_config.yaml",
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--project", type=str, default="Ballu-Morphology-Sweep",
                       help="W&B project name")
    parser.add_argument("--entity", type=str, default="ankitdipto",
                       help="W&B entity (username or team)")
    parser.add_argument("--description", type=str, 
                       default="BALLU robot morphology optimization using W&B Sweeps",
                       help="Project description")
    parser.add_argument("--tags", nargs="*", 
                       default=["morphology", "optimization", "ballu", "robotics"],
                       help="Tags for the W&B project")
    parser.add_argument("--start-agent", action="store_true",
                       help="Automatically start a sweep agent after creating the sweep")
    parser.add_argument("--count", type=int, default=None,
                       help="Number of runs to execute (if starting agent)")
    parser.add_argument("--programmatic", action="store_true",
                       help="Start agents programmatically using wandb.agent() instead of CLI")
    
    args = parser.parse_args()
    
    # Get absolute path to config file
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"🎯 BALLU Morphology Optimization - W&B Sweep Launcher")
    print(f"📁 Config file: {config_path}")
    print(f"📊 Project: {args.project}")
    if args.entity:
        print(f"👤 Entity: {args.entity}")
    
    # Load and validate sweep configuration
    config = load_sweep_config(config_path)
    if not validate_sweep_config(config):
        print("❌ Invalid sweep configuration")
        sys.exit(1)
    
    # Setup W&B project
    print(f"\n🔧 Setting up W&B project...")
    setup_wandb_project(args.project, args.entity, args.description, args.tags)
    
    # Create the sweep
    print(f"\n🚀 Creating W&B sweep...")
    sweep_id = create_sweep(config, args.project, args.entity)
    
    # Print sweep information
    print_sweep_info(sweep_id, args.project, args.entity)
    
    # Optionally start an agent
    if args.start_agent:
        if args.programmatic:
            print(f"\n🤖 Starting sweep agent programmatically...")
            print(f"   Note: Using sequential execution (single GPU)")
            
            # Simple programmatic execution using wandb.agent
            entity_str = f"{args.entity}/" if args.entity else ""
            sweep_path = f"{entity_str}{args.project}/{sweep_id}"
            count = args.count or 1
            
            print(f"   Running {count} experiments sequentially...")
            
            try:
                wandb.agent(
                    sweep_id=sweep_path,
                    count=count,
                    project=args.project,
                    entity=args.entity
                )
                print(f"✅ Completed {count} experiments")
            except Exception as e:
                print(f"❌ Error running programmatic agents: {e}")
        else:
            print(f"\n🤖 Starting sweep agent via CLI...")
            entity_str = f"{args.entity}/" if args.entity else ""
            agent_cmd = f"wandb agent {entity_str}{args.project}/{sweep_id}"
            
            if args.count:
                agent_cmd += f" --count {args.count}"
            
            print(f"   Command: {agent_cmd}")
            os.system(agent_cmd)
    else:
        entity_str = f"{args.entity}/" if args.entity else ""
        print(f"\n💡 To start optimization:")
        print(f"   CLI approach: wandb agent {entity_str}{args.project}/{sweep_id}")
        print(f"   Programmatic: python launch_sweep.py --start-agent --programmatic --count 5")
        print(f"   Note: Sequential execution optimized for single GPU")


if __name__ == "__main__":
    main()