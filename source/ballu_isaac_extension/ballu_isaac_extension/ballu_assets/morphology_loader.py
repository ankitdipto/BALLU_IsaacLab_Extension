"""
Dynamic Morphology Loader for Heterogeneous BALLU Training

This module provides utilities to dynamically load morphologies from a directory
instead of hardcoding paths. This enables easy scaling to 100+ morphologies.

Usage:
    from morphology_loader import load_morphology_library, create_hetero_config
    
    # Load all morphologies from a directory
    morphologies = load_morphology_library("morphologies/hetero_library")
    
    # Create heterogeneous config with loaded morphologies
    hetero_cfg = create_hetero_config(morphologies)
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, SpringPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import math


def degree_to_radian(degree):
    """Convert degrees to radians."""
    return degree * math.pi / 180.0


def find_morphology_registry(directory: str) -> Optional[str]:
    """
    Find morphology registry file in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        Path to registry file or None if not found
    """
    registry_path = os.path.join(directory, "morphology_registry.json")
    if os.path.exists(registry_path):
        return registry_path
    return None


def load_morphology_registry(registry_path: str) -> Dict:
    """
    Load morphology registry from JSON file.
    
    Args:
        registry_path: Path to registry file
        
    Returns:
        Registry dictionary
    """
    with open(registry_path, 'r') as f:
        return json.load(f)


def scan_directory_for_usds(directory: str, pattern: str = "**/*.usd") -> List[str]:
    """
    Scan directory for USD files.
    
    Args:
        directory: Directory to scan
        pattern: Glob pattern for USD files
        
    Returns:
        List of USD file paths
    """
    usd_files = []
    search_path = os.path.join(directory, pattern)
    
    for usd_path in glob.glob(search_path, recursive=True):
        if os.path.isfile(usd_path):
            usd_files.append(usd_path)
    
    return sorted(usd_files)


def load_morphology_library(
    directory: str,
    max_morphologies: Optional[int] = None,
    use_registry: bool = True,
    filter_fn: Optional[callable] = None
) -> List[Dict]:
    """
    Load morphology library from directory.
    
    Args:
        directory: Directory containing morphologies
        max_morphologies: Maximum number of morphologies to load (None = all)
        use_registry: Whether to use registry file if available
        filter_fn: Optional function to filter morphologies (takes metadata dict, returns bool)
        
    Returns:
        List of morphology metadata dictionaries with 'usd_path' key
    """
    morphologies = []
    
    # Try to load from registry first
    if use_registry:
        registry_path = find_morphology_registry(directory)
        if registry_path:
            print(f"[MorphologyLoader] Loading from registry: {registry_path}")
            registry = load_morphology_registry(registry_path)
            morphologies = registry.get("morphologies", [])
            
            # Apply filter if provided
            if filter_fn:
                morphologies = [m for m in morphologies if filter_fn(m)]
            
            # Limit number if specified
            if max_morphologies:
                morphologies = morphologies[:max_morphologies]
            
            print(f"[MorphologyLoader] Loaded {len(morphologies)} morphologies from registry")
            return morphologies
    
    # Fallback: scan directory for USD files
    print(f"[MorphologyLoader] Scanning directory for USD files: {directory}")
    usd_files = scan_directory_for_usds(directory)
    
    for i, usd_path in enumerate(usd_files):
        if max_morphologies and i >= max_morphologies:
            break
        
        # Create minimal metadata
        morphology = {
            "morphology_id": Path(usd_path).stem,
            "usd_path": usd_path,
            "index": i
        }
        
        # Apply filter if provided
        if filter_fn is None or filter_fn(morphology):
            morphologies.append(morphology)
    
    print(f"[MorphologyLoader] Found {len(morphologies)} USD files")
    return morphologies


def create_hetero_config(
    morphologies: List[Dict],
    init_pos: Tuple[float, float, float] = (0.0, 0.0, 1.4),
    init_joint_angles: Optional[Dict[str, float]] = None,
    spring_coeff: float = 0.00507,
    spring_damping: float = 1.0e-3,
    pd_p: float = 0.09,
    pd_d: float = 0.02,
    random_choice: bool = True
) -> ArticulationCfg:
    """
    Create heterogeneous ArticulationCfg from morphology list.
    
    Args:
        morphologies: List of morphology metadata dicts with 'usd_path' key
        init_pos: Initial position (x, y, z)
        init_joint_angles: Initial joint angles (None = use defaults)
        spring_coeff: Spring coefficient for knee actuators
        spring_damping: Spring damping for knee actuators
        pd_p: PD controller proportional gain
        pd_d: PD controller derivative gain
        random_choice: Whether to randomly select morphologies per environment
        
    Returns:
        ArticulationCfg configured for heterogeneous morphologies
    """
    if not morphologies:
        raise ValueError("No morphologies provided")
    
    # Extract USD paths
    usd_paths = [m["usd_path"] for m in morphologies]
    
    print(f"[MorphologyLoader] Creating hetero config with {len(usd_paths)} morphologies")
    print(f"[MorphologyLoader] Random choice: {random_choice}")
    
    # Default joint angles if not provided
    if init_joint_angles is None:
        init_joint_angles = {
            "NECK": 0.0,
            "HIP_LEFT": degree_to_radian(1),
            "HIP_RIGHT": degree_to_radian(1),
            "KNEE_LEFT": degree_to_radian(27.35),
            "KNEE_RIGHT": degree_to_radian(27.35),
            "MOTOR_LEFT": degree_to_radian(10),
            "MOTOR_RIGHT": degree_to_radian(10)
        }
    
    # Create ArticulationCfg
    cfg = ArticulationCfg(
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=random_choice,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos=init_joint_angles
        ),
        actuators={
            # Motor actuators (position control)
            "motor_actuators": ImplicitActuatorCfg(
                joint_names_expr=["MOTOR_LEFT", "MOTOR_RIGHT"],
                effort_limit_sim=1.44 * 9.81 * 1e-2 * 0.6,  # 0.1412 * 0.6 Nm
                velocity_limit_sim=degree_to_radian(60) / 0.14,  # 428.57 rad/s
                stiffness=1.0,
                damping=0.01,
            ),
            # Knee actuators (spring-PD control)
            "knee_effort_actuators": SpringPDActuatorCfg(
                joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
                effort_limit=1.44 * 9.81 * 1e-2 * 0.6,  # 0.1412 * 0.6 Nm
                velocity_limit=degree_to_radian(60) / 0.14,  # 428.57 rad/s
                spring_coeff=spring_coeff,
                spring_damping=spring_damping,
                spring_preload=degree_to_radian(180 - 135 + 27.35),
                pd_p=pd_p,
                pd_d=pd_d,
                stiffness=float("inf"),
                damping=float("inf"),
            ),
            # Passive joints
            "other_passive_joints": ImplicitActuatorCfg(
                joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT"],
                stiffness=0.0,
                damping=0.001,
            ),
        },
    )
    
    return cfg


def get_morphology_library_path(library_name: str = "hetero_library") -> Optional[str]:
    """
    Get path to morphology library by name.
    
    Searches in:
    1. Environment variable BALLU_MORPHOLOGY_LIBRARY_PATH
    2. ballu_assets/robots/morphologies/{library_name}
    
    Args:
        library_name: Name of the library directory
        
    Returns:
        Path to library directory or None if not found
    """
    # Check environment variable
    env_path = os.environ.get("BALLU_MORPHOLOGY_LIBRARY_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    
    # Check default location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "robots", "morphologies", library_name)
    if os.path.isdir(default_path):
        return default_path
    
    return None


def load_default_hetero_config(
    library_name: str = "hetero_library",
    max_morphologies: Optional[int] = None,
    **config_kwargs
) -> ArticulationCfg:
    """
    Load default heterogeneous configuration from library.
    
    Args:
        library_name: Name of the morphology library
        max_morphologies: Maximum number of morphologies to load
        **config_kwargs: Additional arguments for create_hetero_config
        
    Returns:
        ArticulationCfg for heterogeneous training
    """
    library_path = get_morphology_library_path(library_name)
    
    if library_path is None:
        raise FileNotFoundError(
            f"Morphology library '{library_name}' not found. "
            f"Set BALLU_MORPHOLOGY_LIBRARY_PATH environment variable or "
            f"generate library using generate_morphology_library.py"
        )
    
    morphologies = load_morphology_library(library_path, max_morphologies=max_morphologies)
    
    if not morphologies:
        raise ValueError(f"No morphologies found in library: {library_path}")
    
    return create_hetero_config(morphologies, **config_kwargs)


# Example filter functions
def filter_by_leg_length(min_length: float, max_length: float):
    """Create filter function for leg length range."""
    def filter_fn(morphology: Dict) -> bool:
        derived = morphology.get("derived_properties", {})
        leg_length = derived.get("total_leg_length", 0)
        return min_length <= leg_length <= max_length
    return filter_fn


def filter_by_parameter(param_name: str, min_val: float, max_val: float):
    """Create filter function for parameter range."""
    def filter_fn(morphology: Dict) -> bool:
        params = morphology.get("parameters", {})
        value = params.get(param_name, 0)
        return min_val <= value <= max_val
    return filter_fn


__all__ = [
    "load_morphology_library",
    "create_hetero_config",
    "load_default_hetero_config",
    "get_morphology_library_path",
    "filter_by_leg_length",
    "filter_by_parameter",
]

