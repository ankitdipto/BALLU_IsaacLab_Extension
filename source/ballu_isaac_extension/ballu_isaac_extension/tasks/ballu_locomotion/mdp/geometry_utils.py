# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot geometry utilities for extracting rigid body dimensions from USD at runtime.

This module provides high-level APIs to extract robot geometry dimensions in a tensorized
format suitable for RL training and observation space integration.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Union, List, Optional
import omni.usd
from pxr import UsdGeom, Gf


@dataclass
class CylinderGeometry:
    """Cylinder geometry with tensorized properties."""
    radius: torch.Tensor  # Shape: [num_envs] or [1]
    height: torch.Tensor  # Shape: [num_envs] or [1]


@dataclass
class BoxGeometry:
    """Box geometry with tensorized properties."""
    size: torch.Tensor  # Shape: [num_envs, 3] or [1, 3] - (width, height, depth)


@dataclass
class SphereGeometry:
    """Sphere geometry with tensorized properties."""
    radius: torch.Tensor  # Shape: [num_envs] or [1]

@dataclass
class ElectronicsGeometry:
    """Electronics geometry with tensorized properties."""
    cylinder: CylinderGeometry  # Electronics box (mesh_0)
    sphere: SphereGeometry  # Foot sphere (mesh_1)


@dataclass
class RobotDimensions:
    """Complete robot dimensions with separate left/right limbs."""
    pelvis: CylinderGeometry
    femur_left: CylinderGeometry
    femur_right: CylinderGeometry
    tibia_left: CylinderGeometry
    tibia_right: CylinderGeometry
    electronics_left: ElectronicsGeometry
    electronics_right: ElectronicsGeometry


class RobotGeometryExtractor:
    """Main class for extracting robot geometry from USD stage."""
    
    def __init__(self):
        """Initialize the geometry extractor."""
        self.stage = None
        self._update_stage()
    
    def _update_stage(self) -> None:
        """Update the USD stage reference."""
        context = omni.usd.get_context()
        if context is None:
            raise RuntimeError("USD context not available. Ensure Isaac Sim is running.")
        
        self.stage = context.get_stage()
        if self.stage is None:
            raise RuntimeError("USD stage not available. Ensure simulation is loaded.")
    
    def _get_env_path(self, env_idx: int) -> str:
        """Get the base path for a specific environment."""
        return f"/World/envs/env_{env_idx}/Robot"
    
    def _extract_cylinder_geometry(self, prim_path: str) -> tuple[float, float]:
        """Extract cylinder radius and height from USD prim."""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {prim_path}")
        
        geom = UsdGeom.Cylinder(prim)
        if not geom:
            raise RuntimeError(f"Prim at {prim_path} is not a valid cylinder")
        
        radius_attr = geom.GetRadiusAttr()
        height_attr = geom.GetHeightAttr()
        
        if not radius_attr or not height_attr:
            raise RuntimeError(f"Missing radius or height attributes for cylinder at {prim_path}")
        
        radius = radius_attr.Get()
        height = height_attr.Get()
        
        if radius is None or height is None:
            raise RuntimeError(f"Failed to get radius or height values for cylinder at {prim_path}")
        
        return float(radius), float(height)
    
    def _extract_box_geometry(self, prim_path: str) -> tuple[float, float, float]:
        """Extract box dimensions from USD prim."""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {prim_path}")
        
        # Try Cube first (as seen in manual_run.py)
        geom = UsdGeom.Cube(prim)
        if geom:
            extent_attr = geom.GetExtentAttr()
            if extent_attr:
                extent = extent_attr.Get()
                if extent is not None:
                    # Cube extent is half-size, so full size is 2 * extent
                    # Handle Vec3fArray by accessing individual components
                    if hasattr(extent, '__len__') and len(extent) >= 2:
                        # extent is a Vec3fArray with min/max bounds
                        # Calculate size as max - min for each dimension
                        size_x = float(extent[1][0] - extent[0][0])
                        size_y = float(extent[1][1] - extent[0][1]) 
                        size_z = float(extent[1][2] - extent[0][2])
                        return size_x, size_y, size_z
                    else:
                        # Single extent value - cube with uniform size
                        size = 2.0 * float(extent)
                        return size, size, size
        
        # Try Box if Cube fails
        geom = UsdGeom.Box(prim)
        if geom:
            size_attr = geom.GetSizeAttr()
            if size_attr:
                size = size_attr.Get()
                if size is not None:
                    return float(size[0]), float(size[1]), float(size[2])
        
        raise RuntimeError(f"Prim at {prim_path} is not a valid box or cube")
    
    def _extract_sphere_geometry(self, prim_path: str) -> float:
        """Extract sphere radius from USD prim."""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {prim_path}")
        
        geom = UsdGeom.Sphere(prim)
        if not geom:
            raise RuntimeError(f"Prim at {prim_path} is not a valid sphere")
        
        radius_attr = geom.GetRadiusAttr()
        if not radius_attr:
            raise RuntimeError(f"Missing radius attribute for sphere at {prim_path}")
        
        radius = radius_attr.Get()
        if radius is None:
            raise RuntimeError(f"Failed to get radius value for sphere at {prim_path}")
        
        return float(radius)
    
    def extract_pelvis_dimensions(self, env_indices: Union[int, List[int], slice]) -> CylinderGeometry:
        """Extract pelvis cylinder dimensions."""
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        elif isinstance(env_indices, slice):
            # Convert slice to list - assume reasonable range
            start = env_indices.start or 0
            stop = env_indices.stop or 100  # Default reasonable upper bound
            step = env_indices.step or 1
            env_indices = list(range(start, stop, step))
        
        radii = []
        heights = []
        
        for env_idx in env_indices:
            prim_path = f"{self._get_env_path(env_idx)}/PELVIS/collisions/mesh_0/cylinder"
            radius, height = self._extract_cylinder_geometry(prim_path)
            radii.append(radius)
            heights.append(height)
        
        return CylinderGeometry(
            radius=torch.tensor(radii, dtype=torch.float32),
            height=torch.tensor(heights, dtype=torch.float32)
        )
    
    def extract_femur_dimensions(self, env_indices: Union[int, List[int], slice], side: str) -> CylinderGeometry:
        """Extract femur cylinder dimensions for specified side."""
        if side not in ['LEFT', 'RIGHT']:
            raise ValueError(f"Side must be 'LEFT' or 'RIGHT', got: {side}")
        
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        elif isinstance(env_indices, slice):
            start = env_indices.start or 0
            stop = env_indices.stop or 100
            step = env_indices.step or 1
            env_indices = list(range(start, stop, step))
        
        radii = []
        heights = []
        
        for env_idx in env_indices:
            prim_path = f"{self._get_env_path(env_idx)}/FEMUR_{side}/collisions/mesh_0/cylinder"
            radius, height = self._extract_cylinder_geometry(prim_path)
            radii.append(radius)
            heights.append(height)
        
        return CylinderGeometry(
            radius=torch.tensor(radii, dtype=torch.float32),
            height=torch.tensor(heights, dtype=torch.float32)
        )
    
    def extract_tibia_dimensions(self, env_indices: Union[int, List[int], slice], side: str) -> CylinderGeometry:
        """Extract all tibia collision geometries for specified side."""
        if side not in ['LEFT', 'RIGHT']:
            raise ValueError(f"Side must be 'LEFT' or 'RIGHT', got: {side}")
        
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        elif isinstance(env_indices, slice):
            start = env_indices.start or 0
            stop = env_indices.stop or 100
            step = env_indices.step or 1
            env_indices = list(range(start, stop, step))
        
        # Cylinder (mesh_0)
        cyl_radii = []
        cyl_heights = []
        
        for env_idx in env_indices:
            base_path = f"{self._get_env_path(env_idx)}/TIBIA_{side}/collisions"
            
            # Extract cylinder
            cyl_path = f"{base_path}/mesh_0/cylinder"
            radius, height = self._extract_cylinder_geometry(cyl_path)
            cyl_radii.append(radius)
            cyl_heights.append(height)
        
        return CylinderGeometry(
                radius=torch.tensor(cyl_radii, dtype=torch.float32),
                height=torch.tensor(cyl_heights, dtype=torch.float32)
        )

    def extract_electronics_dimensions(self, env_indices: Union[int, List[int], slice], side: str) -> ElectronicsGeometry:
        """Extract electronics collision geometries (box, sphere) for specified side.

        The electronics link has two collision meshes:
        - mesh_0: box
        - mesh_1: sphere
        """
        if side not in ['LEFT', 'RIGHT']:
            raise ValueError(f"Side must be 'LEFT' or 'RIGHT', got: {side}")

        if isinstance(env_indices, int):
            env_indices = [env_indices]
        elif isinstance(env_indices, slice):
            start = env_indices.start or 0
            stop = env_indices.stop or 100
            step = env_indices.step or 1
            env_indices = list(range(start, stop, step))

        cyl_radii: List[float] = []
        cyl_heights: List[float] = []
        sphere_radii: List[float] = []

        for env_idx in env_indices:
            base_path = f"{self._get_env_path(env_idx)}/ELECTRONICS_{side}/collisions"

            # mesh_0: cylinder
            cylinder_path = f"{base_path}/mesh_0/cylinder"
            radius, height = self._extract_cylinder_geometry(cylinder_path)
            cyl_radii.append(radius)
            cyl_heights.append(height)

            # mesh_1: sphere
            sphere_path = f"{base_path}/mesh_1/sphere"
            sphere_radius = self._extract_sphere_geometry(sphere_path)
            sphere_radii.append(sphere_radius)

        return ElectronicsGeometry(
            cylinder=CylinderGeometry(
                radius=torch.tensor(cyl_radii, dtype=torch.float32), 
                height=torch.tensor(cyl_heights, dtype=torch.float32)
            ),
            sphere=SphereGeometry(radius=torch.tensor(sphere_radii, dtype=torch.float32)),
        )
    
    def extract_robot_dimensions(self, env_indices: Union[int, List[int], slice]) -> RobotDimensions:
        """Extract complete robot dimensions for all body parts."""
        return RobotDimensions(
            pelvis=self.extract_pelvis_dimensions(env_indices),
            femur_left=self.extract_femur_dimensions(env_indices, 'LEFT'),
            femur_right=self.extract_femur_dimensions(env_indices, 'RIGHT'),
            tibia_left=self.extract_tibia_dimensions(env_indices, 'LEFT'),
            tibia_right=self.extract_tibia_dimensions(env_indices, 'RIGHT'),
            electronics_left=self.extract_electronics_dimensions(env_indices, 'LEFT'),
            electronics_right=self.extract_electronics_dimensions(env_indices, 'RIGHT'),
        )


# Global extractor instance
_extractor: Optional[RobotGeometryExtractor] = None


def _get_extractor() -> RobotGeometryExtractor:
    """Get or create the global geometry extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = RobotGeometryExtractor()
    else:
        # Update stage reference in case it changed
        _extractor._update_stage()
    return _extractor


# High-level API functions
def get_robot_dimensions(env_indices: Union[int, List[int], slice]) -> RobotDimensions:
    """
    Extract complete robot dimensions for all body parts.
    
    Args:
        env_indices: Environment index, list of indices, or slice
        
    Returns:
        RobotDimensions with tensorized geometry data
        
    Raises:
        RuntimeError: If USD stage is not available or geometry extraction fails
        ValueError: If invalid environment indices provided
    """
    return _get_extractor().extract_robot_dimensions(env_indices)


def get_pelvis_dimensions(env_indices: Union[int, List[int], slice]) -> CylinderGeometry:
    """
    Extract pelvis cylinder dimensions.
    
    Args:
        env_indices: Environment index, list of indices, or slice
        
    Returns:
        CylinderGeometry with tensorized radius and height
    """
    return _get_extractor().extract_pelvis_dimensions(env_indices)


def get_femur_dimensions(env_indices: Union[int, List[int], slice], side: str) -> CylinderGeometry:
    """
    Extract femur cylinder dimensions for specified side.
    
    Args:
        env_indices: Environment index, list of indices, or slice
        side: 'LEFT' or 'RIGHT'
        
    Returns:
        CylinderGeometry with tensorized radius and height
    """
    return _get_extractor().extract_femur_dimensions(env_indices, side)


def get_tibia_dimensions(env_indices: Union[int, List[int], slice], side: str) -> CylinderGeometry:
    """
    Extract all tibia collision geometries for specified side.
    
    Args:
        env_indices: Environment index, list of indices, or slice
        side: 'LEFT' or 'RIGHT'
        
    Returns:
        CylinderGeometry with cylinder dimensions
    """
    return _get_extractor().extract_tibia_dimensions(env_indices, side)


def get_electronics_dimensions(env_indices: Union[int, List[int], slice], side: str) -> ElectronicsGeometry:
    """
    Extract electronics collision geometries for specified side.

    Args:
        env_indices: Environment index, list of indices, or slice
        side: 'LEFT' or 'RIGHT'

    Returns:
        ElectronicsGeometry with box and sphere collisions
    """
    return _get_extractor().extract_electronics_dimensions(env_indices, side)
