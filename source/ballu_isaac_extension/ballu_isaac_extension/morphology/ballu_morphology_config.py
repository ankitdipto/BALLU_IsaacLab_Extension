"""
BALLU Morphology Configuration System

This module defines a comprehensive morphology parameter space for the BALLU robot.
It provides:
- Complete morphology parameter definitions (geometry, mass, inertial, joint, contact)
- Parameter validation and constraint checking
- Serialization/deserialization (JSON, dict)
- Default configurations and parameter ranges
- Morphology sampling utilities

Usage:
    # Create default morphology
    morph = BalluMorphology.default()
    
    # Create custom morphology
    morph = BalluMorphology(femur_length=0.4, tibia_length=0.35, ...)
    
    # Validate
    is_valid, errors = morph.validate()
    
    # Serialize
    morph.to_json("morphology.json")
    morph_dict = morph.to_dict()
    
    # Load
    morph = BalluMorphology.from_json("morphology.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Final
import math


@dataclass(frozen=True)
class LinkDensities:
    """Constant material densities (kg/m³) for BALLU links.
    
    These are calculated from the original URDF's geometry and mass values to
    preserve the physical properties of the baseline robot. The balloon's density
    is an "effective" density representing its net buoyancy.
    """
    PELVIS: Final[float] = 329.3 * (5.0 / 8.28)# 1805.5
    FEMUR: Final[float] = 329.3 * (5.0 / 8.28)
    TIBIA: Final[float] = 329.3 * (5.0 / 8.28)  # Updated to match femur density; previous was 400 
    ELECTRONICS: Final[float] = 5900.0 * (9.0 / 7.0)  # New density reported by Yusuke
    MOTORARM: Final[float] = 1666.7
    BALLOON_EFFECTIVE: Final[float] = (0.706 / 3.0) * 1.8

# Global constant for easy access
DENSITIES: Final = LinkDensities()


@dataclass
class GeometryParams:
    """Geometric parameters for BALLU morphology."""
    
    # Limb dimensions (in meters)
    femur_length: float = 0.36501  # Upper leg length
    tibia_length: float = 0.32     # Effective tibia length (to electronics attachment)
    limb_radius: float = 0.005     # Leg cylinder radius
    hip_width: float = 0.11605     # Distance between hip joints (2 * 0.058025)
    foot_radius: float = 0.004     # Contact sphere radius
    
    # Electronics dimensions (in meters)
    electronics_length: float = 0.06   # Electronics box length along Y-axis
    electronics_width: float = 0.01    # Electronics box width along X-axis
    electronics_height: float = 0.01   # Electronics box height along Z-axis
    
    # Pelvis/Body dimensions
    pelvis_height: float = 0.15    # Main body cylinder height
    pelvis_radius: float = 0.005   # Main body cylinder radius
    
    # Balloon dimensions
    balloon_radius: float = 0.32   # Buoyancy cylinder radius
    balloon_height: float = 0.7    # Buoyancy cylinder height
    neck_offset_x: float = 0.0044439  # Forward offset of balloon attachment
    neck_offset_z: float = 0.015   # Vertical offset of balloon attachment
    
    # Motorarm dimensions
    motorarm_length: float = 0.012  # Linkage arm length (box dimension)
    motorarm_width: float = 0.004   # Linkage arm width
    motorarm_height: float = 0.002  # Linkage arm height
    
    def get_total_leg_length(self) -> float:
        """Calculate total leg length (femur + tibia + electronics)."""
        return self.femur_length + self.tibia_length + self.electronics_length
    
    def get_effective_tibia_length(self) -> float:
        """Get the effective tibia length (same as tibia_length, for clarity)."""
        return self.tibia_length
    
    def get_femur_to_limb_ratio(self) -> float:
        """Calculate femur-to-total-leg ratio."""
        total_length = self.get_total_leg_length()
        return self.femur_length / total_length if total_length > 0 else 0.0

@dataclass
class VisualParams:
    """Visual parameters for BALLU morphology."""
    femur_density: float = 1 / 0.36501
    tibia_density: float = 1 / 0.38485
    pelvis_density: float = 1 / 0.15
    balloon_density: float = 1 / 0.7
    motorarm_density: float = 1 / 0.012

@dataclass
class MassParams:
    """Computed mass parameters for BALLU morphology.
    
    Note: These values are not set directly but are computed from
    geometry and constant link densities.
    """
    
    # Link masses (in kg), computed automatically
    pelvis_mass: float
    femur_mass: float      # Per leg
    tibia_mass: float      # Per leg
    electronics_mass: float # Per leg
    balloon_mass: float    # Represents net buoyancy force
    motorarm_mass: float   # Per motorarm
    
    def get_total_mass(self) -> float:
        """Calculate total robot mass."""
        return (
            self.pelvis_mass +
            2 * self.femur_mass +
            2 * self.tibia_mass +
            2 * self.electronics_mass +
            self.balloon_mass +
            2 * self.motorarm_mass
        )
    
    def get_leg_mass(self) -> float:
        """Calculate mass of one complete leg."""
        return self.femur_mass + self.tibia_mass + self.electronics_mass + self.motorarm_mass


@dataclass
class JointParams:
    """Joint configuration parameters for BALLU morphology."""
    
    # Joint limits (in radians)
    hip_lower_limit: float = -1.57079632679   # -π/2
    hip_upper_limit: float = 1.57079632679    #  π/2
    knee_lower_limit: float = 0.0
    knee_upper_limit: float = 1.7453292519943295  # 100 degrees
    neck_lower_limit: float = -1.57079632679  # -π/2
    neck_upper_limit: float = 1.57079632679   #  π/2
    motor_lower_limit: float = 0.0
    motor_upper_limit: float = 3.14159265359  # π
    
    # Joint dynamics
    hip_damping: float = 1e-2
    hip_friction: float = 1e-2
    knee_damping: float = 1e-2
    knee_friction: float = 1e-2
    neck_damping: float = 0.0
    neck_friction: float = 0.0
    motor_damping: float = 0.0
    motor_friction: float = 0.0
    
    # Initial joint positions (in radians)
    init_hip_angle: float = 0.01745  # ~1 degree
    init_knee_angle: float = 0.4773  # ~27.35 degrees
    init_motor_angle: float = 0.1745  # ~10 degrees
    init_neck_angle: float = 0.0


@dataclass
class ContactParams:
    """Contact and friction parameters for BALLU morphology."""
    
    # Friction coefficients
    foot_friction: float = 0.9      # Ground contact
    leg_friction: float = 0.9       # Leg-object contact
    pelvis_friction: float = 0.5    # Pelvis-object contact
    balloon_friction: float = 0.3   # Balloon-object contact
    
    # Contact stiffness/damping (optional, for advanced physics)
    contact_stiffness: float = 1e5
    contact_damping: float = 1e3


@dataclass
class BalluMorphology:
    """
    Complete morphology configuration for the BALLU robot.
    
    This class encapsulates all parameters that define a BALLU morphology variant,
    including geometry, mass distribution, joint configuration, and contact properties.
    Mass is automatically computed from geometry and constant material densities.
    """
    
    # Morphology identifier
    morphology_id: str = "default"
    description: str = "Default BALLU morphology from original URDF"
    
    # Parameter groups
    geometry: GeometryParams = field(default_factory=GeometryParams)
    joints: JointParams = field(default_factory=JointParams)
    contact: ContactParams = field(default_factory=ContactParams)
    visual: VisualParams = field(default_factory=VisualParams)
    
    # Computed properties (not part of constructor)
    mass: MassParams = field(init=False)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-compute mass properties after initialization."""
        self._compute_masses()
    
    def _compute_masses(self):
        """Auto-compute mass properties based on geometry and constant densities."""
        g = self.geometry
        
        # Calculate volumes based on primitive shapes used in URDF generation
        pelvis_volume = math.pi * g.pelvis_radius**2 * g.pelvis_height
        femur_volume = math.pi * g.limb_radius**2 * g.femur_length
        tibia_volume = math.pi * g.limb_radius**2 * g.tibia_length
        electronics_volume = g.electronics_width * g.electronics_length * g.electronics_height
        balloon_volume = math.pi * g.balloon_radius**2 * g.balloon_height
        motorarm_volume = g.motorarm_length * g.motorarm_width * g.motorarm_height
        
        # Compute masses and assign to a new MassParams object
        self.mass = MassParams(
            pelvis_mass=pelvis_volume * DENSITIES.PELVIS,
            femur_mass=femur_volume * DENSITIES.FEMUR,
            tibia_mass=tibia_volume * DENSITIES.TIBIA,
            electronics_mass=electronics_volume * DENSITIES.ELECTRONICS,
            balloon_mass=balloon_volume * DENSITIES.BALLOON_EFFECTIVE,
            motorarm_mass=motorarm_volume * DENSITIES.MOTORARM
        )
    
    @classmethod
    def default(cls) -> BalluMorphology:
        """Create default morphology matching the original URDF."""
        return cls(
            morphology_id="default",
            description="Default BALLU morphology from original URDF"
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BalluMorphology:
        """Create morphology from dictionary."""
        # Extract nested dataclass fields
        geometry_data = data.get("geometry", {})
        joints_data = data.get("joints", {})
        contact_data = data.get("contact", {})
        visual_data = data.get("visual", {})

        # Mass is not loaded from the dictionary; it will be recomputed.
        return cls(
            morphology_id=data.get("morphology_id", "unknown"),
            description=data.get("description", ""),
            geometry=GeometryParams(**geometry_data),
            joints=JointParams(**joints_data),
            contact=ContactParams(**contact_data),
            visual=VisualParams(**visual_data),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> BalluMorphology:
        """Load morphology from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert morphology to dictionary."""
        return asdict(self)
    
    def to_json(self, json_path: str, indent: int = 2) -> None:
        """Save morphology to JSON file."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate morphology parameters against physical constraints.
        
        Returns:
            (is_valid, error_messages): Tuple of validation result and list of error messages
        """
        errors = []
        
        # Geometry constraints
        if self.geometry.femur_length <= 0:
            errors.append("femur_length must be positive")
        if self.geometry.tibia_length <= 0:
            errors.append("tibia_length must be positive")
        if self.geometry.limb_radius <= 0:
            errors.append("limb_radius must be positive")
        if self.geometry.hip_width <= 0:
            errors.append("hip_width must be positive")
        if self.geometry.foot_radius <= 0:
            errors.append("foot_radius must be positive")
        if self.geometry.pelvis_height <= 0:
            errors.append("pelvis_height must be positive")
        if self.geometry.balloon_radius <= 0:
            errors.append("balloon_radius must be positive")
        if self.geometry.balloon_height <= 0:
            errors.append("balloon_height must be positive")
        
        # Electronics constraints
        if self.geometry.electronics_length <= 0:
            errors.append("electronics_length must be positive")
        if self.geometry.electronics_width <= 0:
            errors.append("electronics_width must be positive")
        if self.geometry.electronics_height <= 0:
            errors.append("electronics_height must be positive")
                    
        # Geometric ratios
        total_leg_length = self.geometry.get_total_leg_length()
        if total_leg_length < 0.1 or total_leg_length > 2.0:
            errors.append(f"Total leg length {total_leg_length:.3f}m is unrealistic (expected 0.1-2.0m)")
        
        if total_leg_length > 0:
            femur_ratio = self.geometry.get_femur_to_limb_ratio()
            if femur_ratio < 0.1 or femur_ratio > 0.9:
                errors.append(f"Femur-to-limb ratio {femur_ratio:.2f} is extreme (expected 0.1-0.9)")
        
        # Mass constraints (validating computed masses)
        if self.mass.pelvis_mass <= 0:
            errors.append("Computed pelvis_mass must be positive")
        if self.mass.femur_mass <= 0:
            errors.append("Computed femur_mass must be positive")
        if self.mass.tibia_mass <= 0:
            errors.append("Computed tibia_mass must be positive")
        if self.mass.electronics_mass <= 0:
            errors.append("Computed electronics_mass must be positive")
        if self.mass.balloon_mass <= 0:
            errors.append("Computed balloon_mass must be positive")
        
        # Mass ratios (balloon should be significant for buoyancy)
        total_mass = self.mass.get_total_mass()
        if total_mass > 0:
            balloon_mass_ratio = self.mass.balloon_mass / total_mass
            if balloon_mass_ratio < 0.1:
                errors.append(f"Balloon mass ratio {balloon_mass_ratio:.2f} too low (expected > 0.1 for buoyancy)")
            if balloon_mass_ratio > 0.9:
                errors.append(f"Balloon mass ratio {balloon_mass_ratio:.2f} too high (expected < 0.9)")
        else:
            errors.append("Total mass is zero or negative.")
        
        # Joint limit constraints
        if self.joints.hip_lower_limit >= self.joints.hip_upper_limit:
            errors.append("hip_lower_limit must be less than hip_upper_limit")
        if self.joints.knee_lower_limit >= self.joints.knee_upper_limit:
            errors.append("knee_lower_limit must be less than knee_upper_limit")
        if self.joints.neck_lower_limit >= self.joints.neck_upper_limit:
            errors.append("neck_lower_limit must be less than neck_upper_limit")
        if self.joints.motor_lower_limit >= self.joints.motor_upper_limit:
            errors.append("motor_lower_limit must be less than motor_upper_limit")
        
        # Initial joint positions within limits
        if not (self.joints.hip_lower_limit <= self.joints.init_hip_angle <= self.joints.hip_upper_limit):
            errors.append("init_hip_angle outside joint limits")
        if not (self.joints.knee_lower_limit <= self.joints.init_knee_angle <= self.joints.knee_upper_limit):
            errors.append("init_knee_angle outside joint limits")
        if not (self.joints.neck_lower_limit <= self.joints.init_neck_angle <= self.joints.neck_upper_limit):
            errors.append("init_neck_angle outside joint limits")
        if not (self.joints.motor_lower_limit <= self.joints.init_motor_angle <= self.joints.motor_upper_limit):
            errors.append("init_motor_angle outside joint limits")
        
        # Contact constraints
        if self.contact.foot_friction < 0 or self.contact.foot_friction > 2.0:
            errors.append(f"foot_friction {self.contact.foot_friction} outside reasonable range (0-2.0)")
        
        # Physical consistency check: initial pose should have positive ground clearance
        # Simplified check: assume legs point downward at initialization
        init_height = self.geometry.get_total_leg_length() * math.cos(self.joints.init_hip_angle)
        init_height -= self.geometry.get_total_leg_length() * (1 - math.cos(self.joints.init_knee_angle))
        if init_height < 0.1:
            errors.append(f"Initial ground clearance {init_height:.3f}m too low (may cause penetration)")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_derived_properties(self) -> Dict[str, float]:
        """
        Compute derived properties from the morphology parameters.
        
        Returns useful metrics for analysis and comparison.
        """
        total_mass = self.mass.get_total_mass()
        return {
            "total_mass": total_mass,
            "leg_mass": self.mass.get_leg_mass(),
            "total_leg_length": self.geometry.get_total_leg_length(),
            "femur_to_limb_ratio": self.geometry.get_femur_to_limb_ratio(),
            "balloon_mass_ratio": self.mass.balloon_mass / total_mass if total_mass > 0 else 0.0,
            "balloon_volume": math.pi * self.geometry.balloon_radius**2 * self.geometry.balloon_height,
            "leg_slenderness": self.geometry.get_total_leg_length() / self.geometry.limb_radius if self.geometry.limb_radius > 0 else 0.0,
            "hip_width_to_leg_ratio": self.geometry.hip_width / self.geometry.get_total_leg_length() if self.geometry.get_total_leg_length() > 0 else 0.0,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        derived = self.get_derived_properties()
        return (
            f"BalluMorphology(id={self.morphology_id})\n"
            f"  Total leg length: {derived['total_leg_length']:.4f}m "
            f"(femur: {self.geometry.femur_length:.4f}m, tibia: {self.geometry.tibia_length:.4f}m)\n"
            f"  Femur ratio: {derived['femur_to_limb_ratio']:.2f}\n"
            f"  Total mass: {derived['total_mass']:.4f}kg "
            f"(balloon: {self.mass.balloon_mass:.4f}kg, {derived['balloon_mass_ratio']*100:.1f}%)\n"
            f"  Balloon: r={self.geometry.balloon_radius:.3f}m, h={self.geometry.balloon_height:.3f}m\n"
            f"  Hip width: {self.geometry.hip_width:.4f}m\n"
        )


@dataclass 
class MorphologyParameterRanges:
    """
    Define valid parameter ranges for morphology exploration.
    
    Each parameter has (min, max, default) values. These ranges can be used
    for sampling, optimization, or constraint validation.
    """
    
    # Geometry ranges (min, max, default)
    femur_length: Tuple[float, float, float] = (0.2, 0.6, 0.36501)
    tibia_length: Tuple[float, float, float] = (0.2, 0.5, 0.32)
    limb_radius: Tuple[float, float, float] = (0.003, 0.01, 0.005)
    hip_width: Tuple[float, float, float] = (0.05, 0.20, 0.11605)
    foot_radius: Tuple[float, float, float] = (0.002, 0.01, 0.004)
    
    # Electronics ranges (min, max, default)
    electronics_length: Tuple[float, float, float] = (0.03, 0.12, 0.06)
    electronics_width: Tuple[float, float, float] = (0.005, 0.02, 0.01)
    electronics_height: Tuple[float, float, float] = (0.005, 0.02, 0.01)
    
    pelvis_height: Tuple[float, float, float] = (0.08, 0.25, 0.15)
    pelvis_radius: Tuple[float, float, float] = (0.003, 0.01, 0.005)
    
    balloon_radius: Tuple[float, float, float] = (0.2, 0.5, 0.32)
    balloon_height: Tuple[float, float, float] = (0.4, 1.2, 0.7)
    
    # Mass ranges are removed as mass is now a computed property
    
    # Joint limit ranges (min, max, default) - for lower/upper limits
    hip_range: Tuple[float, float, float] = (1.0, 2.0, 1.57079632679)  # Range of motion
    knee_range: Tuple[float, float, float] = (1.0, 2.5, 1.7453292519943295)
    neck_range: Tuple[float, float, float] = (1.0, 2.0, 1.57079632679)
    motor_range: Tuple[float, float, float] = (2.0, 3.5, 3.14159265359)
    
    # Friction ranges (min, max, default)
    foot_friction: Tuple[float, float, float] = (0.5, 1.5, 0.9)
    
    def get_ranges_dict(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all parameter ranges as a dictionary."""
        return asdict(self)
    
    def sample_uniform(self, param_name: str) -> float:
        """Sample a parameter uniformly from its range."""
        import random
        ranges = self.get_ranges_dict()
        if param_name not in ranges:
            raise ValueError(f"Unknown parameter: {param_name}")
        min_val, max_val, _ = ranges[param_name]
        return random.uniform(min_val, max_val)
    
    def get_default_morphology(self) -> BalluMorphology:
        """Create a morphology using all default values from ranges."""
        ranges = self.get_ranges_dict()
        
        return BalluMorphology(
            morphology_id="default_from_ranges",
            description="Morphology created from parameter range defaults",
            geometry=GeometryParams(
                femur_length=ranges["femur_length"][2],
                tibia_length=ranges["tibia_length"][2],
                limb_radius=ranges["limb_radius"][2],
                hip_width=ranges["hip_width"][2],
                foot_radius=ranges["foot_radius"][2],
                electronics_length=ranges["electronics_length"][2],
                electronics_width=ranges["electronics_width"][2],
                electronics_height=ranges["electronics_height"][2],
                pelvis_height=ranges["pelvis_height"][2],
                pelvis_radius=ranges["pelvis_radius"][2],
                balloon_radius=ranges["balloon_radius"][2],
                balloon_height=ranges["balloon_height"][2],
            ),
            # MassParams are not passed; they will be auto-computed.
            contact=ContactParams(
                foot_friction=ranges["foot_friction"][2],
            )
        )


# Convenience functions
def create_morphology_variant(
    morphology_id: str,
    femur_to_limb_ratio: Optional[float] = None,
    total_leg_length: Optional[float] = None,
    base_morphology: Optional[BalluMorphology] = None,
    **kwargs
) -> BalluMorphology:
    """
    Create a morphology variant by modifying specific parameters.
    
    Args:
        morphology_id: Unique identifier for this morphology
        femur_to_limb_ratio: If provided, adjust femur/tibia lengths to match ratio
        total_leg_length: If provided, scale leg to this total length
        base_morphology: Base morphology to modify (default: original URDF)
        **kwargs: Additional geometry, joint, or contact parameter overrides.
                 Mass parameters are ignored as they are computed automatically.
    
    Returns:
        Modified BalluMorphology instance
    """
    if base_morphology is None:
        morph = BalluMorphology.default()
    else:
        morph = BalluMorphology.from_dict(base_morphology.to_dict())
    
    morph.morphology_id = morphology_id
    
    # Handle femur-to-limb ratio adjustment
    if femur_to_limb_ratio is not None:
        current_total = morph.geometry.get_total_leg_length()
        if total_leg_length is not None:
            current_total = total_leg_length
        morph.geometry.femur_length = current_total * femur_to_limb_ratio
        morph.geometry.tibia_length = current_total * (1 - femur_to_limb_ratio) - morph.geometry.electronics_length
    elif total_leg_length is not None:
        # Scale both proportionally
        current_ratio = morph.geometry.get_femur_to_limb_ratio()
        morph.geometry.femur_length = total_leg_length * current_ratio
        morph.geometry.tibia_length = total_leg_length * (1 - current_ratio) - morph.geometry.electronics_length
    
    # Handle arbitrary parameter overrides for geometry, joints, and contact
    for key, value in kwargs.items():
        # Try to find and set the parameter in nested dataclasses
        if hasattr(morph.geometry, key):
            setattr(morph.geometry, key, value)
        elif hasattr(morph.joints, key):
            setattr(morph.joints, key, value)
        elif hasattr(morph.contact, key):
            setattr(morph.contact, key, value)
        elif hasattr(morph.mass, key):
            # Politely ignore any attempts to set mass directly
            print(f"Warning: Attempted to set '{key}' directly. Mass is auto-computed and this value will be ignored.")
        else:
            # Store in metadata if not a known parameter
            morph.metadata[key] = value
            
    # Re-compute masses since geometry may have changed
    morph._compute_masses()
    
    return morph


__all__ = [
    "GeometryParams",
    "MassParams",
    "JointParams",
    "ContactParams",
    "BalluMorphology",
    "MorphologyParameterRanges",
    "create_morphology_variant",
]

