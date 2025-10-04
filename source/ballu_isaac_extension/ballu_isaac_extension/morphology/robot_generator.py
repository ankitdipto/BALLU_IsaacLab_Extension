"""
BALLU URDF Generator

Generates complete URDF files from BalluMorphology configurations.

Features:
- Auto-computation of joint positions from geometry
- Auto-computation of inertia tensors from mass and geometry
- Support for all BALLU links and joints
- Configurable visual mesh paths
- Material/color definitions

Usage:
    from ballu_morphology_config import BalluMorphology
    from urdf_generator import BalluURDFGenerator
    
    morph = BalluMorphology.default()
    generator = BalluURDFGenerator(morph)
    urdf_path = generator.generate_urdf("output_path.urdf")
"""

from __future__ import annotations

import os
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Tuple, Optional
from dataclasses import dataclass

# Handle both relative and absolute imports
try:
    from .ballu_morphology_config import BalluMorphology
except ImportError:
    from ballu_morphology_config import BalluMorphology

BALLU_ASSETS_DIR = "/home/asinha389/shared/BALLU_Project/ballu_isclb_extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets"

@dataclass
class InertiaProperties:
    """Inertia tensor and center of mass properties."""
    mass: float
    com_x: float
    com_y: float
    com_z: float
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


class InertiaCalculator:
    """
    Calculate inertia tensors for common geometric shapes.
    
    All calculations assume uniform density and are computed at the 
    center of mass of each shape.
    """
    
    @staticmethod
    def cylinder_y_axis(mass: float, radius: float, length: float, com_y: float = 0.0) -> InertiaProperties:
        """
        Calculate inertia for a cylinder oriented along Y-axis.
        
        The cylinder extends from y=0 to y=length, with COM at y=com_y.
        If com_y=0, COM is assumed at geometric center (length/2).
        
        Args:
            mass: Mass in kg
            radius: Cylinder radius in m
            length: Cylinder length along Y in m
            com_y: Y-coordinate of COM (default: length/2)
        
        Returns:
            InertiaProperties with inertia tensor at COM
        """
        if com_y == 0.0:
            com_y = length / 2.0
        
        # Inertia formulas for cylinder along Y-axis at COM
        # I_yy = (1/2) * m * r^2  (rotation about cylinder axis)
        # I_xx = I_zz = (1/12) * m * (3*r^2 + h^2)  (rotation perpendicular to axis)
        iyy = 0.5 * mass * radius**2
        ixx = izz = (1.0/12.0) * mass * (3 * radius**2 + length**2)
        
        return InertiaProperties(
            mass=mass,
            com_x=0.0,
            com_y=com_y,
            com_z=0.0,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            ixy=0.0,
            ixz=0.0,
            iyz=0.0
        )
    
    @staticmethod
    def sphere(mass: float, radius: float) -> InertiaProperties:
        """
        Calculate inertia for a sphere.
        
        Args:
            mass: Mass in kg
            radius: Sphere radius in m
        
        Returns:
            InertiaProperties with inertia tensor at COM (center)
        """
        # I = (2/5) * m * r^2 for all axes (sphere is symmetric)
        inertia = (2.0/5.0) * mass * radius**2
        
        return InertiaProperties(
            mass=mass,
            com_x=0.0,
            com_y=0.0,
            com_z=0.0,
            ixx=inertia,
            iyy=inertia,
            izz=inertia,
        )
    
    @staticmethod
    def box(mass: float, x_size: float, y_size: float, z_size: float) -> InertiaProperties:
        """
        Calculate inertia for a box centered at origin.
        
        Args:
            mass: Mass in kg
            x_size: Size along X-axis in m
            y_size: Size along Y-axis in m
            z_size: Size along Z-axis in m
        
        Returns:
            InertiaProperties with inertia tensor at COM (center)
        """
        # I_xx = (1/12) * m * (y^2 + z^2), etc.
        ixx = (1.0/12.0) * mass * (y_size**2 + z_size**2)
        iyy = (1.0/12.0) * mass * (x_size**2 + z_size**2)
        izz = (1.0/12.0) * mass * (x_size**2 + y_size**2)
        
        return InertiaProperties(
            mass=mass,
            com_x=0.0,
            com_y=0.0,
            com_z=0.0,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
        )


class BalluRobotGenerator:
    """
    Generate complete URDF and USD files from BalluMorphology configurations.
    
    This class computes all geometric properties (joint positions, collision
    geometries) and physical properties (inertia tensors, centers of mass)
    from a morphology configuration.
    """
    
    def __init__(
        self,
        morphology: BalluMorphology,
        mesh_package: str = "package://urdf/meshes",
        use_visual_meshes: bool = True,
        urdf_output_dir: str = os.path.join(BALLU_ASSETS_DIR, "old", "urdf", "urdf", "morphologies"),
        usd_output_dir: str = os.path.join(BALLU_ASSETS_DIR, "robots", "morphologies")
    ):
        """
        Initialize robot generator.
        
        Args:
            morphology: BalluMorphology configuration
            mesh_package: ROS package path for visual meshes
            use_visual_meshes: If True, use STL meshes for visuals; if False, use primitive shapes
            urdf_output_dir: Directory to save the URDF file
            usd_output_dir: Directory to save the USD file
        """
        self.morph = morphology
        self.mesh_package = mesh_package
        self.use_visual_meshes = use_visual_meshes
        self.urdf_output_dir = urdf_output_dir
        self.usd_output_dir = usd_output_dir
        
        # Validate morphology
        is_valid, errors = self.morph.validate()
        if not is_valid:
            raise ValueError(f"Invalid morphology: {errors}")
        
        # Pre-compute derived geometry
        self._compute_joint_positions()
        self._compute_inertial_properties()
    
    def _compute_joint_positions(self):
        """Pre-compute all joint origin positions from geometry."""
        g = self.morph.geometry  # shorthand
        
        # Hip joints: offset from pelvis center along Z-axis
        self.hip_left_origin = (0.0, g.hip_width/2.0, 0.0)
        self.hip_right_origin = (0.0, -g.hip_width/2.0, 0.0)
        
        # Knee joints: at end of femur along Y-axis (in femur frame)
        self.knee_origin = (0.0, g.femur_length, 0.0)
        
        # Motor joints: positioned along tibia
        # Place at ~85% of tibia length (matches original design)
        motor_y_pos = 0.85 * g.tibia_length
        motor_x_offset = -0.008375  # Original X offset
        motor_z_left = 0.001   # Original Z offset for left
        motor_z_right = -0.001  # Original Z offset for right
        
        self.motor_left_origin = (motor_x_offset, motor_y_pos, motor_z_left)
        self.motor_right_origin = (motor_x_offset, motor_y_pos, motor_z_right)
        
        # Neck joint: offset from pelvis center
        self.neck_origin = (g.neck_offset_x, 0.0, g.neck_offset_z)
        
        # Collision geometry centers (in link local frame)
        # Femur collision: centered at middle of femur
        self.femur_collision_center = (0.0, g.femur_length/2.0, 0.0)
        
        # Tibia collision: centered at middle of tibia
        self.tibia_collision_center = (0.0, g.tibia_length/2.0, 0.0)
        
        # Foot collision: at end of tibia
        self.foot_position = (0.0, g.tibia_length, 0.0)
        
        # Balloon collision: offset downward from pelvis
        self.balloon_collision_center = (0.0, -0.38, 0.0)  # Original offset
        
        # Pelvis collision: centered at origin
        self.pelvis_collision_center = (0.0, 0.0, 0.0)
    
    def _compute_inertial_properties(self):
        """Compute inertia tensors for all links."""
        g = self.morph.geometry
        m = self.morph.mass
        
        calc = InertiaCalculator()
        
        # Pelvis: cylinder along Y-axis
        self.pelvis_inertia = calc.cylinder_y_axis(
            mass=m.pelvis_mass,
            radius=g.pelvis_radius,
            length=g.pelvis_height,
        )
        
        # Femur: cylinder along Y-axis
        self.femur_inertia = calc.cylinder_y_axis(
            mass=m.femur_mass,
            radius=g.limb_radius,
            length=g.femur_length,
        )
        
        # Tibia: cylinder along Y-axis (includes foot mass for simplicity)
        # In reality, foot is separate, but we approximate here
        self.tibia_inertia = calc.cylinder_y_axis(
            mass=m.tibia_mass,
            radius=g.limb_radius,
            length=g.tibia_length,
        )
        
        # Balloon: large cylinder along Y-axis (oriented downward)
        self.balloon_inertia = calc.cylinder_y_axis(
            mass=m.balloon_mass,
            radius=g.balloon_radius,
            length=g.balloon_height,
            com_y=-0.38  # Positioned below pelvis
        )
        
        # Motorarm: small box
        self.motorarm_inertia = calc.box(
            mass=m.motorarm_mass,
            x_size=g.motorarm_width,
            y_size=g.motorarm_length,
            z_size=g.motorarm_height,
        )
    
    def _format_origin(self, xyz: Tuple[float, float, float], rpy: Tuple[float, float, float] = (0, 0, 0)) -> str:
        """Format origin as 'x y z' string."""
        return f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
    
    def _format_rpy(self, rpy: Tuple[float, float, float]) -> str:
        """Format roll-pitch-yaw as 'r p y' string."""
        return f"{rpy[0]:.11f} {rpy[1]:.11f} {rpy[2]:.11f}"
    
    def _create_inertial_element(self, parent: ET.Element, inertia: InertiaProperties):
        """Add inertial element to parent."""
        inertial = ET.SubElement(parent, "inertial")
        
        # Origin (COM position)
        origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", self._format_origin((inertia.com_x, inertia.com_y, inertia.com_z)))
        origin.set("rpy", "0 0 0")
        
        # Mass
        mass = ET.SubElement(inertial, "mass")
        mass.set("value", f"{inertia.mass:.6f}")
        
        # Inertia tensor
        inertia_elem = ET.SubElement(inertial, "inertia")
        inertia_elem.set("ixx", f"{inertia.ixx:.6e}")
        inertia_elem.set("iyy", f"{inertia.iyy:.6e}")
        inertia_elem.set("izz", f"{inertia.izz:.6e}")
        inertia_elem.set("ixy", f"{inertia.ixy:.6e}")
        inertia_elem.set("ixz", f"{inertia.ixz:.6e}")
        inertia_elem.set("iyz", f"{inertia.iyz:.6e}")
    
    def _create_visual_mesh(self, parent: ET.Element, mesh_name: str, material_name: str):
        """Add visual element with mesh geometry."""
        visual = ET.SubElement(parent, "visual")
        
        origin = ET.SubElement(visual, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", f"{self.mesh_package}/{mesh_name}.STL")
        
        material = ET.SubElement(visual, "material")
        material.set("name", material_name)
    
    def _create_visual_primitive(
        self, 
        parent: ET.Element, 
        shape: str,  # "cylinder", "sphere", "box"
        material_name: str,
        origin: Tuple[float, float, float] = (0, 0, 0),
        rpy: Tuple[float, float, float] = (0, 0, 0),
        **shape_params
    ):
        """Add visual element with primitive geometry."""
        visual = ET.SubElement(parent, "visual")
        
        origin_elem = ET.SubElement(visual, "origin")
        origin_elem.set("xyz", self._format_origin(origin))
        origin_elem.set("rpy", self._format_rpy(rpy))
        
        geometry = ET.SubElement(visual, "geometry")
        
        if shape == "cylinder":
            cylinder = ET.SubElement(geometry, "cylinder")
            cylinder.set("radius", f"{shape_params['radius']:.6f}")
            cylinder.set("length", f"{shape_params['length']:.6f}")
        elif shape == "sphere":
            sphere = ET.SubElement(geometry, "sphere")
            sphere.set("radius", f"{shape_params['radius']:.6f}")
        elif shape == "box":
            box = ET.SubElement(geometry, "box")
            box.set("size", f"{shape_params['x']} {shape_params['y']} {shape_params['z']}")
        
        material = ET.SubElement(visual, "material")
        material.set("name", material_name)
    
    def _create_collision_cylinder(
        self,
        parent: ET.Element,
        radius: float,
        length: float,
        origin: Tuple[float, float, float],
        rpy: Tuple[float, float, float] = (math.pi/2, 0, 0),
        friction: Optional[float] = None
    ):
        """Add collision element with cylinder geometry."""
        collision = ET.SubElement(parent, "collision")
        
        origin_elem = ET.SubElement(collision, "origin")
        origin_elem.set("xyz", self._format_origin(origin))
        origin_elem.set("rpy", self._format_rpy(rpy))
        
        geometry = ET.SubElement(collision, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        cylinder.set("radius", f"{radius:.6f}")
        cylinder.set("length", f"{length:.6f}")
        
        if friction is not None:
            contact = ET.SubElement(collision, "contact_coefficients")
            contact.set("mu", f"{friction:.2f}")
    
    def _create_collision_sphere(
        self,
        parent: ET.Element,
        radius: float,
        origin: Tuple[float, float, float],
        friction: Optional[float] = None
    ):
        """Add collision element with sphere geometry."""
        collision = ET.SubElement(parent, "collision")
        
        origin_elem = ET.SubElement(collision, "origin")
        origin_elem.set("xyz", self._format_origin(origin))
        origin_elem.set("rpy", "0 0 0")
        
        geometry = ET.SubElement(collision, "geometry")
        sphere = ET.SubElement(geometry, "sphere")
        sphere.set("radius", f"{radius:.6f}")
        
        if friction is not None:
            contact = ET.SubElement(collision, "contact_coefficients")
            contact.set("mu", f"{friction:.2f}")
    
    def _create_collision_box(
        self,
        parent: ET.Element,
        x_size: float,
        y_size: float,
        z_size: float,
        origin: Tuple[float, float, float] = (0, 0, 0),
        rpy: Tuple[float, float, float] = (0, 0, 0),
    ):
        """Add collision element with box geometry."""
        collision = ET.SubElement(parent, "collision")
        
        origin_elem = ET.SubElement(collision, "origin")
        origin_elem.set("xyz", self._format_origin(origin))
        origin_elem.set("rpy", self._format_rpy(rpy))
        
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        box.set("size", f"{x_size:.6f} {y_size:.6f} {z_size:.6f}")
    
    def generate_urdf(self) -> str:
        """
        Generate complete URDF file.
        
        Args:
            output_path: Path where URDF file will be written
        
        Returns:
            Absolute path to generated URDF file
        """
        # Create root element
        robot = ET.Element("robot")
        robot.set("name", "ballu")
        
        # Add ASCII art header (optional, for style!)
        header_comment = ET.Comment("""
┏━━━━━┓┏━━━━━━┳━┓    ┏━┓    ┏━┓  ┏━┓
┃ ┏━┓ ┃┃ ┏━━┓ ┃ ┃    ┃ ┃    ┃ ┃  ┃ ┃
┃ ┗━┛ ┗┫ ┃  ┃ ┃ ┃    ┃ ┃    ┃ ┃  ┃ ┃
┃ ┏━━┓ ┃ ┗━━┛ ┃ ┃  ┏━┫ ┃  ┏━┫ ┃  ┃ ┃
┃ ┗━━┛ ┃ ┏━━┓ ┃ ┗━━┛ ┃ ┗━━┛ ┃ ┗━━┛ ┃
┗━━━━━━┻━┛  ┗━┻━━━━━━┻━━━━━━┻━━━━━━┛

Generated from BalluMorphology: {0}
Description: {1}
""".format(self.morph.morphology_id, self.morph.description))
        robot.insert(0, header_comment)
        
        # Add comment sections
        link_comment = ET.Comment(" LINK DEFINITIONS ")
        robot.append(link_comment)
        
        # Create links
        self._create_base_link(robot)
        self._create_pelvis_link(robot)
        self._create_femur_links(robot)
        self._create_tibia_links(robot)
        self._create_motorarm_links(robot)
        self._create_balloon_link(robot)
        
        # Add joint section comment
        joint_comment = ET.Comment(" JOINT DEFINITIONS ")
        robot.append(joint_comment)
        
        # Create joints
        self._create_hip_joints(robot)
        self._create_knee_joints(robot)
        self._create_motor_joints(robot)
        self._create_neck_joint(robot)
        
        # Add material definitions
        material_comment = ET.Comment(" MATERIAL DEFINITIONS ")
        robot.append(material_comment)
        self._create_materials(robot)
        
        # Convert to pretty XML
        xml_str = self._prettify_xml(robot)
        
        # Write to file
        output_path = os.path.join(self.urdf_output_dir, f"{self.morph.morphology_id}.urdf")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(xml_str)
        
        print(f"Generated URDF: {output_path}")
        print(f"  Morphology ID: {self.morph.morphology_id}")
        print(f"  Total leg length: {self.morph.geometry.get_total_leg_length():.4f}m")
        print(f"  Femur ratio: {self.morph.geometry.get_femur_to_limb_ratio():.2f}")
        print(f"  Total mass: {self.morph.mass.get_total_mass():.4f}kg")
        
        return output_path
    
    def _create_base_link(self, robot: ET.Element):
        """Create base_link (empty, for kinematic tree root)."""
        link = ET.SubElement(robot, "link")
        link.set("name", "base_link")
    
    def _create_pelvis_link(self, robot: ET.Element):
        """Create PELVIS link."""
        link = ET.SubElement(robot, "link")
        link.set("name", "PELVIS")
        
        # Inertial
        self._create_inertial_element(link, self.pelvis_inertia)
        
        # Visual
        if self.use_visual_meshes:
            self._create_visual_mesh(link, "PELVIS", "color_pelvis")
        else:
            self._create_visual_primitive(
                link, "cylinder", "color_pelvis",
                origin=self.pelvis_collision_center,
                rpy=(math.pi/2, 0, 0),
                radius=self.morph.geometry.pelvis_radius,
                length=self.morph.geometry.pelvis_height
            )
        
        # Collision
        self._create_collision_cylinder(
            link,
            radius=self.morph.geometry.pelvis_radius,
            length=self.morph.geometry.pelvis_height,
            origin=self.pelvis_collision_center,
            friction=self.morph.contact.pelvis_friction
        )
    
    def _create_femur_links(self, robot: ET.Element):
        """Create FEMUR_LEFT and FEMUR_RIGHT links."""
        for side in ["LEFT", "RIGHT"]:
            link = ET.SubElement(robot, "link")
            link.set("name", f"FEMUR_{side}")
            
            # Inertial
            self._create_inertial_element(link, self.femur_inertia)
            
            # Visual
            if self.use_visual_meshes:
                material = "color_femur_left" if side == "LEFT" else "color_femur_right"
                self._create_visual_mesh(link, f"FEMUR_{side}", material)
            else:
                material = "color_femur_left" if side == "LEFT" else "color_femur_right"
                self._create_visual_primitive(
                    link, "cylinder", material,
                    origin=self.femur_collision_center,
                    rpy=(math.pi/2, 0, 0),
                    radius=self.morph.geometry.limb_radius,
                    length=self.morph.geometry.femur_length
                )
            
            # Collision
            self._create_collision_cylinder(
                link,
                radius=self.morph.geometry.limb_radius,
                length=self.morph.geometry.femur_length,
                origin=self.femur_collision_center,
                friction=self.morph.contact.leg_friction
            )
    
    def _create_tibia_links(self, robot: ET.Element):
        """Create TIBIA_LEFT and TIBIA_RIGHT links (with integrated feet)."""
        for side in ["LEFT", "RIGHT"]:
            link = ET.SubElement(robot, "link")
            link.set("name", f"TIBIA_{side}")
            
            # Inertial
            self._create_inertial_element(link, self.tibia_inertia)
            
            # Visual
            if self.use_visual_meshes:
                material = "color_tibia_left" if side == "LEFT" else "color_tibia_right"
                self._create_visual_mesh(link, f"TIBIA_{side}", material)
            else:
                material = "color_tibia_left" if side == "LEFT" else "color_tibia_right"
                self._create_visual_primitive(
                    link, "cylinder", material,
                    origin=self.tibia_collision_center,
                    rpy=(math.pi/2, 0, 0),
                    radius=self.morph.geometry.limb_radius,
                    length=self.morph.geometry.tibia_length
                )
            
            # Collision 1: Tibia cylinder
            self._create_collision_cylinder(
                link,
                radius=self.morph.geometry.limb_radius,
                length=self.morph.geometry.tibia_length,
                origin=self.tibia_collision_center,
                friction=self.morph.contact.leg_friction
            )
            
            # Collision 2: Foot sphere
            self._create_collision_sphere(
                link,
                radius=self.morph.geometry.foot_radius,
                origin=self.foot_position,
                friction=self.morph.contact.foot_friction
            )
    
    def _create_motorarm_links(self, robot: ET.Element):
        """Create MOTORARM_LEFT and MOTORARM_RIGHT links."""
        for side in ["LEFT", "RIGHT"]:
            link = ET.SubElement(robot, "link")
            link.set("name", f"MOTORARM_{side}")
            
            # Inertial
            self._create_inertial_element(link, self.motorarm_inertia)
            
            # Visual
            if self.use_visual_meshes:
                self._create_visual_mesh(link, f"MOTORARM_{side}", "color_motorarm")
            else:
                self._create_visual_primitive(
                    link, "box", "color_motorarm",
                    origin=(0, 0, 0),
                    rpy=(0, 0, math.pi/2),
                    x=self.morph.geometry.motorarm_width,
                    y=self.morph.geometry.motorarm_length,
                    z=self.morph.geometry.motorarm_height
                )
            
            # Collision
            self._create_collision_box(
                link,
                x_size=self.morph.geometry.motorarm_width,
                y_size=self.morph.geometry.motorarm_length,
                z_size=self.morph.geometry.motorarm_height,
                origin=(0, 0, 0),
                rpy=(0, 0, math.pi/2)
            )
    
    def _create_balloon_link(self, robot: ET.Element):
        """Create BALLOON link."""
        link = ET.SubElement(robot, "link")
        link.set("name", "BALLOON")
        
        # Inertial
        self._create_inertial_element(link, self.balloon_inertia)
        
        # Visual
        if self.use_visual_meshes:
            self._create_visual_mesh(link, "BALLOON", "color_balloons")
        else:
            self._create_visual_primitive(
                link, "cylinder", "color_balloons",
                origin=self.balloon_collision_center,
                rpy=(math.pi/2, 0, 0),
                radius=self.morph.geometry.balloon_radius,
                length=self.morph.geometry.balloon_height
            )
        
        # Collision
        self._create_collision_cylinder(
            link,
            radius=self.morph.geometry.balloon_radius,
            length=self.morph.geometry.balloon_height,
            origin=self.balloon_collision_center,
            friction=self.morph.contact.balloon_friction
        )
    
    def _create_hip_joints(self, robot: ET.Element):
        """Create HIP_LEFT and HIP_RIGHT joints."""
        j = self.morph.joints
        
        for side, origin in [("LEFT", self.hip_left_origin), ("RIGHT", self.hip_right_origin)]:
            joint = ET.SubElement(robot, "joint")
            joint.set("name", f"HIP_{side}")
            joint.set("type", "revolute")
            
            origin_elem = ET.SubElement(joint, "origin")
            origin_elem.set("xyz", self._format_origin(origin))
            origin_elem.set("rpy", self._format_rpy((-math.pi/2, 0, 0)))
            
            parent = ET.SubElement(joint, "parent")
            parent.set("link", "PELVIS")
            
            child = ET.SubElement(joint, "child")
            child.set("link", f"FEMUR_{side}")
            
            axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1")
            
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{j.hip_lower_limit:.11f}")
            limit.set("upper", f"{j.hip_upper_limit:.11f}")
            limit.set("effort", "0")
            limit.set("velocity", "0")
            
            dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", f"{j.hip_damping:.2e}")
            dynamics.set("friction", f"{j.hip_friction:.2e}")
    
    def _create_knee_joints(self, robot: ET.Element):
        """Create KNEE_LEFT and KNEE_RIGHT joints."""
        j = self.morph.joints
        
        for side in ["LEFT", "RIGHT"]:
            joint = ET.SubElement(robot, "joint")
            joint.set("name", f"KNEE_{side}")
            joint.set("type", "revolute")
            
            origin_elem = ET.SubElement(joint, "origin")
            origin_elem.set("xyz", self._format_origin(self.knee_origin))
            origin_elem.set("rpy", "0 0 0")
            
            parent = ET.SubElement(joint, "parent")
            parent.set("link", f"FEMUR_{side}")
            
            child = ET.SubElement(joint, "child")
            child.set("link", f"TIBIA_{side}")
            
            axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1")
            
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{j.knee_lower_limit:.11f}")
            limit.set("upper", f"{j.knee_upper_limit:.11f}")
            limit.set("effort", "0")
            limit.set("velocity", "0")
            
            dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", f"{j.knee_damping:.2e}")
            dynamics.set("friction", f"{j.knee_friction:.2e}")
    
    def _create_motor_joints(self, robot: ET.Element):
        """Create MOTOR_LEFT and MOTOR_RIGHT joints."""
        j = self.morph.joints
        
        # Different RPY for left vs right due to mirroring
        rpy_left = (-math.pi, 0, -math.pi/2)
        rpy_right = (math.pi, 0, -math.pi/2)
        
        for side, origin, rpy in [
            ("LEFT", self.motor_left_origin, rpy_left),
            ("RIGHT", self.motor_right_origin, rpy_right)
        ]:
            joint = ET.SubElement(robot, "joint")
            joint.set("name", f"MOTOR_{side}")
            joint.set("type", "revolute")
            
            origin_elem = ET.SubElement(joint, "origin")
            origin_elem.set("xyz", self._format_origin(origin))
            origin_elem.set("rpy", self._format_rpy(rpy))
            
            parent = ET.SubElement(joint, "parent")
            parent.set("link", f"TIBIA_{side}")
            
            child = ET.SubElement(joint, "child")
            child.set("link", f"MOTORARM_{side}")
            
            axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1")
            
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", f"{j.motor_lower_limit:.11f}")
            limit.set("upper", f"{j.motor_upper_limit:.11f}")
            limit.set("effort", "0")
            limit.set("velocity", "0")
            
            dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("damping", f"{j.motor_damping:.2e}")
            dynamics.set("friction", f"{j.motor_friction:.2e}")
    
    def _create_neck_joint(self, robot: ET.Element):
        """Create NECK joint."""
        j = self.morph.joints
        
        joint = ET.SubElement(robot, "joint")
        joint.set("name", "NECK")
        joint.set("type", "revolute")
        
        origin_elem = ET.SubElement(joint, "origin")
        origin_elem.set("xyz", self._format_origin(self.neck_origin))
        origin_elem.set("rpy", self._format_rpy((-math.pi/2, 0, 0)))
        
        parent = ET.SubElement(joint, "parent")
        parent.set("link", "PELVIS")
        
        child = ET.SubElement(joint, "child")
        child.set("link", "BALLOON")
        
        axis = ET.SubElement(joint, "axis")
        axis.set("xyz", "0 0 1")
        
        limit = ET.SubElement(joint, "limit")
        limit.set("lower", f"{j.neck_lower_limit:.11f}")
        limit.set("upper", f"{j.neck_upper_limit:.11f}")
        limit.set("effort", "0")
        limit.set("velocity", "0")
        
        dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", f"{j.neck_damping:.2e}")
        dynamics.set("friction", f"{j.neck_friction:.2e}")
    
    def _create_materials(self, robot: ET.Element):
        """Add material color definitions."""
        materials = [
            ("color_pelvis", "0 0.8 0 1"),           # Green
            ("color_femur_left", "0.8 0 0 1"),       # Red
            ("color_femur_right", "0 0 0.8 1"),      # Blue
            ("color_tibia_left", "0.75 0.3 0.3 1"),  # Light red
            ("color_tibia_right", "0.3 0.3 0.75 1"), # Light blue
            ("color_motorarm", "1 0.93725 0.13725 1"),  # Yellow
            ("color_balloons", "0.7 0.7 0.7 0.7"),   # Translucent gray
        ]
        
        for name, rgba in materials:
            material = ET.SubElement(robot, "material")
            material.set("name", name)
            
            color = ET.SubElement(material, "color")
            color.set("rgba", rgba)
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def generate_usd(self, urdf_file_path: str) -> int:
        """
        Generate complete USD file.
        """
        import subprocess
        import sys

        # Get the absolute path to the convert_urdf.py script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        convert_script_path = os.path.join(script_dir, "convert_urdf.py")

        if not os.path.exists(convert_script_path):
            raise FileNotFoundError(f"convert_urdf.py not found at: {convert_script_path}")
        
        if not os.path.exists(urdf_file_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_file_path}")
        
        usd_file_path = os.path.join(self.usd_output_dir, f"{self.morph.morphology_id}", f"{self.morph.morphology_id}.usd")
        
        cmd = [
            sys.executable,
            convert_script_path,
            urdf_file_path,
            usd_file_path,
            "--merge-joints",
            "--headless"
        ]
        
        # Run the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        try:
            for line in process.stdout:
                print("[URDF->USD] ", line, end='')

            process.wait()
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(cmd)

        return process.returncode


__all__ = [
    "InertiaCalculator",
    "InertiaProperties",
    "BalluUSDGenerator",
]

