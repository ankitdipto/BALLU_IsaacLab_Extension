"""
BALLU morphology utilities.

Functionality:
- Adjust femur-to-total-limb length ratio for both legs in the BALLU URDF while keeping link masses unchanged.

Notes:
- This function preserves total limb length (femur + tibia) and only redistributes it between femur and tibia.
- It updates kinematic joint origins and primary collision geometries to reflect new lengths.
- It does NOT modify any masses. Inertial tensors and CoM are updated proportionally to maintain mass distribution.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Optional, Tuple


class BalluMorphologyModifier:
    """
    Class for modifying BALLU robot morphology by adjusting femur-to-tibia ratios.

    This class reads the original BALLU URDF and creates modified versions with different
    femur-to-total-limb ratios while preserving total leg length and link masses.
    """

    def __init__(self, 
        original_urdf_path: str = "/home/asinha389/shared/BALLU_Project/ballu_isclb_extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/old/urdf/urdf/original.urdf"
    ):
        """
        Initialize the morphology modifier.

        Args:
            original_urdf_path: Absolute path to the original BALLU URDF file
        """
        if not os.path.isabs(original_urdf_path):
            original_urdf_path = os.path.abspath(original_urdf_path)
        if not os.path.exists(original_urdf_path):
            raise FileNotFoundError(f"URDF not found: {original_urdf_path}")

        self.original_urdf_path = original_urdf_path
        self.tree = ET.parse(original_urdf_path)
        self.root = self.tree.getroot()

        # Read original lengths
        self.femur_len_old, self.tibia_len_old = self._read_current_lengths()
        self.total_len = self.femur_len_old + self.tibia_len_old

        self.output_urdf_path = None
        self.usd_root_dir = "/home/asinha389/shared/BALLU_Project/ballu_isclb_extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots"
        self.morphology_name: str | None = None

    def _get_float_triple(self, text: str) -> Tuple[float, float, float]:
        """Parse xyz string into float tuple."""
        parts = text.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Expected 3 components in xyz string, got: {text}")
        return float(parts[0]), float(parts[1]), float(parts[2])

    def _set_float_triple(self, x: float, y: float, z: float) -> str:
        """Format float triple as xyz string."""
        return f"{x:.5f} {y:.5f} {z:.5f}"

    def _find_required(self, elem: ET.Element, path: str) -> ET.Element:
        """Find required element or raise error."""
        found = elem.find(path)
        if found is None:
            raise ValueError(f"Required element not found at path: {path}")
        return found

    def _scale_inertia_along_y(
        self,
        ixx: float,
        iyy: float,
        izz: float,
        ixy: float,
        ixz: float,
        iyz: float,
        s: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Scale inertia tensor for anisotropic scaling along Y by factor s (mass constant), at CoM.

        Uses:
          I'_xx = J_z + s^2 J_y
          I'_yy = J_x + J_z (independent of s)
          I'_zz = J_x + s^2 J_y
          I'xy = s I_xy, I'xz = I_xz, I'yz = s I_yz
        where J_x = (I_yy + I_zz - I_xx)/2, etc.
        """
        # Second moments
        jx = (iyy + izz - ixx) * 0.5
        jy = (ixx + izz - iyy) * 0.5
        jz = (ixx + iyy - izz) * 0.5

        s2 = s * s

        ixx_p = jz + s2 * jy
        iyy_p = jx + jz  # equals original iyy
        izz_p = jx + s2 * jy

        ixy_p = s * ixy
        ixz_p = ixz
        iyz_p = s * iyz

        return ixx_p, iyy_p, izz_p, ixy_p, ixz_p, iyz_p

    def _read_current_lengths(self) -> Tuple[float, float]:
        """
        Read current femur and tibia lengths from the URDF.

        Strategy:
        - Femur length: y of `origin` in `KNEE_LEFT` joint (relative to femur origin)
        - Tibia length: y of `origin` of the foot collision sphere under `TIBIA_LEFT`
        """
        # Femur length from knee joint origin
        knee_left = self.root.find(".//joint[@name='KNEE_LEFT']")
        if knee_left is None:
            raise ValueError("KNEE_LEFT joint not found; cannot infer femur length")
        knee_origin = self._find_required(knee_left, "origin")
        _, femur_len, _ = self._get_float_triple(knee_origin.get("xyz", "0 0 0"))

        # Tibia length from foot sphere origin (end of tibia)
        tibia_left = self.root.find(".//link[@name='TIBIA_LEFT']")
        if tibia_left is None:
            raise ValueError("TIBIA_LEFT link not found; cannot infer tibia length")

        tibia_len: Optional[float] = None
        for collision in tibia_left.findall("collision"):
            geom = collision.find("geometry")
            if geom is None:
                continue
            sphere = geom.find("sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            if origin is None:
                continue
            _, y, _ = self._get_float_triple(origin.get("xyz", "0 0 0"))
            tibia_len = y
            break

        if tibia_len is None:
            # Fallback: try tibia motor y as proportion of tibia-cylinder length is not reliable
            motor_left = self.root.find(".//joint[@name='MOTOR_LEFT']")
            cyl_len = None
            tibia_cyl = None
            if tibia_left is not None:
                for collision in tibia_left.findall("collision"):
                    geom = collision.find("geometry")
                    if geom is None:
                        continue
                    cylinder = geom.find("cylinder")
                    if cylinder is not None:
                        tibia_cyl = cylinder
                        cyl_len = float(cylinder.get("length", "0.0"))
                        break
            if motor_left is not None:
                m_origin = motor_left.find("origin")
                if m_origin is not None and cyl_len and cyl_len > 0:
                    _, m_y, _ = self._get_float_triple(m_origin.get("xyz", "0 0 0"))
                    # Original model places motor at ~0.85 of tibia; infer tibia_len ≈ m_y / 0.85
                    tibia_len = m_y / 0.85
            if tibia_len is None:
                raise ValueError("Could not infer tibia length from URDF")

        return float(femur_len), float(tibia_len)

    def adjust_femur_to_limb_ratio(
        self,
        femur_ratio: float
    ) -> str:
        """
        Adjust the femur-to-total-limb length ratio for both legs in the BALLU URDF.

        - Keeps total limb length constant (femur + tibia per leg)
        - Updates: femur collision cylinder length/position, knee joint origin, tibia collision
          cylinder length/position, foot sphere origin, and motor joint origin (proportionally).
        - Does not change any link masses.
        - Updates CoM and inertia tensors proportionally to maintain mass distribution.

        Args:
            femur_ratio: Desired femur-to-total-limb ratio in (0, 1). For example, 0.5 for 1:1.
            output_path: Optional path for the modified URDF. If not provided, a sibling file is created.

        Returns:
            Path to the written modified URDF file.
        """
        if not (0.0 < femur_ratio < 1.0):
            raise ValueError("femur_ratio must be in (0, 1)")

        femur_len_new = self.total_len * femur_ratio
        tibia_len_new = self.total_len - femur_len_new

        # Update FEMUR: collision cylinder length and centered origin; KNEE joint origin
        for side in ("LEFT", "RIGHT"):
            # Link collision geometry
            femur_link = self.root.find(f".//link[@name='FEMUR_{side}']")
            if femur_link is not None:
                # Update CoM to preserve original y-ratio relative to link length
                inertial = femur_link.find("inertial")
                if inertial is not None:
                    i_origin = inertial.find("origin")
                    if i_origin is not None:
                        cx, cy, cz = self._get_float_triple(i_origin.get("xyz", "0 0 0"))
                        y_ratio = cy / self.femur_len_old if self.femur_len_old > 0 else 0.0
                        i_origin.set("xyz", self._set_float_triple(cx, y_ratio * femur_len_new, cz))

                    # Update inertia tensor for scaling along Y by s_f
                    inertia_elem = inertial.find("inertia")
                    if inertia_elem is not None:
                        s_f = femur_len_new / self.femur_len_old if self.femur_len_old > 0 else 1.0
                        ixx = float(inertia_elem.get("ixx", "0"))
                        iyy = float(inertia_elem.get("iyy", "0"))
                        izz = float(inertia_elem.get("izz", "0"))
                        ixy = float(inertia_elem.get("ixy", "0"))
                        ixz = float(inertia_elem.get("ixz", "0"))
                        iyz = float(inertia_elem.get("iyz", "0"))
                        ixx_p, iyy_p, izz_p, ixy_p, ixz_p, iyz_p = self._scale_inertia_along_y(
                            ixx, iyy, izz, ixy, ixz, iyz, s_f
                        )
                        inertia_elem.set("ixx", f"{ixx_p:.6e}")
                        inertia_elem.set("iyy", f"{iyy_p:.6e}")
                        inertia_elem.set("izz", f"{izz_p:.6e}")
                        inertia_elem.set("ixy", f"{ixy_p:.6e}")
                        inertia_elem.set("ixz", f"{ixz_p:.6e}")
                        inertia_elem.set("iyz", f"{iyz_p:.6e}")

                collision = femur_link.find("collision")
                if collision is not None:
                    geometry = collision.find("geometry")
                    if geometry is not None:
                        cylinder = geometry.find("cylinder")
                        if cylinder is not None:
                            cylinder.set("length", f"{femur_len_new:.5f}")
                    origin = collision.find("origin")
                    if origin is not None:
                        # center the cylinder at half the new length along +Y
                        origin.set("xyz", self._set_float_triple(0.0, femur_len_new / 2.0, 0.0))

                # Update visual mesh scaling for femur
                visual = femur_link.find("visual")
                if visual is not None:
                    origin = visual.find("origin")
                    if origin is not None:
                        # Scale the visual mesh along Y-axis proportionally
                        ox, oy, oz = self._get_float_triple(origin.get("xyz", "0 0 0"))
                        # For visual meshes, we scale the Y position proportionally
                        # and add a scale attribute to the visual element
                        origin.set("xyz", self._set_float_triple(ox, oy, oz))
                    
                    # Add scale attribute to visual element for proper mesh scaling
                    scale_factor = femur_len_new / self.femur_len_old if self.femur_len_old > 0 else 1.0
                    geometry_v = visual.find("geometry")
                    if geometry_v is not None:
                        mesh = geometry_v.find("mesh")
                        if mesh is not None:
                            mesh.set("scale", f"1.0 {scale_factor:.5f} 1.0")

            # Knee joint at end of femur
            knee_joint = self.root.find(f".//joint[@name='KNEE_{side}']")
            if knee_joint is not None:
                k_origin = knee_joint.find("origin")
                if k_origin is not None:
                    k_origin.set("xyz", self._set_float_triple(0.0, femur_len_new, 0.0))

        # Update TIBIA: collision cylinder length, centered origin; foot sphere origin at end
        # Also scale motor joint position proportionally along the tibia length
        motor_pos_ratio = None
        # Compute once from LEFT if possible
        motor_left = self.root.find(".//joint[@name='MOTOR_LEFT']")
        if motor_left is not None:
            m_origin = motor_left.find("origin")
            if m_origin is not None and self.tibia_len_old > 0:
                _, m_y, _ = self._get_float_triple(m_origin.get("xyz", "0 0 0"))
                motor_pos_ratio = max(0.0, min(1.0, m_y / self.tibia_len_old))
        if motor_pos_ratio is None:
            motor_pos_ratio = 0.85  # sensible default per original model

        for side in ("LEFT", "RIGHT"):
            tibia_link = self.root.find(f".//link[@name='TIBIA_{side}']")
            if tibia_link is not None:
                # Update CoM to preserve original y-ratio relative to link length
                inertial = tibia_link.find("inertial")
                if inertial is not None:
                    i_origin = inertial.find("origin")
                    if i_origin is not None:
                        cx, cy, cz = self._get_float_triple(i_origin.get("xyz", "0 0 0"))
                        y_ratio = cy / self.tibia_len_old if self.tibia_len_old > 0 else 0.0
                        i_origin.set("xyz", self._set_float_triple(cx, y_ratio * tibia_len_new, cz))

                    # Update inertia tensor for scaling along Y by s_t
                    inertia_elem = inertial.find("inertia")
                    if inertia_elem is not None:
                        s_t = tibia_len_new / self.tibia_len_old if self.tibia_len_old > 0 else 1.0
                        ixx = float(inertia_elem.get("ixx", "0"))
                        iyy = float(inertia_elem.get("iyy", "0"))
                        izz = float(inertia_elem.get("izz", "0"))
                        ixy = float(inertia_elem.get("ixy", "0"))
                        ixz = float(inertia_elem.get("ixz", "0"))
                        iyz = float(inertia_elem.get("iyz", "0"))
                        ixx_p, iyy_p, izz_p, ixy_p, ixz_p, iyz_p = self._scale_inertia_along_y(
                            ixx, iyy, izz, ixy, ixz, iyz, s_t
                        )
                        inertia_elem.set("ixx", f"{ixx_p:.6e}")
                        inertia_elem.set("iyy", f"{iyy_p:.6e}")
                        inertia_elem.set("izz", f"{izz_p:.6e}")
                        inertia_elem.set("ixy", f"{ixy_p:.6e}")
                        inertia_elem.set("ixz", f"{ixz_p:.6e}")
                        inertia_elem.set("iyz", f"{iyz_p:.6e}")

                # Update all collisions for tibia
                for collision in tibia_link.findall("collision"):
                    geometry = collision.find("geometry")
                    if geometry is None:
                        continue
                    origin = collision.find("origin")
                    cylinder = geometry.find("cylinder")
                    sphere = geometry.find("sphere")
                    if cylinder is not None:
                        cylinder.set("length", f"{tibia_len_new:.5f}")
                        if origin is not None:
                            origin.set("xyz", self._set_float_triple(0.0, tibia_len_new / 2.0, 0.0))
                    elif sphere is not None:
                        # Foot at tibia end
                        if origin is not None:
                            origin.set("xyz", self._set_float_triple(0.0, tibia_len_new, 0.0))

                # Update visual mesh scaling for tibia
                visual = tibia_link.find("visual")
                if visual is not None:
                    origin = visual.find("origin")
                    if origin is not None:
                        # Scale the visual mesh along Y-axis proportionally
                        ox, oy, oz = self._get_float_triple(origin.get("xyz", "0 0 0"))
                        # For visual meshes, we scale the Y position proportionally
                        # and add a scale attribute to the visual element
                        origin.set("xyz", self._set_float_triple(ox, oy, oz))
                    
                    # Add scale attribute to visual element for proper mesh scaling
                    scale_factor = tibia_len_new / self.tibia_len_old if self.tibia_len_old > 0 else 1.0
                    geometry_v = visual.find("geometry")
                    if geometry_v is not None:
                        mesh = geometry_v.find("mesh")
                        if mesh is not None:
                            mesh.set("scale", f"1.0 {scale_factor:.5f} 1.0")

            # Motor joint along tibia
            motor_joint = self.root.find(f".//joint[@name='MOTOR_{side}']")
            if motor_joint is not None:
                m_origin = motor_joint.find("origin")
                if m_origin is not None:
                    x, _, z = self._get_float_triple(m_origin.get("xyz", "0 0 0"))
                    new_m_y = tibia_len_new * motor_pos_ratio
                    m_origin.set("xyz", self._set_float_triple(x, new_m_y, z))

        # Determine output 
        base_dir = os.path.dirname(self.original_urdf_path)
        self.morphology_name = f"original_FL_{femur_ratio:.2f}"
        self.output_urdf_path = os.path.join(base_dir, f"{self.morphology_name}.urdf")
        
        os.makedirs(os.path.dirname(self.output_urdf_path), exist_ok=True)
        self.tree.write(self.output_urdf_path, encoding="utf-8", xml_declaration=True)

        return self.output_urdf_path

    def convert_to_usd(self) -> int:
        """
        Convert a URDF file to USD format using the convert_urdf.py script.

        This method runs the conversion script as a subprocess with --merge-joints
        and --headless flags enabled.

        Returns:
            Exit code of the subprocess (0 for success, non-zero for failure)
        """
        import subprocess
        import sys

        # Get the absolute path to the convert_urdf.py script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        convert_script_path = os.path.join(script_dir, "convert_urdf.py")

        if not os.path.exists(convert_script_path):
            raise FileNotFoundError(f"convert_urdf.py not found at: {convert_script_path}")

        # Build the command with required flags
        cmd = [
            sys.executable,  # Use the same Python interpreter
            convert_script_path,
            self.output_urdf_path,
            os.path.join(self.usd_root_dir, self.morphology_name, f"{self.morphology_name}.usd"),
            "--merge-joints",
            "--headless"
        ]

        # Run the subprocess
        # try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        try:    
            # Print output for debugging
            for line in process.stdout:
                print("[URDF->USD] ", line, end='')

            process.wait()    
        
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(cmd)
        
        return process.returncode
        # except subprocess.CalledProcessError as e:
        #     print(f"Subprocess failed with return code: {e.returncode}")
        #     print(f"Error output: {e.stderr}")
        #     return e.returncode
        # except Exception as e:
        #     print(f"Unexpected error running subprocess: {e}")
        #     return 1

__all__ = [
    "BalluMorphologyModifier",
]
