#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Create a BALLU URDF file with custom femur-to-tibia ratio.

This script reads the original BALLU URDF and creates a new URDF with the specified
femur-to-tibia ratio while preserving total leg length. All physical properties
(mass, inertia, center of mass), visual elements, and joint frames are updated accordingly.

Usage:
    python create_leg_ratio_ballu.py --ratio 3:7 --output modified_ballu_3_7.urdf
    python create_leg_ratio_ballu.py --ratio 1:2 --output modified_ballu_1_2.urdf
    python create_leg_ratio_ballu.py --femur-ratio 2 --tibia-ratio 3 --output modified_ballu_2_3.urdf
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, List
import math
import shutil
from pathlib import Path


class BalluLegRatioModifier:
    """
    Utility class for modifying BALLU robot's femur-to-tibia ratio in URDF format.
    
    This class maintains total leg length while changing the proportion between femur and tibia,
    updating all associated physics properties, joint positions, and visual elements.
    """
    
    def __init__(self, original_urdf_path: str):
        """
        Initialize the leg ratio modifier.
        
        Args:
            original_urdf_path: Path to the original BALLU URDF file
        """
        self.original_urdf_path = original_urdf_path
        
        # Original measurements from BALLU URDF analysis
        self.original_femur_length = 0.36501  # meters (hip to knee joint)
        self.original_tibia_length = 0.38485  # meters (knee to foot)
        self.total_leg_length = self.original_femur_length + self.original_tibia_length  # 0.74986 m
        
        # Original physics properties
        self.original_femur_mass = 0.00944  # kg
        self.original_tibia_mass = 0.04367  # kg
        self.total_leg_mass = self.original_femur_mass + self.original_tibia_mass
        
        # Original inertia values [ixx, iyy, izz, ixy, ixz, iyz]
        self.original_femur_inertia = [1.705039E-04, 1.024278E-07, 1.704907E-04, -8.965456E-08, 0.0, 5.521509E-09]
        self.original_tibia_inertia = [1.554562E-04, 5.653536E-06, 1.589192E-04, 2.581390E-06, 2.577077E-07, -2.545764E-06]
        
        # Original centers of mass
        self.original_femur_com = [-0.00005, 0.17956, 0.0]  # Relative to femur origin
        self.original_tibia_com = [0.0068, 0.22605, -0.00419]  # Relative to tibia origin
        
        print(f"🤖 BALLU Leg Ratio Modifier initialized")
        print(f"📏 Original leg dimensions:")
        print(f"   Femur length: {self.original_femur_length:.5f} m")
        print(f"   Tibia length: {self.original_tibia_length:.5f} m")
        print(f"   Total length: {self.total_leg_length:.5f} m")
        
    def calculate_new_dimensions(self, femur_ratio: float, tibia_ratio: float) -> Tuple[float, float]:
        """
        Calculate new femur and tibia lengths based on the desired ratio.
        
        Args:
            femur_ratio: Femur portion of the ratio (e.g., 3 for 3:7)
            tibia_ratio: Tibia portion of the ratio (e.g., 7 for 3:7)
            
        Returns:
            Tuple of (new_femur_length, new_tibia_length)
        """
        ratio_sum = femur_ratio + tibia_ratio
        
        new_femur_length = self.total_leg_length * (femur_ratio / ratio_sum)
        new_tibia_length = self.total_leg_length * (tibia_ratio / ratio_sum)
        
        print(f"📐 New dimensions for ratio {femur_ratio}:{tibia_ratio}:")
        print(f"   New femur length: {new_femur_length:.5f} m")
        print(f"   New tibia length: {new_tibia_length:.5f} m")
        print(f"   Verification - total: {new_femur_length + new_tibia_length:.5f} m")
        
        return new_femur_length, new_tibia_length
    
    def calculate_scaled_physics_properties(self, original_length: float, new_length: float, 
                                          original_mass: float, original_inertia: List[float], 
                                          original_com: List[float]) -> Tuple[float, List[float], List[float]]:
        """
        Calculate scaled physics properties based on new length.
        
        Args:
            original_length: Original length of the link
            new_length: New length of the link
            original_mass: Original mass
            original_inertia: Original inertia tensor [ixx, iyy, izz, ixy, ixz, iyz]
            original_com: Original center of mass [x, y, z]
            
        Returns:
            Tuple of (new_mass, new_inertia, new_com)
        """
        length_scale = new_length / original_length
        
        # Mass scales linearly with length (assuming constant cross-section and density)
        new_mass = original_mass * length_scale
        
        # Inertia scaling for cylindrical bodies:
        # For length axis (Y): I_yy scales with length
        # For other axes: I_xx, I_zz scale with length^3 (mass * length^2)
        new_inertia = [
            original_inertia[0] * (length_scale ** 3),  # ixx
            original_inertia[1] * length_scale,         # iyy (length axis)
            original_inertia[2] * (length_scale ** 3),  # izz
            original_inertia[3] * (length_scale ** 3),  # ixy
            original_inertia[4] * (length_scale ** 3),  # ixz
            original_inertia[5] * (length_scale ** 3),  # iyz
        ]
        
        # Center of mass Y-coordinate scales with length, others remain proportional
        new_com = [
            original_com[0],                           # x unchanged
            original_com[1] * length_scale,            # y scales with length
            original_com[2]                            # z unchanged
        ]
        
        return new_mass, new_inertia, new_com
    
    def update_femur_properties(self, root: ET.Element, new_femur_length: float):
        """Update femur link properties in the URDF."""
        femur_scale = new_femur_length / self.original_femur_length
        
        print(f"🔧 Updating FEMUR properties (scale factor: {femur_scale:.3f}):")
        
        # Calculate new physics properties
        new_mass, new_inertia, new_com = self.calculate_scaled_physics_properties(
            self.original_femur_length, new_femur_length,
            self.original_femur_mass, self.original_femur_inertia, self.original_femur_com
        )
        
        # Update both left and right femur
        for side in ['LEFT', 'RIGHT']:
            link_name = f'FEMUR_{side}'
            link = root.find(f".//link[@name='{link_name}']")
            
            if link is not None:
                # Update inertial properties
                inertial = link.find('inertial')
                if inertial is not None:
                    # Update mass
                    mass_elem = inertial.find('mass')
                    if mass_elem is not None:
                        mass_elem.set('value', f"{new_mass:.5f}")
                    
                    # Update center of mass
                    origin_elem = inertial.find('origin')
                    if origin_elem is not None:
                        origin_elem.set('xyz', f"{new_com[0]:.5f} {new_com[1]:.5f} {new_com[2]:.5f}")
                    
                    # Update inertia tensor
                    inertia_elem = inertial.find('inertia')
                    if inertia_elem is not None:
                        inertia_elem.set('ixx', f"{new_inertia[0]:.6e}")
                        inertia_elem.set('iyy', f"{new_inertia[1]:.6e}")
                        inertia_elem.set('izz', f"{new_inertia[2]:.6e}")
                        inertia_elem.set('ixy', f"{new_inertia[3]:.6e}")
                        inertia_elem.set('ixz', f"{new_inertia[4]:.6e}")
                        inertia_elem.set('iyz', f"{new_inertia[5]:.6e}")
                
                # Update visual mesh scaling
                visual = link.find('visual')
                if visual is not None:
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            mesh.set('scale', f"1.0 {femur_scale:.6f} 1.0")
                
                # Update collision geometry
                collision = link.find('collision')
                if collision is not None:
                    # Update collision cylinder dimensions and position
                    geometry = collision.find('geometry')
                    if geometry is not None:
                        cylinder = geometry.find('cylinder')
                        if cylinder is not None:
                            cylinder.set('length', f"{new_femur_length:.5f}")
                    
                    # Update collision origin position
                    origin = collision.find('origin')
                    if origin is not None:
                        # Center the collision at half the new length
                        origin.set('xyz', f"0 {new_femur_length/2:.5f} 0")
                
                print(f"   ✅ Updated {link_name}: mass={new_mass:.5f}kg, length={new_femur_length:.5f}m")
        
        # Update knee joint position (where femur connects to tibia)
        for side in ['LEFT', 'RIGHT']:
            joint_name = f'KNEE_{side}'
            joint = root.find(f".//joint[@name='{joint_name}']")
            
            if joint is not None:
                origin = joint.find('origin')
                if origin is not None:
                    # Knee joint is at the end of the femur
                    origin.set('xyz', f"0 {new_femur_length:.5f} 0")
                    print(f"   ✅ Updated {joint_name} position to y={new_femur_length:.5f}")
    
    def update_tibia_properties(self, root: ET.Element, new_tibia_length: float):
        """Update tibia link properties in the URDF."""
        tibia_scale = new_tibia_length / self.original_tibia_length
        
        print(f"🔧 Updating TIBIA properties (scale factor: {tibia_scale:.3f}):")
        
        # Calculate new physics properties
        new_mass, new_inertia, new_com = self.calculate_scaled_physics_properties(
            self.original_tibia_length, new_tibia_length,
            self.original_tibia_mass, self.original_tibia_inertia, self.original_tibia_com
        )
        
        # Update both left and right tibia
        for side in ['LEFT', 'RIGHT']:
            link_name = f'TIBIA_{side}'
            link = root.find(f".//link[@name='{link_name}']")
            
            if link is not None:
                # Update inertial properties
                inertial = link.find('inertial')
                if inertial is not None:
                    # Update mass
                    mass_elem = inertial.find('mass')
                    if mass_elem is not None:
                        mass_elem.set('value', f"{new_mass:.5f}")
                    
                    # Update center of mass
                    origin_elem = inertial.find('origin')
                    if origin_elem is not None:
                        origin_elem.set('xyz', f"{new_com[0]:.5f} {new_com[1]:.5f} {new_com[2]:.5f}")
                    
                    # Update inertia tensor
                    inertia_elem = inertial.find('inertia')
                    if inertia_elem is not None:
                        inertia_elem.set('ixx', f"{new_inertia[0]:.6e}")
                        inertia_elem.set('iyy', f"{new_inertia[1]:.6e}")
                        inertia_elem.set('izz', f"{new_inertia[2]:.6e}")
                        inertia_elem.set('ixy', f"{new_inertia[3]:.6e}")
                        inertia_elem.set('ixz', f"{new_inertia[4]:.6e}")
                        inertia_elem.set('iyz', f"{new_inertia[5]:.6e}")
                
                # Update visual mesh scaling
                visual = link.find('visual')
                if visual is not None:
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            mesh.set('scale', f"1.0 {tibia_scale:.6f} 1.0")
                
                # Update collision geometries (tibia has multiple collision elements)
                collisions = link.findall('collision')
                for i, collision in enumerate(collisions):
                    geometry = collision.find('geometry')
                    origin = collision.find('origin')
                    
                    if geometry is not None:
                        cylinder = geometry.find('cylinder')
                        sphere = geometry.find('sphere')
                        
                        if cylinder is not None:
                            # This is the main tibia collision cylinder
                            cylinder.set('length', f"{new_tibia_length:.5f}")
                            if origin is not None:
                                # Center the collision cylinder
                                origin.set('xyz', f"0 {new_tibia_length/2:.5f} 0")
                        
                        elif sphere is not None:
                            # This is the foot collision sphere
                            if origin is not None:
                                # Position foot at the end of the new tibia length
                                origin.set('xyz', f"0 {new_tibia_length:.5f} 0")
                
                print(f"   ✅ Updated {link_name}: mass={new_mass:.5f}kg, length={new_tibia_length:.5f}m")
        
        # Update motor joint positions (motors are attached to tibia)
        for side in ['LEFT', 'RIGHT']:
            joint_name = f'MOTOR_{side}'
            joint = root.find(f".//joint[@name='{joint_name}']")
            
            if joint is not None:
                origin = joint.find('origin')
                if origin is not None:
                    # Scale the motor position proportionally along the tibia
                    # Original motor position was at y=0.32732, which is 85% along original tibia
                    original_motor_y = 0.32732
                    motor_position_ratio = original_motor_y / self.original_tibia_length
                    new_motor_y = new_tibia_length * motor_position_ratio
                    
                    current_xyz = origin.get('xyz', '0 0 0').split()
                    new_xyz = f"{current_xyz[0]} {new_motor_y:.5f} {current_xyz[2]}"
                    origin.set('xyz', new_xyz)
                    print(f"   ✅ Updated {joint_name} position to y={new_motor_y:.5f}")
    
    def add_modification_comment(self, root: ET.Element, femur_ratio: float, tibia_ratio: float):
        """Add a comment to the URDF describing the modifications."""
        # Find existing comment or create new one
        comment_text = f"""
MODIFIED VERSION: Femur:Tibia ratio = {femur_ratio}:{tibia_ratio}
Original total leg length preserved: {self.total_leg_length:.5f} m
New femur length: {self.total_leg_length * (femur_ratio/(femur_ratio + tibia_ratio)):.5f} m
New tibia length: {self.total_leg_length * (tibia_ratio/(femur_ratio + tibia_ratio)):.5f} m
Generated by BalluLegRatioModifier
"""
        # Note: ET doesn't handle comments well during parsing/writing, so we'll add this after file generation
        return comment_text
    
    def create_modified_urdf(self, femur_ratio: float, tibia_ratio: float, output_path: str) -> bool:
        """
        Create a new URDF file with the specified femur-to-tibia ratio.
        
        Args:
            femur_ratio: Femur portion of the ratio
            tibia_ratio: Tibia portion of the ratio
            output_path: Path for the output URDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"\n{'='*60}")
            print(f"🤖 Creating BALLU URDF with {femur_ratio}:{tibia_ratio} leg ratio")
            print(f"{'='*60}")
            print(f"📂 Input:  {os.path.basename(self.original_urdf_path)}")
            print(f"💾 Output: {os.path.basename(output_path)}")
            
            # Parse the original URDF
            tree = ET.parse(self.original_urdf_path)
            root = tree.getroot()
            
            # Calculate new dimensions
            new_femur_length, new_tibia_length = self.calculate_new_dimensions(femur_ratio, tibia_ratio)
            
            # Update femur properties
            self.update_femur_properties(root, new_femur_length)
            
            # Update tibia properties
            self.update_tibia_properties(root, new_tibia_length)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the modified URDF
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            # Add modification comment by reading and rewriting the file
            self._add_comment_to_file(output_path, femur_ratio, tibia_ratio)
            
            # Verify file creation
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"\n✅ Successfully created modified URDF!")
                print(f"📁 File: {output_path}")
                print(f"📊 Size: {file_size} bytes")
                print(f"🎯 Ratio: {femur_ratio}:{tibia_ratio}")
                print(f"📏 Femur: {new_femur_length:.5f}m, Tibia: {new_tibia_length:.5f}m")
                return True
            else:
                print(f"❌ Failed to create output file")
                return False
                
        except Exception as e:
            print(f"❌ Error creating modified URDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_comment_to_file(self, file_path: str, femur_ratio: float, tibia_ratio: float):
        """Add modification comment to the URDF file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Create the modification comment
            new_femur_length = self.total_leg_length * (femur_ratio/(femur_ratio + tibia_ratio))
            new_tibia_length = self.total_leg_length * (tibia_ratio/(femur_ratio + tibia_ratio))
            
            comment = f'''<!--
MODIFIED VERSION: Femur:Tibia ratio = {femur_ratio}:{tibia_ratio}
Original total leg length preserved: {self.total_leg_length:.5f} m
New femur length: {new_femur_length:.5f} m
New tibia length: {new_tibia_length:.5f} m
Generated by BalluLegRatioModifier
-->

'''
            
            # Insert comment after XML declaration
            lines = content.split('\n')
            if lines[0].startswith('<?xml'):
                lines.insert(1, comment)
            else:
                lines.insert(0, comment)
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            print(f"⚠️  Could not add comment to file: {e}")


def parse_ratio(ratio_str: str) -> Tuple[float, float]:
    """Parse ratio string like '3:7' into (3.0, 7.0)."""
    try:
        parts = ratio_str.split(':')
        if len(parts) != 2:
            raise ValueError("Ratio must be in format 'X:Y'")
        
        femur_ratio = float(parts[0])
        tibia_ratio = float(parts[1])
        
        if femur_ratio <= 0 or tibia_ratio <= 0:
            raise ValueError("Ratio values must be positive")
            
        return femur_ratio, tibia_ratio
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid ratio format '{ratio_str}': {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create a BALLU URDF with custom femur-to-tibia ratio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ratio 3:7 --output modified_ballu_3_7.urdf
  %(prog)s --ratio 1:2 --output modified_ballu_1_2.urdf
  %(prog)s --femur-ratio 2 --tibia-ratio 3 --output modified_ballu_2_3.urdf
  %(prog)s --ratio 1:1  # Creates 1:1 ratio with default output name
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        '--input', 
        type=str,
        default='source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/old/urdf/urdf/original.urdf',
        help='Path to input URDF file (default: original.urdf)'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Path to output URDF file'
    )
    
    # Ratio specification (two methods)
    ratio_group = parser.add_mutually_exclusive_group(required=True)
    ratio_group.add_argument(
        '--ratio', 
        type=str,
        help='Femur:tibia ratio (e.g., "3:7", "1:2")'
    )
    ratio_group.add_argument(
        '--femur-ratio', 
        type=float,
        help='Femur ratio value (use with --tibia-ratio)'
    )
    parser.add_argument(
        '--tibia-ratio', 
        type=float,
        help='Tibia ratio value (use with --femur-ratio)'
    )
    
    args = parser.parse_args()
    
    # Validate ratio specification
    if args.ratio:
        try:
            femur_ratio, tibia_ratio = parse_ratio(args.ratio)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    elif args.femur_ratio is not None and args.tibia_ratio is not None:
        if args.femur_ratio <= 0 or args.tibia_ratio <= 0:
            print(f"❌ Error: Ratio values must be positive")
            sys.exit(1)
        femur_ratio, tibia_ratio = args.femur_ratio, args.tibia_ratio
    else:
        print(f"❌ Error: Must specify either --ratio or both --femur-ratio and --tibia-ratio")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Generate default output name
        input_parent = Path(args.input).parent
        ratio_str = f"{int(femur_ratio)}_{int(tibia_ratio)}"
        output_path = input_parent / f"modified_ballu_{ratio_str}_ratio.urdf"
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("output path: ", output_path)
    # Create modifier and generate new URDF
    modifier = BalluLegRatioModifier(args.input)
    success = modifier.create_modified_urdf(femur_ratio, tibia_ratio, output_path)
    
    if success:
        print(f"\n{'='*60}")
        print(f"🎉 SUCCESS!")
        print(f"{'='*60}")
        print(f"✅ Modified BALLU URDF created successfully")
        print(f"📂 Location: {output_path}")
        print(f"🎯 Femur:Tibia ratio: {femur_ratio}:{tibia_ratio}")
        print(f"🚀 Ready for use with Isaac Lab!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ FAILED")
        print(f"{'='*60}")
        print(f"Could not create modified URDF file")
        print(f"Check the error messages above for details")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main() 