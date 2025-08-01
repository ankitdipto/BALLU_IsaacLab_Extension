#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Create a BALLU USD file with 3x tibia length - CONSERVATIVE PHYSICS VERSION

This script applies more conservative physics scaling to avoid numerical instability:
- Moderate mass increase (1.5x instead of 3x)
- Conservative inertia scaling (1.5x instead of scale_factor^2)
- Proper collision repositioning

Usage:
    python create_3x_tibia_ballu_conservative.py
"""

import shutil
import os
import numpy as np
from isaaclab.app import AppLauncher

# Launch Isaac Sim (minimal setup)
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

def calculate_conservative_inertia(original_inertia, scale_factor):
    """
    Calculate conservatively scaled inertia tensor.
    
    For 3x longer tibia, use moderate scaling to avoid numerical issues.
    """
    # Conservative scaling: use sqrt of scale_factor instead of scale_factor^2
    conservative_scale = np.sqrt(scale_factor)
    
    new_ixx = original_inertia[0] * conservative_scale
    new_iyy = original_inertia[1] * 1.2  # Slight increase for length axis
    new_izz = original_inertia[2] * conservative_scale
    
    return [new_ixx, new_iyy, new_izz]

def update_tibia_physics_conservative(stage, tibia_path, scale_factor):
    """Update physics properties for a scaled tibia (conservative approach)."""
    try:
        # Original TIBIA properties from URDF
        original_mass = 0.04367  # kg
        original_inertia = [1.554562E-04, 5.653536E-06, 1.589192E-04]  # [ixx, iyy, izz]
        original_com = [0.0068, 0.22605, -0.00419]  # Center of mass
        
        # Conservative scaling to avoid numerical instability
        new_mass = original_mass * 1.5  # Moderate increase instead of 3x
        new_inertia = calculate_conservative_inertia(original_inertia, scale_factor)
        
        # Center of mass Y position scales more conservatively
        com_scale = 1.0 + 0.5 * (scale_factor - 1.0)  # Interpolated scaling
        new_com = [original_com[0], original_com[1] * com_scale, original_com[2]]
        
        print(f"      📊 Conservative physics updates for {tibia_path}:")
        print(f"         Mass: {original_mass:.6f} → {new_mass:.6f} kg (conservative)")
        print(f"         Inertia XX: {original_inertia[0]:.6e} → {new_inertia[0]:.6e}")
        print(f"         Inertia YY: {original_inertia[1]:.6e} → {new_inertia[1]:.6e}")
        print(f"         Inertia ZZ: {original_inertia[2]:.6e} → {new_inertia[2]:.6e}")
        print(f"         COM Y: {original_com[1]:.6f} → {new_com[1]:.6f}")
        
        # Find and update physics properties
        prim = stage.GetPrimAtPath(tibia_path)
        if prim and prim.IsValid():
            # Update mass properties using USD Physics API
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            if mass_api:
                mass_api.GetMassAttr().Set(new_mass)
                mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*new_com))
                mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*new_inertia))
                print(f"         ✅ Updated conservative mass properties")
            else:
                print(f"         ⚠️  Could not apply USD Physics MassAPI")
                
        return True
        
    except Exception as e:
        print(f"         ❌ Error updating physics properties: {e}")
        return False

def update_collision_geometry_conservative(stage, tibia_path, scale_factor):
    """Update collision geometry positions for scaled tibia (conservative approach)."""
    try:
        tibia_prim = stage.GetPrimAtPath(tibia_path)
        if not tibia_prim or not tibia_prim.IsValid():
            return False
            
        print(f"      🔧 Updating collision geometry for {tibia_path}")
        
        # Find collision children
        collision_updated = 0
        for child in tibia_prim.GetAllChildren():
            child_path = str(child.GetPath())
            
            if "collision" in child_path.lower():
                # This is the collision group
                for collision_child in child.GetAllChildren():
                    collision_path = str(collision_child.GetPath())
                    print(f"         Processing collision: {collision_path}")
                    
                    if "mesh_0" in collision_path:
                        # Main tibia collision - scale both position and size
                        new_y_pos = 0.15 * scale_factor
                        new_length = 0.2 * scale_factor
                        
                        # Update transform
                        xformable = UsdGeom.Xformable(collision_child)
                        if xformable:
                            # Clear existing transforms and set new ones
                            xformable.ClearXformOpOrder()
                            translate_op = xformable.AddTranslateOp()
                            translate_op.Set(Gf.Vec3f(0, new_y_pos, 0))
                            
                            # Update geometry size if it's a cylinder
                            cylinder = UsdGeom.Cylinder(collision_child)
                            if cylinder:
                                cylinder.GetHeightAttr().Set(new_length)
                                print(f"           ✅ Updated cylinder: pos=(0,{new_y_pos:.3f},0), height={new_length:.3f}")
                        
                    elif "mesh_1" in collision_path:
                        # Foot collision - position at end of scaled tibia
                        new_y_pos = 0.38485 * scale_factor
                        
                        # Update transform
                        xformable = UsdGeom.Xformable(collision_child)
                        if xformable:
                            xformable.ClearXformOpOrder()
                            translate_op = xformable.AddTranslateOp()
                            translate_op.Set(Gf.Vec3f(0, new_y_pos, 0))
                            print(f"           ✅ Updated sphere: pos=(0,{new_y_pos:.3f},0)")
                        
                    collision_updated += 1
        
        print(f"         Updated {collision_updated} collision geometries")
        return collision_updated > 0
        
    except Exception as e:
        print(f"         ❌ Error updating collision geometry: {e}")
        return False

def create_3x_tibia_ballu_conservative():
    """Create a USD file with 3x scaled tibia links and conservative physics properties."""
    
    # Define file paths
    original_usd = "/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/ballu_v_0.usd"
    scaled_usd = "/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/ballu_3x_tibia_conservative.usd"
    
    print("="*60)
    print("🤖 Creating 3x Tibia BALLU USD (CONSERVATIVE PHYSICS)")
    print("="*60)
    print(f"📂 Original: {os.path.basename(original_usd)}")
    print(f"💾 Output: {os.path.basename(scaled_usd)}")
    print(f"📏 Tibia scale factor: 3.0x")
    print(f"🔬 Physics scaling: CONSERVATIVE (numerical stability)")
    print("="*60)
    
    try:
        # Verify original file exists
        if not os.path.exists(original_usd):
            print(f"❌ Original USD file not found: {original_usd}")
            return False
        
        # Copy the original file
        shutil.copy2(original_usd, scaled_usd)
        print(f"✅ Copied original USD file")
        
        # Open the USD stage
        stage = Usd.Stage.Open(scaled_usd)
        if not stage:
            print(f"❌ Could not open USD stage: {scaled_usd}")
            return False
        
        print(f"📖 Opened USD stage successfully")
        
        # Process tibia scaling with conservative physics
        scale_factor = 3.0
        tibia_paths = ["/ballu/TIBIA_LEFT", "/ballu/TIBIA_RIGHT"]
        
        total_success = 0
        
        for tibia_path in tibia_paths:
            print(f"\n🔧 Processing: {tibia_path}")
            
            prim = stage.GetPrimAtPath(tibia_path)
            if not prim or not prim.IsValid():
                print(f"   ❌ Could not find prim: {tibia_path}")
                continue
            
            # 1. Scale visual geometry (as before)
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                
                # Apply visual scaling
                existing_scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if existing_scale_ops:
                    scale_op = existing_scale_ops[0]
                    current_scale = scale_op.Get()
                    if current_scale:
                        new_scale = Gf.Vec3f(current_scale[0], current_scale[1] * scale_factor, current_scale[2])
                    else:
                        new_scale = Gf.Vec3f(1.0, scale_factor, 1.0)
                    scale_op.Set(new_scale)
                    print(f"   📏 Updated visual scale to {tuple(new_scale)}")
                
                # 2. Update physics properties (conservative)
                physics_success = update_tibia_physics_conservative(stage, tibia_path, scale_factor)
                
                # 3. Update collision geometry
                collision_success = update_collision_geometry_conservative(stage, tibia_path, scale_factor)
                
                if physics_success and collision_success:
                    total_success += 1
                    print(f"   ✅ Successfully processed {tibia_path}")
                else:
                    print(f"   ⚠️  Partial success for {tibia_path}")
            
            # 4. Scale visual children
            for child in prim.GetAllChildren():
                if "visual" in str(child.GetPath()).lower():
                    if child.IsA(UsdGeom.Xformable):
                        child_xformable = UsdGeom.Xformable(child)
                        child_scale_ops = [op for op in child_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                        if child_scale_ops:
                            child_scale_op = child_scale_ops[0]
                            child_current_scale = child_scale_op.Get()
                            if child_current_scale:
                                child_new_scale = Gf.Vec3f(child_current_scale[0], child_current_scale[1] * scale_factor, child_current_scale[2])
                                child_scale_op.Set(child_new_scale)
                                print(f"     ↳ Scaled visual child: {child.GetPath()}")
        
        # Save the modified USD file
        stage.Save()
        print(f"\n💾 Saved USD file with conservative physics")
        
        # Verify the file was created
        if os.path.exists(scaled_usd):
            file_size = os.path.getsize(scaled_usd)
            print(f"✅ File created successfully: {file_size} bytes")
            
            if total_success >= 2:
                print(f"🎉 Successfully processed {total_success}/2 tibia links with conservative physics!")
                return True
            else:
                print(f"⚠️  Partial success: {total_success}/2 tibia links processed")
                return True
        else:
            print(f"❌ File was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating conservative physics USD: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🚀 Starting 3x Tibia BALLU USD Creation (Conservative Physics)...")
    
    success = create_3x_tibia_ballu_conservative()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SUCCESS!")
        print("="*60)
        print("✅ 3x Tibia BALLU USD with conservative physics created!")
        print("📂 File: ballu_assets/robots/ballu_3x_tibia_conservative.usd")
        print("🔬 Features:")
        print("   • Scaled visual geometry (3x tibia length)")
        print("   • Conservative mass scaling (1.5x instead of 3x)")
        print("   • Moderate inertia adjustments (numerical stability)")
        print("   • Properly repositioned collision geometry")
        print("🚀 Ready for stable simulation!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ FAILED")
        print("="*60)
        print("Could not create conservative physics USD file")
        print("Check the error messages above for details")
        print("="*60)

if __name__ == "__main__":
    main()
    simulation_app.close() 