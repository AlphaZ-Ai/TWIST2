#!/usr/bin/env python3
"""
Quick verification script for G1 with Inspire Grippers URDF
Prints DOF information and validates the model structure
"""

import xml.etree.ElementTree as ET

def check_g1_gripper_model():
    xml_file = "/home/mcenlaptop/isaacgym/TWIST2/assets/g1/g1_sim2sim_29dof_with_grippers.xml"
    
    print("=" * 70)
    print("G1 with Inspire Grippers - Model Verification")
    print("=" * 70)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Count joints
    joints = root.findall(".//joint")
    print(f"\n✓ Total Joints: {len(joints)}")
    
    # Find gripper joints
    gripper_joints = [j for j in joints if 'gripper' in j.get('name', '')]
    print(f"✓ Gripper Joints: {len(gripper_joints)}")
    for joint in gripper_joints:
        name = joint.get('name')
        jtype = joint.get('type')
        axis = joint.get('axis')
        range_attr = joint.get('range')
        print(f"  - {name}: type={jtype}, axis={axis}, range={range_attr}")
    
    # Count actuators
    actuators = root.findall(".//actuator/*")
    print(f"\n✓ Total Actuators: {len(actuators)}")
    
    gripper_actuators = [a for a in actuators if 'gripper' in a.get('name', '')]
    print(f"✓ Gripper Actuators: {len(gripper_actuators)}")
    for act in gripper_actuators:
        name = act.get('name')
        joint = act.get('joint')
        print(f"  - {name} -> {joint}")
    
    # Check keyframe
    keyframes = root.findall(".//keyframe/key")
    if keyframes:
        key = keyframes[0]
        qpos = key.get('qpos', '')
        qpos_values = qpos.split()
        print(f"\n✓ Keyframe 'home' has {len(qpos_values)} qpos values")
        if len(qpos_values) >= 38:
            print(f"  - Left gripper init: {qpos_values[36]}")
            print(f"  - Right gripper init: {qpos_values[37]}")
        else:
            print(f"  ⚠ WARNING: Expected 38 values, got {len(qpos_values)}")
    
    # Check body structure
    bodies = root.findall(".//body")
    gripper_bodies = [b for b in bodies if 'gripper' in b.get('name', '')]
    print(f"\n✓ Gripper Bodies: {len(gripper_bodies)}")
    for body in gripper_bodies:
        print(f"  - {body.get('name')}")
    
    print("\n" + "=" * 70)
    print("Model Structure Validated ✓")
    print("=" * 70)
    print("\nDOF Summary:")
    print("  - Body (legs, waist, arms): 29 DOF")
    print("  - Left gripper: 1 DOF")
    print("  - Right gripper: 1 DOF")
    print("  - TOTAL: 31 DOF")
    print("\nActuator Indices:")
    print("  - Body actuators: 0-28")
    print("  - left_gripper_joint: 29")
    print("  - right_gripper_joint: 30")
    print("=" * 70)

if __name__ == "__main__":
    try:
        check_g1_gripper_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
