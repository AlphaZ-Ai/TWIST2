# Inspire Gripper Retargeting Configuration

This directory contains the retargeting configuration for the Inspire parallel gripper (1-DOF per gripper).

## Overview

The Inspire gripper is a simple parallel gripper with only 1 degree of freedom per side, unlike the Inspire Hand which has 6 DOF per hand. This configuration maps human hand pinch gestures (thumb-to-index finger) to gripper open/close commands.

## Required Files

To use this configuration, you need to provide the URDF files for the Inspire gripper:

1. **`inspire_gripper_left.urdf`** - URDF model for left gripper
2. **`inspire_gripper_right.urdf`** - URDF model for right gripper

### URDF Requirements

Your URDF files should include:

- **Joint name**: The main actuated joint should be named `left_gripper_joint` and `right_gripper_joint` respectively
- **Link names**: 
  - `left_gripper_base` / `right_gripper_base` - Base link of the gripper
  - `left_gripper_finger1` / `right_gripper_finger1` - First finger link
  - `left_gripper_finger2` / `right_gripper_finger2` - Second finger link

If your URDF uses different names, update the `inspire_gripper.yml` file accordingly.

## Configuration Details

### Retargeting Type: Vector

The configuration uses **vector retargeting**, which works by measuring the distance between two links:
- **Origin link**: `left_gripper_base` (fixed base)
- **Task link**: `left_gripper_finger2` (moving finger)

### Human Hand Mapping

The gripper maps the human hand pinch gesture:
- **Index 4**: Thumb tip
- **Index 8**: Index finger tip

When you pinch these fingers together in VR/mocap, the gripper closes. When you open them, the gripper opens.

### Parameters

- **`scaling_factor: 1.2`** - Adjust if gripper size differs from human hand scale
- **`low_pass_alpha: 0.2`** - Smoothing filter (0.1 = heavy smoothing, 1.0 = no smoothing)

## Usage

### In Python Code

```python
from robot_control.hand_retargeting import HandType, HandRetargeting

# Initialize gripper retargeting
gripper_retarget = HandRetargeting(HandType.INSPIRE_GRIPPER)

# Get retargeted positions
left_qpos, right_qpos = gripper_retarget.retarget(hand_keypoints)
```

### Joint Limits

Based on `params.py`, the Inspire gripper has the following limits:
- **Open**: 0.0 rad
- **Closed**: 0.85 rad (~48.7 degrees)
- **Half**: 0.4 rad (intermediate position)

## Customization

If you need to adjust the configuration:

1. **Different joint names**: Update `target_joint_names` in `inspire_gripper.yml`
2. **Different link names**: Update `target_origin_link_names` and `target_task_link_names`
3. **Different finger mapping**: Update `target_link_human_indices` (see MediaPipe hand landmark indices)
4. **Gripper sensitivity**: Adjust `scaling_factor`

## MediaPipe Hand Landmarks Reference

```
0  = Wrist
1-4   = Thumb (1=CMC, 2=MCP, 3=IP, 4=Tip)
5-8   = Index (5=MCP, 6=PIP, 7=DIP, 8=Tip)
9-12  = Middle (9=MCP, 10=PIP, 11=DIP, 12=Tip)
13-16 = Ring (13=MCP, 14=PIP, 15=DIP, 16=Tip)
17-20 = Pinky (17=MCP, 18=PIP, 19=DIP, 20=Tip)
```

Current mapping uses:
- **[4]**: Thumb tip
- **[8]**: Index finger tip

You can change this to use different fingers (e.g., thumb-to-middle: `[4]` and `[12]`).

## Differences from Inspire Hand

| Feature | Inspire Hand | Inspire Gripper |
|---------|-------------|-----------------|
| DOF per side | 6 | 1 |
| Retargeting type | DexPilot | Vector |
| Joint count | 6 joints per hand | 1 joint per gripper |
| Complexity | Full finger control | Simple open/close |
| Config file | `inspire_hand.yml` | `inspire_gripper.yml` |
| HandType enum | `INSPIRE_HAND` | `INSPIRE_GRIPPER` |

## Troubleshooting

1. **"URDF not found" error**: Make sure your URDF files are in this directory
2. **"Joint name not found" error**: Update joint names in YAML to match your URDF
3. **Gripper not responding**: Check scaling_factor and low_pass_alpha values
4. **Inverted motion**: Swap the indices in `target_link_human_indices`
