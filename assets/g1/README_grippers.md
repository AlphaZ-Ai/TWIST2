# G1 Sim2Sim 29DOF with Inspire Grippers

This URDF file defines a G1 humanoid robot (29 DOF body) with simple Inspire parallel grippers instead of dexterous hands.

## File Details

- **Filename**: `g1_sim2sim_29dof_with_grippers.xml`
- **Location**: `/home/mcenlaptop/isaacgym/TWIST2/assets/g1/`
- **Total DOF**: 31 (29 body + 2 grippers)

## Changes from Base g1_sim2sim_29dof.xml

### Added Components

1. **Left Gripper** (after left_wrist_yaw_link):
   - `left_gripper_base` - Gripper mounting base
   - `left_gripper_finger1` - Fixed finger
   - `left_gripper_finger2` - Actuated finger (slider joint)
   - `left_gripper_joint` - Prismatic joint (0 to 0.04m range)

2. **Right Gripper** (after right_wrist_yaw_link):
   - `right_gripper_base` - Gripper mounting base
   - `right_gripper_finger1` - Fixed finger
   - `right_gripper_finger2` - Actuated finger (slider joint)
   - `right_gripper_joint` - Prismatic joint (0 to 0.04m range)

### Gripper Specifications

- **Type**: Parallel gripper with slider joint
- **Range**: 0 to 0.04m (fully open to fully closed)
- **Actuation**: Position-controlled
- **Force Range**: -10 to 10 N
- **Mass**: 
  - Base: 0.15 kg
  - Each finger: 0.02 kg
- **Geometry**: 
  - Base: 15mm x 20mm x 15mm box
  - Fingers: 20mm x 8mm x 5mm boxes with 8mm spherical tips

### Actuator Configuration

Added two position-controlled actuators:
- `left_gripper_joint` (actuator index 29)
- `right_gripper_joint` (actuator index 30)

### Updated Keyframe

The "home" keyframe now includes two additional values (38 total qpos values):
- Previous: 36 values (29 DOF body)
- New: 38 values (29 DOF body + 2 grippers)
- Last two values are gripper positions (both initialized to 0 = open)

## DOF Index Mapping

### Body (0-28): Same as g1_sim2sim_29dof.xml
```
0-6:   Pelvis (freejoint: x, y, z, qw, qx, qy, qz)
7-12:  Left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
13-18: Right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
19-21: Waist (yaw, roll, pitch)
22-28: Left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
29-35: Right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
```

### Grippers (29-30): NEW
```
36: left_gripper_joint (0 = open, 0.04 = closed)
37: right_gripper_joint (0 = open, 0.04 = closed)
```

## Usage in Code

### Loading the Model

```python
from isaacgym import gymapi

# Create gym
gym = gymapi.acquire_gym()

# Load asset
asset_root = "assets/g1"
asset_file = "g1_sim2sim_29dof_with_grippers.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
```

### Controlling the Grippers

```python
# Get DOF properties
dof_props = gym.get_actor_dof_properties(env, actor_handle)

# Set gripper targets (position control)
left_gripper_idx = 36  # After 29 body DOF
right_gripper_idx = 37

# Open grippers
dof_props['driveMode'][left_gripper_idx] = gymapi.DOF_MODE_POS
dof_props['driveMode'][right_gripper_idx] = gymapi.DOF_MODE_POS
dof_props['stiffness'][left_gripper_idx] = 1000
dof_props['stiffness'][right_gripper_idx] = 1000
dof_props['damping'][left_gripper_idx] = 100
dof_props['damping'][right_gripper_idx] = 100

gym.set_actor_dof_properties(env, actor_handle, dof_props)

# Set target positions
targets[left_gripper_idx] = 0.0  # Open
targets[right_gripper_idx] = 0.0  # Open

gym.set_actor_dof_position_targets(env, actor_handle, targets)
```

### Integration with Hand Retargeting

To use this with the inspire gripper retargeting config:

```python
from robot_control.hand_retargeting import HandType, HandRetargeting

# Initialize gripper retargeting
gripper_retarget = HandRetargeting(HandType.INSPIRE_GRIPPER)

# Retarget hand poses to gripper commands
left_qpos, right_qpos = gripper_retarget.retarget(hand_keypoints)

# Apply to simulation
targets[36] = left_qpos[0]   # Left gripper
targets[37] = right_qpos[0]  # Right gripper
```

## Differences from Other G1 Models

| Model | Body DOF | End Effector | Total DOF | Actuators |
|-------|----------|--------------|-----------|-----------|
| g1_sim2sim_29dof.xml | 29 | Rubber hand (fixed) | 29 | 29 |
| g1_sim2sim_29dof_with_hands.xml | 29 | Dexterous hands (7DOF each) | 43 | 43 |
| **g1_sim2sim_29dof_with_grippers.xml** | **29** | **Parallel grippers (1DOF each)** | **31** | **31** |

## Gripper Coordinate Frame

The grippers are attached to the wrist_yaw_link with the following offsets:
- **Left gripper**: pos="0.12 0.003 0"
- **Right gripper**: pos="0.12 -0.003 0"

The gripper fingers move along the Z-axis:
- Z+ direction = closing
- Z- direction = opening
- Range: 0 (fully open) to 0.04m (fully closed)

## Visual Appearance

The grippers are rendered using basic geometric primitives:
- **Base**: Dark gray box (0.3, 0.3, 0.3)
- **Fingers**: Medium gray boxes (0.5, 0.5, 0.5)
- **Fingertips**: Light gray spheres (0.6, 0.6, 0.6)

## Notes

- This is a simplified gripper model for simulation purposes
- For more accurate gripper dynamics, you may need to adjust mass, inertia, and friction properties
- The gripper geometry is purely geometric (boxes and spheres) - no mesh files required
- Compatible with all G1 training configurations that use 29 DOF body control

## Related Files

- Base model: [g1_sim2sim_29dof.xml](g1_sim2sim_29dof.xml)
- With dexterous hands: [g1_sim2sim_29dof_with_hands.xml](g1_sim2sim_29dof_with_hands.xml)
- Gripper retargeting config: [../inspire_gripper/inspire_gripper.yml](../../inspire_gripper/inspire_gripper.yml)
