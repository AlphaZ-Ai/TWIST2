# Quick Reference: G1 with Inspire Grippers

## Model File
```
/home/mcenlaptop/isaacgym/TWIST2/assets/g1/g1_sim2sim_29dof_with_grippers.xml
```

## Key Specs
- **Total DOF**: 31 (29 body + 2 grippers)
- **Gripper Type**: 1-DOF parallel (slider joint)
- **Gripper Range**: 0 to 0.04m (open to closed)
- **Control Mode**: Position control
- **Max Force**: ±10N

## DOF Indices
```python
# Body DOF: 0-35 (same as base model)
LEFT_GRIPPER = 36   # 0=open, 0.04=closed
RIGHT_GRIPPER = 37  # 0=open, 0.04=closed
```

## Usage Example
```python
# Load model
asset = gym.load_asset(sim, "assets/g1", "g1_sim2sim_29dof_with_grippers.xml", options)

# Control grippers
targets = np.zeros(38)  # 7 (pelvis freejoint) + 29 (body) + 2 (grippers)
targets[36] = 0.0    # Left gripper open
targets[37] = 0.02   # Right gripper half-closed

gym.set_actor_dof_position_targets(env, actor, targets)
```

## Retargeting Setup
```python
from robot_control.hand_retargeting import HandType, HandRetargeting

gripper = HandRetargeting(HandType.INSPIRE_GRIPPER)
left_q, right_q = gripper.retarget(hand_keypoints)

targets[36] = left_q[0]
targets[37] = right_q[0]
```

## Joint Names
- `left_gripper_joint` (slider, Z-axis)
- `right_gripper_joint` (slider, Z-axis)

## Link Names  
- `left_gripper_base`, `left_gripper_finger1`, `left_gripper_finger2`
- `right_gripper_base`, `right_gripper_finger1`, `right_gripper_finger2`

## Configuration
- **Retargeting**: `assets/inspire_gripper/inspire_gripper.yml`
- **Hand mapping**: Thumb tip [4] ↔ Index tip [8]
- **Scaling**: 1.2x
- **Smoothing**: alpha=0.2

## Verify Model
```bash
python assets/g1/verify_gripper_model.py
```

## See Also
- Full docs: `assets/g1/README_grippers.md`
- Summary: `assets/inspire_gripper/IMPLEMENTATION_SUMMARY.md`
- Retargeting guide: `assets/inspire_gripper/README.md`
