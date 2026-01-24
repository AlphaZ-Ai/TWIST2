import numpy as np
import threading
import time
from multiprocessing import Process, Array, Lock
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from inspire_sdkpy import inspire_dds
    import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
    INSPIRE_SDK_AVAILABLE = True
except ImportError:
    INSPIRE_SDK_AVAILABLE = False
    print("[WARNING] inspire_sdkpy not available. Running in simulation mode only.")

try:
    from robot_control.hand_retargeting import HandRetargeting, HandType
    DEX_RETARGETING_AVAILABLE = True
except ImportError:
    DEX_RETARGETING_AVAILABLE = False
    print("[WARNING] dex-retargeting not available. Using simple pinch mapping.")


# Inspire FTP DDS topics
kTopicInspireFTPLeftCommand = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"

# Number of motors per Inspire FTP hand
INSPIRE_FTP_NUM_MOTORS = 6

# XRobot hand joint indices (26 joints per hand: 0-25)
# Based on XR hand tracking standard (similar to OpenXR/MediaPipe)
class XRobotHandJoint(IntEnum):
    WRIST = 0
    PALM = 1
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    PINKY_METACARPAL = 21
    PINKY_PROXIMAL = 22
    PINKY_INTERMEDIATE = 23
    PINKY_DISTAL = 24
    PINKY_TIP = 25  # Max index is 25 (0-25), 26 joints total


# Inspire FTP joint indices
class InspireFTPJointIndex(IntEnum):
    PINKY = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB_BEND = 4
    THUMB_ROTATION = 5


@dataclass
class HandTrackingConfig:
    """Configuration for hand tracking behavior"""
    # Scaling factor for retargeting (gripper vs human hand size)
    scaling_factor: float = 1.2
    
    # Low pass filter alpha (0.0 = heavy smoothing, 1.0 = no smoothing)
    low_pass_alpha: float = 0.3
    
    # Finger joint angle ranges for normalization
    # Inspire FTP: output is 0-1000 where 1000=open, 0=closed
    finger_range: Tuple[float, float] = (0.0, 1.7)  # radians for indices 0-3
    thumb_bend_range: Tuple[float, float] = (0.0, 0.5)  # radians
    thumb_rotation_range: Tuple[float, float] = (-0.1, 1.3)  # radians
    
    # Control loop frequency
    fps: float = 100.0
    
    # Whether to use dex-retargeting or simple pinch mapping
    use_dex_retargeting: bool = True


class SimplePinchMapper:
    """
    Simple hand tracking mapper that converts pinch gestures to gripper commands.
    Used as fallback when dex-retargeting is not available.
    """
    
    def __init__(self, config: HandTrackingConfig):
        self.config = config
        self.last_left_output = np.ones(INSPIRE_FTP_NUM_MOTORS)
        self.last_right_output = np.ones(INSPIRE_FTP_NUM_MOTORS)
        
    def compute_finger_curl(self, hand_data: np.ndarray, finger_base_idx: int, finger_tip_idx: int) -> float:
        """
        Compute finger curl amount based on tip-to-base distance.
        Returns value in [0, 1] where 0=straight, 1=fully curled
        """
        if hand_data is None or len(hand_data) < 26:
            return 0.0
            
        base_pos = hand_data[finger_base_idx]
        tip_pos = hand_data[finger_tip_idx]
        palm_pos = hand_data[XRobotHandJoint.PALM]
        
        # Compute distance from tip to palm (normalized by palm-to-base distance)
        palm_to_base = np.linalg.norm(base_pos - palm_pos)
        palm_to_tip = np.linalg.norm(tip_pos - palm_pos)
        
        if palm_to_base < 0.001:
            return 0.0
            
        # When finger is straight, tip is far from palm
        # When finger is curled, tip is close to palm
        curl_ratio = 1.0 - np.clip(palm_to_tip / (palm_to_base * 3.0), 0.0, 1.0)
        
        return curl_ratio
    
    def compute_thumb_pinch(self, hand_data: np.ndarray) -> Tuple[float, float]:
        """
        Compute thumb bend and rotation based on thumb-to-index distance.
        Returns (bend, rotation) both in [0, 1]
        """
        if hand_data is None or len(hand_data) < 26:
            return 0.0, 0.5
            
        thumb_tip = hand_data[XRobotHandJoint.THUMB_TIP]
        index_tip = hand_data[XRobotHandJoint.INDEX_TIP]
        palm = hand_data[XRobotHandJoint.PALM]
        
        # Distance between thumb and index tip
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        palm_size = np.linalg.norm(hand_data[XRobotHandJoint.MIDDLE_METACARPAL] - palm)
        
        if palm_size < 0.001:
            return 0.0, 0.5
            
        # Normalize by palm size
        normalized_dist = pinch_dist / palm_size
        
        # Thumb bend: close when pinching
        bend = 1.0 - np.clip(normalized_dist / 1.5, 0.0, 1.0)
        
        # Thumb rotation: estimate from thumb position relative to palm
        thumb_metacarpal = hand_data[XRobotHandJoint.THUMB_METACARPAL]
        rotation = 0.5  # Default neutral position
        
        return bend, rotation
    
    def retarget(self, hand_data: np.ndarray, is_left: bool = True) -> np.ndarray:
        """
        Convert hand tracking data to Inspire FTP joint commands.
        
        Args:
            hand_data: (26, 3) array of hand joint positions
            is_left: True for left hand, False for right hand
            
        Returns:
            (6,) array of normalized joint commands [0, 1] where 1=open, 0=closed
        """
        output = np.ones(INSPIRE_FTP_NUM_MOTORS)
        
        if hand_data is None or np.all(hand_data == 0.0):
            # Return last valid output or default open
            return self.last_left_output if is_left else self.last_right_output
        
        # Compute finger curls
        output[InspireFTPJointIndex.INDEX] = 1.0 - self.compute_finger_curl(
            hand_data, XRobotHandJoint.INDEX_METACARPAL, XRobotHandJoint.INDEX_TIP)
        output[InspireFTPJointIndex.MIDDLE] = 1.0 - self.compute_finger_curl(
            hand_data, XRobotHandJoint.MIDDLE_METACARPAL, XRobotHandJoint.MIDDLE_TIP)
        output[InspireFTPJointIndex.RING] = 1.0 - self.compute_finger_curl(
            hand_data, XRobotHandJoint.RING_METACARPAL, XRobotHandJoint.RING_TIP)
        output[InspireFTPJointIndex.PINKY] = 1.0 - self.compute_finger_curl(
            hand_data, XRobotHandJoint.PINKY_METACARPAL, XRobotHandJoint.PINKY_TIP)
        
        # Compute thumb
        thumb_bend, thumb_rotation = self.compute_thumb_pinch(hand_data)
        output[InspireFTPJointIndex.THUMB_BEND] = 1.0 - thumb_bend
        output[InspireFTPJointIndex.THUMB_ROTATION] = thumb_rotation
        
        # Apply low pass filtering
        alpha = self.config.low_pass_alpha
        if is_left:
            output = alpha * output + (1 - alpha) * self.last_left_output
            self.last_left_output = output.copy()
        else:
            output = alpha * output + (1 - alpha) * self.last_right_output
            self.last_right_output = output.copy()
        
        return np.clip(output, 0.0, 1.0)


class DexRetargetingMapper:
    """
    Full dex-retargeting mapper using the dex-retargeting library.
    Provides more accurate finger-to-robot mapping.
    """
    
    def __init__(self, config: HandTrackingConfig):
        self.config = config
        
        if not DEX_RETARGETING_AVAILABLE:
            raise ImportError("dex-retargeting not available")
        
        # Initialize retargeting for Inspire Hand (which has same DOF structure as FTP)
        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        
        self.left_indices = self.hand_retargeting.left_indices
        self.right_indices = self.hand_retargeting.right_indices
        self.left_dex_retargeting_to_hardware = self.hand_retargeting.left_dex_retargeting_to_hardware
        self.right_dex_retargeting_to_hardware = self.hand_retargeting.right_dex_retargeting_to_hardware
    
    def normalize_joint(self, val: float, idx: int) -> float:
        """Normalize joint angle to [0, 1] range for Inspire FTP"""
        config = self.config
        
        if idx <= 3:  # Finger joints
            min_val, max_val = config.finger_range
        elif idx == 4:  # Thumb bend
            min_val, max_val = config.thumb_bend_range
        else:  # Thumb rotation
            min_val, max_val = config.thumb_rotation_range
            
        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)
    
    def retarget(self, hand_data: np.ndarray, is_left: bool = True) -> np.ndarray:
        """
        Use dex-retargeting to convert hand tracking to robot joint angles.
        
        Args:
            hand_data: (26, 3) array of hand joint positions
            is_left: True for left hand, False for right hand
            
        Returns:
            (6,) array of normalized joint commands [0, 1] where 1=open, 0=closed
        """
        if hand_data is None or np.all(hand_data == 0.0):
            return np.ones(INSPIRE_FTP_NUM_MOTORS)
        
        if is_left:
            indices = self.left_indices
            retargeting = self.hand_retargeting.left_retargeting
            hardware_indices = self.left_dex_retargeting_to_hardware
        else:
            indices = self.right_indices
            retargeting = self.hand_retargeting.right_retargeting
            hardware_indices = self.right_dex_retargeting_to_hardware
        
        # Compute reference vectors for retargeting
        ref_value = hand_data[indices[1, :]] - hand_data[indices[0, :]]
        
        # Run retargeting optimization
        q_target = retargeting.retarget(ref_value)[hardware_indices]
        
        # Normalize to [0, 1] range
        output = np.zeros(INSPIRE_FTP_NUM_MOTORS)
        for idx in range(INSPIRE_FTP_NUM_MOTORS):
            output[idx] = self.normalize_joint(q_target[idx], idx)
        
        return output


class InspireFTPHandTracker:
    """
    Main class for Inspire FTP hand tracking integration.
    
    Supports two modes:
    1. Controller mode: Simple open/close using VR controller triggers
    2. Hand tracking mode: Full finger tracking using XRobot hand data
    
    Usage:
        # Initialize
        tracker = InspireFTPHandTracker(
            config=HandTrackingConfig(),
            simulation_mode=False,
            use_controller_mode=False
        )
        
        # In your main loop:
        tracker.update(left_hand_data, right_hand_data)
        
        # Or for controller mode:
        tracker.update_from_controller(left_trigger, right_trigger)
    """
    
    def __init__(
        self,
        config: Optional[HandTrackingConfig] = None,
        simulation_mode: bool = False,
        use_controller_mode: bool = False,
        controller_left_value: Optional[Any] = None,
        controller_right_value: Optional[Any] = None,
    ):
        self.config = config or HandTrackingConfig()
        self.simulation_mode = simulation_mode
        self.use_controller_mode = use_controller_mode
        self.controller_left_value = controller_left_value
        self.controller_right_value = controller_right_value
        
        # Initialize state arrays
        self.left_hand_state = np.zeros(INSPIRE_FTP_NUM_MOTORS)
        self.right_hand_state = np.zeros(INSPIRE_FTP_NUM_MOTORS)
        self.left_hand_target = np.ones(INSPIRE_FTP_NUM_MOTORS)
        self.right_hand_target = np.ones(INSPIRE_FTP_NUM_MOTORS)
        
        # Default poses for open/close
        self.left_open_pose = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.left_closed_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.right_open_pose = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.right_closed_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Initialize retargeting mapper
        if not use_controller_mode:
            if self.config.use_dex_retargeting and DEX_RETARGETING_AVAILABLE:
                print("[InspireFTPHandTracker] Using dex-retargeting for hand tracking")
                self.mapper = DexRetargetingMapper(self.config)
            else:
                print("[InspireFTPHandTracker] Using simple pinch mapping for hand tracking")
                self.mapper = SimplePinchMapper(self.config)
        else:
            self.mapper = None
            print("[InspireFTPHandTracker] Using controller mode (no hand tracking)")
        
        # Initialize DDS communication if not in simulation mode
        if not simulation_mode and INSPIRE_SDK_AVAILABLE:
            self._init_dds_communication()
        else:
            self.left_cmd_publisher = None
            self.right_cmd_publisher = None
            self.left_state_subscriber = None
            self.right_state_subscriber = None
            
            if not simulation_mode:
                print("[InspireFTPHandTracker] WARNING: Running without inspire_sdkpy - DDS disabled")
        
        # Start state subscription thread
        self._running = True
        if not simulation_mode and INSPIRE_SDK_AVAILABLE:
            self._subscribe_thread = threading.Thread(target=self._subscribe_hand_state)
            self._subscribe_thread.daemon = True
            self._subscribe_thread.start()
    
    def _init_dds_communication(self):
        """Initialize DDS publishers and subscribers"""
        print("[InspireFTPHandTracker] Initializing DDS communication...")
        
        # Note: ChannelFactory should already be initialized by XRobotStreamer
        # Do NOT call ChannelFactoryInitialize again as it will fail
        
        # Command publishers
        self.left_cmd_publisher = ChannelPublisher(
            kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.left_cmd_publisher.Init()
        
        self.right_cmd_publisher = ChannelPublisher(
            kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.right_cmd_publisher.Init()
        
        # State subscribers
        self.left_state_subscriber = ChannelSubscriber(
            kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.left_state_subscriber.Init()
        
        self.right_state_subscriber = ChannelSubscriber(
            kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.right_state_subscriber.Init()
        
        print("[InspireFTPHandTracker] DDS communication initialized")
    
    def _subscribe_hand_state(self):
        """Background thread to subscribe to hand state updates"""
        while self._running:
            if self.left_state_subscriber is not None:
                left_msg = self.left_state_subscriber.Read()
                if left_msg is not None and hasattr(left_msg, 'angle_act'):
                    if len(left_msg.angle_act) == INSPIRE_FTP_NUM_MOTORS:
                        for i in range(INSPIRE_FTP_NUM_MOTORS):
                            self.left_hand_state[i] = left_msg.angle_act[i] / 1000.0
            
            if self.right_state_subscriber is not None:
                right_msg = self.right_state_subscriber.Read()
                if right_msg is not None and hasattr(right_msg, 'angle_act'):
                    if len(right_msg.angle_act) == INSPIRE_FTP_NUM_MOTORS:
                        for i in range(INSPIRE_FTP_NUM_MOTORS):
                            self.right_hand_state[i] = right_msg.angle_act[i] / 1000.0
            
            time.sleep(0.002)  # 500Hz polling
    
    def update_from_tracking(
        self,
        left_hand_data: Optional[np.ndarray],
        right_hand_data: Optional[np.ndarray]
    ):
        """
        Update hand targets from XRobot hand tracking data.
        
        Args:
            left_hand_data: (26, 3) array of left hand joint positions, or None
            right_hand_data: (26, 3) array of right hand joint positions, or None
        """
        if self.mapper is None:
            return
        
        if left_hand_data is not None and not np.all(left_hand_data == 0.0):
            self.left_hand_target = self.mapper.retarget(left_hand_data, is_left=True)
        
        if right_hand_data is not None and not np.all(right_hand_data == 0.0):
            self.right_hand_target = self.mapper.retarget(right_hand_data, is_left=False)
        
        # Send commands
        self._send_commands()
    
    def update_from_controller(
        self,
        left_trigger: float = 0.0,
        right_trigger: float = 0.0
    ):
        """
        Update hand targets from VR controller triggers.
        
        Args:
            left_trigger: Left trigger value [0, 1] where 0=released, 1=pressed
            right_trigger: Right trigger value [0, 1] where 0=released, 1=pressed
        """
        # Interpolate between open and closed poses
        self.left_hand_target = (
            self.left_open_pose * (1.0 - left_trigger) + 
            self.left_closed_pose * left_trigger
        )
        self.right_hand_target = (
            self.right_open_pose * (1.0 - right_trigger) + 
            self.right_closed_pose * right_trigger
        )
        
        # Send commands
        self._send_commands()
    
    def _send_commands(self):
        """Send current targets to the Inspire FTP hands"""
        if self.simulation_mode:
            return
        
        # Scale to 0-1000 range for Inspire FTP
        left_cmd_scaled = [int(np.clip(val * 1000, 0, 1000)) for val in self.left_hand_target]
        right_cmd_scaled = [int(np.clip(val * 1000, 0, 1000)) for val in self.right_hand_target]
        
        # Debug print
        if not hasattr(self, '_dds_debug_count'):
            self._dds_debug_count = 0
        self._dds_debug_count += 1
        if self._dds_debug_count % 100 == 0:
            print(f"[FTP DDS] Sending L={left_cmd_scaled} R={right_cmd_scaled}")
        
        # Send left hand command
        if self.left_cmd_publisher is not None:
            left_msg = inspire_hand_default.get_inspire_hand_ctrl()
            left_msg.angle_set = left_cmd_scaled
            left_msg.mode = 0b0001  # Angle control mode
            self.left_cmd_publisher.Write(left_msg)
        
        # Send right hand command
        if self.right_cmd_publisher is not None:
            right_msg = inspire_hand_default.get_inspire_hand_ctrl()
            right_msg.angle_set = right_cmd_scaled
            right_msg.mode = 0b0001  # Angle control mode
            self.right_cmd_publisher.Write(right_msg)
    
    def get_hand_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current hand states"""
        return self.left_hand_state.copy(), self.right_hand_state.copy()
    
    def get_hand_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current hand targets"""
        return self.left_hand_target.copy(), self.right_hand_target.copy()
    
    def close(self):
        """Clean shutdown"""
        self._running = False
        if hasattr(self, '_subscribe_thread'):
            self._subscribe_thread.join(timeout=1.0)
        print("[InspireFTPHandTracker] Closed")


def convert_xrobot_hand_dict_to_array(hand_dict: Dict[str, Any], hand_prefix: str = "Left") -> np.ndarray:
    """
    Convert XRobot hand data dictionary to numpy array.
    
    XRobot returns hand data as dict with keys like "LeftHandWrist", "LeftHandThumbTip", etc.
    This function converts it to a (26, 3) numpy array matching the XRobotHandJoint order.
    
    XRobot hand joint order (26 joints total):
    - Wrist, Palm (2)
    - Thumb: Metacarpal, Proximal, Distal, Tip (4)
    - Index: Metacarpal, Proximal, Intermediate, Distal, Tip (5)
    - Middle: Metacarpal, Proximal, Intermediate, Distal, Tip (5)
    - Ring: Metacarpal, Proximal, Intermediate, Distal, Tip (5)
    - Little: Metacarpal, Proximal, Intermediate, Distal, Tip (5)
    
    Args:
        hand_dict: Dictionary from XRobotStreamer.get_left_hand_data() or get_right_hand_data()
        hand_prefix: "Left" or "Right"
        
    Returns:
        (26, 3) numpy array of joint positions
    """
    joint_order = [
        "Wrist", "Palm",
        "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
        "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
        "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
        "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
        "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
    ]
    
    result = np.zeros((26, 3))
    
    for i, joint_name in enumerate(joint_order):
        key = f"{hand_prefix}Hand{joint_name}"
        if key in hand_dict:
            pos = hand_dict[key][0]  # [position, rotation]
            result[i] = np.array(pos)
    
    return result


if __name__ == "__main__":
    # Test the hand tracker
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--controller', action='store_true', help='Use controller mode instead of hand tracking')
    args = parser.parse_args()
    
    print("Testing InspireFTPHandTracker...")
    
    config = HandTrackingConfig(
        use_dex_retargeting=True,
        low_pass_alpha=0.3,
        fps=100.0
    )
    
    tracker = InspireFTPHandTracker(
        config=config,
        simulation_mode=args.simulation,
        use_controller_mode=args.controller
    )
    
    try:
        if args.controller:
            print("Testing controller mode...")
            for i in range(50):
                trigger = i / 50.0
                tracker.update_from_controller(trigger, trigger)
                left_state, right_state = tracker.get_hand_state()
                left_target, right_target = tracker.get_hand_target()
                print(f"Trigger: {trigger:.2f} | L Target: {left_target[0]:.2f} | R Target: {right_target[0]:.2f}")
                time.sleep(0.05)
        else:
            print("Testing hand tracking mode with synthetic data...")
            # Create some synthetic hand data
            for i in range(50):
                # Simulate hand opening and closing
                t = i / 50.0
                left_hand = np.zeros((26, 3))
                right_hand = np.zeros((26, 3))
                
                # Set some basic positions
                left_hand[XRobotHandJoint.PALM] = [0, 0, 0]
                left_hand[XRobotHandJoint.INDEX_TIP] = [0.1 - t * 0.1, 0.1, 0]
                left_hand[XRobotHandJoint.THUMB_TIP] = [t * 0.1, 0.1, 0]
                
                right_hand[XRobotHandJoint.PALM] = [0, 0, 0]
                right_hand[XRobotHandJoint.INDEX_TIP] = [0.1 - t * 0.1, 0.1, 0]
                right_hand[XRobotHandJoint.THUMB_TIP] = [t * 0.1, 0.1, 0]
                
                tracker.update_from_tracking(left_hand, right_hand)
                left_target, right_target = tracker.get_hand_target()
                print(f"Step {i}: L Target: {left_target} | R Target: {right_target}")
                time.sleep(0.05)
                
    finally:
        tracker.close()
        print("Test complete!")
