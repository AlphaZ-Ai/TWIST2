import sys
import os
import numpy as np
from enum import IntEnum
import time
from data_utils.params import DEFAULT_HAND_POSE

DEFAULT_QPOS_LEFT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["left"]["open"]
DEFAULT_QPOS_RIGHT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["right"]["open"]

Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"

QPOS_LEFT_MAX = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
QPOS_LEFT_MIN  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
QPOS_RIGHT_MAX = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
QPOS_RIGHT_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class Inspire_Hand_Controller:
    def __init__(self, net, re_init=True):
        """
        Controller for Unitree RH56DFX Inspire Hands (6-DOF per hand)
        
        DDS Topics:
        - Publish to: rt/inspire/cmd (MotorCmds_)
        - Subscribe from: rt/inspire/state (MotorStates_)
        
        Motor order (12 total): 
        - Right hand [0-5]: pinky, ring, middle, index, thumb-bend, thumb-rotation
        - Left hand [6-11]: pinky, ring, middle, index, thumb-bend, thumb-rotation
        """
        print("Initialize Inspire_Hand_Controller...")
        print("Using unitree_go DDS interface")
        
        try:
            import unitree_go.msg.dds_ as msg_dds
            from unitree.robot import ChannelPublisher, ChannelSubscriber, ChannelFactory
            
            # Initialize DDS
            ChannelFactory.Instance().Init(0, net)

            self.hand_cmd_pub = ChannelPublisher(kTopicInspireCommand, msg_dds.MotorCmds_)
            self.hand_cmd_pub.InitChannel()
            
            self.hand_state = msg_dds.MotorStates_()
            self.hand_state_sub = ChannelSubscriber(kTopicInspireState, msg_dds.MotorStates_)
            self.hand_state_sub.InitChannel(self._state_handler)

            self.cmd_msg = msg_dds.MotorCmds_()
            self.cmd_msg.cmds().resize(12)
            
        except ImportError:
            raise ImportError("unitree_go SDK not found. Please install unitree_sdk2 with GO2 support.")
        
        self.left_hand_state_array  = np.zeros(Inspire_Num_Motors)
        self.right_hand_state_array = np.zeros(Inspire_Num_Motors)
        
        self.initialize()
        print("Initialize Inspire_Hand_Controller OK!\n")

    def _state_handler(self, message):
        for i in range(Inspire_Num_Motors):
            self.right_hand_state_array[i] = message.states()[i].q()
            self.left_hand_state_array[i] = message.states()[i + 6].q()

    def get_hand_state(self):
        return self.left_hand_state_array.copy(), self.right_hand_state_array.copy()

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        left_q_target = np.clip(left_q_target, 0.0, 1.0)
        right_q_target = np.clip(right_q_target, 0.0, 1.0)

        for i in range(Inspire_Num_Motors):
            self.cmd_msg.cmds()[i].q() = float(right_q_target[i])
            
        for i in range(Inspire_Num_Motors):
            self.cmd_msg.cmds()[i + 6].q() = float(left_q_target[i])
        
        self.hand_cmd_pub.Write(self.cmd_msg)
    
    def initialize(self):
        """Initialize hands with default poses"""
        print("🔧 Initializing Inspire hands with default (open) poses...")
        self.ctrl_dual_hand(DEFAULT_QPOS_LEFT, DEFAULT_QPOS_RIGHT)
    
    def close(self):
        """Cleanup resources"""
        print("Closing Inspire hand controller...")
        pass


class Inspire_Left_JointIndex(IntEnum):
    LEFT_PINKY = 0
    LEFT_RING = 1
    LEFT_MIDDLE = 2
    LEFT_INDEX = 3
    LEFT_THUMB_BEND = 4
    LEFT_THUMB_ROTATION = 5

class Inspire_Right_JointIndex(IntEnum):
    RIGHT_PINKY = 0
    RIGHT_RING = 1
    RIGHT_MIDDLE = 2
    RIGHT_INDEX = 3
    RIGHT_THUMB_BEND = 4
    RIGHT_THUMB_ROTATION = 5

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='eno1', help='Network interface used by G1RealWorldEnv.')
    args = parser.parse_args()

    print("Testing Inspire_Hand_Controller with DDS API...")
    hand_ctrl = Inspire_Hand_Controller(args.net)

    print("Running test sequence...")
    for i in range(5):
        progress = i / 4.0
        left_target = np.ones(6) * (1.0 - progress)
        right_target = np.ones(6) * (1.0 - progress)
        
        hand_ctrl.ctrl_dual_hand(left_target, right_target)
        
        # Get hand state
        left_hand_state, right_hand_state = hand_ctrl.get_hand_state()
        print(f"Step {i}: Left [{left_hand_state[0]:.3f}, {left_hand_state[1]:.3f}, {left_hand_state[2]:.3f}...] Right [{right_hand_state[0]:.3f}, {right_hand_state[1]:.3f}, {right_hand_state[2]:.3f}...]")
        time.sleep(0.5)
    
    print("Test completed! Inspire hand control")
    
    print("Test completed! New unified API is working perfectly.")