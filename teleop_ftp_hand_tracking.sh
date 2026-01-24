#!/bin/bash
# Teleop with Inspire FTP Hand Tracking
# 
# This script runs the teleop system with full finger tracking for the Inspire FTP gripper.
# It uses XRobot hand tracking data and dex-retargeting for individual finger control.
#
# Prerequisites:
# 1. XRobot PC Service must be running
# 2. PICO headset connected and streaming hand data
# 3. Redis server running
# 4. sim2real.sh or sim2sim.sh running in another terminal

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gmr

cd deploy_real

export PYTHONPATH="${PYTHONPATH}:/home/mcenlaptop/isaacgym/TWIST2/inspire_hand_ws/inspire_hand_sdk"

# Configuration
redis_ip="localhost"
actual_human_height=1.7272

# Run teleop with Inspire FTP hand tracking enabled
python xrobot_teleop_ftp_hand_tracking.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 1 \
    --hand_tracking \
    --use_dex_retargeting \
    --hand_smooth_alpha 0.3
    # Optional flags:
    # --smooth                   # Enable body motion smoothing
    # --pinch_mode               # Use pinch mode for non-tracked hands
    # --record_video             # Record video output
