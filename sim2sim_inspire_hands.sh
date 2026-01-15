#!/bin/bash

# Set LD_LIBRARY_PATH to include conda environment lib directory
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname $(realpath $0))
#Needs a new checkpoint for the inspire hands
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/inspire_robot/g1_29dof_with_inspire_rev_1_0.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
