

source ~/miniforge3/etc/profile.d/conda.sh
conda activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=eno1

# Hand type: "dex3" for Dex3 7-DOF hands, "ftp" for Inspire FTP 6-DOF hands
hand_type=ftp

# Add Inspire SDK to PYTHONPATH for FTP hand support
export PYTHONPATH="${SCRIPT_DIR}/inspire_hand_ws/inspire_hand_sdk:$PYTHONPATH"

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hand_type ${hand_type}
    # --smooth_body 0.5
    # --record_proprio \
