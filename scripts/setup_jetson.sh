#!/usr/bin/env bash
# Install bundled Python packages and verify the complete onboard runtime.

# ROS 2 Foxy setup scripts read optional unset variables, so nounset is unsafe here.
set -eo pipefail

setup_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${setup_repo_dir}/scripts/jetson_env.sh"

python3 -m pip install --no-deps -e "${setup_repo_dir}/rsl_rl"

python3 - <<'PY'
import numpy
import pyrealsense2
import rclpy
import torch
from rsl_rl.modules import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from unitree_api.msg import Request
from unitree_go.msg import LowCmd, LowState, WirelessController

import run_extreme_parkour
import unitree_ros2_real
import visual_extreme_parkour

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the selected Python environment.")

print("Jetson runtime check: OK")
print(f"Python Torch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"rclpy: {rclpy.__file__}")
print(f"pyrealsense2: {pyrealsense2.__file__}")
PY
