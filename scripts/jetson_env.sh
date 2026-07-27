#!/usr/bin/env bash
# Load the Jetson ROS 2, Unitree message, CUDA Python, and project environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead: source scripts/jetson_env.sh" >&2
    exit 2
fi

extreme_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required_setups=(
    "/opt/ros/foxy/setup.bash"
    "${HOME}/cyclonedds_ws/install/setup.bash"
    "${HOME}/unitree_msgs_ws/install/setup.bash"
    "${HOME}/parkour/parkour_venv/bin/activate"
)

for setup_file in "${required_setups[@]}"; do
    if [[ ! -r "${setup_file}" ]]; then
        echo "Missing required environment file: ${setup_file}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${setup_file}"
done

export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export CYCLONEDDS_URI="${HOME}/cyclonedds_ws/cyclonedds.xml"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export PYTHONPATH="${extreme_repo_dir}/rsl_rl:${extreme_repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

unset extreme_repo_dir setup_file required_setups
