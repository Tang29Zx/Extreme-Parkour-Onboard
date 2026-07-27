"""Message-independent Unitree LowState/LowCmd policy boundary."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

from joint_mapping import (
    SIM_TO_REAL_DOF,
    foot_real_to_sim,
    real_to_sim,
    sim_to_real,
)
from real_control_safety import RealControlError, filter_foot_contacts


GO2_JOINT_LIMITS_HIGH = np.asarray(
    [
        1.0472, 3.4907, -0.83776,
        1.0472, 3.4907, -0.83776,
        1.0472, 4.5379, -0.83776,
        1.0472, 4.5379, -0.83776,
    ],
    dtype=np.float64,
)
GO2_JOINT_LIMITS_LOW = np.asarray(
    [
        -1.0472, -1.5708, -2.7227,
        -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1.0472, -0.5236, -2.7227,
    ],
    dtype=np.float64,
)
GO2_TORQUE_LIMITS = np.asarray(
    [23.7, 23.7, 35.55] * 4,
    dtype=np.float64,
)
GO2_JOINT_VELOCITY_LIMITS = np.asarray(
    [30.1, 30.1, 20.07] * 4,
    dtype=np.float64,
)


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RealControlError(f"{name} must contain {size} finite values")
    return result.copy()


@dataclass(frozen=True)
class BoundaryLowState:
    """Unitree LowState fields consumed by the parkour policy."""

    motor_q: Sequence[float]
    motor_dq: Sequence[float]
    foot_force: Sequence[float]
    gyroscope: Sequence[float]
    imu_quaternion_wxyz: Sequence[float]


@dataclass(frozen=True)
class DecodedLowState:
    """LowState values expressed at the actor boundary."""

    joint_q: np.ndarray
    joint_dq: np.ndarray
    foot_force: np.ndarray
    gyroscope: np.ndarray
    roll_pitch: np.ndarray


@dataclass(frozen=True)
class BoundaryLowCmd:
    """Unitree LowCmd motor fields in FR/FL/RR/RL order."""

    motor_q: np.ndarray
    motor_dq: np.ndarray
    motor_tau: np.ndarray
    motor_kp: np.ndarray
    motor_kd: np.ndarray


def quaternion_wxyz_to_roll_pitch(
    quaternion_wxyz: Sequence[float],
) -> np.ndarray:
    """Convert a Unitree wxyz quaternion to roll and pitch in radians."""
    quaternion = _finite_vector(
        quaternion_wxyz,
        4,
        "IMU quaternion",
    )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RealControlError("IMU quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    return np.asarray([roll, pitch], dtype=np.float64)


def decode_low_state(state: BoundaryLowState) -> DecodedLowState:
    """Decode Unitree LowState arrays into actor-order values."""
    motor_q = _finite_vector(state.motor_q, 12, "LowState motor position")
    motor_dq = _finite_vector(state.motor_dq, 12, "LowState motor velocity")
    foot_force = _finite_vector(state.foot_force, 4, "LowState foot force")
    gyroscope = _finite_vector(state.gyroscope, 3, "LowState gyroscope")
    return DecodedLowState(
        joint_q=np.asarray(real_to_sim(motor_q), dtype=np.float64),
        joint_dq=np.asarray(real_to_sim(motor_dq), dtype=np.float64),
        foot_force=np.asarray(foot_real_to_sim(foot_force), dtype=np.float64),
        gyroscope=gyroscope,
        roll_pitch=quaternion_wxyz_to_roll_pitch(
            state.imu_quaternion_wxyz
        ),
    )


def _policy_to_motor_unsigned(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    policy = _finite_vector(values, 12, name)
    # Kp/Kd are reordered like joints but do not use joint direction signs.
    unitree = np.zeros(12, dtype=np.float64)
    for policy_index, motor_index in enumerate(SIM_TO_REAL_DOF):
        unitree[motor_index] = policy[policy_index]
    return unitree


def encode_low_cmd(
    target_q: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
) -> BoundaryLowCmd:
    """Encode one actor-order position target as Unitree LowCmd fields."""
    target = _finite_vector(target_q, 12, "LowCmd target")
    p_gain = _finite_vector(kp, 12, "LowCmd Kp")
    d_gain = _finite_vector(kd, 12, "LowCmd Kd")
    if bool(np.any(p_gain <= 0.0)):
        raise RealControlError("LowCmd Kp must be positive")
    if bool(np.any(d_gain < 0.0)):
        raise RealControlError("LowCmd Kd must be non-negative")
    return BoundaryLowCmd(
        motor_q=np.asarray(sim_to_real(target), dtype=np.float64),
        motor_dq=np.zeros(12, dtype=np.float64),
        motor_tau=np.zeros(12, dtype=np.float64),
        motor_kp=_policy_to_motor_unsigned(p_gain, "LowCmd Kp"),
        motor_kd=_policy_to_motor_unsigned(d_gain, "LowCmd Kd"),
    )


def build_policy_proprio(
    state: DecodedLowState,
    default_q: Sequence[float],
    last_action: Sequence[float],
    previous_contacts: Sequence[bool],
    command_forward_mps: float,
    ang_vel_scale: float,
    dof_pos_scale: float,
    dof_vel_scale: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 53-value production proprioception and current contacts."""
    default = _finite_vector(default_q, 12, "default joint position")
    action = _finite_vector(last_action, 12, "last actor action")
    previous = np.asarray(previous_contacts, dtype=np.bool_)
    if previous.shape != (4,):
        raise RealControlError("previous contacts must contain 4 values")
    command = float(command_forward_mps)
    scales = (
        float(ang_vel_scale),
        float(dof_pos_scale),
        float(dof_vel_scale),
    )
    if not math.isfinite(command) or not all(math.isfinite(value) for value in scales):
        raise RealControlError("policy command and observation scales must be finite")
    if mode == "parkour":
        skill = np.asarray([1.0, 0.0], dtype=np.float64)
    elif mode == "walk":
        skill = np.asarray([0.0, 1.0], dtype=np.float64)
    else:
        raise RealControlError(f"unsupported policy mode '{mode}'")

    filtered_contacts, current_contacts = filter_foot_contacts(
        state.foot_force,
        previous,
    )
    contact_observation = np.where(
        filtered_contacts,
        0.5,
        -0.5,
    ).astype(np.float64)
    proprio = np.concatenate(
        (
            state.gyroscope * scales[0],
            state.roll_pitch,
            np.zeros(3, dtype=np.float64),
            np.asarray([0.0, 0.0, command], dtype=np.float64),
            skill,
            (state.joint_q - default) * scales[1],
            state.joint_dq * scales[2],
            action,
            contact_observation,
        )
    )
    if proprio.shape != (53,) or not np.isfinite(proprio).all():
        raise RealControlError("policy proprioception must contain 53 finite values")
    return proprio.astype(np.float32), current_contacts.astype(np.bool_)
