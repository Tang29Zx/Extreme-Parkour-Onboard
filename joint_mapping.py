"""Joint and foot ordering at the Parkour policy/Unitree boundary."""

from typing import Sequence, Tuple


NUM_DOF = 12

SIM_DOF_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

# Index by policy/simulation joint; value is the matching Unitree motor index.
SIM_TO_REAL_DOF = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
DOF_SIGNS = (1.0,) * NUM_DOF

# Unitree foot order FR/FL/RR/RL -> policy order FL/FR/RL/RR.
FOOT_REAL_TO_SIM = (1, 0, 3, 2)


def _vector(values: Sequence[float], size: int, name: str) -> Tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(values)}")
    return tuple(float(value) for value in values)


def sim_to_real(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert a policy-order joint vector to Unitree motor order."""
    simulation = _vector(values, NUM_DOF, "simulation joint vector")
    real = [0.0] * NUM_DOF
    for sim_index, real_index in enumerate(SIM_TO_REAL_DOF):
        real[real_index] = simulation[sim_index] * DOF_SIGNS[sim_index]
    return tuple(real)


def real_to_sim(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert a Unitree motor-order joint vector to policy order."""
    real = _vector(values, NUM_DOF, "Unitree joint vector")
    return tuple(
        real[real_index] * DOF_SIGNS[sim_index]
        for sim_index, real_index in enumerate(SIM_TO_REAL_DOF)
    )


def foot_real_to_sim(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert Unitree FR/FL/RR/RL foot values to FL/FR/RL/RR."""
    real = _vector(values, 4, "Unitree foot vector")
    return tuple(real[index] for index in FOOT_REAL_TO_SIM)
