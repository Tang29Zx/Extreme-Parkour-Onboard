"""Joint and foot ordering at the Parkour policy/Unitree boundary."""

from typing import Sequence, Tuple


NUM_DOF = 12

ISAAC_DOF_NAMES = (
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

# Training reindexes Isaac Gym vectors before they cross the actor boundary.
ISAAC_TO_POLICY_DOF = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
POLICY_DOF_NAMES = tuple(ISAAC_DOF_NAMES[index] for index in ISAAC_TO_POLICY_DOF)
POLICY_TO_ISAAC_DOF = tuple(
    ISAAC_TO_POLICY_DOF.index(index) for index in range(NUM_DOF)
)

# Unitree motor state and LowCmd use the same FR/FL/RR/RL order as the actor.
POLICY_TO_REAL_DOF = tuple(range(NUM_DOF))
DOF_SIGNS = (1.0,) * NUM_DOF

# Unitree foot force and actor contact use the same FR/FL/RR/RL order.
FOOT_REAL_TO_POLICY = tuple(range(4))
ISAAC_TO_UNITREE_FOOT = (1, 0, 3, 2)
UNITREE_TO_ISAAC_FOOT = tuple(
    ISAAC_TO_UNITREE_FOOT.index(index) for index in range(4)
)

# Backwards-compatible names used by the current control code.
SIM_DOF_NAMES = POLICY_DOF_NAMES
SIM_TO_REAL_DOF = POLICY_TO_REAL_DOF
FOOT_REAL_TO_SIM = FOOT_REAL_TO_POLICY


def _vector(values: Sequence[float], size: int, name: str) -> Tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(values)}")
    return tuple(float(value) for value in values)


def sim_to_real(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert an actor-order joint vector to Unitree motor order."""
    simulation = _vector(values, NUM_DOF, "policy joint vector")
    real = [0.0] * NUM_DOF
    for sim_index, real_index in enumerate(SIM_TO_REAL_DOF):
        real[real_index] = simulation[sim_index] * DOF_SIGNS[sim_index]
    return tuple(real)


def real_to_sim(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert a Unitree motor-order joint vector to actor order."""
    real = _vector(values, NUM_DOF, "Unitree joint vector")
    return tuple(
        real[real_index] * DOF_SIGNS[sim_index]
        for sim_index, real_index in enumerate(SIM_TO_REAL_DOF)
    )


def foot_real_to_sim(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert Unitree FR/FL/RR/RL foot values to actor order."""
    real = _vector(values, 4, "Unitree foot vector")
    return tuple(real[index] for index in FOOT_REAL_TO_SIM)


def isaac_to_policy(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert an Isaac FL/FR/RL/RR joint vector to actor order."""
    isaac = _vector(values, NUM_DOF, "Isaac joint vector")
    return tuple(isaac[index] for index in ISAAC_TO_POLICY_DOF)


def policy_to_isaac(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert an actor-order joint vector to Isaac FL/FR/RL/RR order."""
    policy = _vector(values, NUM_DOF, "policy joint vector")
    return tuple(policy[index] for index in POLICY_TO_ISAAC_DOF)


def isaac_feet_to_unitree(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert Isaac FL/FR/RL/RR foot values to Unitree FR/FL/RR/RL."""
    isaac = _vector(values, 4, "Isaac foot vector")
    return tuple(isaac[index] for index in ISAAC_TO_UNITREE_FOOT)


def unitree_feet_to_isaac(values: Sequence[float]) -> Tuple[float, ...]:
    """Convert Unitree FR/FL/RR/RL foot values to Isaac FL/FR/RL/RR."""
    unitree = _vector(values, 4, "Unitree foot vector")
    return tuple(unitree[index] for index in UNITREE_TO_ISAAC_FOOT)
