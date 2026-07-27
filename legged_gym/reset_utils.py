"""Dependency-light reset configuration helpers."""

import math
from typing import Optional, Sequence, Tuple


DEFAULT_DOF_POS_RESET_RANGE = (0.0, 0.9)


def resolve_dof_pos_reset_range(
    configured: Optional[Sequence[float]],
) -> Tuple[float, float]:
    """Return a validated additive DOF reset range."""
    values = DEFAULT_DOF_POS_RESET_RANGE if configured is None else configured
    if len(values) != 2:
        raise ValueError("dof_pos_reset_range must contain two values")
    lower, upper = (float(values[0]), float(values[1]))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("dof_pos_reset_range must contain finite values")
    if lower > upper:
        raise ValueError("dof_pos_reset_range lower bound exceeds upper bound")
    return lower, upper
