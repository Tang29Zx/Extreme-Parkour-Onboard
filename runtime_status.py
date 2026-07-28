"""Build the bounded JSON payload published for live control diagnostics."""

import json
from typing import Optional, Sequence

import numpy as np


RUNTIME_STATUS_SCHEMA_VERSION = 1
RUNTIME_STATUS_TOPIC = "/extreme_parkour/runtime_status"
RUNTIME_STATUS_PERIOD_S = 0.5


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(
    values: Sequence[float],
    size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _rounded(value: float, digits: int = 5) -> float:
    return float(round(float(value), digits))


def _rounded_list(values: np.ndarray, digits: int = 5):
    return [_rounded(value, digits) for value in values]


def should_publish_runtime_status(
    last_publish_time: Optional[float],
    now: float,
    period_s: float = RUNTIME_STATUS_PERIOD_S,
) -> bool:
    """Return whether a monotonic timestamp has reached the publish period."""
    current = _finite_scalar(now, "current monotonic time")
    period = _finite_scalar(period_s, "runtime status period")
    if period <= 0.0:
        raise ValueError("runtime status period must be positive")
    if last_publish_time is None:
        return True
    previous = _finite_scalar(last_publish_time, "last publish time")
    if current < previous:
        raise ValueError("runtime status monotonic clock moved backwards")
    return current - previous >= period


def build_runtime_status(
    *,
    timestamp_unix_s: float,
    phase: str,
    dryrun: bool,
    real_lowcmd_authorized: bool,
    engagement_active: bool,
    input_ages_s: Sequence[float],
    roll_pitch_rad: Sequence[float],
    foot_force: Sequence[float],
    contact_state: Sequence[float],
    measured_q: Sequence[float],
    commanded_q: Sequence[float],
    measured_dq: Sequence[float],
    raw_action: Sequence[float],
    requested_q: Sequence[float],
    motor_temperature: Sequence[float],
    motor_lost: Sequence[float],
    motor_tau_est: Sequence[float],
    torque_limits: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    loop_samples_s: Sequence[float],
) -> str:
    """Validate one live snapshot and serialize schema version 1 as JSON."""
    timestamp = _finite_scalar(timestamp_unix_s, "wall-clock timestamp")
    ages = _finite_vector(input_ages_s, 3, "input ages")
    roll_pitch = _finite_vector(roll_pitch_rad, 2, "roll and pitch")
    forces = _finite_vector(foot_force, 4, "foot force")
    contacts = _finite_vector(contact_state, 4, "contact state")
    measured = _finite_vector(measured_q, 12, "measured q")
    commanded = _finite_vector(commanded_q, 12, "commanded q")
    velocity = _finite_vector(measured_dq, 12, "measured dq")
    action = _finite_vector(raw_action, 12, "raw action")
    requested = _finite_vector(requested_q, 12, "requested q")
    temperature = _finite_vector(motor_temperature, 12, "motor temperature")
    lost = _finite_vector(motor_lost, 12, "motor lost")
    tau_est = _finite_vector(motor_tau_est, 12, "estimated motor torque")
    limits = _finite_vector(torque_limits, 12, "torque limits")
    p_gains = _finite_vector(kp, 12, "kp")
    d_gains = _finite_vector(kd, 12, "kd")
    loops = np.asarray(loop_samples_s, dtype=np.float64)
    if loops.ndim != 1 or loops.size == 0 or not np.isfinite(loops).all():
        raise ValueError("loop samples must contain finite values")

    if np.any(ages < 0.0) or np.any(loops < 0.0):
        raise ValueError("input ages and loop samples must be non-negative")
    if np.any(lost < 0.0) or not np.equal(lost, np.rint(lost)).all():
        raise ValueError("motor lost counters must be non-negative integers")
    if np.any(limits <= 0.0) or np.any(p_gains <= 0.0):
        raise ValueError("torque limits and kp must be positive")
    if np.any(d_gains < 0.0):
        raise ValueError("kd must be non-negative")

    tracking_error = commanded - measured
    predicted_pd_tau = p_gains * tracking_error - d_gains * velocity
    loop_ms = loops * 1000.0
    status = {
        "schema_version": RUNTIME_STATUS_SCHEMA_VERSION,
        "timestamp_unix_s": _rounded(timestamp, 3),
        "phase": str(phase),
        "output": {
            "dryrun": bool(dryrun),
            "real_lowcmd_authorized": bool(real_lowcmd_authorized),
            "engagement_active": bool(engagement_active),
        },
        "input_age_ms": {
            "low_state": _rounded(ages[0] * 1000.0, 3),
            "remote": _rounded(ages[1] * 1000.0, 3),
            "depth": _rounded(ages[2] * 1000.0, 3),
        },
        "body": {
            "roll_deg": _rounded(np.rad2deg(roll_pitch[0]), 3),
            "pitch_deg": _rounded(np.rad2deg(roll_pitch[1]), 3),
        },
        "feet": {
            "force": _rounded_list(forces, 3),
            "contact": [bool(value) for value in contacts],
        },
        "joint": {
            "measured_q_rad": _rounded_list(measured),
            "commanded_q_rad": _rounded_list(commanded),
            "measured_dq_rad_s": _rounded_list(velocity),
            "max_tracking_error_rad": _rounded(
                np.max(np.abs(tracking_error))
            ),
        },
        "policy": {
            "max_abs_raw_action": _rounded(np.max(np.abs(action))),
            "max_request_command_delta_rad": _rounded(
                np.max(np.abs(requested - commanded))
            ),
        },
        "motor": {
            "temperature_c": _rounded_list(temperature, 1),
            "max_temperature_c": _rounded(np.max(temperature), 1),
            "lost": [int(value) for value in lost],
            "tau_est_nm": _rounded_list(tau_est, 3),
            "max_abs_tau_ratio": _rounded(
                np.max(np.abs(tau_est) / limits)
            ),
            "max_abs_pd_tau_ratio": _rounded(
                np.max(np.abs(predicted_pd_tau) / limits)
            ),
        },
        "loop_ms": {
            "last": _rounded(loop_ms[-1], 3),
            "p50": _rounded(np.percentile(loop_ms, 50), 3),
            "p95": _rounded(np.percentile(loop_ms, 95), 3),
            "max": _rounded(np.max(loop_ms), 3),
            "samples": int(loop_ms.size),
        },
    }
    return json.dumps(
        status,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
