"""ROS-independent guards for real Go2 low-level takeover."""

import json
import math
from typing import Optional, Sequence

import numpy as np


MOTION_REQUEST_TOPIC = "/api/motion_switcher/request"
MOTION_RESPONSE_TOPIC = "/api/motion_switcher/response"
REAL_LOW_COMMAND_TOPIC = "/lowcmd"
CHECK_MODE_API_ID = 1001
RELEASE_MODE_API_ID = 1003
RPC_TIMEOUT_S = 2.0
RPC_MAX_ATTEMPTS = 3
PUBLISHER_CLEAR_TIMEOUT_S = 15.0
PUBLISHER_CLEAR_STABLE_S = 0.5
TAKEOVER_HOLD_S = 1.0
INPUT_TIMEOUT_S = 0.25
TAKEOVER_MAX_JOINT_VELOCITY = 0.5
STARTUP_RAMP_S = 3.0
POLICY_ENGAGEMENT_RAMP_S = 1.0
POLICY_TARGET_MAX_STEP_RAD = 0.05


class RealControlError(RuntimeError):
    """Raised when a real-output boundary cannot be proven safe."""


class PublisherClearGate:
    """Require zero external publishers for one continuous time window."""

    def __init__(
        self,
        start_time: float,
        timeout_s: float = PUBLISHER_CLEAR_TIMEOUT_S,
        stable_s: float = PUBLISHER_CLEAR_STABLE_S,
    ) -> None:
        start = float(start_time)
        self.timeout_s = float(timeout_s)
        self.stable_s = float(stable_s)
        if not all(
            math.isfinite(value)
            for value in (start, self.timeout_s, self.stable_s)
        ):
            raise ValueError("publisher clear timing must be finite.")
        if self.timeout_s <= 0.0 or self.stable_s <= 0.0:
            raise ValueError("publisher clear timing must be positive.")
        self.deadline = start + self.timeout_s
        self.clear_since = None

    def observe(self, external_publisher_count: int, now: float) -> bool:
        count = int(external_publisher_count)
        timestamp = float(now)
        if count < 0:
            raise ValueError("external publisher count cannot be negative.")
        if not math.isfinite(timestamp):
            raise ValueError("publisher observation time must be finite.")
        if timestamp > self.deadline:
            raise RealControlError(
                "external /lowcmd publisher did not clear within "
                f"{self.timeout_s:.1f} seconds"
            )
        if count == 0:
            if self.clear_since is None:
                self.clear_since = timestamp
            if timestamp - self.clear_since >= self.stable_s:
                return True
        else:
            self.clear_since = None
        return False


def release_mode_required(active_mode: str) -> bool:
    """Return whether MotionSwitcher still has an active mode to release."""
    return bool(str(active_mode).strip())


def describe_publisher_endpoint(endpoint) -> str:
    """Return stable publisher identity fields for takeover diagnostics."""
    node_name = str(getattr(endpoint, "node_name", "unknown"))
    node_namespace = str(getattr(endpoint, "node_namespace", "unknown"))
    endpoint_gid = getattr(endpoint, "endpoint_gid", ())
    try:
        gid = bytes(endpoint_gid).hex()
    except (TypeError, ValueError):
        gid = str(endpoint_gid)
    return f"node={node_namespace}/{node_name}, gid={gid or 'unknown'}"


def build_motion_request(request, api_id: int, request_id: int):
    """Fill a Unitree MotionSwitcher request-like message."""
    request.header.identity.id = int(request_id)
    request.header.identity.api_id = int(api_id)
    request.header.lease.id = 0
    request.header.policy.priority = 0
    request.header.policy.noreply = False
    request.parameter = json.dumps({}, separators=(",", ":"))
    request.binary = []
    return request


def parse_motion_response(response, request_id: int, api_id: int) -> dict:
    """Validate and decode one matching MotionSwitcher response."""
    if int(response.header.identity.id) != int(request_id):
        raise RealControlError("MotionSwitcher response request ID mismatch.")
    if int(response.header.identity.api_id) != int(api_id):
        raise RealControlError("MotionSwitcher response API ID mismatch.")
    status = int(response.header.status.code)
    if status != 0:
        raise RealControlError(
            f"MotionSwitcher API {api_id} returned status {status}."
        )
    data = str(response.data)
    if not data:
        return {}
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as error:
        raise RealControlError("MotionSwitcher returned invalid JSON.") from error
    if not isinstance(decoded, dict):
        raise RealControlError(
            "MotionSwitcher response must contain a JSON object."
        )
    return decoded


def validate_low_command_boundary(
    real_output_enabled: bool,
    external_publisher_count: int,
) -> None:
    """Reject creation of the real publisher without exclusive ownership."""
    if not real_output_enabled:
        raise RealControlError("real /lowcmd output is not enabled")
    count = int(external_publisher_count)
    if count != 0:
        raise RealControlError(
            f"refusing /lowcmd while {count} external publisher(s) remain"
        )


def _joint_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise RealControlError(f"{name} must contain 12 finite values")
    return result


def validate_takeover_inputs(
    joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    low_state_age_s: float,
    remote_age_s: float,
) -> None:
    """Require fresh, finite, nearly stationary state before releasing Sport Mode."""
    _joint_vector(joint_position, "joint position")
    velocity = _joint_vector(joint_velocity, "joint velocity")
    ages = (float(low_state_age_s), float(remote_age_s))
    if not all(math.isfinite(age) and 0.0 <= age <= INPUT_TIMEOUT_S for age in ages):
        raise RealControlError("LowState and remote input must be fresh")
    if float(np.max(np.abs(velocity))) > TAKEOVER_MAX_JOINT_VELOCITY:
        raise RealControlError(
            "joint velocity is too high for low-level takeover"
        )


def quintic_smoothstep(progress: float) -> float:
    """Return a clamped quintic blend with quiet trajectory endpoints."""
    value = float(progress)
    if not math.isfinite(value):
        raise RealControlError("trajectory progress must be finite")
    value = min(max(value, 0.0), 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def interpolate_pose(
    start_q: Sequence[float],
    end_q: Sequence[float],
    elapsed_s: float,
    duration_s: float,
) -> np.ndarray:
    """Interpolate a 12-joint pose with a quintic time law."""
    start = _joint_vector(start_q, "trajectory start")
    end = _joint_vector(end_q, "trajectory end")
    elapsed = float(elapsed_s)
    duration = float(duration_s)
    if not math.isfinite(elapsed):
        raise RealControlError("trajectory elapsed time must be finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise RealControlError("trajectory duration must be positive")
    blend = quintic_smoothstep(elapsed / duration)
    return start + (end - start) * blend


class PolicyTransitionGuard:
    """Blend policy entry and bound every commanded target step."""

    def __init__(
        self,
        ramp_s: float = POLICY_ENGAGEMENT_RAMP_S,
        max_step_rad: float = POLICY_TARGET_MAX_STEP_RAD,
    ) -> None:
        self.ramp_s = float(ramp_s)
        self.max_step_rad = float(max_step_rad)
        if not math.isfinite(self.ramp_s) or self.ramp_s <= 0.0:
            raise ValueError("policy ramp duration must be positive")
        if not math.isfinite(self.max_step_rad) or self.max_step_rad <= 0.0:
            raise ValueError("policy target step must be positive")
        self.start_time = None
        self.start_q = None
        self.previous_q = None

    @property
    def active(self) -> bool:
        return self.start_time is not None

    def begin(self, start_q: Sequence[float], now: float) -> None:
        start = _joint_vector(start_q, "policy transition start")
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise RealControlError("policy transition time must be finite")
        self.start_time = timestamp
        self.start_q = start.copy()
        self.previous_q = start.copy()

    def apply(self, requested_q: Sequence[float], now: float) -> np.ndarray:
        if not self.active or self.start_q is None or self.previous_q is None:
            raise RealControlError("policy transition was not initialized")
        requested = _joint_vector(requested_q, "policy target")
        elapsed = max(0.0, float(now) - float(self.start_time))
        blend = quintic_smoothstep(elapsed / self.ramp_s)
        blended = self.start_q + blend * (requested - self.start_q)
        delta = np.clip(
            blended - self.previous_q,
            -self.max_step_rad,
            self.max_step_rad,
        )
        target = self.previous_q + delta
        self.previous_q = target.copy()
        return target

    def reset(self) -> None:
        self.start_time = None
        self.start_q = None
        self.previous_q = None
