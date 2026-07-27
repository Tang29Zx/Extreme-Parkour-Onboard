"""ROS-independent guards for real Go2 low-level takeover."""

import json
import math
from typing import Sequence, Tuple

import numpy as np


MOTION_REQUEST_TOPIC = "/api/motion_switcher/request"
MOTION_RESPONSE_TOPIC = "/api/motion_switcher/response"
REAL_LOW_COMMAND_TOPIC = "/lowcmd"
CHECK_MODE_API_ID = 1001
RELEASE_MODE_API_ID = 1003
RPC_TIMEOUT_S = 2.0
RPC_MAX_ATTEMPTS = 3
TAKEOVER_HOLD_S = 1.0
INPUT_TIMEOUT_S = 0.25
TAKEOVER_MAX_JOINT_VELOCITY = 0.5
STARTUP_RAMP_S = 3.0
POLICY_PRIME_S = 0.5
POLICY_PRIME_PROPRIO_SAMPLES = 10
POLICY_PRIME_DEPTH_SAMPLES = 5
POLICY_ENGAGEMENT_RAMP_S = 1.0
POLICY_TARGET_MAX_STEP_RAD = 0.05


class RealControlError(RuntimeError):
    """Raised when a real-output boundary cannot be proven safe."""


def release_mode_required(active_mode: str) -> bool:
    """Return whether MotionSwitcher still has an active mode to release."""
    return bool(str(active_mode).strip())


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


def validate_real_low_command_publish(
    real_output_enabled: bool,
    takeover_authorized: bool,
) -> None:
    """Reject real LowCmd publication until MotionSwitcher released control."""
    if not real_output_enabled:
        raise RealControlError("real /lowcmd output is not enabled")
    if not takeover_authorized:
        raise RealControlError(
            "real /lowcmd publishing is blocked until CheckMode confirms release"
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


def validate_policy_prime_inputs(
    low_state_age_s: float,
    remote_age_s: float,
    depth_age_s: float,
) -> None:
    """Require fresh state, remote, and depth while priming policy memory."""
    ages = (
        float(low_state_age_s),
        float(remote_age_s),
        float(depth_age_s),
    )
    if not all(
        math.isfinite(age) and 0.0 <= age <= INPUT_TIMEOUT_S
        for age in ages
    ):
        raise RealControlError(
            "LowState, remote input, and depth must be fresh for policy prime"
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


class PolicyPrimeGate:
    """Count fresh policy context samples over a minimum hold duration."""

    def __init__(
        self,
        start_time: float,
        duration_s: float = POLICY_PRIME_S,
        proprio_samples: int = POLICY_PRIME_PROPRIO_SAMPLES,
        depth_samples: int = POLICY_PRIME_DEPTH_SAMPLES,
    ) -> None:
        self.duration_s = float(duration_s)
        self.required_proprio_samples = int(proprio_samples)
        self.required_depth_samples = int(depth_samples)
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("policy prime duration must be positive")
        if self.required_proprio_samples <= 0 or self.required_depth_samples <= 0:
            raise ValueError("policy prime sample counts must be positive")
        self.restart(start_time)

    def restart(self, start_time: float) -> None:
        timestamp = float(start_time)
        if not math.isfinite(timestamp):
            raise ValueError("policy prime start time must be finite")
        self.start_time = timestamp
        self.proprio_samples = 0
        self.depth_samples = 0

    @property
    def has_samples(self) -> bool:
        return self.proprio_samples > 0 or self.depth_samples > 0

    def record_proprio(self) -> None:
        self.proprio_samples += 1

    def record_depth(self) -> None:
        self.depth_samples += 1

    def ready(self, now: float) -> bool:
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("policy prime observation time must be finite")
        return (
            timestamp - self.start_time >= self.duration_s
            and self.proprio_samples >= self.required_proprio_samples
            and self.depth_samples >= self.required_depth_samples
        )


class PolicyTransitionGuard:
    """Smooth only the handoff from stand hold to policy targets."""

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
        self.first_apply = False

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
        self.first_apply = True

    def apply(self, requested_q: Sequence[float], now: float) -> np.ndarray:
        if not self.active or self.start_q is None or self.previous_q is None:
            raise RealControlError("policy transition was not initialized")
        requested = _joint_vector(requested_q, "policy target")
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise RealControlError("policy transition time must be finite")
        if self.first_apply:
            self.first_apply = False
            return self.start_q.copy()
        elapsed = max(0.0, timestamp - float(self.start_time))
        blend = quintic_smoothstep(elapsed / self.ramp_s)
        blended = self.start_q + blend * (requested - self.start_q)
        delta = np.clip(
            blended - self.previous_q,
            -self.max_step_rad,
            self.max_step_rad,
        )
        target = self.previous_q + delta
        self.previous_q = target.copy()
        if (
            elapsed >= self.ramp_s
            and float(np.max(np.abs(requested - target))) <= 1e-12
        ):
            self.reset()
        return target

    def reset(self) -> None:
        self.start_time = None
        self.start_q = None
        self.previous_q = None
        self.first_apply = False


def executed_target_to_action(
    target_q: Sequence[float],
    default_q: Sequence[float],
    action_scale: float,
) -> np.ndarray:
    """Convert an executed joint target back to policy-space action units."""
    target = _joint_vector(target_q, "executed target")
    default = _joint_vector(default_q, "default target")
    scale = float(action_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RealControlError("action scale must be positive")
    return (target - default) / scale


def prepare_policy_action(
    raw_action: Sequence[float],
    default_q: Sequence[float],
    clip_actions: float,
    action_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the training-equivalent action clip and joint-target mapping."""
    action = _joint_vector(raw_action, "policy action")
    default = _joint_vector(default_q, "default target")
    clip_limit = float(clip_actions)
    scale = float(action_scale)
    if not math.isfinite(clip_limit) or clip_limit <= 0.0:
        raise RealControlError("action clip limit must be positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise RealControlError("action scale must be positive")
    policy_clip_limit = clip_limit / scale
    clipped_action = np.clip(action, -policy_clip_limit, policy_clip_limit)
    target_q = default + clipped_action * scale
    return clipped_action, target_q
