"""ROS-independent guards for real Go2 low-level takeover."""

import json
import math
from typing import Optional, Sequence, Tuple, Union

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
POLICY_TRANSITION_MAX_STEP_RAD = 0.05
POLICY_TARGET_MAX_STEP_RAD = 0.21
POLICY_CALF_TARGET_MAX_STEP_RAD = 0.20
POLICY_TARGET_MAX_STEP_RAD_BY_JOINT = (
    POLICY_TARGET_MAX_STEP_RAD,
    POLICY_TARGET_MAX_STEP_RAD,
    POLICY_CALF_TARGET_MAX_STEP_RAD,
) * 4
POLICY_TORQUE_ESCAPE_MAX_STEP_RAD = 0.30
POLICY_TORQUE_ESCAPE_MAX_STEP_RAD_BY_JOINT = (
    POLICY_TORQUE_ESCAPE_MAX_STEP_RAD,
    POLICY_TORQUE_ESCAPE_MAX_STEP_RAD,
    POLICY_CALF_TARGET_MAX_STEP_RAD,
) * 4
POLICY_TORQUE_ESCAPE_TOLERANCE = 1e-9
POLICY_JOINT_VELOCITY_LIMIT_REL_TOLERANCE = 0.005
POLICY_TARGET_MAX_DEVIATION_RAD = 0.30
POLICY_STATE_LIMIT_TOLERANCE_RAD = 0.05
REAL_FOOT_CONTACT_THRESHOLD = 5.0
POLICY_ENTRY_MAX_TILT_RAD = math.radians(8.0)
POLICY_ENTRY_MAX_TRACKING_ERROR_RAD = 0.20


class RealControlError(RuntimeError):
    """Raised when a real-output boundary cannot be proven safe."""


class LowStateStaleError(RealControlError):
    """Raised when policy control has lost fresh motor feedback."""


class DepthStaleError(RealControlError):
    """Raised when policy control has lost fresh depth input."""


class PolicyTargetInfeasibleError(RealControlError):
    """Raised when no position target can satisfy every output bound."""

    def __init__(self, message: str, joint_indices: Sequence[int]):
        super().__init__(message)
        self.joint_indices = tuple(int(index) for index in joint_indices)


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
    depth_age_s: float,
) -> None:
    """Require fresh state and depth while priming policy memory."""
    ages = (
        float(low_state_age_s),
        float(depth_age_s),
    )
    if not all(
        math.isfinite(age) and 0.0 <= age <= INPUT_TIMEOUT_S
        for age in ages
    ):
        raise RealControlError(
            "LowState and depth must be fresh for policy prime"
        )


def validate_policy_runtime_inputs(
    low_state_age_s: float,
    depth_age_s: float,
    joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    joint_limits_low: Sequence[float],
    joint_limits_high: Sequence[float],
    joint_velocity_limits: Sequence[float],
) -> None:
    """Require fresh, finite, physically bounded policy inputs every cycle."""
    low_state_age = float(low_state_age_s)
    if (
        not math.isfinite(low_state_age)
        or low_state_age < 0.0
        or low_state_age > INPUT_TIMEOUT_S
    ):
        raise LowStateStaleError("LowState is stale during policy control")

    depth_age = float(depth_age_s)
    if (
        not math.isfinite(depth_age)
        or depth_age < 0.0
        or depth_age > INPUT_TIMEOUT_S
    ):
        raise DepthStaleError("depth is stale during policy control")

    position = _joint_vector(joint_position, "measured joint position")
    velocity = _joint_vector(joint_velocity, "measured joint velocity")
    lower = _joint_vector(joint_limits_low, "joint lower limits")
    upper = _joint_vector(joint_limits_high, "joint upper limits")
    velocity_limits = _joint_vector(
        joint_velocity_limits,
        "joint velocity limits",
    )
    if bool(np.any(lower >= upper)):
        raise RealControlError("joint lower limits must be below upper limits")
    if bool(np.any(velocity_limits <= 0.0)):
        raise RealControlError("joint velocity limits must be positive")

    tolerance = POLICY_STATE_LIMIT_TOLERANCE_RAD
    if bool(
        np.any(position < lower - tolerance)
        or np.any(position > upper + tolerance)
    ):
        raise RealControlError("measured joint position exceeded its limit")
    velocity_threshold = velocity_limits * (
        1.0 + POLICY_JOINT_VELOCITY_LIMIT_REL_TOLERANCE
    )
    if bool(np.any(np.abs(velocity) > velocity_threshold)):
        raise RealControlError("measured joint velocity exceeded its limit")


def validate_policy_request_input(remote_age_s: float) -> None:
    """Require a fresh remote sample at the instant policy entry is requested."""
    age = float(remote_age_s)
    if not math.isfinite(age) or not 0.0 <= age <= INPUT_TIMEOUT_S:
        raise RealControlError("remote input must be fresh when Y is pressed")


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RealControlError(f"{name} must contain {size} finite values")
    return result


def classify_foot_contacts(
    foot_force: Sequence[float],
    threshold: float = REAL_FOOT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """Classify model-order real foot-force samples without temporal filtering."""
    forces = _finite_vector(foot_force, 4, "foot force")
    contact_threshold = float(threshold)
    if not math.isfinite(contact_threshold) or contact_threshold < 0.0:
        raise RealControlError("foot contact threshold must be finite and non-negative")
    return forces > contact_threshold


def filter_foot_contacts(
    foot_force: Sequence[float],
    previous_contact: Sequence[bool],
    threshold: float = REAL_FOOT_CONTACT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the training-equivalent current-or-previous contact filter."""
    current = classify_foot_contacts(foot_force, threshold)
    previous = np.asarray(previous_contact, dtype=np.bool_)
    if previous.shape != (4,):
        raise RealControlError("previous contact must contain 4 values")
    return np.logical_or(current, previous), current


def update_motor_lost_baseline(
    motor_lost: Sequence[float],
    motor_lost_baseline: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance cumulative lost counters and report indices that increased."""
    current = _finite_vector(motor_lost, 12, "motor lost status")
    baseline = _finite_vector(
        motor_lost_baseline,
        12,
        "motor lost baseline",
    )
    if bool(np.any(current < 0.0) or np.any(baseline < 0.0)):
        raise RealControlError("motor lost counters must be non-negative")
    return current.copy(), np.flatnonzero(current > baseline)


def validate_policy_entry_state(
    foot_force: Sequence[float],
    roll_rad: float,
    pitch_rad: float,
    measured_q: Sequence[float],
    commanded_q: Sequence[float],
    motor_temperature: Sequence[float],
    motor_lost: Sequence[float],
    motor_lost_baseline: Sequence[float],
) -> None:
    """Require a loaded, upright, healthy, tracked stand before accepting Y."""
    contacts = classify_foot_contacts(foot_force)
    if not bool(np.all(contacts)):
        missing = np.flatnonzero(~contacts).tolist()
        raise RealControlError(f"all four feet must be loaded; missing contacts {missing}")

    tilt = _finite_vector((roll_rad, pitch_rad), 2, "roll and pitch")
    if float(np.max(np.abs(tilt))) > POLICY_ENTRY_MAX_TILT_RAD:
        raise RealControlError("body tilt is too large for policy entry")

    measured = _joint_vector(measured_q, "measured joint position")
    commanded = _joint_vector(commanded_q, "commanded joint position")
    if (
        float(np.max(np.abs(measured - commanded)))
        > POLICY_ENTRY_MAX_TRACKING_ERROR_RAD
    ):
        raise RealControlError("joint tracking error is too large for policy entry")

    _finite_vector(motor_temperature, 12, "motor temperature")

    _, increased = update_motor_lost_baseline(
        motor_lost,
        motor_lost_baseline,
    )
    if increased.size:
        raise RealControlError(
            "one or more motor lost counters increased at indices "
            f"{increased.tolist()}"
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


class RemoteEdgeTracker:
    """Track fresh remote samples and one-shot button rising edges."""

    def __init__(self) -> None:
        self.keys = 0
        self.rising_edges = 0
        self.latest_time = None

    def update(self, keys: int, now: float) -> None:
        value = int(keys)
        timestamp = float(now)
        if value < 0:
            raise RealControlError("remote keys must be non-negative")
        if not math.isfinite(timestamp):
            raise RealControlError("remote sample time must be finite")
        self.rising_edges |= value & ~self.keys
        self.keys = value
        self.latest_time = timestamp

    def consume_rising(self, button: int) -> bool:
        mask = int(button)
        if mask <= 0:
            raise RealControlError("remote button mask must be positive")
        pressed = bool(self.rising_edges & mask)
        self.rising_edges &= ~mask
        return pressed


class PolicyTransitionGuard:
    """Smooth only the handoff from stand hold to policy targets."""

    def __init__(
        self,
        ramp_s: float = POLICY_ENGAGEMENT_RAMP_S,
        max_step_rad: float = POLICY_TRANSITION_MAX_STEP_RAD,
        max_deviation_rad: float = POLICY_TARGET_MAX_DEVIATION_RAD,
    ) -> None:
        self.ramp_s = float(ramp_s)
        self.max_step_rad = float(max_step_rad)
        self.max_deviation_rad = float(max_deviation_rad)
        if not math.isfinite(self.ramp_s) or self.ramp_s <= 0.0:
            raise ValueError("policy ramp duration must be positive")
        if not math.isfinite(self.max_step_rad) or self.max_step_rad <= 0.0:
            raise ValueError("policy target step must be positive")
        if (
            not math.isfinite(self.max_deviation_rad)
            or self.max_deviation_rad <= 0.0
        ):
            raise ValueError("policy target deviation must be positive")
        self.start_time = None
        self.start_q = None
        self.previous_q = None
        self.first_apply = False
        self.pending_requested_q = None
        self.pending_target_q = None
        self.pending_elapsed_s = None

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
        self.pending_requested_q = None
        self.pending_target_q = None
        self.pending_elapsed_s = None

    def apply(self, requested_q: Sequence[float], now: float) -> np.ndarray:
        if not self.active or self.start_q is None or self.previous_q is None:
            raise RealControlError("policy transition was not initialized")
        if self.pending_target_q is not None:
            raise RealControlError(
                "executed policy target feedback is required before the next "
                "transition target"
            )
        requested = _joint_vector(requested_q, "policy target")
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise RealControlError("policy transition time must be finite")
        elapsed = max(0.0, timestamp - float(self.start_time))
        if self.first_apply:
            self.first_apply = False
            target = self.start_q.copy()
        else:
            blend = quintic_smoothstep(elapsed / self.ramp_s)
            blended = self.start_q + blend * (requested - self.start_q)
            if elapsed < self.ramp_s:
                bounded = np.clip(
                    blended,
                    self.start_q - self.max_deviation_rad,
                    self.start_q + self.max_deviation_rad,
                )
            else:
                bounded = requested
            delta = np.clip(
                bounded - self.previous_q,
                -self.max_step_rad,
                self.max_step_rad,
            )
            target = self.previous_q + delta
        self.pending_requested_q = requested.copy()
        self.pending_target_q = target.copy()
        self.pending_elapsed_s = elapsed
        return target

    def record_executed_target(self, executed_q: Sequence[float]) -> None:
        """Commit the target that passed all downstream safety constraints."""
        if (
            not self.active
            or self.previous_q is None
            or self.pending_requested_q is None
            or self.pending_target_q is None
            or self.pending_elapsed_s is None
        ):
            raise RealControlError("policy transition has no pending target")
        executed = _joint_vector(executed_q, "executed policy target")
        executed_step = float(np.max(np.abs(executed - self.previous_q)))
        if executed_step > self.max_step_rad + 1e-12:
            raise RealControlError(
                "executed policy target exceeded the transition step limit"
            )
        elapsed = self.pending_elapsed_s
        self.previous_q = executed.copy()
        if elapsed >= self.ramp_s:
            self.reset()
            return
        self.pending_requested_q = None
        self.pending_target_q = None
        self.pending_elapsed_s = None

    def reset(self) -> None:
        self.start_time = None
        self.start_q = None
        self.previous_q = None
        self.first_apply = False
        self.pending_requested_q = None
        self.pending_target_q = None
        self.pending_elapsed_s = None


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw history action, clipped action, and scaled joint target."""
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
    return action.copy(), clipped_action, target_q


def constrain_policy_target(
    requested_q: Sequence[float],
    previous_q: Sequence[float],
    measured_q: Sequence[float],
    measured_dq: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    joint_limits_low: Sequence[float],
    joint_limits_high: Sequence[float],
    torque_limits: Sequence[float],
    max_step_rad: Union[float, Sequence[float]] = (
        POLICY_TARGET_MAX_STEP_RAD_BY_JOINT
    ),
    escape_max_step_rad: Optional[Union[float, Sequence[float]]] = None,
) -> np.ndarray:
    """Intersect step, joint, and estimated PD-torque bounds for one target.

    A larger escape step may be supplied for joints whose previous target has
    become torque-unsafe. It is used only when the ordinary intersection is
    empty and the escaped command moves toward measured position while reducing
    absolute predicted PD torque.
    """
    requested = _joint_vector(requested_q, "requested joint target")
    previous = _joint_vector(previous_q, "previous joint target")
    measured = _joint_vector(measured_q, "measured joint position")
    measured_velocity = _joint_vector(
        measured_dq,
        "measured joint velocity",
    )
    p_gain = _joint_vector(kp, "joint proportional gains")
    d_gain = _joint_vector(kd, "joint derivative gains")
    joint_lower = _joint_vector(joint_limits_low, "joint lower limits")
    joint_upper = _joint_vector(joint_limits_high, "joint upper limits")
    torque = _joint_vector(torque_limits, "joint torque limits")
    max_step_value = np.asarray(max_step_rad, dtype=np.float64)
    if max_step_value.ndim == 0:
        max_step = np.full(12, float(max_step_value), dtype=np.float64)
    elif max_step_value.shape == (12,):
        max_step = max_step_value.copy()
    else:
        raise RealControlError(
            "policy target step must be a scalar or contain 12 values"
        )

    if not np.isfinite(max_step).all() or bool(np.any(max_step <= 0.0)):
        raise RealControlError("policy target steps must be positive and finite")

    escape_max_step = None
    if escape_max_step_rad is not None:
        escape_max_step_value = np.asarray(
            escape_max_step_rad,
            dtype=np.float64,
        )
        if escape_max_step_value.ndim == 0:
            escape_max_step = np.full(
                12,
                float(escape_max_step_value),
                dtype=np.float64,
            )
        elif escape_max_step_value.shape == (12,):
            escape_max_step = escape_max_step_value.copy()
        else:
            raise RealControlError(
                "policy target escape step must be a scalar or contain "
                "12 values"
            )
        if not np.isfinite(escape_max_step).all() or bool(
            np.any(escape_max_step <= 0.0)
        ):
            raise RealControlError(
                "policy target escape steps must be positive and finite"
            )
        if bool(
            np.any(
                escape_max_step
                < max_step - POLICY_TORQUE_ESCAPE_TOLERANCE
            )
        ):
            raise RealControlError(
                "policy target escape steps must not be below ordinary steps"
            )
    if bool(np.any(p_gain <= 0.0)):
        raise RealControlError("joint proportional gains must be positive")
    if bool(np.any(d_gain < 0.0)):
        raise RealControlError("joint derivative gains must be non-negative")
    if bool(np.any(joint_lower >= joint_upper)):
        raise RealControlError("joint lower limits must be below upper limits")
    if bool(np.any(torque <= 0.0)):
        raise RealControlError("joint torque limits must be positive")

    torque_lower = measured + (-torque + d_gain * measured_velocity) / p_gain
    torque_upper = measured + (torque + d_gain * measured_velocity) / p_gain
    torque_safe_lower = np.maximum(joint_lower, torque_lower)
    torque_safe_upper = np.minimum(joint_upper, torque_upper)
    lower = np.maximum(previous - max_step, torque_safe_lower)
    upper = np.minimum(previous + max_step, torque_safe_upper)

    if escape_max_step is not None and bool(np.any(lower > upper)):
        previous_predicted_torque = (
            p_gain * (previous - measured) - d_gain * measured_velocity
        )
        escape_lower = np.maximum(
            previous - escape_max_step,
            torque_safe_lower,
        )
        escape_upper = np.minimum(
            previous + escape_max_step,
            torque_safe_upper,
        )
        escape_candidate = np.clip(requested, escape_lower, escape_upper)
        escape_predicted_torque = (
            p_gain * (escape_candidate - measured)
            - d_gain * measured_velocity
        )
        tolerance = POLICY_TORQUE_ESCAPE_TOLERANCE
        escape_allowed = (
            (lower > upper)
            & (torque_safe_lower <= torque_safe_upper)
            & (escape_lower <= escape_upper)
            & (escape_max_step > max_step + tolerance)
            & (np.abs(previous_predicted_torque) > torque + tolerance)
            & (
                np.abs(escape_candidate - measured)
                < np.abs(previous - measured) - tolerance
            )
            & (
                np.abs(escape_predicted_torque)
                < np.abs(previous_predicted_torque) - tolerance
            )
        )
        lower = np.where(escape_allowed, escape_lower, lower)
        upper = np.where(escape_allowed, escape_upper, upper)

    if bool(np.any(lower > upper)):
        joints = np.flatnonzero(lower > upper)
        details = []
        for joint in joints:
            details.append(
                f"joint={int(joint)} "
                f"requested={requested[joint]:+.6f} "
                f"previous={previous[joint]:+.6f} "
                f"measured_q={measured[joint]:+.6f} "
                f"measured_dq={measured_velocity[joint]:+.6f} "
                f"step=[{previous[joint] - max_step[joint]:+.6f},"
                f"{previous[joint] + max_step[joint]:+.6f}] "
                f"joint_limit=[{joint_lower[joint]:+.6f},"
                f"{joint_upper[joint]:+.6f}] "
                f"pd_target=[{torque_lower[joint]:+.6f},"
                f"{torque_upper[joint]:+.6f}] "
                f"intersection=[{lower[joint]:+.6f},"
                f"{upper[joint]:+.6f}]"
            )
        raise PolicyTargetInfeasibleError(
            "no safe policy target satisfies all bounds; " + "; ".join(details),
            joints,
        )
    return np.clip(requested, lower, upper)
