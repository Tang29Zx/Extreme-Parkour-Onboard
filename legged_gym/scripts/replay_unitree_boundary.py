"""Replay exported parkour models through the Unitree low-level boundary."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from joint_mapping import (
    POLICY_DOF_NAMES,
    isaac_feet_to_unitree,
    isaac_to_policy,
    policy_to_isaac,
    real_to_sim,
)
from legged_gym.scripts.replay_geometry import single_box_rear_x
from legged_gym.scripts.replay_traced import (
    FORWARD_COMMAND_MPS,
    VISUAL_UPDATE_INTERVAL,
    configure_replay_environment,
    fixed_command_observation,
    install_single_box_terrain,
    load_exported_models,
    render_single_box_depth,
)
from legged_gym.utils import get_args, task_registry
from policy_context import update_proprio_history
from real_control_safety import (
    POLICY_TARGET_MAX_STEP_RAD,
    POLICY_TRANSITION_MAX_STEP_RAD,
    STARTUP_RAMP_S,
    PolicyPrimeGate,
    PolicyTransitionGuard,
    RemoteEdgeTracker,
    constrain_policy_target,
    interpolate_pose,
    prepare_policy_action,
    validate_policy_entry_state,
    validate_policy_prime_inputs,
    validate_policy_request_input,
    validate_policy_runtime_inputs,
    validate_takeover_inputs,
)
from unitree_boundary import (
    BoundaryLowState,
    DecodedLowState,
    GO2_JOINT_LIMITS_HIGH,
    GO2_JOINT_LIMITS_LOW,
    GO2_JOINT_VELOCITY_LIMITS,
    GO2_TORQUE_LIMITS,
    build_policy_proprio,
    decode_low_state,
    encode_low_cmd,
)


LEGACY_JOINT_REINDEX = np.asarray(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
    dtype=np.int64,
)
LEGACY_FOOT_REINDEX = np.asarray([1, 0, 3, 2], dtype=np.int64)
PHASE_CODES = {
    "dryrun": 0,
    "startup": 1,
    "prime": 2,
    "stand_hold": 3,
    "policy": 4,
    "rejected": 5,
}
SIM_REMOTE_L1 = 2
SIM_REMOTE_Y = 2048
SIM_REMOTE_L1_STEP = 5
LANE_COLORS = {
    "direct": gymapi.Vec3(0.15, 0.35, 1.0),
    "boundary": gymapi.Vec3(0.1, 0.85, 0.2),
    "legacy": gymapi.Vec3(0.95, 0.15, 0.1),
}


@dataclass
class PolicyContract:
    default_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: float
    clip_actions: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float


@dataclass
class LaneState:
    kind: str
    start_q: np.ndarray
    previous_target_q: np.ndarray
    previous_physical_q: np.ndarray
    last_action: np.ndarray
    last_contacts: np.ndarray
    transition: PolicyTransitionGuard
    prime_gate: Optional[PolicyPrimeGate] = None
    policy_enabled: bool = True
    entry_rejection: str = ""
    faulted: bool = False
    fault_reason: str = ""
    clamp_count: int = 0
    reset_count: int = 0
    crossed_box: bool = False
    completed: bool = False
    termination_reason: str = ""
    max_target_step: float = 0.0
    max_transition_target_step: float = 0.0
    max_steady_target_step: float = 0.0
    max_torque_ratio: float = 0.0
    max_target_parity_error: float = 0.0
    joint_limit_violation: float = 0.0


class ReplayRecorder:
    """Collect homogeneous arrays and save a no-pickle NPZ artifact."""

    def __init__(self, lane_kinds: Sequence[str]) -> None:
        self.lane_kinds = tuple(lane_kinds)
        self.values: Dict[str, List[np.ndarray]] = defaultdict(list)

    def append(self, **values) -> None:
        for name, value in values.items():
            # Torch CPU tensors expose mutable NumPy views.  Keep a frame-owned
            # copy so later simulator steps cannot rewrite the replay history.
            self.values[name].append(np.asarray(value).copy())

    def save(self, log_dir: Path, summary: Dict[str, np.ndarray]) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = log_dir / f"unitree-boundary-s2s-{timestamp}.npz"
        arrays = {
            name: np.stack(samples, axis=0)
            for name, samples in self.values.items()
            if samples
        }
        arrays.update(summary)
        arrays["format_version"] = np.asarray(1, dtype=np.int64)
        arrays["lane_kind"] = np.asarray(self.lane_kinds, dtype="<U16")
        arrays["policy_order"] = np.asarray(POLICY_DOF_NAMES, dtype="<U24")
        arrays["unitree_motor_order"] = np.asarray(
            POLICY_DOF_NAMES,
            dtype="<U24",
        )
        arrays["isaac_order"] = np.asarray(
            [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
            dtype="<U24",
        )
        np.savez_compressed(path, **arrays)
        return path


def _joint_gains(values: Dict[str, float]) -> np.ndarray:
    result = []
    for name in POLICY_DOF_NAMES:
        matches = [float(value) for key, value in values.items() if key in name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one gain for joint '{name}'")
        result.append(matches[0])
    return np.asarray(result, dtype=np.float64)


def load_policy_contract(traced_dir: Path) -> PolicyContract:
    with (traced_dir / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    angles = config["init_state"]["default_joint_angles"]
    scales = config["normalization"]["obs_scales"]
    return PolicyContract(
        default_q=np.asarray(
            [float(angles[name]) for name in POLICY_DOF_NAMES],
            dtype=np.float64,
        ),
        kp=_joint_gains(config["control"]["stiffness"]),
        kd=_joint_gains(config["control"]["damping"]),
        action_scale=float(config["control"]["action_scale"]),
        clip_actions=float(config["normalization"]["clip_actions"]),
        ang_vel_scale=float(scales["ang_vel"]),
        dof_pos_scale=float(scales["dof_pos"]),
        dof_vel_scale=float(scales["dof_vel"]),
    )


def lane_kinds_from_environment() -> Tuple[str, ...]:
    comparison = os.environ.get("EXTREME_BOUNDARY_COMPARISON", "abc").lower()
    if comparison == "abc":
        return "direct", "boundary", "legacy"
    if comparison == "fixed":
        return ("boundary",)
    raise ValueError(
        "EXTREME_BOUNDARY_COMPARISON must be either 'abc' or 'fixed'"
    )


def align_lanes_to_first_terrain_row(env) -> None:
    """Move every lane to row zero while preserving its local robot state."""
    old_origins = env.env_origins.clone()
    env.terrain_levels.zero_()
    new_origins = env.terrain_origins[
        env.terrain_levels,
        env.terrain_types,
    ]
    local_position = env.root_states[:, :3] - old_origins
    env.env_origins[:] = new_origins
    env.root_states[:, :3] = new_origins + local_position
    env.env_class[:] = env.terrain_class[
        env.terrain_levels,
        env.terrain_types,
    ]
    goals = env.terrain_goals[env.terrain_levels, env.terrain_types]
    last_goal = goals[:, -1].unsqueeze(1)
    env.env_goals[:] = torch.cat(
        (
            goals,
            last_goal.repeat(1, env.cfg.env.num_future_goal_obs, 1),
        ),
        dim=1,
    )
    env.cur_goal_idx.zero_()
    env.cur_goals = env._gather_cur_goals()
    env.next_goals = env._gather_cur_goals(future=1)
    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
    )


def reset_lanes_to_default_pose(env, default_isaac: np.ndarray) -> None:
    """Remove the embedded replay copy's unconditional 0..0.9 rad reset noise."""
    target = torch.as_tensor(
        default_isaac,
        device=env.device,
        dtype=env.dof_pos.dtype,
    )
    env.dof_pos[:] = target.unsqueeze(0)
    env.dof_vel.zero_()
    env.gym.set_dof_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
    )


def color_and_frame_lanes(env, lane_kinds: Sequence[str]) -> None:
    if env.viewer is None:
        return
    for lane_index, kind in enumerate(lane_kinds):
        color = LANE_COLORS[kind]
        for body_index in range(env.num_bodies):
            env.gym.set_rigid_body_color(
                env.envs[lane_index],
                env.actor_handles[lane_index],
                body_index,
                gymapi.MESH_VISUAL,
                color,
            )
    origins = env.env_origins.detach().cpu().numpy()
    center_y = float(np.mean(origins[:, 1]))
    start_x = float(np.mean(origins[:, 0]))
    env.set_camera(
        [start_x - 3.0, center_y - 10.0, 5.5],
        [start_x + 2.0, center_y, 0.4],
    )


def synthesize_low_states(env) -> Tuple[List[BoundaryLowState], np.ndarray]:
    isaac_q = env.dof_pos.detach().cpu().numpy().astype(np.float64)
    isaac_dq = env.dof_vel.detach().cpu().numpy().astype(np.float64)
    isaac_foot_force = torch.linalg.norm(
        env.contact_forces[:, env.feet_indices],
        dim=-1,
    ).detach().cpu().numpy().astype(np.float64)
    gyroscope = env.base_ang_vel.detach().cpu().numpy().astype(np.float64)
    quaternion_xyzw = env.root_states[:, 3:7].detach().cpu().numpy().astype(
        np.float64
    )
    states = []
    for index in range(env.num_envs):
        quaternion = quaternion_xyzw[index]
        states.append(
            BoundaryLowState(
                motor_q=isaac_to_policy(isaac_q[index]),
                motor_dq=isaac_to_policy(isaac_dq[index]),
                foot_force=isaac_feet_to_unitree(isaac_foot_force[index]),
                gyroscope=gyroscope[index],
                imu_quaternion_wxyz=[
                    quaternion[3],
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                ],
            )
        )
    return states, isaac_foot_force


def legacy_decoded_state(state: DecodedLowState) -> DecodedLowState:
    """Reproduce the old double-reorder fault without exposing it to production."""
    return DecodedLowState(
        joint_q=state.joint_q[LEGACY_JOINT_REINDEX].copy(),
        joint_dq=state.joint_dq[LEGACY_JOINT_REINDEX].copy(),
        foot_force=state.foot_force[LEGACY_FOOT_REINDEX].copy(),
        gyroscope=state.gyroscope.copy(),
        roll_pitch=state.roll_pitch.copy(),
    )


def encode_lane_target(
    kind: str,
    target_q: np.ndarray,
    contract: PolicyContract,
):
    if kind == "legacy":
        return encode_low_cmd(
            target_q[LEGACY_JOINT_REINDEX],
            contract.kp[LEGACY_JOINT_REINDEX],
            contract.kd[LEGACY_JOINT_REINDEX],
        )
    return encode_low_cmd(target_q, contract.kp, contract.kd)


def reset_recurrent_rows(depth_encoder, done: torch.Tensor) -> None:
    if depth_encoder.hidden_states is None or not bool(done.any()):
        return
    depth_encoder.hidden_states[:, done, :] = 0.0


def boundary_summary(
    lane_states: Sequence[LaneState],
    max_steps: int,
    steps_executed: int,
    guard_mode: str,
    l1_event_step: int,
    y_event_step: int,
) -> Tuple[Dict[str, np.ndarray], bool, str]:
    boundary_index = [state.kind for state in lane_states].index("boundary")
    boundary = lane_states[boundary_index]
    checks = {
        "no_reset": boundary.reset_count == 0,
        "no_fault": not boundary.faulted,
        "crossed_box": boundary.crossed_box,
        "target_parity": boundary.max_target_parity_error <= 1e-6,
        "remote_l1": l1_event_step >= 0,
        "remote_y": y_event_step >= 0,
    }
    if guard_mode == "full":
        checks.update(
            {
                "transition_target_step": (
                    boundary.max_transition_target_step
                    <= POLICY_TRANSITION_MAX_STEP_RAD + 1e-6
                ),
                "joint_limits": boundary.joint_limit_violation <= 1e-8,
                "torque_limits": boundary.max_torque_ratio <= 1.000001,
            }
        )
    complete_run = boundary.completed or (
        max_steps >= 1000 and steps_executed >= max_steps
    )
    passed = bool(all(checks.values())) if complete_run else False
    status = "PASS" if passed else ("INCOMPLETE" if not complete_run else "FAIL")
    summary = {
        "summary_status": np.asarray(status, dtype="<U16"),
        "summary_complete_run": np.asarray(complete_run, dtype=np.bool_),
        "summary_steps_executed": np.asarray(steps_executed, dtype=np.int64),
        "summary_l1_event_step": np.asarray(l1_event_step, dtype=np.int64),
        "summary_y_event_step": np.asarray(y_event_step, dtype=np.int64),
        "summary_passed": np.asarray(passed, dtype=np.bool_),
        "summary_guard_mode": np.asarray(guard_mode, dtype="<U16"),
        "summary_boundary_checks": np.asarray(
            [checks[name] for name in sorted(checks)],
            dtype=np.bool_,
        ),
        "summary_boundary_check_names": np.asarray(
            sorted(checks),
            dtype="<U24",
        ),
        "summary_reset_count": np.asarray(
            [state.reset_count for state in lane_states],
            dtype=np.int64,
        ),
        "summary_crossed_box": np.asarray(
            [state.crossed_box for state in lane_states],
            dtype=np.bool_,
        ),
        "summary_max_target_step": np.asarray(
            [state.max_target_step for state in lane_states],
            dtype=np.float64,
        ),
        "summary_max_torque_ratio": np.asarray(
            [state.max_torque_ratio for state in lane_states],
            dtype=np.float64,
        ),
        "summary_max_target_parity_error": np.asarray(
            [state.max_target_parity_error for state in lane_states],
            dtype=np.float64,
        ),
        "summary_clamp_count": np.asarray(
            [state.clamp_count for state in lane_states],
            dtype=np.int64,
        ),
        "summary_faulted": np.asarray(
            [state.faulted for state in lane_states],
            dtype=np.bool_,
        ),
        "summary_fault_reason": np.asarray(
            [state.fault_reason for state in lane_states],
            dtype="<U160",
        ),
        "summary_entry_rejection": np.asarray(
            [state.entry_rejection for state in lane_states],
            dtype="<U160",
        ),
        "summary_completed": np.asarray(
            [state.completed for state in lane_states],
            dtype=np.bool_,
        ),
        "summary_termination_reason": np.asarray(
            [state.termination_reason for state in lane_states],
            dtype="<U32",
        ),
        "summary_max_transition_target_step": np.asarray(
            [state.max_transition_target_step for state in lane_states],
            dtype=np.float64,
        ),
        "summary_max_steady_target_step": np.asarray(
            [state.max_steady_target_step for state in lane_states],
            dtype=np.float64,
        ),
    }
    detail = ", ".join(f"{name}={value}" for name, value in checks.items())
    return summary, passed, detail


@torch.inference_mode()
def replay(args) -> None:
    lane_kinds = lane_kinds_from_environment()
    num_lanes = len(lane_kinds)
    max_steps = int(os.environ.get("EXTREME_REPLAY_STEPS", "1000"))
    guard_mode = os.environ.get(
        "EXTREME_BOUNDARY_GUARDS",
        "full",
    ).lower()
    if guard_mode not in ("full", "mapping"):
        raise ValueError(
            "EXTREME_BOUNDARY_GUARDS must be either 'full' or 'mapping'"
        )
    ground_noise_m = float(
        os.environ.get("EXTREME_GROUND_NOISE_M", "0.0")
    )
    ground_noise_seed = int(
        os.environ.get("EXTREME_GROUND_NOISE_SEED", "17")
    )
    ground_noise_patch_m = float(
        os.environ.get("EXTREME_GROUND_NOISE_PATCH_M", "0.2")
    )
    if ground_noise_m < 0.0:
        raise ValueError("EXTREME_GROUND_NOISE_M must be non-negative")
    log_dir = Path(
        os.environ.get(
            "EXTREME_BOUNDARY_LOG_DIR",
            "~/extreme-boundary-s2s",
        )
    ).expanduser().resolve()
    traced_dir = Path(
        os.environ.get(
            "EXTREME_TRACED_DIR",
            Path(__file__).resolve().parents[2] / "traced",
        )
    ).expanduser().resolve()
    contract = load_policy_contract(traced_dir)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_replay_environment(
        env_cfg,
        num_envs=num_lanes,
        num_terrain_columns=num_lanes,
    )
    install_single_box_terrain(
        ground_noise_m,
        ground_noise_seed,
        ground_noise_patch_m,
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    align_lanes_to_first_terrain_row(env)
    color_and_frame_lanes(env, lane_kinds)
    device = torch.device(env.device)

    base_model, depth_encoder = load_exported_models(traced_dir, device)
    estimator = base_model.estimator.estimator
    history_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    activation = nn.ELU()

    default_isaac = env.default_dof_pos_all[0].detach().cpu().numpy().astype(
        np.float64
    )
    expected_default_isaac = np.asarray(
        policy_to_isaac(contract.default_q),
        dtype=np.float64,
    )
    if not np.allclose(default_isaac, expected_default_isaac, atol=1e-7):
        raise RuntimeError("traced and Isaac default joint positions differ")
    reset_lanes_to_default_pose(env, default_isaac)
    dt = float(env.dt)
    observation = env.get_observations()
    boundary_states, _ = synthesize_low_states(env)
    decoded_states = [decode_low_state(state) for state in boundary_states]
    lane_states = []
    for index, kind in enumerate(lane_kinds):
        decoded = decoded_states[index]
        if kind == "legacy":
            decoded = legacy_decoded_state(decoded)
        start_q = decoded.joint_q.copy()
        physical_start = np.asarray(
            real_to_sim(encode_lane_target(kind, start_q, contract).motor_q),
            dtype=np.float64,
        )
        lane_states.append(
            LaneState(
                kind=kind,
                start_q=start_q,
                previous_target_q=start_q.copy(),
                previous_physical_q=physical_start,
                last_action=np.zeros(12, dtype=np.float64),
                last_contacts=np.zeros(4, dtype=np.bool_),
                transition=PolicyTransitionGuard(),
            )
        )

    history = torch.zeros(
        num_lanes,
        env.cfg.env.history_len,
        env.cfg.env.n_proprio,
        device=device,
        dtype=torch.float32,
    )
    episode_length = torch.zeros(num_lanes, device=device, dtype=torch.float32)
    visual_output = None
    previous_depth = None
    latest_depth_time = None
    context_cycle = 0
    phase = "dryrun"
    startup_start_time = None
    remote = RemoteEdgeTracker()
    l1_event_step = -1
    y_event_step = -1
    recorder = ReplayRecorder(lane_kinds)
    box_rear = single_box_rear_x(env.env_origins).detach().cpu().numpy()

    print(f"Loaded exported models from: {traced_dir}")
    print("Boundary S2S lanes: " + ", ".join(
        f"{index}={kind}" for index, kind in enumerate(lane_kinds)
    ))
    if lane_kinds == ("direct", "boundary", "legacy"):
        print("Viewer colors: blue=direct, green=fixed boundary, red=legacy fault")
    print("Simulated remote sequence: L1 -> 3.0 s stand -> 0.5 s prime -> Y")
    print(
        f"Ground noise: +/-{ground_noise_m:.4f} m, "
        f"patch={ground_noise_patch_m:.3f} m, seed={ground_noise_seed}"
    )
    print(f"Boundary guard mode: {guard_mode}")
    print(f"NPZ log directory: {log_dir}")
    if not args.headless:
        print("Isaac Gym Viewer is running. Press Esc in the viewer to exit.")

    step = 0
    run_error: Optional[BaseException] = None
    try:
        while max_steps <= 0 or step < max_steps:
            sim_time = step * dt
            env.commands.zero_()
            env.commands[:, 0] = FORWARD_COMMAND_MPS

            low_states, isaac_foot_force = synthesize_low_states(env)
            decoded = [decode_low_state(state) for state in low_states]
            for index, lane in enumerate(lane_states):
                if lane.kind == "legacy":
                    decoded[index] = legacy_decoded_state(decoded[index])

            simulated_keys = 0
            if step == SIM_REMOTE_L1_STEP:
                simulated_keys = SIM_REMOTE_L1
            elif phase == "stand_hold" and y_event_step < 0:
                simulated_keys = SIM_REMOTE_Y
            remote.update(simulated_keys, sim_time)
            l1_rising = remote.consume_rising(SIM_REMOTE_L1)
            y_rising = remote.consume_rising(SIM_REMOTE_Y)

            if phase == "dryrun" and l1_rising:
                boundary_index = [
                    state.kind for state in lane_states
                ].index("boundary")
                validate_takeover_inputs(
                    decoded[boundary_index].joint_q,
                    decoded[boundary_index].joint_dq,
                    0.0,
                    0.0,
                )
                l1_event_step = step
                startup_start_time = sim_time
                phase = "startup"
                for index, lane in enumerate(lane_states):
                    lane.start_q = decoded[index].joint_q.copy()
                    lane.previous_target_q = lane.start_q.copy()
                    lane.previous_physical_q = np.asarray(
                        real_to_sim(
                            encode_lane_target(
                                lane.kind,
                                lane.start_q,
                                contract,
                            ).motor_q
                        ),
                        dtype=np.float64,
                    )
                print(f"step={step} simulated L1 takeover accepted")

            if phase == "startup":
                if startup_start_time is None:
                    raise RuntimeError("startup time is unavailable")
                if sim_time - startup_start_time >= STARTUP_RAMP_S:
                    phase = "prime"
                    history.zero_()
                    episode_length.zero_()
                    depth_encoder.hidden_states = None
                    visual_output = None
                    previous_depth = None
                    latest_depth_time = None
                    context_cycle = 0
                    for lane in lane_states:
                        lane.last_action.fill(0.0)
                        lane.last_contacts.fill(False)
                        lane.prime_gate = PolicyPrimeGate(sim_time)
                    print(f"step={step} startup complete; policy prime started")

            if phase == "stand_hold" and y_rising:
                if remote.latest_time is None:
                    raise RuntimeError("simulated remote time is unavailable")
                validate_policy_request_input(sim_time - remote.latest_time)
                for index, lane in enumerate(lane_states):
                    if lane.kind != "direct":
                        try:
                            validate_policy_entry_state(
                                decoded[index].foot_force,
                                decoded[index].roll_pitch[0],
                                decoded[index].roll_pitch[1],
                                decoded[index].joint_q,
                                lane.previous_target_q,
                                np.zeros(12),
                                np.zeros(12),
                                np.zeros(12),
                            )
                        except RuntimeError as error:
                            if lane.kind == "boundary":
                                raise
                            lane.policy_enabled = False
                            lane.entry_rejection = str(error)
                            print(
                                f"step={step} lane={lane.kind} policy "
                                f"entry rejected: {error}"
                            )
                    lane.transition.begin(
                        lane.previous_target_q,
                        sim_time,
                    )
                y_event_step = step
                phase = "policy"
                print(f"step={step} simulated Y policy request accepted")

            proprio = None
            depth_updated = False
            if phase in ("prime", "policy"):
                proprio_rows = []
                for index, lane in enumerate(lane_states):
                    if lane.kind == "direct":
                        row = fixed_command_observation(
                            observation[index : index + 1, : env.cfg.env.n_proprio]
                        )[0].detach().cpu().numpy().astype(np.float32)
                        lane.last_contacts = row[-4:] > 0.0
                    else:
                        row, current_contacts = build_policy_proprio(
                            decoded[index],
                            contract.default_q,
                            lane.last_action,
                            lane.last_contacts,
                            FORWARD_COMMAND_MPS,
                            contract.ang_vel_scale,
                            contract.dof_pos_scale,
                            contract.dof_vel_scale,
                            "parkour",
                        )
                        lane.last_contacts = current_contacts
                    proprio_rows.append(row)
                proprio = torch.as_tensor(
                    np.stack(proprio_rows),
                    device=device,
                    dtype=torch.float32,
                )
                history = update_proprio_history(
                    history,
                    proprio,
                    episode_length,
                )
                episode_length += 1

                if context_cycle % VISUAL_UPDATE_INTERVAL == 0:
                    current_depth = render_single_box_depth(env)
                    depth_input = (
                        current_depth if previous_depth is None else previous_depth
                    )
                    visual_output = depth_encoder(depth_input, proprio)
                    previous_depth = current_depth
                    latest_depth_time = sim_time
                    depth_updated = True
                    if not torch.isfinite(visual_output).all():
                        raise RuntimeError("vision encoder produced NaN or Inf")

                if phase == "prime":
                    if latest_depth_time is None:
                        raise RuntimeError("policy prime has no depth sample")
                    validate_policy_prime_inputs(0.0, sim_time - latest_depth_time)
                    ready = []
                    for lane in lane_states:
                        if lane.prime_gate is None:
                            raise RuntimeError("policy prime gate is unavailable")
                        lane.prime_gate.record_proprio()
                        if depth_updated:
                            lane.prime_gate.record_depth()
                        ready.append(lane.prime_gate.ready(sim_time))
                    if all(ready):
                        phase = "stand_hold"
                        print(
                            f"step={step} prime complete; waiting for simulated Y"
                        )
                context_cycle += 1

            raw_actions = np.zeros((num_lanes, 12), dtype=np.float64)
            if phase == "policy":
                if proprio is None or visual_output is None:
                    raise RuntimeError("policy context is unavailable")
                proprio_for_actor = proprio.clone()
                proprio_for_actor[:, 6:8] = visual_output[:, -2:] * 1.5
                actor_observation = torch.cat(
                    (
                        proprio_for_actor,
                        visual_output[:, :-2],
                        estimator(proprio_for_actor),
                        history_encoder(activation, history),
                    ),
                    dim=-1,
                )
                action_tensor = actor(actor_observation)
                if action_tensor.shape != (num_lanes, 12):
                    raise RuntimeError("actor action shape is invalid")
                if not torch.isfinite(action_tensor).all():
                    raise RuntimeError("actor produced NaN or Inf")
                raw_actions = action_tensor.detach().cpu().numpy().astype(
                    np.float64
                )

            env_actions = np.zeros((num_lanes, 12), dtype=np.float64)
            requested_targets = np.zeros((num_lanes, 12), dtype=np.float64)
            commanded_targets = np.zeros((num_lanes, 12), dtype=np.float64)
            lowcmd_motor_q = np.zeros((num_lanes, 12), dtype=np.float64)
            isaac_targets = np.zeros((num_lanes, 12), dtype=np.float64)
            estimated_torque = np.zeros((num_lanes, 12), dtype=np.float64)
            target_steps = np.zeros(num_lanes, dtype=np.float64)
            guard_clamped = np.zeros(num_lanes, dtype=np.bool_)
            transition_active = np.zeros(num_lanes, dtype=np.bool_)

            for index, lane in enumerate(lane_states):
                if phase == "dryrun":
                    requested = lane.start_q.copy()
                    commanded = lane.start_q.copy()
                elif phase == "startup":
                    if startup_start_time is None:
                        raise RuntimeError("startup time is unavailable")
                    logical_target = interpolate_pose(
                        lane.start_q,
                        contract.default_q,
                        sim_time - startup_start_time,
                        STARTUP_RAMP_S,
                    )
                    requested = logical_target.copy()
                    commanded = logical_target.copy()
                elif phase in ("prime", "stand_hold") or not lane.policy_enabled:
                    requested = contract.default_q.copy()
                    commanded = contract.default_q.copy()
                elif lane.kind == "direct":
                    _, _, requested = prepare_policy_action(
                        raw_actions[index],
                        contract.default_q,
                        contract.clip_actions,
                        contract.action_scale,
                    )
                    commanded = requested.copy()
                else:
                    requested = lane.previous_target_q.copy()
                    if not lane.faulted:
                        try:
                            transition_active[index] = lane.transition.active
                            observed_action, _, requested = prepare_policy_action(
                                raw_actions[index],
                                contract.default_q,
                                contract.clip_actions,
                                contract.action_scale,
                            )
                            lane.last_action = observed_action
                            if guard_mode == "mapping":
                                engagement_active = lane.transition.active
                                commanded = (
                                    lane.transition.apply(requested, sim_time)
                                    if engagement_active
                                    else requested.copy()
                                )
                                if engagement_active:
                                    lane.transition.record_executed_target(
                                        commanded
                                    )
                            else:
                                if latest_depth_time is None:
                                    raise RuntimeError(
                                        "runtime depth timestamp is unavailable"
                                    )
                                validate_policy_runtime_inputs(
                                    0.0,
                                    sim_time - latest_depth_time,
                                    decoded[index].joint_q,
                                    decoded[index].joint_dq,
                                    GO2_JOINT_LIMITS_LOW,
                                    GO2_JOINT_LIMITS_HIGH,
                                    GO2_JOINT_VELOCITY_LIMITS,
                                )
                                engagement_active = lane.transition.active
                                transition_target = (
                                    lane.transition.apply(requested, sim_time)
                                    if engagement_active
                                    else requested
                                )
                                commanded = constrain_policy_target(
                                    transition_target,
                                    lane.previous_target_q,
                                    decoded[index].joint_q,
                                    decoded[index].joint_dq,
                                    contract.kp,
                                    contract.kd,
                                    GO2_JOINT_LIMITS_LOW,
                                    GO2_JOINT_LIMITS_HIGH,
                                    GO2_TORQUE_LIMITS,
                                    max_step_rad=(
                                        POLICY_TRANSITION_MAX_STEP_RAD
                                        if engagement_active
                                        else POLICY_TARGET_MAX_STEP_RAD
                                    ),
                                )
                                if engagement_active:
                                    lane.transition.record_executed_target(
                                        commanded
                                    )
                            guard_clamped[index] = not np.allclose(
                                commanded,
                                requested,
                                atol=1e-12,
                                rtol=0.0,
                            )
                            lane.clamp_count += int(guard_clamped[index])
                        except RuntimeError as error:
                            lane.faulted = True
                            lane.fault_reason = str(error)
                            print(
                                f"step={step} lane={lane.kind} production "
                                f"guard fault: {error}"
                            )
                    if lane.faulted:
                        # env.step has no motor-off mode. This zero-torque
                        # target lets the other comparison lanes continue.
                        commanded = np.clip(
                            decoded[index].joint_q
                            + contract.kd * decoded[index].joint_dq / contract.kp,
                            np.maximum(
                                GO2_JOINT_LIMITS_LOW,
                                contract.default_q - contract.clip_actions,
                            ),
                            np.minimum(
                                GO2_JOINT_LIMITS_HIGH,
                                contract.default_q + contract.clip_actions,
                            ),
                        )

                if lane.kind == "direct" and phase == "policy":
                    lane.last_action = raw_actions[index].copy()
                lowcmd = encode_lane_target(lane.kind, commanded, contract)
                physical_policy_target = np.asarray(
                    real_to_sim(lowcmd.motor_q),
                    dtype=np.float64,
                )
                isaac_target = np.asarray(
                    policy_to_isaac(physical_policy_target),
                    dtype=np.float64,
                )
                env_action = (
                    physical_policy_target - contract.default_q
                ) / contract.action_scale
                predicted_isaac_target = default_isaac + np.asarray(
                    policy_to_isaac(env_action),
                    dtype=np.float64,
                ) * contract.action_scale
                parity_error = float(
                    np.max(np.abs(predicted_isaac_target - isaac_target))
                )
                if parity_error > 1e-7:
                    raise RuntimeError("LowCmd to Isaac target conversion differs")

                motor_q = np.asarray(low_states[index].motor_q, dtype=np.float64)
                motor_dq = np.asarray(low_states[index].motor_dq, dtype=np.float64)
                torque = (
                    lowcmd.motor_kp * (lowcmd.motor_q - motor_q)
                    - lowcmd.motor_kd * motor_dq
                )
                target_step = float(
                    np.max(
                        np.abs(
                            physical_policy_target - lane.previous_physical_q
                        )
                    )
                )
                if not lane.faulted:
                    lane.max_target_step = max(lane.max_target_step, target_step)
                    if phase == "policy":
                        if transition_active[index]:
                            lane.max_transition_target_step = max(
                                lane.max_transition_target_step,
                                target_step,
                            )
                        else:
                            lane.max_steady_target_step = max(
                                lane.max_steady_target_step,
                                target_step,
                            )
                    lane.max_torque_ratio = max(
                        lane.max_torque_ratio,
                        float(np.max(np.abs(torque) / GO2_TORQUE_LIMITS)),
                    )
                lower_violation = np.maximum(
                    GO2_JOINT_LIMITS_LOW - physical_policy_target,
                    0.0,
                )
                upper_violation = np.maximum(
                    physical_policy_target - GO2_JOINT_LIMITS_HIGH,
                    0.0,
                )
                if not lane.faulted:
                    lane.joint_limit_violation = max(
                        lane.joint_limit_violation,
                        float(
                            np.max(np.maximum(lower_violation, upper_violation))
                        ),
                    )
                lane.max_target_parity_error = max(
                    lane.max_target_parity_error,
                    parity_error,
                )
                lane.previous_target_q = commanded.copy()
                lane.previous_physical_q = physical_policy_target.copy()

                env_actions[index] = env_action
                requested_targets[index] = requested
                commanded_targets[index] = commanded
                lowcmd_motor_q[index] = lowcmd.motor_q
                isaac_targets[index] = isaac_target
                estimated_torque[index] = torque
                target_steps[index] = target_step

            observation, _, _, done, _ = env.step(
                torch.as_tensor(
                    env_actions,
                    device=device,
                    dtype=torch.float32,
                )
            )
            actual_isaac_target = (
                env.default_dof_pos_all + env.actions * contract.action_scale
            ).detach().cpu().numpy().astype(np.float64)
            parity = np.max(
                np.abs(actual_isaac_target - isaac_targets),
                axis=1,
            )
            if bool(np.any(parity > 1e-6)):
                bad = np.flatnonzero(parity > 1e-6).tolist()
                raise RuntimeError(
                    f"PhysX PD target differs from LowCmd for lanes {bad}"
                )
            for index, lane in enumerate(lane_states):
                lane.max_target_parity_error = max(
                    lane.max_target_parity_error,
                    float(parity[index]),
                )
                if bool(done[index]):
                    if lane.crossed_box and bool(env.time_out_buf[index]):
                        lane.completed = True
                        lane.termination_reason = "goal_complete"
                    else:
                        lane.reset_count += 1
                        if bool(env.time_out_buf[index]):
                            lane.termination_reason = "timeout_before_box"
                        elif abs(float(env.roll[index])) > 1.5:
                            lane.termination_reason = "roll_limit"
                        elif abs(float(env.pitch[index])) > 1.5:
                            lane.termination_reason = "pitch_limit"
                        else:
                            lane.termination_reason = "height_or_other"
                    history[index].zero_()
                    episode_length[index] = 0.0
                    lane.last_action.fill(0.0)
                    lane.last_contacts.fill(False)
                position_x = float(env.root_states[index, 0])
                if position_x > float(box_rear[index]):
                    lane.crossed_box = True
                    if lane.kind == "boundary":
                        lane.completed = True
                        lane.termination_reason = "crossed_box"
            reset_recurrent_rows(depth_encoder, done)

            decoded_q = np.stack([state.joint_q for state in decoded])
            decoded_dq = np.stack([state.joint_dq for state in decoded])
            recorder.append(
                simulation_time=np.asarray(sim_time, dtype=np.float64),
                phase=np.asarray(
                    [
                        PHASE_CODES[
                            "rejected"
                            if not lane.policy_enabled or lane.faulted
                            else phase
                        ]
                        for lane in lane_states
                    ],
                    dtype=np.int8,
                ),
                root_position=env.root_states[:, :3].detach().cpu().numpy(),
                root_orientation_xyzw=(
                    env.root_states[:, 3:7].detach().cpu().numpy()
                ),
                measured_isaac_q=env.dof_pos.detach().cpu().numpy(),
                lowstate_motor_q=np.stack(
                    [np.asarray(state.motor_q) for state in low_states]
                ),
                lowstate_motor_dq=np.stack(
                    [np.asarray(state.motor_dq) for state in low_states]
                ),
                actor_q=decoded_q,
                actor_dq=decoded_dq,
                foot_force_isaac=isaac_foot_force,
                foot_force_unitree=np.stack(
                    [np.asarray(state.foot_force) for state in low_states]
                ),
                contact_state=np.stack(
                    [lane.last_contacts for lane in lane_states]
                ),
                raw_actor_action=raw_actions,
                requested_q=requested_targets,
                commanded_q=commanded_targets,
                lowcmd_motor_q=lowcmd_motor_q,
                isaac_target_q=isaac_targets,
                actual_isaac_target_q=actual_isaac_target,
                estimated_torque=estimated_torque,
                target_step=target_steps,
                guard_clamped=guard_clamped,
                faulted=np.asarray(
                    [lane.faulted for lane in lane_states],
                    dtype=np.bool_,
                ),
                done=done.detach().cpu().numpy(),
                completed=np.asarray(
                    [lane.completed for lane in lane_states],
                    dtype=np.bool_,
                ),
                simulated_remote_keys=np.asarray(
                    remote.keys,
                    dtype=np.int64,
                ),
                simulated_l1_rising=np.asarray(
                    l1_rising,
                    dtype=np.bool_,
                ),
                simulated_y_rising=np.asarray(
                    y_rising,
                    dtype=np.bool_,
                ),
                lowcmd_authorized=np.asarray(
                    phase != "dryrun",
                    dtype=np.bool_,
                ),
            )

            step += 1
            boundary = lane_states[
                [state.kind for state in lane_states].index("boundary")
            ]
            if boundary.completed:
                print(
                    f"step={step} boundary completed: "
                    f"{boundary.termination_reason}"
                )
                break
            if step % 250 == 0:
                positions = env.root_states[:, 0].detach().cpu().numpy()
                details = " | ".join(
                    (
                        f"{lane.kind}:x={positions[index]:.2f},"
                        f"reset={lane.reset_count},"
                        f"step={lane.max_target_step:.3f},"
                        f"clamp={lane.clamp_count}"
                    )
                    for index, lane in enumerate(lane_states)
                )
                print(f"step={step} phase={phase} | {details}")
    except BaseException as error:
        run_error = error
    finally:
        summary, passed, detail = boundary_summary(
            lane_states,
            max_steps,
            step,
            guard_mode,
            l1_event_step,
            y_event_step,
        )
        summary["summary_ground_noise_m"] = np.asarray(
            ground_noise_m,
            dtype=np.float64,
        )
        summary["summary_ground_noise_seed"] = np.asarray(
            ground_noise_seed,
            dtype=np.int64,
        )
        summary["summary_ground_noise_patch_m"] = np.asarray(
            ground_noise_patch_m,
            dtype=np.float64,
        )
        if recorder.values:
            log_path = recorder.save(log_dir, summary)
            print(f"Boundary S2S log saved: {log_path}")
        else:
            log_path = None
        print(f"Boundary S2S result: {summary['summary_status']} ({detail})")

    if run_error is not None:
        raise run_error
    if max_steps >= 1000 and not passed:
        raise RuntimeError(
            "fixed Unitree boundary failed the 1000-step acceptance checks"
        )


if __name__ == "__main__":
    replay(get_args())
