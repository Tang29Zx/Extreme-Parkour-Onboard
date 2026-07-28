import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    SportModeState,
    LowCmd,
)
from unitree_api.msg import Request, Response

from std_msgs.msg import Float32MultiArray

from joint_mapping import (
    DOF_SIGNS,
    FOOT_REAL_TO_SIM,
    NUM_DOF as GO2_NUM_DOF,
    SIM_DOF_NAMES,
    SIM_TO_REAL_DOF,
)
from output_routing import resolve_output_topics
from policy_context import reset_policy_context, update_proprio_history
from real_control_safety import (
    CHECK_MODE_API_ID,
    MOTION_REQUEST_TOPIC,
    MOTION_RESPONSE_TOPIC,
    REAL_LOW_COMMAND_TOPIC,
    RELEASE_MODE_API_ID,
    RPC_MAX_ATTEMPTS,
    RPC_TIMEOUT_S,
    POLICY_TARGET_MAX_STEP_RAD_BY_JOINT,
    STARTUP_RAMP_S,
    TAKEOVER_HOLD_S,
    DepthStaleError,
    LowStateStaleError,
    PolicyPrimeGate,
    PolicyTransitionGuard,
    RemoteEdgeTracker,
    RealControlError,
    build_motion_request,
    constrain_policy_target,
    executed_target_to_action,
    filter_foot_contacts,
    interpolate_pose,
    parse_motion_response,
    prepare_policy_action,
    release_mode_required,
    update_motor_lost_baseline,
    validate_policy_prime_inputs,
    validate_policy_runtime_inputs,
    validate_policy_request_input,
    validate_policy_entry_state,
    validate_real_low_command_publish,
    validate_takeover_inputs,
)
from unitree_boundary import (
    BoundaryLowState,
    GO2_JOINT_LIMITS_HIGH,
    GO2_JOINT_LIMITS_LOW,
    GO2_JOINT_VELOCITY_LIMITS,
    GO2_TORQUE_LIMITS,
    build_policy_proprio,
    decode_low_state,
    encode_low_cmd,
)

if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "x86",
    ))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "aarch64",
    ))
from crc_module import get_crc

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch
import time


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class RobotCfgs:
    class H1:
        pass

    class Go2:
        NUM_DOF = GO2_NUM_DOF
        NUM_ACTIONS = 12
        dof_map = SIM_TO_REAL_DOF
        dof_names = SIM_DOF_NAMES
        dof_signs = DOF_SIGNS
        joint_limits_high = torch.as_tensor(
            GO2_JOINT_LIMITS_HIGH.copy(),
            device="cpu",
            dtype=torch.float32,
        )
        joint_limits_low = torch.as_tensor(
            GO2_JOINT_LIMITS_LOW.copy(),
            device="cpu",
            dtype=torch.float32,
        )
        torque_limits = torch.as_tensor(
            GO2_TORQUE_LIMITS.copy(),
            device="cpu",
            dtype=torch.float32,
        )
        joint_velocity_limits = torch.as_tensor(
            GO2_JOINT_VELOCITY_LIMITS.copy(),
            device="cpu",
            dtype=torch.float32,
        )
        turn_on_motor_mode = [0x01] * 12
        

class UnitreeRos2Real(Node):
    """ A proxy implementation of the real H1 robot. """
    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self,
            robot_namespace= None,
            low_state_topic= "/lowstate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            depth_data_topic= "/forward_depth_image",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            move_by_wireless_remote= True, # if True, the robot will be controlled by a wireless remote
            model_device= "cpu",
            dof_pos_protect_ratio= 1.1, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "Go2",
            dryrun= True, # if True, the robot will not send commands to the real robot
            mode= "parkour",
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        dryrun_suffix = int(np.random.randint(0, 65535)) if dryrun else None
        (
            self.low_cmd_topic,
            self.sport_state_topic,
            self.sport_mode_topic,
        ) = resolve_output_topics(low_cmd_topic, dryrun, dryrun_suffix)
        self.sport_state_pub = None
        self.sport_mode_pub = None
        self.low_cmd_pub = None
        self.low_cmd_buffer = None
        self.real_lowcmd_authorized = False
        self.motion_request_pub = None
        self.motion_responses = []
        self.latest_low_state_time = None
        self.latest_remote_time = None
        self.latest_depth_time = None
        self.remote_keys = 0
        self.remote_rising_edges = 0
        self.remote_tracker = RemoteEdgeTracker()
        self.real_control_phase = "dryrun" if dryrun else "sport"
        self.startup_ramp_start_time = None
        self.startup_ramp_start_q = None
        self.stand_recovery_start_time = None
        self.stand_recovery_start_q = None
        self.last_command_target_q = None
        self.policy_prime_gate = None
        self.policy_prime_cycle = 0
        self.policy_prime_rejection_reason = None
        self.policy_lost_baseline = None
        self.policy_depth_reset = None
        self.policy_transition = PolicyTransitionGuard()
        self.joy_stick_topic = joy_stick_topic
        self.depth_data_topic = depth_data_topic
        self.cfg = cfg
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun
        self.mode = mode

        self.dof_map = getattr(RobotCfgs, robot_class_name).dof_map
        self.dof_names = getattr(RobotCfgs, robot_class_name).dof_names
        self.dof_signs = getattr(RobotCfgs, robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, robot_class_name).turn_on_motor_mode

        self.n_proprio = 53
        self.n_depth_latent = 32
        self.n_hist_len = 10

        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.forward_depth_latent_yaw_buffer = torch.zeros(1, self.n_depth_latent+2, device=self.model_device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device= self.model_device, dtype= torch.float32)
        self.contact_filt = torch.full((1, 4), -0.5, device=self.model_device, dtype=torch.float32)
        self.last_contacts = np.zeros(4, dtype=np.bool_)

        self.parse_config()
        self.init_stand_config()

    def init_stand_config(self):
        self.startPos = [0.0] * 12
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self.stand_action = [0.0] * 12

        self.duration_1 = 10
        self.duration_2 = 100
        self.percent_1 = 0
        self.percent_2 = 0

        self.firstrun_target_1 = True
        self.firstRun = True

    def reset_obs(self):
        self.startPos = [0.0] * 12
        self.stand_action = [0.0] * 12

        self.percent_1 = 0
        self.percent_2 = 0

        self.firstrun_target_1 = True
        self.firstRun = True

        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)    
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.forward_depth_latent_yaw_buffer = torch.zeros(1, self.n_depth_latent+2, device=self.model_device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device= self.model_device, dtype= torch.float32)
        self.contact_filt = torch.full((1, 4), -0.5, device=self.model_device, dtype=torch.float32)
        self.last_contacts = np.zeros(4, dtype=np.bool_)


    def parse_config(self):
        """ parse, set attributes from config dict, initialize buffers to speed up the computation """

        # observation
        self.clip_obs = self.cfg["normalization"]["clip_observations"]

        # controls
        self.control_type = self.cfg["control"]["control_type"]
        if not (self.control_type == "P"):
            raise NotImplementedError("Only position control is supported for now.")
        
        self.p_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            for k, v in self.cfg["control"]["stiffness"].items():
                if k in name:
                    self.p_gains.append(v)
                    break 
        self.p_gains = torch.tensor(self.p_gains, device= self.model_device, dtype= torch.float32)
        self.p_gains_np = (
            self.p_gains.detach().cpu().numpy().astype(np.float64)
        )

        self.d_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] 
            for k, v in self.cfg["control"]["damping"].items():
                if k in name:
                    self.d_gains.append(v)
                    break
        self.d_gains = torch.tensor(self.d_gains, device= self.model_device, dtype= torch.float32)
        self.d_gains_np = (
            self.d_gains.detach().cpu().numpy().astype(np.float64)
        )

        self.default_dof_pos = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            self.default_dof_pos[i] = default_joint_angle
        self.default_dof_pos_np = (
            self.default_dof_pos.detach().cpu().numpy().astype(np.float64)
        )

        # actions
        self.num_actions = self.NUM_ACTIONS
        self.action_scale = self.cfg["control"]["action_scale"]
        self.get_logger().info("[Env] action scale: {:.2f}".format(self.action_scale))
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"].get("clip_actions_method", None) == "hard":
            self.get_logger().info("clip_actions_method with hard mode")
            self.get_logger().info("clip_actions_high: " + str(self.cfg["normalization"]["clip_actions_high"]))
            self.get_logger().info("clip_actions_low: " + str(self.cfg["normalization"]["clip_actions_low"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = torch.tensor(self.cfg["normalization"]["clip_actions_low"], device= self.model_device, dtype= torch.float32)
            self.clip_actions_high = torch.tensor(self.cfg["normalization"]["clip_actions_high"], device= self.model_device, dtype= torch.float32)
        else:
            self.get_logger().info("clip_actions_method is " + str(self.cfg["normalization"].get("clip_actions_method", None)))
        
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)    

        ###################### hardware related #####################
        robot_cfg = getattr(RobotCfgs, self.robot_class_name)
        self.joint_limits_high = robot_cfg.joint_limits_high.to(self.model_device)
        self.joint_limits_low = robot_cfg.joint_limits_low.to(self.model_device)
        self.joint_limits_high_np = (
            self.joint_limits_high.detach().cpu().numpy().astype(np.float64)
        )
        self.joint_limits_low_np = (
            self.joint_limits_low.detach().cpu().numpy().astype(np.float64)
        )
        self.torque_limits_np = (
            robot_cfg.torque_limits.detach().cpu().numpy().astype(np.float64)
        )
        self.joint_velocity_limits_np = (
            robot_cfg.joint_velocity_limits.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio

    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """
        # Dry-run uses an isolated LowCmd-like topic immediately. Real output is
        # created only after MotionSwitcher and graph-ownership checks pass.
        if self.dryrun:
            self.low_cmd_pub = self.create_publisher(
                LowCmd,
                self.low_cmd_topic,
                1,
            )
            self.low_cmd_buffer = LowCmd()

        # ROS subscribers
        self.low_state_sub = self.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.get_logger().info("Low state subscriber started, waiting to receive low state messages.")

        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )
        self.get_logger().info("Wireless controller subscriber started, waiting to receive wireless controller messages.")

        if not self.dryrun:
            self.sport_state_pub = self.create_publisher(
                Request,
                self.sport_state_topic,
                1,
            )
            self.sport_mode_pub = self.create_publisher(
                Request,
                self.sport_mode_topic,
                1,
            )
            self.motion_request_pub = self.create_publisher(
                Request,
                MOTION_REQUEST_TOPIC,
                10,
            )
            self.motion_response_sub = self.create_subscription(
                Response,
                MOTION_RESPONSE_TOPIC,
                self._motion_response_callback,
                10,
            )

        self.depth_input_sub = self.create_subscription(
            Float32MultiArray,
            self.depth_data_topic,
            self._depth_data_callback,
            1
        )

        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(
                "Real output requested, but /lowcmd does not exist yet. Model "
                "warm-up and the explicit L1 takeover gate must complete first."
            )
        else:
            self.get_logger().warn(
                f"Dry-run output isolation enabled: LowCmd is published only to "
                f"'{self.low_cmd_topic}' and Sport Mode API output is disabled."
            )
        while rclpy.ok():
            rclpy.spin_once(self)
            if (
                hasattr(self, "low_state_buffer")
                and hasattr(self, "joy_stick_buffer")
                and hasattr(self, "depth_data")
            ):
                break
        self.get_logger().info(
            "Low state, wireless controller, and depth input received."
        )

    def _motion_response_callback(self, message):
        self.motion_responses.append(message)

    def consume_button_rising(self, button):
        pressed = self.remote_tracker.consume_rising(button)
        self.remote_rising_edges = self.remote_tracker.rising_edges
        return pressed

    def _call_motion_switcher(self, api_id):
        if self.dryrun or self.motion_request_pub is None:
            raise RealControlError("MotionSwitcher is unavailable in dry-run")
        reason = "timed out"
        for _ in range(RPC_MAX_ATTEMPTS):
            request_id = time.monotonic_ns()
            request = build_motion_request(Request(), api_id, request_id)
            self.motion_responses.clear()
            self.motion_request_pub.publish(request)
            deadline = time.monotonic() + RPC_TIMEOUT_S
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                matching = [
                    response
                    for response in self.motion_responses
                    if int(response.header.identity.id) == request_id
                    and int(response.header.identity.api_id) == int(api_id)
                ]
                self.motion_responses.clear()
                if matching:
                    return parse_motion_response(
                        matching[-1],
                        request_id,
                        api_id,
                    )
            reason = "timed out"
        raise RealControlError(
            f"MotionSwitcher API {api_id} failed after "
            f"{RPC_MAX_ATTEMPTS} attempts: {reason}"
        )

    def _current_joint_state(self):
        position = self.dof_pos_[0].detach().cpu().numpy().astype(np.float64)
        velocity = self.dof_vel_[0].detach().cpu().numpy().astype(np.float64)
        return position, velocity

    def _current_foot_force(self):
        if not hasattr(self, "decoded_low_state"):
            raise RealControlError("decoded LowState is unavailable")
        return self.decoded_low_state.foot_force.copy()

    def _current_motor_diagnostics(self):
        temperatures = np.asarray(
            [
                self.low_state_buffer.motor_state[self.dof_map[index]].temperature
                for index in range(self.NUM_DOF)
            ],
            dtype=np.float64,
        )
        lost = np.asarray(
            [
                self.low_state_buffer.motor_state[self.dof_map[index]].lost
                for index in range(self.NUM_DOF)
            ],
            dtype=np.float64,
        )
        return temperatures, lost

    def _current_roll_pitch(self):
        return self._get_imu_obs()[0].detach().cpu().numpy().astype(np.float64)

    def _validate_policy_entry_now(self):
        if self.last_command_target_q is None:
            raise RealControlError("stand target is unavailable")
        position, _ = self._current_joint_state()
        roll, pitch = self._current_roll_pitch()
        temperatures, lost = self._current_motor_diagnostics()
        if self.policy_lost_baseline is None:
            self.policy_lost_baseline = lost.copy()
        try:
            validate_policy_entry_state(
                self._current_foot_force(),
                roll,
                pitch,
                position,
                self.last_command_target_q,
                temperatures,
                lost,
                self.policy_lost_baseline,
            )
        finally:
            self.policy_lost_baseline = lost.copy()

    def monitor_stand_hold_motor_lost(self):
        """Re-prime immediately when a lost counter advances while waiting for Y."""
        if self.real_control_phase != "stand_hold":
            return True
        _, lost = self._current_motor_diagnostics()
        if self.policy_lost_baseline is None:
            self.policy_lost_baseline = lost.copy()
            return True
        baseline, increased = update_motor_lost_baseline(
            lost,
            self.policy_lost_baseline,
        )
        self.policy_lost_baseline = baseline
        if not increased.size:
            return True
        self.get_logger().warning(
            "Motor lost counters increased while waiting for Y at indices "
            f"{increased.tolist()}; rebuilding policy context now."
        )
        self.begin_policy_prime()
        return False

    def _validate_takeover_now(self, now):
        if self.latest_low_state_time is None or self.latest_remote_time is None:
            raise RealControlError("LowState and remote input are required")
        position, velocity = self._current_joint_state()
        validate_takeover_inputs(
            position,
            velocity,
            now - self.latest_low_state_time,
            now - self.latest_remote_time,
        )

    def set_policy_depth_reset(self, callback):
        """Register the visual GRU reset hook after models are constructed."""
        if not callable(callback):
            raise TypeError("policy depth reset hook must be callable")
        self.policy_depth_reset = callback

    def _reset_policy_prime_memory(self, now):
        if self.policy_depth_reset is None:
            raise RealControlError("policy depth reset hook is not registered")
        self.actions = torch.zeros(
            self.NUM_ACTIONS,
            device=self.model_device,
            dtype=torch.float32,
        )
        reset_policy_context(
            self.actions,
            self.proprio_history_buf,
            self.episode_length_buf,
            self.policy_depth_reset,
        )
        self.policy_prime_gate = PolicyPrimeGate(now)
        self.policy_prime_cycle = 0
        self.last_contacts = np.zeros(4, dtype=np.bool_)
        self.contact_filt.fill_(-0.5)

    def begin_policy_prime(self, now=None):
        """Hold stand while rebuilding current proprioception and visual state."""
        timestamp = time.monotonic() if now is None else float(now)
        self.policy_transition.reset()
        self.policy_prime_rejection_reason = None
        _, lost = self._current_motor_diagnostics()
        self.policy_lost_baseline = lost.copy()
        self._reset_policy_prime_memory(timestamp)
        self.real_control_phase = "policy_prime"
        self.get_logger().warning(
            "Stand ramp complete; rebuilding policy context for at least 0.5 "
            "seconds before Y is accepted."
        )

    def restart_policy_prime(self, now, reason):
        """Discard partial prime state after any stale input sample."""
        rejection_reason = str(reason)
        had_samples = (
            self.policy_prime_gate is not None
            and self.policy_prime_gate.has_samples
        )
        if had_samples or self.policy_prime_gate is None:
            self._reset_policy_prime_memory(float(now))
        else:
            self.policy_prime_gate.restart(float(now))
            self.policy_prime_cycle = 0
        if rejection_reason != self.policy_prime_rejection_reason:
            self.get_logger().warning(
                f"Policy prime waiting for a safe state: {rejection_reason}"
            )
            self.policy_prime_rejection_reason = rejection_reason

    def validate_policy_prime_now(self, now):
        if (
            self.latest_low_state_time is None
            or self.latest_depth_time is None
        ):
            raise RealControlError(
                "LowState and depth are required for policy prime"
            )
        validate_policy_prime_inputs(
            now - self.latest_low_state_time,
            now - self.latest_depth_time,
        )
        self._validate_policy_entry_now()
        self.policy_prime_rejection_reason = None

    def validate_policy_runtime_now(self, now):
        """Validate every sensor sample consumed by an active policy cycle."""
        if self.latest_low_state_time is None:
            raise LowStateStaleError(
                "LowState is unavailable during policy control"
            )
        if self.latest_depth_time is None:
            raise DepthStaleError("depth is unavailable during policy control")
        position, velocity = self._current_joint_state()
        validate_policy_runtime_inputs(
            now - self.latest_low_state_time,
            now - self.latest_depth_time,
            position,
            velocity,
            self.joint_limits_low_np,
            self.joint_limits_high_np,
            self.joint_velocity_limits_np,
        )

    def record_policy_prime_sample(self, now, *, depth=False):
        if self.policy_prime_gate is None:
            raise RealControlError("policy prime gate is not initialized")
        self.policy_prime_gate.record_proprio()
        if depth:
            self.policy_prime_gate.record_depth()
        if self.policy_prime_gate.ready(now):
            self.real_control_phase = "stand_hold"
            self.get_logger().warning(
                "Policy prime complete; holding the default pose. Press Y "
                "once to enter policy control."
            )
            return True
        return False

    def prepare_dryrun_takeover(self):
        """Start the real takeover trajectory on the isolated dry-run topic."""
        if not self.dryrun or self.real_control_phase != "dryrun":
            raise RealControlError("dry-run takeover is unavailable")
        position, _ = self._current_joint_state()
        self.startup_ramp_start_q = position.copy()
        self.startup_ramp_start_time = time.monotonic()
        self.last_command_target_q = position.copy()
        self.real_control_phase = "startup_ramp"
        self.get_logger().warning(
            "Dry-run takeover started on the isolated LowCmd topic."
        )

    def _wait_for_l1_takeover_hold(self):
        self.get_logger().warning(
            "Release L1, then hold it continuously for one second to authorize "
            "Sport Mode release and real /lowcmd takeover."
        )
        while rclpy.ok() and (self.remote_keys & self.WirelessButtons.L1):
            rclpy.spin_once(self, timeout_sec=0.05)

        hold_started = None
        last_reason = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            l1_pressed = bool(self.remote_keys & self.WirelessButtons.L1)
            if not l1_pressed:
                hold_started = None
                continue
            try:
                self._validate_takeover_now(now)
            except RealControlError as error:
                hold_started = None
                reason = str(error)
                if reason != last_reason:
                    self.get_logger().warning(
                        f"L1 takeover hold reset: {reason}"
                    )
                    last_reason = reason
                continue
            last_reason = None
            if hold_started is None:
                hold_started = now
            if now - hold_started >= TAKEOVER_HOLD_S:
                return
        raise RealControlError("ROS shutdown before takeover authorization")

    def prepare_real_takeover(self):
        """Pre-create /lowcmd, then publish only after Sport Mode is released."""
        if self.dryrun:
            return
        if self.real_control_phase != "sport":
            raise RealControlError("real takeover was already attempted")

        active_mode = str(
            self._call_motion_switcher(CHECK_MODE_API_ID).get("name", "")
        )
        self.get_logger().info(
            f"MotionSwitcher active mode: '{active_mode or 'none'}'."
        )
        self._wait_for_l1_takeover_hold()
        self._validate_takeover_now(time.monotonic())

        active_mode = str(
            self._call_motion_switcher(CHECK_MODE_API_ID).get("name", "")
        )
        self._validate_takeover_now(time.monotonic())
        handoff_position, _ = self._current_joint_state()
        self.low_cmd_pub = self.create_publisher(
            LowCmd,
            REAL_LOW_COMMAND_TOPIC,
            1,
        )
        self.low_cmd_buffer = LowCmd()
        self.get_logger().warning(
            "Created /lowcmd publisher before Sport Mode release; real "
            "publishing remains blocked until CheckMode confirms no active mode."
        )

        release_cycles = 0
        while release_mode_required(active_mode):
            if release_cycles >= RPC_MAX_ATTEMPTS:
                raise RealControlError(
                    "Sport Mode remained active after ReleaseMode retries"
                )
            self._call_motion_switcher(RELEASE_MODE_API_ID)
            release_cycles += 1
            active_mode = str(
                self._call_motion_switcher(CHECK_MODE_API_ID).get("name", "")
            )
        if release_cycles == 0:
            self.get_logger().info(
                "MotionSwitcher was already released; skipped ReleaseMode."
            )
        else:
            self.get_logger().warning("Sport Mode release verified.")

        self.startup_ramp_start_q = handoff_position.copy()
        self.startup_ramp_start_time = time.monotonic()
        self.last_command_target_q = handoff_position.copy()
        self.real_lowcmd_authorized = True
        self.real_control_phase = "startup_ramp"
        self._publish_legs_cmd(
            torch.as_tensor(handoff_position, device=self.model_device),
            stand=True,
        )
        self.get_logger().warning(
            "CheckMode is empty; published the latched-pose hold immediately "
            "and started the 3-second measured-pose ramp."
        )

    def begin_policy_transition(self):
        if self.real_control_phase != "stand_hold":
            self.get_logger().warning(
                "Policy request ignored until policy context priming completes."
            )
            return False
        try:
            if self.latest_remote_time is None:
                raise RealControlError("remote input is required when Y is pressed")
            validate_policy_request_input(
                time.monotonic() - self.latest_remote_time
            )
            self.validate_policy_prime_now(time.monotonic())
        except RealControlError as error:
            self.get_logger().warning(f"Policy request rejected: {error}")
            self.begin_policy_prime()
            return False
        start = self.last_command_target_q
        self.policy_transition.begin(start, time.monotonic())
        self.real_control_phase = "policy"
        return True

    def begin_stand_recovery(self):
        start = self.last_command_target_q
        if start is None:
            start, _ = self._current_joint_state()
        self.stand_recovery_start_q = np.asarray(start, dtype=np.float64).copy()
        self.stand_recovery_start_time = time.monotonic()
        self.policy_transition.reset()
        self.real_control_phase = "stand_recovery"

    """ ROS callbacks and handlers that update the buffer """

    def _low_state_callback(self, msg):
        # self.get_logger().warn("Low state message received.")
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state
        self.latest_low_state_time = time.monotonic()

        boundary_state = BoundaryLowState(
            motor_q=[
                self.low_state_buffer.motor_state[index].q
                for index in range(self.NUM_DOF)
            ],
            motor_dq=[
                self.low_state_buffer.motor_state[index].dq
                for index in range(self.NUM_DOF)
            ],
            foot_force=[
                self.low_state_buffer.foot_force[index]
                for index in range(4)
            ],
            gyroscope=self.low_state_buffer.imu_state.gyroscope,
            imu_quaternion_wxyz=self.low_state_buffer.imu_state.quaternion,
        )
        self.decoded_low_state = decode_low_state(boundary_state)
        self.dof_pos_[0].copy_(
            torch.as_tensor(
                self.decoded_low_state.joint_q,
                device=self.model_device,
                dtype=torch.float32,
            )
        )
        self.dof_vel_[0].copy_(
            torch.as_tensor(
                self.decoded_low_state.joint_dq,
                device=self.model_device,
                dtype=torch.float32,
            )
        )

    def _joy_stick_callback(self, msg):
        # self.get_logger().warn("Wireless controller message received.")
        self.joy_stick_buffer = msg
        keys = int(msg.keys)
        self.remote_tracker.update(keys, time.monotonic())
        self.remote_keys = self.remote_tracker.keys
        self.remote_rising_edges = self.remote_tracker.rising_edges
        self.latest_remote_time = self.remote_tracker.latest_time
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for turning left/right
            lx = -msg.lx
            if lx > self.ang_vel_deadband:
                yaw = (lx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif lx < -self.ang_vel_deadband:
                yaw = (lx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            # right-x for side moving left/right
            rx = -msg.rx
            if rx > self.lin_vel_deadband:
                vy = (rx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif rx < -self.lin_vel_deadband:
                vy = (rx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            self.xyyaw_command = torch.tensor([[0.5, vy, yaw]], device= self.model_device, dtype= torch.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        
        # if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
        # if  msg.keys & self.WirelessButtons.L2: # R2 or L2 is pressed
        #     self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
        #     self._turn_off_motors()
        #     raise SystemExit()

        # roll-pitch target
        if hasattr(self, "roll_pitch_yaw_cmd"):
            if (msg.keys & self.WirelessButtons.up):
                self.roll_pitch_yaw_cmd[0, 1] += 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.down):
                self.roll_pitch_yaw_cmd[0, 1] -= 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.left):
                self.roll_pitch_yaw_cmd[0, 0] -= 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.right):
                self.roll_pitch_yaw_cmd[0, 0] += 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))

    def _depth_data_callback(self, msg):
        self.depth_data = torch.tensor(msg.data, dtype=torch.float32).reshape(1, 58, 87).to(self.model_device)
        self.latest_depth_time = time.monotonic()

    
    def _sport_mode_change(self, mode):
        if self.dryrun:
            self.get_logger().warn(
                "Dry-run blocked a Sport Mode API request.",
                once=True,
            )
            return False
        if self.sport_mode_pub is None:
            raise RuntimeError("Sport Mode publisher is not initialized.")

        msg = Request()

        msg.header.identity.id = 0
        msg.header.identity.api_id = mode
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        msg.parameter = ''
        msg.binary = []

        self.sport_mode_pub.publish(msg)
        return True
    
    def _sport_state_change(self, mode):
        if self.dryrun:
            self.get_logger().warn(
                "Dry-run blocked a robot-state API request.",
                once=True,
            )
            return False
        if self.sport_state_pub is None:
            raise RuntimeError("Robot-state publisher is not initialized.")

        msg = Request()

        # Fill the header
        msg.header.identity.id = 0
        msg.header.identity.api_id = 1001
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        if mode == 0:
            msg.parameter = '{"name":"sport_mode","switch":0}'
        elif mode == 1:
            msg.parameter = '{"name":"sport_mode","switch":1}'
        
        msg.binary = []

        # Publish the request
        self.sport_state_pub.publish(msg)
        return True

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """
    
    def _get_ang_vel_obs(self):
        if not hasattr(self, "decoded_low_state"):
            raise RealControlError("decoded LowState is unavailable")
        ang_vel = torch.as_tensor(
            self.decoded_low_state.gyroscope,
            device=self.model_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        return ang_vel * self.cfg["normalization"]["obs_scales"]["ang_vel"]
    
    def _get_imu_obs(self):
        if not hasattr(self, "decoded_low_state"):
            raise RealControlError("decoded LowState is unavailable")
        return torch.as_tensor(
            self.decoded_low_state.roll_pitch,
            device=self.model_device,
            dtype=torch.float32,
        ).unsqueeze(0)

    def _get_delta_yaw_obs(self):
        yaw = 0
        delta_yaw, delta_next_yaw = 0, 0
        yaw_info = torch.tensor([[0, delta_yaw, delta_next_yaw]], device= self.model_device, dtype= torch.float32)
        return yaw_info

    #  maybe only vx used
    def _get_commands_obs(self):
        if self.move_by_wireless_remote:
            vx, _, _ = self.xyyaw_command[0, :]
            commands = torch.tensor([[0, 0, vx]], device= self.model_device, dtype= torch.float32)
            return commands
        else:
            return torch.tensor([[0., 0., 0.]], device=self.model_device)

    def _get_dof_pos_obs(self):
        return (self.dof_pos_ - self.default_dof_pos.unsqueeze(0)) * self.cfg["normalization"]["obs_scales"]["dof_pos"]

    def _get_dof_vel_obs(self):
        return self.dof_vel_ * self.cfg["normalization"]["obs_scales"]["dof_vel"]

    def _get_last_actions_obs(self):
        return self.actions

    def _get_contact_filt_obs(self):
        filtered, current = filter_foot_contacts(
            self._current_foot_force(),
            self.last_contacts,
        )
        values = np.where(filtered, 0.5, -0.5).astype(np.float32)
        self.contact_filt.copy_(
            torch.as_tensor(values, device=self.model_device).unsqueeze(0)
        )
        self.last_contacts = current
        return self.contact_filt

    def _get_depth_image(self):
        return self.depth_data

    def _get_history_proprio(self):
        return self.proprio_history_buf
    

    def get_proprio(self):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        if not hasattr(self, "decoded_low_state"):
            raise RealControlError("decoded LowState is unavailable")
        command_forward = (
            float(self.xyyaw_command[0, 0])
            if self.move_by_wireless_remote
            else 0.0
        )
        proprio_values, current_contacts = build_policy_proprio(
            self.decoded_low_state,
            self.default_dof_pos_np,
            self.actions.detach().cpu().numpy(),
            self.last_contacts,
            command_forward,
            float(self.cfg["normalization"]["obs_scales"]["ang_vel"]),
            float(self.cfg["normalization"]["obs_scales"]["dof_pos"]),
            float(self.cfg["normalization"]["obs_scales"]["dof_vel"]),
            self.mode,
        )
        proprio = torch.as_tensor(
            proprio_values,
            device=self.model_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.contact_filt.copy_(proprio[:, -4:])
        self.last_contacts = current_contacts

        self.proprio_history_buf = update_proprio_history(
            self.proprio_history_buf,
            proprio,
            self.episode_length_buf,
        )
        self.episode_length_buf += 1

        return proprio


    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        if isinstance(actions, list):
            actions = torch.tensor(
                actions,
                device=self.model_device,
                dtype=torch.float32,
            ).unsqueeze(0)
        if self.real_control_phase != "policy":
            raise RealControlError(
                "policy output is blocked outside the policy phase"
            )

        raw_action = actions.detach().cpu().numpy().astype(np.float64)
        if raw_action.shape == (1, self.NUM_ACTIONS):
            raw_action = raw_action[0]
        action_scale = float(self.cfg["control"]["action_scale"])
        observed_action, _, requested_target = prepare_policy_action(
            raw_action,
            self.default_dof_pos_np,
            float(self.cfg["normalization"]["clip_actions"]),
            action_scale,
        )
        self.actions = torch.as_tensor(
            observed_action,
            dtype=torch.float32,
            device=self.model_device,
        )

        engagement_active = self.policy_transition.active
        if engagement_active:
            transition_target = self.policy_transition.apply(
                requested_target,
                time.monotonic(),
            )
        else:
            transition_target = requested_target

        if self.last_command_target_q is None:
            raise RealControlError("previous joint target is unavailable")
        measured_q, measured_dq = self._current_joint_state()
        commanded_target = constrain_policy_target(
            transition_target,
            self.last_command_target_q,
            measured_q,
            measured_dq,
            self.p_gains_np,
            self.d_gains_np,
            self.joint_limits_low_np,
            self.joint_limits_high_np,
            self.torque_limits_np,
            max_step_rad=(
                self.policy_transition.max_step_rad
                if engagement_active
                else POLICY_TARGET_MAX_STEP_RAD_BY_JOINT
            ),
        )
        if engagement_active:
            self.policy_transition.record_executed_target(commanded_target)
            if not self.policy_transition.active:
                self.get_logger().warning(
                    "Policy engagement complete; continuous policy target "
                    "constraints remain active."
                )

        self._publish_legs_cmd(commanded_target, stand=False)
        return {
            "phase": self.real_control_phase,
            "raw_action": raw_action.copy(),
            "executed_action": executed_target_to_action(
                commanded_target,
                self.default_dof_pos_np,
                action_scale,
            ),
            "requested_q": requested_target.copy(),
            "commanded_q": commanded_target.copy(),
            "engagement_active": engagement_active,
        }

    def send_stand_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        target = np.asarray(actions, dtype=np.float64)
        self._publish_legs_cmd(target, stand=True)
        executed_action = executed_target_to_action(
            target,
            self.default_dof_pos_np,
            float(self.cfg["control"]["action_scale"]),
        )
        return {
            "phase": self.real_control_phase,
            "raw_action": np.zeros(self.NUM_ACTIONS, dtype=np.float64),
            "executed_action": executed_action,
            "requested_q": target.copy(),
            "commanded_q": target.copy(),
            "engagement_active": False,
        }

    def get_stand_action(self):
        if self.real_control_phase in (
            "startup_ramp",
            "stand_recovery",
            "policy_prime",
            "stand_hold",
        ):
            now = time.monotonic()
            target = self.default_dof_pos_np
            if self.real_control_phase == "startup_ramp":
                if (
                    self.startup_ramp_start_time is None
                    or self.startup_ramp_start_q is None
                ):
                    raise RealControlError("startup ramp state is missing")
                elapsed = now - self.startup_ramp_start_time
                result = interpolate_pose(
                    self.startup_ramp_start_q,
                    target,
                    elapsed,
                    STARTUP_RAMP_S,
                )
                if elapsed >= STARTUP_RAMP_S:
                    self.begin_policy_prime(now)
                return result.tolist()
            if self.real_control_phase == "stand_recovery":
                if (
                    self.stand_recovery_start_time is None
                    or self.stand_recovery_start_q is None
                ):
                    raise RealControlError("stand recovery state is missing")
                elapsed = now - self.stand_recovery_start_time
                result = interpolate_pose(
                    self.stand_recovery_start_q,
                    target,
                    elapsed,
                    1.0,
                )
                if elapsed >= 1.0:
                    self.stand_recovery_start_time = None
                    self.stand_recovery_start_q = None
                    self.begin_policy_prime(now)
                return result.tolist()
            if self.real_control_phase == "policy_prime":
                return target.tolist()
            if self.real_control_phase == "stand_hold":
                return target.tolist()
            raise RealControlError(
                f"stand output is blocked in phase '{self.real_control_phase}'"
            )

        if self.firstRun:
            for sim_idx in range(self.NUM_DOF):
                real_idx = self.dof_map[sim_idx]
                self.startPos[sim_idx] = (
                    self.low_state_buffer.motor_state[real_idx].q
                    * self.dof_signs[sim_idx]
                )
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.stand_action[i] = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]

            if self.firstrun_target_1:
                self.get_logger().info('Going to target Pos 1.', once=True)
                self.firstrun_target_1 = False
                self.firstrun_target_2 = True
        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.stand_action[i] = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]

            self.get_logger().info('Staying in target Pos 2.', once=True)

        return self.stand_action

    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action, stand):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in actor/Unitree order.
        """
        if self.low_cmd_pub is None or self.low_cmd_buffer is None:
            raise RealControlError("/lowcmd publisher is not initialized")
        if not self.dryrun:
            validate_real_low_command_publish(
                real_output_enabled=True,
                takeover_authorized=self.real_lowcmd_authorized,
            )
        if isinstance(robot_coordinates_action, torch.Tensor):
            target = robot_coordinates_action.detach().cpu().numpy()
        else:
            target = np.asarray(robot_coordinates_action)
        target = target.astype(np.float64, copy=False)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise RealControlError("joint command must contain 12 finite values")

        encoded = encode_low_cmd(target, self.p_gains_np, self.d_gains_np)
        for real_idx in range(self.NUM_DOF):
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = (
                    self.turn_on_motor_mode[real_idx]
                )
            self.low_cmd_buffer.motor_cmd[real_idx].q = float(
                encoded.motor_q[real_idx]
            )
            self.low_cmd_buffer.motor_cmd[real_idx].dq = float(
                encoded.motor_dq[real_idx]
            )
            self.low_cmd_buffer.motor_cmd[real_idx].tau = float(
                encoded.motor_tau[real_idx]
            )
            self.low_cmd_buffer.motor_cmd[real_idx].kp = float(
                encoded.motor_kp[real_idx]
            )
            self.low_cmd_buffer.motor_cmd[real_idx].kd = float(
                encoded.motor_kd[real_idx]
            )
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
        self.last_command_target_q = target.astype(np.float64).copy()

    def _turn_off_motors(self):
        """ Turn off the motors """
        if self.low_cmd_pub is None or self.low_cmd_buffer is None:
            return
        if not self.dryrun:
            validate_real_low_command_publish(
                real_output_enabled=True,
                takeover_authorized=self.real_lowcmd_authorized,
            )
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def shutdown_outputs(self):
        """Publish a short all-motors-off tail before destroying a real node."""
        if (
            self.dryrun
            or self.low_cmd_pub is None
            or not self.real_lowcmd_authorized
        ):
            return
        for _ in range(10):
            self._turn_off_motors()
            time.sleep(0.02)
    """ Done: functions that actually publish the commands and take effect """
