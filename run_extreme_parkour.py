import rclpy
from unitree_ros2_real import UnitreeRos2Real

import os
import os.path as osp
import json
import time
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from flight_recorder import FlightRecorder
from joint_mapping import FOOT_REAL_TO_SIM
from real_control_safety import (
    LowStateStaleError,
    PolicyTargetInfeasibleError,
    RealControlError,
    classify_foot_contacts,
)
from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

from sport_api_constants import *

class Go2Node(UnitreeRos2Real):
    def __init__(self, *args, flight_log_dir, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        self.global_counter = 0
        self.visual_update_interval = 5
        self.flight_recorder = FlightRecorder(flight_log_dir)
        self.flight_record_error_reported = False
        self.pending_flight_reason = None

        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = True

    # This warm up is useful in my experiment on Go2
    # The first two iterations are very slow, but the rest is fast
    def warm_up(self):
        for _ in range(2):
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            proprio_history = self._get_history_proprio() 
            get_hist_pro_time = time.monotonic()

            depth_image = self._get_depth_image()
            self.depth_latent_yaw = self.depth_encode(depth_image, proprio)
            self._record_visual_sample(depth_image, self.depth_latent_yaw)

            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)

            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()

            publish_time = time.monotonic()
            print("warm up: ",
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

    def register_models(
        self,
        turn_obs,
        depth_encode,
        policy,
        reset_depth_hidden,
    ):
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy
        self.set_policy_depth_reset(reset_depth_hidden)

    def _record_visual_sample(self, depth_image, visual_output):
        depth = depth_image.detach().cpu().numpy()
        output = visual_output.detach().cpu().numpy().reshape(-1)
        self.flight_recorder.record_visual(
            timestamp=time.monotonic(),
            depth_stats=(float(depth.min()), float(depth.max()), float(depth.mean())),
            visual_output=output,
        )

    def _update_policy_prime(self):
        now = time.monotonic()
        try:
            self.validate_policy_prime_now(now)
        except RealControlError as error:
            self.restart_policy_prime(now, str(error))
            return

        proprio = self.get_proprio()
        depth_updated = self.policy_prime_cycle % self.visual_update_interval == 0
        if depth_updated:
            depth_image = self._get_depth_image()
            self.depth_latent_yaw = self.depth_encode(depth_image, proprio)
            self.last_depth_image = depth_image
            self._record_visual_sample(depth_image, self.depth_latent_yaw)
        self.policy_prime_cycle += 1
        self.record_policy_prime_sample(now, depth=depth_updated)

    def _record_control(self, command, loop_started):
        now = time.monotonic()
        motor_states = [
            self.low_state_buffer.motor_state[self.dof_map[index]]
            for index in range(self.NUM_DOF)
        ]
        position = np.asarray(
            [
                motor_states[index].q * self.dof_signs[index]
                for index in range(self.NUM_DOF)
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                motor_states[index].dq * self.dof_signs[index]
                for index in range(self.NUM_DOF)
            ],
            dtype=np.float64,
        )
        input_ages = (
            now - self.latest_low_state_time,
            now - self.latest_remote_time,
            now - self.latest_depth_time,
        )
        foot_force = [
            self.low_state_buffer.foot_force[index]
            for index in FOOT_REAL_TO_SIM
        ]
        motor_temperature = [state.temperature for state in motor_states]
        motor_lost = [state.lost for state in motor_states]
        motor_tau_est = [
            motor_states[index].tau_est * self.dof_signs[index]
            for index in range(self.NUM_DOF)
        ]
        try:
            self.flight_recorder.record_control(
                timestamp=now,
                phase=command["phase"],
                engagement_active=command["engagement_active"],
                raw_action=command["raw_action"],
                executed_action=command["executed_action"],
                requested_q=command["requested_q"],
                commanded_q=command["commanded_q"],
                measured_q=position,
                measured_dq=velocity,
                imu_quaternion=self.low_state_buffer.imu_state.quaternion,
                foot_force=foot_force,
                contact_state=classify_foot_contacts(foot_force),
                motor_temperature=motor_temperature,
                motor_lost=motor_lost,
                motor_tau_est=motor_tau_est,
                input_ages=input_ages,
                loop_s=now - loop_started,
            )
        except (TypeError, ValueError) as error:
            if not self.flight_record_error_reported:
                self.get_logger().error(f"Flight recorder rejected a sample: {error}")
                self.flight_record_error_reported = True

    def flush_flight_record(self, reason):
        try:
            path = self.flight_recorder.flush(reason)
        except (OSError, ValueError) as error:
            self.get_logger().error(f"Failed to save flight record: {error}")
            return None
        if path is not None:
            self.get_logger().warning(f"Flight record saved: {path}")
        return path

    def _emergency_stop(self, reason):
        self.get_logger().error(
            f"Emergency stop ({reason}): publishing the motor-off tail."
        )
        self.policy_transition.reset()
        self.shutdown_outputs()
        self.flush_flight_record(reason)
        self.real_control_phase = "emergency_stop"
        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = False

    def _handle_policy_fault(self, error):
        """Leave policy control without issuing the rejected target."""
        if isinstance(
            error,
            (LowStateStaleError, PolicyTargetInfeasibleError),
        ):
            reason = (
                "policy_low_state_stale"
                if isinstance(error, LowStateStaleError)
                else "policy_target_infeasible"
            )
            self._emergency_stop(reason)
            return
        self.get_logger().error(
            f"Policy guard rejected the control cycle: {error}. "
            "Returning to stand through the recovery ramp."
        )
        self.pending_flight_reason = "policy_guard_recovery"
        self.begin_stand_recovery()
        self.use_stand_policy = True
        self.use_parkour_policy = False
        self.use_sport_mode = False

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):
        r1_pressed = self.consume_button_rising(self.WirelessButtons.R1)
        r2_pressed = self.consume_button_rising(self.WirelessButtons.R2)
        x_pressed = self.consume_button_rising(self.WirelessButtons.X)
        l1_pressed = self.consume_button_rising(self.WirelessButtons.L1)
        y_pressed = self.consume_button_rising(self.WirelessButtons.Y)
        l2_pressed = self.consume_button_rising(self.WirelessButtons.L2)

        if (
            not self.dryrun
            and self.real_control_phase not in ("sport", "dryrun")
            and r2_pressed
        ):
            self._emergency_stop("r2_emergency_stop")
            return

        if self.use_sport_mode:
            if r1_pressed:
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if r2_pressed:
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)

            if x_pressed:
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

            if l1_pressed and self.dryrun:
                self.get_logger().info("Exist the sport mode. Switch to stand policy.")
                self.prepare_dryrun_takeover()
                self.use_sport_mode = False
                self.use_stand_policy = True
                self.use_parkour_policy = False
        
        if l2_pressed and self.real_control_phase == "policy":
            self.get_logger().warning(
                "L2 pressed: leaving policy through the one-second stand ramp."
            )
            self.pending_flight_reason = "l2_stand_recovery"
            self.begin_stand_recovery()
            self.use_stand_policy = True
            self.use_parkour_policy = False
            self.use_sport_mode = False

        if y_pressed:
            if self.begin_policy_transition():
                self.get_logger().info("Y pressed, use the parkour policy")
                self.use_stand_policy = False
                self.use_parkour_policy = True
                self.use_sport_mode = False
                self.global_counter = 0

        if self.use_stand_policy:
            loop_started = time.monotonic()
            stand_action = self.get_stand_action()
            command = self.send_stand_action(stand_action)
            if self.real_control_phase == "policy_prime":
                if self.pending_flight_reason is not None:
                    self.flush_flight_record(self.pending_flight_reason)
                    self.pending_flight_reason = None
                self._update_policy_prime()
            self._record_control(command, loop_started)

        if self.use_parkour_policy:
            self.use_stand_policy = False
            self.use_sport_mode = False
            loop_started = time.monotonic()
            try:
                self.validate_policy_runtime_now(loop_started)
                proprio = self.get_proprio()
                proprio_history = self._get_history_proprio()

                if self.global_counter % self.visual_update_interval == 0:
                    depth_image = self._get_depth_image()
                    if self.global_counter == 0:
                        self.last_depth_image = depth_image
                    self.depth_latent_yaw = self.depth_encode(
                        self.last_depth_image,
                        proprio,
                    )
                    self._record_visual_sample(
                        self.last_depth_image,
                        self.depth_latent_yaw,
                    )
                    self.last_depth_image = depth_image

                obs = self.turn_obs(
                    proprio,
                    self.depth_latent_yaw,
                    proprio_history,
                    self.n_proprio,
                    self.n_depth_latent,
                    self.n_hist_len,
                )
                action = self.policy(obs)
                command = self.send_action(action)
            except RealControlError as error:
                self._handle_policy_fault(error)
                return
            self._record_control(command, loop_started)
            self.global_counter += 1


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    config_dict["control"]["computer_clip_torque"] = True
    
    # duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # different from parkour
    device = "cuda"
    duration = 0.02

    env_node = Go2Node(
        "go2",
        cfg= config_dict,
        model_device= device,
        dryrun= not args.nodryrun,
        mode = args.mode,
        flight_log_dir=args.flight_log_dir,
    )

    env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))

    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(args.logdir, base_model_name)

    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(args.logdir, vision_model_name)

    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()

    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        proprio[:, 6:8] = yaw

        lin_vel_latent = estimator(proprio)

        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))

        
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)

        return obs

    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if not torch.isfinite(depth_latent_yaw).all():
            raise RuntimeError("depth encoder output contains NaN or Inf")
        return depth_latent_yaw

    def reset_depth_hidden():
        depth_encoder.hidden_states = None
    
    def actor_model(obs):
        action = actor(obs)
        return action

    env_node.register_models(
        turn_obs=turn_obs,
        depth_encode=encode_depth,
        policy=actor_model,
        reset_depth_hidden=reset_depth_hidden,
    )


    exit_reason = "shutdown"
    try:
        env_node.start_ros_handlers()
        env_node.warm_up()
        if args.nodryrun:
            env_node.prepare_real_takeover()
            env_node.use_sport_mode = False
            env_node.use_stand_policy = True
            env_node.use_parkour_policy = False

        if args.loop_mode == "while":
            rclpy.spin_once(env_node, timeout_sec=0.0)
            env_node.get_logger().info("Model and Policy are ready")
            while rclpy.ok():
                main_loop_time = time.monotonic()
                env_node.main_loop()
                rclpy.spin_once(env_node, timeout_sec=0.0)
                time.sleep(
                    max(0, duration - (time.monotonic() - main_loop_time))
                )
        elif args.loop_mode == "timer":
            env_node.get_logger().info("Model and Policy are ready")
            env_node.start_main_loop_timer(duration)
            rclpy.spin(env_node)
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception:
        exit_reason = "exception"
        raise
    finally:
        env_node.shutdown_outputs()
        env_node.flush_flight_record(exit_reason)
        env_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--mode", type= str, default= "parkour", choices=["parkour", "walk"])
    parser.add_argument(
        "--flight-log-dir",
        type=str,
        default=osp.expanduser("~/extreme-flight-logs"),
        help="Directory for timestamped policy flight-record NPZ files.",
    )
    args = parser.parse_args()
    
    main(args)
