import unittest

import numpy as np

from real_control_safety import (
    DepthStaleError,
    LowStateStaleError,
    POLICY_CALF_TARGET_MAX_STEP_RAD,
    POLICY_TRANSITION_MAX_STEP_RAD,
    POLICY_TARGET_MAX_DEVIATION_RAD,
    POLICY_TARGET_MAX_STEP_RAD,
    POLICY_TARGET_MAX_STEP_RAD_BY_JOINT,
    POLICY_JOINT_VELOCITY_LIMIT_REL_TOLERANCE,
    PolicyTargetInfeasibleError,
    PolicyPrimeGate,
    PolicyTransitionGuard,
    RemoteEdgeTracker,
    RealControlError,
    classify_foot_contacts,
    constrain_policy_target,
    executed_target_to_action,
    filter_foot_contacts,
    interpolate_pose,
    update_motor_lost_baseline,
    prepare_policy_action,
    release_mode_required,
    validate_policy_prime_inputs,
    validate_policy_runtime_inputs,
    validate_policy_request_input,
    validate_policy_entry_state,
    validate_real_low_command_publish,
    validate_takeover_inputs,
)


class RealControlSafetyTest(unittest.TestCase):
    def test_motor_lost_baseline_consumes_waiting_period_increment(self):
        baseline = np.full(12, 5.0)
        current = baseline.copy()
        current[5] = 6.0

        baseline, increased = update_motor_lost_baseline(current, baseline)
        np.testing.assert_array_equal(increased, [5])

        baseline, increased = update_motor_lost_baseline(current, baseline)
        self.assertEqual(increased.size, 0)

    def test_remote_edges_are_fresh_and_consumed_once(self):
        tracker = RemoteEdgeTracker()
        tracker.update(0, 1.0)
        tracker.update(2, 1.1)
        self.assertEqual(tracker.latest_time, 1.1)
        self.assertTrue(tracker.consume_rising(2))
        self.assertFalse(tracker.consume_rising(2))

        tracker.update(2, 1.2)
        self.assertFalse(tracker.consume_rising(2))
        tracker.update(0, 1.3)
        tracker.update(2 | 2048, 1.4)
        self.assertTrue(tracker.consume_rising(2))
        self.assertTrue(tracker.consume_rising(2048))

    def test_real_publish_requires_motion_switcher_authorization(self):
        with self.assertRaises(RealControlError):
            validate_real_low_command_publish(False, False)
        with self.assertRaisesRegex(RealControlError, "CheckMode"):
            validate_real_low_command_publish(True, False)
        validate_real_low_command_publish(True, True)

    def test_empty_motion_mode_skips_release(self):
        self.assertFalse(release_mode_required(""))
        self.assertTrue(release_mode_required("mcf"))

    def test_takeover_requires_fresh_stationary_inputs(self):
        position = np.zeros(12)
        velocity = np.zeros(12)

        validate_takeover_inputs(position, velocity, 0.1, 0.1)
        velocity[3] = 0.51
        with self.assertRaisesRegex(RealControlError, "velocity"):
            validate_takeover_inputs(position, velocity, 0.1, 0.1)
        with self.assertRaisesRegex(RealControlError, "fresh"):
            validate_takeover_inputs(position, np.zeros(12), 0.3, 0.1)

    def test_startup_interpolation_starts_and_ends_exactly(self):
        start = np.linspace(-0.2, 0.2, 12)
        end = np.linspace(0.2, -0.2, 12)

        np.testing.assert_array_equal(interpolate_pose(start, end, 0.0, 3.0), start)
        np.testing.assert_allclose(interpolate_pose(start, end, 3.0, 3.0), end)

    def test_policy_prime_requires_duration_and_both_sample_counts(self):
        gate = PolicyPrimeGate(10.0)
        for index in range(10):
            gate.record_proprio()
            if index % 2 == 0:
                gate.record_depth()

        self.assertFalse(gate.ready(10.49))
        self.assertTrue(gate.ready(10.5))
        gate.restart(20.0)
        self.assertFalse(gate.has_samples)
        self.assertFalse(gate.ready(21.0))

    def test_policy_prime_rejects_stale_or_nonfinite_inputs(self):
        validate_policy_prime_inputs(0.1, 0.1)
        with self.assertRaisesRegex(RealControlError, "fresh"):
            validate_policy_prime_inputs(0.1, 0.26)
        with self.assertRaisesRegex(RealControlError, "fresh"):
            validate_policy_prime_inputs(np.nan, 0.1)

    def test_policy_runtime_rejects_stale_or_out_of_range_inputs(self):
        lower = np.full(12, -1.0)
        upper = np.full(12, 1.0)
        velocity_limits = np.full(12, 20.0)
        position = np.zeros(12)
        velocity = np.zeros(12)

        validate_policy_runtime_inputs(
            0.1,
            0.1,
            position,
            velocity,
            lower,
            upper,
            velocity_limits,
        )
        with self.assertRaisesRegex(LowStateStaleError, "LowState"):
            validate_policy_runtime_inputs(
                0.26,
                0.1,
                position,
                velocity,
                lower,
                upper,
                velocity_limits,
            )
        with self.assertRaisesRegex(DepthStaleError, "depth"):
            validate_policy_runtime_inputs(
                0.1,
                0.26,
                position,
                velocity,
                lower,
                upper,
                velocity_limits,
            )
        position[2] = 1.051
        with self.assertRaisesRegex(RealControlError, "position"):
            validate_policy_runtime_inputs(
                0.1,
                0.1,
                position,
                velocity,
                lower,
                upper,
                velocity_limits,
            )
        position[2] = 0.0
        velocity[5] = 20.11
        with self.assertRaisesRegex(RealControlError, "velocity"):
            validate_policy_runtime_inputs(
                0.1,
                0.1,
                position,
                velocity,
                lower,
                upper,
                velocity_limits,
            )

    def test_policy_runtime_velocity_tolerance_is_bounded(self):
        lower = np.full(12, -1.0)
        upper = np.full(12, 1.0)
        velocity_limits = np.full(12, 20.0)
        position = np.zeros(12)
        velocity = np.zeros(12)
        velocity[5] = 20.0 * (
            1.0 + POLICY_JOINT_VELOCITY_LIMIT_REL_TOLERANCE
        )

        validate_policy_runtime_inputs(
            0.1,
            0.1,
            position,
            velocity,
            lower,
            upper,
            velocity_limits,
        )
        velocity[5] += 1e-6
        with self.assertRaisesRegex(RealControlError, "velocity"):
            validate_policy_runtime_inputs(
                0.1,
                0.1,
                position,
                velocity,
                lower,
                upper,
                velocity_limits,
            )

    def test_policy_request_requires_remote_only_when_y_is_pressed(self):
        validate_policy_request_input(0.1)
        with self.assertRaisesRegex(RealControlError, "remote"):
            validate_policy_request_input(0.26)

    def test_policy_transition_starts_exactly_and_limits_every_step(self):
        guard = PolicyTransitionGuard(max_deviation_rad=2.0)
        stand = np.zeros(12)
        requested = np.ones(12)
        guard.begin(stand, 10.0)

        first = guard.apply(requested, 10.2)
        guard.record_executed_target(first)
        np.testing.assert_array_equal(first, stand)
        previous = first
        for index in range(1, 101):
            target = guard.apply(requested, 10.2 + index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TRANSITION_MAX_STEP_RAD + 1e-12,
            )
            guard.record_executed_target(target)
            previous = target
            if not guard.active:
                break

        self.assertFalse(guard.active)
        np.testing.assert_allclose(previous, requested)

    def test_policy_transition_rejects_nonfinite_target(self):
        guard = PolicyTransitionGuard()
        guard.begin(np.zeros(12), 0.0)
        invalid = np.zeros(12)
        invalid[4] = np.inf

        with self.assertRaisesRegex(RealControlError, "finite"):
            guard.apply(invalid, 0.1)
        invalid[4] = np.nan
        with self.assertRaisesRegex(RealControlError, "finite"):
            guard.apply(invalid, 0.1)
        with self.assertRaisesRegex(RealControlError, "12 finite values"):
            guard.apply(np.zeros(11), 0.1)

    def test_policy_transition_tracks_the_executed_downstream_target(self):
        guard = PolicyTransitionGuard(max_deviation_rad=2.0)
        stand = np.zeros(12)
        requested = np.ones(12)
        guard.begin(stand, 0.0)

        first = guard.apply(requested, 0.0)
        with self.assertRaisesRegex(RealControlError, "feedback"):
            guard.apply(requested, 0.02)
        guard.record_executed_target(first)

        guard.apply(requested, 0.5)
        downstream_target = np.full(12, 0.01)
        guard.record_executed_target(downstream_target)
        next_target = guard.apply(requested, 0.52)

        self.assertLessEqual(
            float(np.max(np.abs(next_target - downstream_target))),
            POLICY_TRANSITION_MAX_STEP_RAD + 1e-12,
        )
        np.testing.assert_allclose(next_target, 0.06)

    def test_policy_transition_ends_after_the_full_ramp(self):
        guard = PolicyTransitionGuard()
        stand = np.zeros(12)
        requested = np.full(12, 10.0)
        guard.begin(stand, 0.0)

        previous = guard.apply(requested, 0.0)
        guard.record_executed_target(previous)
        for index in range(1, 51):
            target = guard.apply(requested, index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TRANSITION_MAX_STEP_RAD + 1e-12,
            )
            guard.record_executed_target(target)
            previous = target

        self.assertFalse(guard.active)
        self.assertGreater(float(np.max(np.abs(requested - previous))), 0.05)

    def test_policy_transition_limits_the_complete_first_second(self):
        guard = PolicyTransitionGuard()
        stand = np.linspace(-0.2, 0.2, 12)
        requested = stand + 2.0
        guard.begin(stand, 0.0)

        previous = guard.apply(requested, 0.0)
        guard.record_executed_target(previous)
        for index in range(1, 50):
            target = guard.apply(requested, index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TRANSITION_MAX_STEP_RAD + 1e-12,
            )
            self.assertLessEqual(
                float(np.max(np.abs(target - stand))),
                POLICY_TARGET_MAX_DEVIATION_RAD + 1e-12,
            )
            guard.record_executed_target(target)
            previous = target
        self.assertTrue(guard.active)

        target = guard.apply(requested, 1.0)
        self.assertLessEqual(
            float(np.max(np.abs(target - previous))),
            POLICY_TRANSITION_MAX_STEP_RAD + 1e-12,
        )
        guard.record_executed_target(target)
        self.assertFalse(guard.active)

    def test_real_contact_filter_matches_training_one_frame_memory(self):
        forces = np.asarray([8.0, 11.0, 0.0, 6.0])
        np.testing.assert_array_equal(
            classify_foot_contacts(forces),
            [True, True, False, True],
        )
        filtered, current = filter_foot_contacts(
            [0.0, 0.0, 0.0, 0.0],
            [True, False, True, False],
        )
        np.testing.assert_array_equal(filtered, [True, False, True, False])
        filtered, _ = filter_foot_contacts(
            [0.0, 0.0, 0.0, 0.0],
            current,
        )
        np.testing.assert_array_equal(filtered, [False, False, False, False])

    def test_policy_entry_requires_loaded_upright_tracked_healthy_stand(self):
        measured = np.zeros(12)
        commanded = np.zeros(12)
        temperatures = np.full(12, 90.0)
        lost = np.full(12, 5.0)

        # Temperature remains diagnostic-only and must not block policy entry.
        validate_policy_entry_state(
            [8.0, 9.0, 10.0, 11.0],
            0.0,
            0.0,
            measured,
            commanded,
            temperatures,
            lost,
            lost,
        )
        with self.assertRaisesRegex(RealControlError, "four feet"):
            validate_policy_entry_state(
                [8.0, 0.0, 10.0, 11.0],
                0.0,
                0.0,
                measured,
                commanded,
                temperatures,
                lost,
                lost,
            )
        with self.assertRaisesRegex(RealControlError, "tilt"):
            validate_policy_entry_state(
                [8.0, 9.0, 10.0, 11.0],
                np.deg2rad(9.0),
                0.0,
                measured,
                commanded,
                temperatures,
                lost,
                lost,
            )
        commanded[3] = 0.21
        with self.assertRaisesRegex(RealControlError, "tracking"):
            validate_policy_entry_state(
                [8.0, 9.0, 10.0, 11.0],
                0.0,
                0.0,
                measured,
                commanded,
                temperatures,
                lost,
                lost,
            )
        commanded[3] = 0.0
        baseline = lost.copy()
        lost[5] = 6.0
        with self.assertRaisesRegex(RealControlError, "increased"):
            validate_policy_entry_state(
                [8.0, 9.0, 10.0, 11.0],
                0.0,
                0.0,
                measured,
                commanded,
                temperatures,
                lost,
                baseline,
            )

    def test_executed_target_converts_to_observed_action(self):
        default = np.linspace(-0.2, 0.2, 12)
        executed_action = np.linspace(-1.0, 1.0, 12)
        target = default + 0.25 * executed_action

        np.testing.assert_allclose(
            executed_target_to_action(target, default, 0.25),
            executed_action,
        )

    def test_policy_action_is_clipped_before_scaling(self):
        raw_action = np.linspace(-6.0, 6.0, 12)
        default = np.linspace(-0.3, 0.3, 12)

        observed_action, clipped_action, target = prepare_policy_action(
            raw_action,
            default,
            clip_actions=1.2,
            action_scale=0.25,
        )

        expected_action = np.clip(raw_action, -4.8, 4.8)
        np.testing.assert_allclose(observed_action, raw_action)
        np.testing.assert_allclose(clipped_action, expected_action)
        np.testing.assert_allclose(target, default + expected_action * 0.25)
        self.assertLessEqual(
            float(np.max(np.abs(target - default))),
            1.2 + 1e-12,
        )
        np.testing.assert_allclose(
            executed_target_to_action(target, default, 0.25),
            clipped_action,
        )

    def test_policy_action_preprocessing_rejects_invalid_inputs(self):
        default = np.zeros(12)
        invalid_action = np.zeros(12)
        invalid_action[2] = np.nan

        with self.assertRaisesRegex(RealControlError, "finite"):
            prepare_policy_action(invalid_action, default, 1.2, 0.25)
        with self.assertRaisesRegex(RealControlError, "clip limit"):
            prepare_policy_action(np.zeros(12), default, 0.0, 0.25)
        with self.assertRaisesRegex(RealControlError, "action scale"):
            prepare_policy_action(np.zeros(12), default, 1.2, np.inf)

    def test_policy_target_intersects_step_joint_and_torque_bounds(self):
        requested = np.full(12, 10.0)
        previous = np.zeros(12)
        measured = np.zeros(12)
        velocity = np.zeros(12)
        kp = np.full(12, 100.0)
        kd = np.full(12, 1.0)
        lower = np.full(12, -1.0)
        upper = np.full(12, 1.0)
        torque = np.full(12, 1.0)

        target = constrain_policy_target(
            requested,
            previous,
            measured,
            velocity,
            kp,
            kd,
            lower,
            upper,
            torque,
        )

        # Torque is the tightest bound here: 100 * 0.01 == 1 Nm.
        np.testing.assert_allclose(target, 0.01)
        self.assertTrue(
            bool(
                np.all(
                    np.abs(target - previous)
                    <= np.asarray(POLICY_TARGET_MAX_STEP_RAD_BY_JOINT)
                )
            )
        )
        self.assertTrue(bool(np.all(target <= upper)))
        estimated_torque = kp * (target - measured) - kd * velocity
        self.assertTrue(bool(np.all(np.abs(estimated_torque) <= torque)))

    def test_policy_target_uses_the_steady_step_contract_by_default(self):
        target = constrain_policy_target(
            requested_q=np.ones(12),
            previous_q=np.zeros(12),
            measured_q=np.zeros(12),
            measured_dq=np.zeros(12),
            kp=np.ones(12),
            kd=np.ones(12),
            joint_limits_low=np.full(12, -2.0),
            joint_limits_high=np.full(12, 2.0),
            torque_limits=np.full(12, 10.0),
        )

        np.testing.assert_allclose(
            target,
            POLICY_TARGET_MAX_STEP_RAD_BY_JOINT,
        )
        self.assertAlmostEqual(POLICY_TARGET_MAX_STEP_RAD, 0.30)
        self.assertAlmostEqual(POLICY_CALF_TARGET_MAX_STEP_RAD, 0.40)

    def test_policy_target_accepts_a_scalar_transition_step(self):
        target = constrain_policy_target(
            requested_q=np.ones(12),
            previous_q=np.zeros(12),
            measured_q=np.zeros(12),
            measured_dq=np.zeros(12),
            kp=np.ones(12),
            kd=np.ones(12),
            joint_limits_low=np.full(12, -2.0),
            joint_limits_high=np.full(12, 2.0),
            torque_limits=np.full(12, 10.0),
            max_step_rad=POLICY_TRANSITION_MAX_STEP_RAD,
        )

        np.testing.assert_allclose(target, POLICY_TRANSITION_MAX_STEP_RAD)

    def test_policy_target_rejects_an_empty_safe_intersection(self):
        with self.assertRaisesRegex(
            PolicyTargetInfeasibleError,
            "no safe policy target",
        ):
            constrain_policy_target(
                requested_q=np.zeros(12),
                previous_q=np.ones(12),
                measured_q=-np.ones(12),
                measured_dq=np.zeros(12),
                kp=np.full(12, 100.0),
                kd=np.ones(12),
                joint_limits_low=np.full(12, -2.0),
                joint_limits_high=np.full(12, 2.0),
                torque_limits=np.ones(12),
            )

if __name__ == "__main__":
    unittest.main()
