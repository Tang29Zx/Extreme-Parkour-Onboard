import unittest

import numpy as np

from real_control_safety import (
    POLICY_TARGET_MAX_STEP_RAD,
    PolicyPrimeGate,
    PolicyTransitionGuard,
    RealControlError,
    executed_target_to_action,
    interpolate_pose,
    prepare_policy_action,
    release_mode_required,
    validate_policy_prime_inputs,
    validate_real_low_command_publish,
    validate_takeover_inputs,
)


class RealControlSafetyTest(unittest.TestCase):
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
        validate_policy_prime_inputs(0.1, 0.1, 0.1)
        with self.assertRaisesRegex(RealControlError, "fresh"):
            validate_policy_prime_inputs(0.1, 0.1, 0.26)
        with self.assertRaisesRegex(RealControlError, "fresh"):
            validate_policy_prime_inputs(np.nan, 0.1, 0.1)

    def test_policy_transition_starts_exactly_and_limits_every_step(self):
        guard = PolicyTransitionGuard()
        stand = np.zeros(12)
        requested = np.ones(12)
        guard.begin(stand, 10.0)

        first = guard.apply(requested, 10.2)
        np.testing.assert_array_equal(first, stand)
        previous = first
        for index in range(1, 101):
            target = guard.apply(requested, 10.2 + index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TARGET_MAX_STEP_RAD + 1e-12,
            )
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

    def test_policy_transition_stays_active_after_ramp_until_caught_up(self):
        guard = PolicyTransitionGuard()
        stand = np.zeros(12)
        requested = np.full(12, 10.0)
        guard.begin(stand, 0.0)

        previous = guard.apply(requested, 0.0)
        for index in range(1, 60):
            target = guard.apply(requested, index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TARGET_MAX_STEP_RAD + 1e-12,
            )
            previous = target

        self.assertTrue(guard.active)
        self.assertGreater(float(np.max(np.abs(requested - previous))), 0.05)

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

        clipped_action, target = prepare_policy_action(
            raw_action,
            default,
            clip_actions=1.2,
            action_scale=0.25,
        )

        expected_action = np.clip(raw_action, -4.8, 4.8)
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

if __name__ == "__main__":
    unittest.main()
