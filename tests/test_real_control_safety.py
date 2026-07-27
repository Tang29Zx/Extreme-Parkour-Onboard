import unittest

import numpy as np

from real_control_safety import (
    POLICY_TARGET_MAX_STEP_RAD,
    PolicyTransitionGuard,
    PublisherClearGate,
    RealControlError,
    describe_publisher_endpoint,
    interpolate_pose,
    release_mode_required,
    validate_low_command_boundary,
    validate_takeover_inputs,
)


class RealControlSafetyTest(unittest.TestCase):
    def test_publisher_description_contains_node_and_gid(self):
        endpoint = type(
            "Endpoint",
            (),
            {
                "node_name": "native_lowcmd",
                "node_namespace": "/unitree",
                "endpoint_gid": [1, 2, 255],
            },
        )()

        self.assertEqual(
            describe_publisher_endpoint(endpoint),
            "node=/unitree/native_lowcmd, gid=0102ff",
        )

    def test_publisher_gate_requires_continuous_zero_window(self):
        gate = PublisherClearGate(0.0, timeout_s=2.0, stable_s=0.5)

        self.assertFalse(gate.observe(0, 0.0))
        self.assertFalse(gate.observe(1, 0.3))
        self.assertFalse(gate.observe(0, 0.4))
        self.assertFalse(gate.observe(0, 0.89))
        self.assertTrue(gate.observe(0, 0.9))

    def test_publisher_gate_times_out(self):
        gate = PublisherClearGate(0.0, timeout_s=1.0, stable_s=0.5)

        with self.assertRaisesRegex(RealControlError, "did not clear"):
            gate.observe(1, 1.01)

    def test_low_command_boundary_requires_exclusive_real_output(self):
        with self.assertRaises(RealControlError):
            validate_low_command_boundary(False, 0)
        with self.assertRaises(RealControlError):
            validate_low_command_boundary(True, 1)
        validate_low_command_boundary(True, 0)

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

    def test_policy_transition_starts_at_stand_and_limits_each_step(self):
        guard = PolicyTransitionGuard()
        stand = np.zeros(12)
        requested = np.ones(12)
        guard.begin(stand, 10.0)

        np.testing.assert_array_equal(guard.apply(requested, 10.0), stand)
        previous = stand
        for index in range(1, 61):
            target = guard.apply(requested, 10.0 + index * 0.02)
            self.assertLessEqual(
                float(np.max(np.abs(target - previous))),
                POLICY_TARGET_MAX_STEP_RAD + 1e-12,
            )
            previous = target
        np.testing.assert_allclose(previous, requested)

    def test_policy_transition_rejects_nonfinite_target(self):
        guard = PolicyTransitionGuard()
        guard.begin(np.zeros(12), 0.0)
        invalid = np.zeros(12)
        invalid[0] = np.nan

        with self.assertRaisesRegex(RealControlError, "finite"):
            guard.apply(invalid, 0.1)


if __name__ == "__main__":
    unittest.main()
