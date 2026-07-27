import math
import unittest

import numpy as np

from joint_mapping import isaac_feet_to_unitree, isaac_to_policy, policy_to_isaac
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


class UnitreeBoundaryTest(unittest.TestCase):
    def test_distinct_isaac_state_crosses_lowstate_boundary(self):
        isaac_q = np.arange(12, dtype=np.float64) + 0.25
        isaac_dq = np.arange(12, dtype=np.float64) - 6.0
        isaac_feet = np.asarray([11.0, 22.0, 33.0, 44.0])
        state = BoundaryLowState(
            motor_q=isaac_to_policy(isaac_q),
            motor_dq=isaac_to_policy(isaac_dq),
            foot_force=isaac_feet_to_unitree(isaac_feet),
            gyroscope=[0.1, -0.2, 0.3],
            imu_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        )

        decoded = decode_low_state(state)

        np.testing.assert_array_equal(decoded.joint_q, isaac_to_policy(isaac_q))
        np.testing.assert_array_equal(decoded.joint_dq, isaac_to_policy(isaac_dq))
        np.testing.assert_array_equal(decoded.foot_force, [22.0, 11.0, 44.0, 33.0])
        np.testing.assert_array_equal(
            policy_to_isaac(decoded.joint_q),
            isaac_q,
        )

    def test_quaternion_and_53_value_proprio_match_contract(self):
        half_angle = math.pi / 8.0
        state = BoundaryLowState(
            motor_q=np.linspace(-0.3, 0.3, 12),
            motor_dq=np.linspace(-1.1, 1.1, 12),
            foot_force=[8.0, 0.0, 10.0, 0.0],
            gyroscope=[1.0, 2.0, 3.0],
            imu_quaternion_wxyz=[
                math.cos(half_angle),
                math.sin(half_angle),
                0.0,
                0.0,
            ],
        )
        decoded = decode_low_state(state)
        last_action = np.linspace(-1.0, 1.0, 12)

        proprio, current = build_policy_proprio(
            decoded,
            np.zeros(12),
            last_action,
            [False, True, False, True],
            0.5,
            0.25,
            1.0,
            0.05,
            "parkour",
        )

        self.assertEqual(proprio.shape, (53,))
        np.testing.assert_allclose(proprio[:3], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(proprio[3:5], [math.pi / 4.0, 0.0])
        np.testing.assert_array_equal(proprio[5:8], np.zeros(3))
        np.testing.assert_allclose(proprio[8:11], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(proprio[11:13], [1.0, 0.0])
        np.testing.assert_allclose(proprio[13:25], decoded.joint_q)
        np.testing.assert_allclose(proprio[25:37], decoded.joint_dq * 0.05)
        np.testing.assert_allclose(proprio[37:49], last_action)
        np.testing.assert_array_equal(proprio[49:53], [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(current, [True, False, True, False])

    def test_actor_zero_reaches_unitree_motor_zero_and_isaac_fr_hip(self):
        target = np.zeros(12)
        target[0] = 0.5

        command = encode_low_cmd(target, np.full(12, 40.0), np.ones(12))

        np.testing.assert_array_equal(command.motor_q, target)
        isaac_target = np.asarray(policy_to_isaac(command.motor_q))
        self.assertEqual(isaac_target[3], 0.5)
        self.assertEqual(np.count_nonzero(isaac_target), 1)

    def test_lowcmd_fields_and_go2_limits_are_finite(self):
        target = np.linspace(-0.2, 0.2, 12)
        kp = np.linspace(30.0, 41.0, 12)
        kd = np.linspace(0.5, 1.6, 12)

        command = encode_low_cmd(target, kp, kd)

        np.testing.assert_allclose(command.motor_q, target)
        np.testing.assert_array_equal(command.motor_dq, np.zeros(12))
        np.testing.assert_array_equal(command.motor_tau, np.zeros(12))
        np.testing.assert_allclose(command.motor_kp, kp)
        np.testing.assert_allclose(command.motor_kd, kd)
        for values in (
            GO2_JOINT_LIMITS_LOW,
            GO2_JOINT_LIMITS_HIGH,
            GO2_TORQUE_LIMITS,
            GO2_JOINT_VELOCITY_LIMITS,
        ):
            self.assertEqual(values.shape, (12,))
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(
            GO2_TORQUE_LIMITS,
            [23.7, 23.7, 35.55] * 4,
        )

    def test_boundary_rejects_nonfinite_values(self):
        state = BoundaryLowState(
            motor_q=[0.0] * 11 + [np.nan],
            motor_dq=[0.0] * 12,
            foot_force=[0.0] * 4,
            gyroscope=[0.0] * 3,
            imu_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        )

        with self.assertRaisesRegex(RuntimeError, "finite"):
            decode_low_state(state)

        with self.assertRaisesRegex(RuntimeError, "Kp"):
            encode_low_cmd(np.zeros(12), np.zeros(12), np.ones(12))
        invalid_kd = np.ones(12)
        invalid_kd[3] = -0.1
        with self.assertRaisesRegex(RuntimeError, "Kd"):
            encode_low_cmd(np.zeros(12), np.ones(12), invalid_kd)


if __name__ == "__main__":
    unittest.main()
