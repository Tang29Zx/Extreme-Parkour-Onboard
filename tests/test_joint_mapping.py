import json
from pathlib import Path
import unittest

from joint_mapping import (
    DOF_SIGNS,
    FOOT_REAL_TO_SIM,
    ISAAC_DOF_NAMES,
    ISAAC_TO_POLICY_DOF,
    ISAAC_TO_UNITREE_FOOT,
    POLICY_DOF_NAMES,
    POLICY_TO_ISAAC_DOF,
    POLICY_TO_REAL_DOF,
    SIM_DOF_NAMES,
    SIM_TO_REAL_DOF,
    foot_real_to_sim,
    isaac_feet_to_unitree,
    isaac_to_policy,
    policy_to_isaac,
    real_to_sim,
    sim_to_real,
    unitree_feet_to_isaac,
)


class JointMappingTest(unittest.TestCase):
    def test_policy_joint_contract(self):
        expected_policy_names = tuple(
            ISAAC_DOF_NAMES[index] for index in ISAAC_TO_POLICY_DOF
        )
        self.assertEqual(
            POLICY_DOF_NAMES,
            (
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ),
        )
        self.assertEqual(POLICY_DOF_NAMES, expected_policy_names)
        self.assertEqual(POLICY_TO_REAL_DOF, tuple(range(12)))
        self.assertEqual(SIM_DOF_NAMES, POLICY_DOF_NAMES)
        self.assertEqual(SIM_TO_REAL_DOF, POLICY_TO_REAL_DOF)
        self.assertEqual(DOF_SIGNS, (1.0,) * 12)

    def test_joint_mapping_round_trip_with_distinct_values(self):
        policy = tuple(float(index) for index in range(12))
        real = sim_to_real(policy)

        self.assertEqual(real, policy)
        self.assertEqual(real_to_sim(real), policy)

        isaac = tuple(float(index + 10) for index in range(12))
        self.assertEqual(
            policy_to_isaac(isaac_to_policy(isaac)),
            isaac,
        )
        self.assertEqual(
            tuple(POLICY_TO_ISAAC_DOF),
            tuple(ISAAC_TO_POLICY_DOF),
        )

    def test_actor_dimension_zero_commands_front_right_motor_zero(self):
        one_hot = (1.0,) + (0.0,) * 11

        self.assertEqual(sim_to_real(one_hot), one_hot)

    def test_traced_default_pose_maps_to_named_unitree_legs(self):
        config = json.loads(
            Path("traced/config.json").read_text(encoding="utf-8")
        )
        angles = config["init_state"]["default_joint_angles"]
        policy = tuple(angles[name] for name in POLICY_DOF_NAMES)

        self.assertEqual(
            sim_to_real(policy),
            (
                angles["FR_hip_joint"],
                angles["FR_thigh_joint"],
                angles["FR_calf_joint"],
                angles["FL_hip_joint"],
                angles["FL_thigh_joint"],
                angles["FL_calf_joint"],
                angles["RR_hip_joint"],
                angles["RR_thigh_joint"],
                angles["RR_calf_joint"],
                angles["RL_hip_joint"],
                angles["RL_thigh_joint"],
                angles["RL_calf_joint"],
            ),
        )

    def test_foot_force_mapping(self):
        self.assertEqual(FOOT_REAL_TO_SIM, (0, 1, 2, 3))
        self.assertEqual(
            foot_real_to_sim((10.0, 20.0, 30.0, 40.0)),
            (10.0, 20.0, 30.0, 40.0),
        )
        isaac = (1.0, 2.0, 3.0, 4.0)
        self.assertEqual(ISAAC_TO_UNITREE_FOOT, (1, 0, 3, 2))
        self.assertEqual(isaac_feet_to_unitree(isaac), (2.0, 1.0, 4.0, 3.0))
        self.assertEqual(
            unitree_feet_to_isaac(isaac_feet_to_unitree(isaac)),
            isaac,
        )

    def test_mapping_rejects_wrong_lengths(self):
        with self.assertRaisesRegex(ValueError, "12 values"):
            sim_to_real((0.0,) * 11)
        with self.assertRaisesRegex(ValueError, "12 values"):
            real_to_sim((0.0,) * 11)
        with self.assertRaisesRegex(ValueError, "4 values"):
            foot_real_to_sim((0.0,) * 3)
        with self.assertRaisesRegex(ValueError, "12 values"):
            isaac_to_policy((0.0,) * 11)
        with self.assertRaisesRegex(ValueError, "4 values"):
            isaac_feet_to_unitree((0.0,) * 3)


if __name__ == "__main__":
    unittest.main()
