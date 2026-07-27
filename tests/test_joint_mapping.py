import json
from pathlib import Path
import unittest

from joint_mapping import (
    DOF_SIGNS,
    FOOT_REAL_TO_SIM,
    SIM_DOF_NAMES,
    SIM_TO_REAL_DOF,
    foot_real_to_sim,
    real_to_sim,
    sim_to_real,
)


class JointMappingTest(unittest.TestCase):
    def test_policy_joint_contract(self):
        self.assertEqual(
            SIM_DOF_NAMES,
            (
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ),
        )
        self.assertEqual(
            SIM_TO_REAL_DOF,
            (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8),
        )
        self.assertEqual(DOF_SIGNS, (1.0,) * 12)

    def test_joint_mapping_round_trip_with_distinct_values(self):
        simulation = tuple(float(index) for index in range(12))
        real = sim_to_real(simulation)

        self.assertEqual(
            real,
            (3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0, 6.0, 7.0, 8.0),
        )
        self.assertEqual(real_to_sim(real), simulation)

    def test_traced_default_pose_maps_to_named_unitree_legs(self):
        config = json.loads(
            Path("traced/config.json").read_text(encoding="utf-8")
        )
        angles = config["init_state"]["default_joint_angles"]
        simulation = tuple(angles[name] for name in SIM_DOF_NAMES)

        self.assertEqual(
            sim_to_real(simulation),
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
        self.assertEqual(FOOT_REAL_TO_SIM, (1, 0, 3, 2))
        self.assertEqual(
            foot_real_to_sim((10.0, 20.0, 30.0, 40.0)),
            (20.0, 10.0, 40.0, 30.0),
        )

    def test_mapping_rejects_wrong_lengths(self):
        with self.assertRaisesRegex(ValueError, "12 values"):
            sim_to_real((0.0,) * 11)
        with self.assertRaisesRegex(ValueError, "12 values"):
            real_to_sim((0.0,) * 11)
        with self.assertRaisesRegex(ValueError, "4 values"):
            foot_real_to_sim((0.0,) * 3)


if __name__ == "__main__":
    unittest.main()
