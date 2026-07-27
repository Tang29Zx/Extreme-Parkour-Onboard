import unittest

import torch

from legged_gym.reset_utils import resolve_dof_pos_reset_range
from legged_gym.scripts.replay_geometry import (
    single_box_rear_x,
    single_box_world_bounds,
)


class SingleBoxWorldBoundsTests(unittest.TestCase):
    def test_bounds_follow_each_selected_terrain_origin(self):
        origins = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [19.0, 6.0, 0.0],
            ]
        )
        box_min, box_max = single_box_world_bounds(origins)

        torch.testing.assert_close(
            box_min[:, 0],
            torch.tensor(
                [
                    [2.0, 1.4, 0.0],
                    [20.0, 5.4, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            box_max[:, 0],
            torch.tensor(
                [
                    [3.2, 2.6, 0.2],
                    [21.2, 6.6, 0.2],
                ]
            ),
        )
        torch.testing.assert_close(
            single_box_rear_x(origins),
            torch.tensor([3.4, 21.4]),
        )

    def test_replay_can_request_a_deterministic_dof_reset(self):
        self.assertEqual(resolve_dof_pos_reset_range([0.0, 0.0]), (0.0, 0.0))
        self.assertEqual(resolve_dof_pos_reset_range(None), (0.0, 0.9))
        with self.assertRaisesRegex(ValueError, "lower bound"):
            resolve_dof_pos_reset_range([0.1, -0.1])


if __name__ == "__main__":
    unittest.main()
