import unittest

import torch

from policy_context import reset_policy_context, update_proprio_history


class PolicyContextTest(unittest.TestCase):
    def test_episode_start_replaces_all_stale_history(self):
        history = torch.full((1, 10, 3), 99.0)
        proprio = torch.tensor([[1.0, 2.0, 3.0]])
        result = update_proprio_history(history, proprio, torch.zeros(1))

        expected = proprio.unsqueeze(1).repeat(1, 10, 1)
        torch.testing.assert_close(result, expected)

    def test_running_history_shifts_one_frame(self):
        history = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
        proprio = torch.tensor([[100.0, 101.0, 102.0]])
        result = update_proprio_history(history, proprio, torch.tensor([2.0]))

        torch.testing.assert_close(result[:, :-1], history[:, 1:])
        torch.testing.assert_close(result[:, -1], proprio)

    def test_reset_clears_buffers_and_resets_depth_once(self):
        actions = torch.ones(12)
        history = torch.ones(1, 10, 53)
        episode_length = torch.ones(1)
        calls = []

        reset_policy_context(
            actions,
            history,
            episode_length,
            lambda: calls.append("reset"),
        )

        self.assertEqual(calls, ["reset"])
        self.assertEqual(torch.count_nonzero(actions).item(), 0)
        self.assertEqual(torch.count_nonzero(history).item(), 0)
        self.assertEqual(torch.count_nonzero(episode_length).item(), 0)


if __name__ == "__main__":
    unittest.main()
