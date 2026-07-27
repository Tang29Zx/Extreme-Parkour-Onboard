import unittest

from output_routing import (
    SPORT_MODE_TOPIC,
    SPORT_STATE_TOPIC,
    resolve_output_topics,
)


class OutputRoutingTest(unittest.TestCase):
    def test_dryrun_uses_isolated_lowcmd_and_disables_sport_topics(self):
        lowcmd, sport_state, sport_mode = resolve_output_topics(
            "/lowcmd",
            dryrun=True,
            dryrun_suffix=1234,
        )

        self.assertEqual(lowcmd, "/lowcmd_dryrun_1234")
        self.assertIsNone(sport_state)
        self.assertIsNone(sport_mode)

    def test_real_mode_preserves_original_topics(self):
        result = resolve_output_topics("/lowcmd", dryrun=False)

        self.assertEqual(
            result,
            ("/lowcmd", SPORT_STATE_TOPIC, SPORT_MODE_TOPIC),
        )

    def test_dryrun_suffix_is_required(self):
        with self.assertRaisesRegex(ValueError, "dryrun_suffix"):
            resolve_output_topics("/lowcmd", dryrun=True)


if __name__ == "__main__":
    unittest.main()
