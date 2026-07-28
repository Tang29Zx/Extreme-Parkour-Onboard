import json
import unittest

import numpy as np

from runtime_status import (
    RUNTIME_STATUS_SCHEMA_VERSION,
    build_runtime_status,
    should_publish_runtime_status,
)


class RuntimeStatusTest(unittest.TestCase):
    def _build(self, **overrides):
        values = {
            "timestamp_unix_s": 1000.25,
            "phase": "policy",
            "dryrun": False,
            "real_lowcmd_authorized": True,
            "engagement_active": False,
            "input_ages_s": (0.001, 0.002, 0.003),
            "roll_pitch_rad": np.deg2rad([2.0, -3.0]),
            "foot_force": (10.0, 20.0, 30.0, 40.0),
            "contact_state": (True, True, True, True),
            "measured_q": np.zeros(12),
            "commanded_q": np.full(12, 0.1),
            "measured_dq": np.full(12, 0.2),
            "raw_action": np.linspace(-2.0, 2.0, 12),
            "requested_q": np.full(12, 0.15),
            "motor_temperature": np.arange(35.0, 47.0),
            "motor_lost": np.asarray([0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
            "motor_tau_est": np.full(12, 2.0),
            "torque_limits": np.full(12, 20.0),
            "kp": np.full(12, 40.0),
            "kd": np.ones(12),
            "loop_samples_s": (0.01, 0.02, 0.03),
        }
        values.update(overrides)
        return json.loads(build_runtime_status(**values))

    def test_status_contains_compact_live_control_summary(self):
        status = self._build()

        self.assertEqual(status["schema_version"], RUNTIME_STATUS_SCHEMA_VERSION)
        self.assertEqual(status["phase"], "policy")
        self.assertTrue(status["output"]["real_lowcmd_authorized"])
        self.assertEqual(status["input_age_ms"]["depth"], 3.0)
        self.assertEqual(status["body"]["roll_deg"], 2.0)
        self.assertEqual(status["motor"]["lost"][5], 5)
        self.assertEqual(status["joint"]["max_tracking_error_rad"], 0.1)
        self.assertEqual(status["policy"]["max_request_command_delta_rad"], 0.05)
        self.assertAlmostEqual(status["motor"]["max_abs_tau_ratio"], 0.1)
        self.assertAlmostEqual(status["motor"]["max_abs_pd_tau_ratio"], 0.19)
        self.assertEqual(status["loop_ms"]["last"], 30.0)
        self.assertEqual(status["loop_ms"]["p50"], 20.0)
        self.assertEqual(status["loop_ms"]["p95"], 29.0)

    def test_status_rejects_nonfinite_or_invalid_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self._build(raw_action=np.full(12, np.nan))
        invalid_lost = np.zeros(12)
        invalid_lost[3] = 0.5
        with self.assertRaisesRegex(ValueError, "integers"):
            self._build(motor_lost=invalid_lost)
        with self.assertRaisesRegex(ValueError, "positive"):
            self._build(torque_limits=np.zeros(12))

    def test_publish_throttle_uses_monotonic_time(self):
        self.assertTrue(should_publish_runtime_status(None, 10.0))
        self.assertFalse(should_publish_runtime_status(10.0, 10.49))
        self.assertTrue(should_publish_runtime_status(10.0, 10.5))
        with self.assertRaisesRegex(ValueError, "backwards"):
            should_publish_runtime_status(10.0, 9.0)


if __name__ == "__main__":
    unittest.main()
