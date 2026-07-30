import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import flight_recorder as flight_recorder_module
from flight_recorder import CONTROL_CAPACITY, VISUAL_CAPACITY, FlightRecorder


class FlightRecorderTest(unittest.TestCase):
    def _record_control(self, recorder, index):
        values = np.arange(12, dtype=np.float32) + index
        recorder.record_control(
            timestamp=100.0 + index,
            phase="policy",
            engagement_active=index < 2,
            raw_action=values,
            executed_action=values * 0.5,
            requested_q=values + 1.0,
            commanded_q=values + 0.5,
            measured_q=values + 0.25,
            measured_dq=values * 0.01,
            imu_quaternion=(1.0, 0.0, 0.0, 0.0),
            foot_force=(10.0, 20.0, 30.0, 40.0),
            contact_state=(1.0, 1.0, 1.0, 1.0),
            motor_temperature=np.full(12, 45.0),
            motor_lost=np.zeros(12),
            motor_tau_est=values * 0.02,
            input_ages=(0.01, 0.02, 0.03),
            loop_s=0.02,
        )

    def test_ring_capacity_retains_most_recent_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(
                directory,
                control_capacity=2,
                visual_capacity=1,
            )
            for index in range(3):
                self._record_control(recorder, index)
            recorder.record_visual(
                timestamp=100.0,
                depth_input=np.zeros((58, 87), dtype=np.float32),
                visual_output=np.zeros(34),
            )
            latest_depth = np.linspace(
                -0.4,
                0.4,
                num=58 * 87,
                dtype=np.float32,
            ).reshape(58, 87)
            recorder.record_visual(
                timestamp=101.0,
                depth_input=latest_depth,
                visual_output=np.ones(34),
            )

            path = recorder.flush("unit_test")
            self.assertIsNotNone(path)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["control_timestamp"],
                    np.asarray([101.0, 102.0]),
                )
                self.assertEqual(data["raw_action"].shape, (2, 12))
                self.assertEqual(data["motor_temperature"].shape, (2, 12))
                self.assertEqual(data["motor_lost"].shape, (2, 12))
                self.assertEqual(data["motor_tau_est"].shape, (2, 12))
                self.assertEqual(data["contact_state"].shape, (2, 4))
                self.assertEqual(data["format_version"].tolist(), [4])
                self.assertEqual(data["depth_input"].shape, (1, 58, 87))
                np.testing.assert_array_equal(data["depth_input"][0], latest_depth)
                np.testing.assert_allclose(
                    data["depth_stats"][0],
                    (latest_depth.min(), latest_depth.max(), latest_depth.mean()),
                    rtol=0.0,
                    atol=1e-7,
                )
                self.assertEqual(data["visual_output"].shape, (1, 34))
                self.assertEqual(data["reason"].tolist(), ["unit_test"])
                self.assertEqual(data["detail"].tolist(), [""])
                self.assertEqual(data["visual_timestamp"].tolist(), [101.0])

            self.assertEqual(len(recorder.control_records), 0)
            self.assertEqual(len(recorder.visual_records), 0)
            self.assertIsNone(recorder.flush("empty"))

    def test_default_capacity_retains_ten_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            recorder._next_checkpoint_time = float("inf")

            for index in range(CONTROL_CAPACITY + 1):
                self._record_control(recorder, index)
            for index in range(VISUAL_CAPACITY + 1):
                recorder.record_visual(
                    timestamp=100.0 + index,
                    depth_input=np.zeros((58, 87), dtype=np.float32),
                    visual_output=np.zeros(34),
                )

            self.assertEqual(CONTROL_CAPACITY, 500)
            self.assertEqual(VISUAL_CAPACITY, 100)
            self.assertEqual(len(recorder.control_records), 500)
            self.assertEqual(len(recorder.visual_records), 100)
            self.assertEqual(recorder.control_records[0]["timestamp"], 101.0)
            self.assertEqual(recorder.visual_records[0]["timestamp"], 101.0)

            path = recorder.flush("ten_second_window")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["control_timestamp"].shape, (500,))
                self.assertEqual(data["depth_input"].shape, (100, 58, 87))

    def test_nonfinite_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            depth_input = np.zeros((58, 87), dtype=np.float32)
            depth_input[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                recorder.record_visual(
                    timestamp=0.0,
                    depth_input=depth_input,
                    visual_output=np.zeros(34),
                )

    def test_wrong_depth_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            with self.assertRaisesRegex(ValueError, "shape"):
                recorder.record_visual(
                    timestamp=0.0,
                    depth_input=np.zeros((58, 86), dtype=np.float32),
                    visual_output=np.zeros(34),
                )

    def test_visual_only_record_keeps_control_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            recorder.record_visual(
                timestamp=1.0,
                depth_input=np.zeros((58, 87), dtype=np.float32),
                visual_output=np.zeros(34),
            )

            path = recorder.flush("visual_only")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["raw_action"].shape, (0, 12))
                self.assertEqual(data["imu_quaternion"].shape, (0, 4))
                self.assertEqual(data["input_ages"].shape, (0, 3))
                self.assertEqual(data["motor_temperature"].shape, (0, 12))
                self.assertEqual(data["depth_input"].shape, (1, 58, 87))

    def test_control_only_record_keeps_visual_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            self._record_control(recorder, 0)

            path = recorder.flush("control_only")
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["visual_timestamp"].shape, (0,))
                self.assertEqual(data["depth_input"].shape, (0, 58, 87))
                self.assertEqual(data["depth_stats"].shape, (0, 3))
                self.assertEqual(data["visual_output"].shape, (0, 34))

    def test_checkpoint_is_loadable_and_syncs_file_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            self._record_control(recorder, 0)
            recorder.record_visual(
                timestamp=100.0,
                depth_input=np.zeros((58, 87), dtype=np.float32),
                visual_output=np.zeros(34),
            )

            with mock.patch.object(
                flight_recorder_module.os,
                "fsync",
                wraps=os.fsync,
            ) as fsync:
                path = recorder.checkpoint()

            self.assertEqual(path, recorder.checkpoint_path)
            self.assertGreaterEqual(fsync.call_count, 2)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["reason"].tolist(), ["periodic_checkpoint"])
                self.assertEqual(data["detail"].tolist(), [""])
                self.assertEqual(data["control_timestamp"].tolist(), [100.0])
                self.assertEqual(data["depth_input"].shape, (1, 58, 87))

    def test_due_checkpoint_runs_in_a_background_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(
                directory,
                checkpoint_interval_s=1e-9,
            )
            self._record_control(recorder, 0)
            recorder._wait_for_pending_checkpoint()

            self.assertTrue(recorder.checkpoint_path.exists())
            with np.load(recorder.checkpoint_path, allow_pickle=False) as data:
                self.assertEqual(data["reason"].tolist(), ["periodic_checkpoint"])
                self.assertEqual(data["control_timestamp"].tolist(), [100.0])

    def test_incomplete_temporary_write_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            self._record_control(recorder, 0)
            path = recorder.checkpoint()
            temporary_path = path.with_name(f".{path.name}.tmp")
            temporary_path.write_bytes(b"incomplete reset-interrupted data")

            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["control_timestamp"].tolist(), [100.0])
                self.assertEqual(data["reason"].tolist(), ["periodic_checkpoint"])

    def test_flush_failure_preserves_checkpoint_and_ring_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            self._record_control(recorder, 0)
            checkpoint_path = recorder.checkpoint()
            self._record_control(recorder, 1)

            with mock.patch.object(
                recorder,
                "_write_payload_durable",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    recorder.flush("unit_test")

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(len(recorder.control_records), 2)
            with np.load(checkpoint_path, allow_pickle=False) as data:
                self.assertEqual(data["control_timestamp"].tolist(), [100.0])

    def test_successful_flush_removes_durable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(directory)
            self._record_control(recorder, 0)
            checkpoint_path = recorder.checkpoint()

            final_path = recorder.flush("unit_test", "joint=RR_thigh")

            self.assertFalse(checkpoint_path.exists())
            self.assertTrue(final_path.exists())
            with np.load(final_path, allow_pickle=False) as data:
                self.assertEqual(data["reason"].tolist(), ["unit_test"])
                self.assertEqual(data["detail"].tolist(), ["joint=RR_thigh"])
                self.assertEqual(data["control_timestamp"].tolist(), [100.0])


if __name__ == "__main__":
    unittest.main()
