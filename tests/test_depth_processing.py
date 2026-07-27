import json
from pathlib import Path
import unittest

import numpy as np
import torch

from depth_processing import (
    DepthProcessingConfig,
    DepthProcessingError,
    prepare_realsense_depth,
    preprocess_depth,
)


class DepthProcessingTest(unittest.TestCase):
    def test_traced_config_matches_supported_depth_contract(self):
        config = json.loads(Path("traced/config.json").read_text(encoding="utf-8"))

        result = DepthProcessingConfig.from_mapping(config)

        self.assertEqual((result.original_height, result.original_width), (60, 106))
        self.assertEqual((result.output_height, result.output_width), (58, 87))
        self.assertEqual((result.near_clip, result.far_clip), (0.0, 2.0))

    def test_four_by_four_area_downsample_is_exact(self):
        expected_mm = (
            500 + np.arange(60 * 106, dtype=np.uint16).reshape(60, 106) % 1000
        )
        raw_depth = np.repeat(
            np.repeat(expected_mm, 4, axis=0),
            4,
            axis=1,
        )

        result = prepare_realsense_depth(raw_depth, depth_scale=0.001)

        self.assertEqual(result.shape, (60, 106))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(
            result,
            expected_mm.astype(np.float32) / 1000.0,
            rtol=2e-7,
            atol=1e-7,
        )

    def test_invalid_and_overrange_depth_become_two_meters(self):
        raw_depth = np.full((240, 424), 1000, dtype=np.uint16)
        raw_depth[:4, :4] = 0
        raw_depth[:4, 4:8] = 3000
        raw_depth[:4, 8:12] = 65535

        result = prepare_realsense_depth(raw_depth, depth_scale=0.001)

        np.testing.assert_array_equal(result[0, :4], (2.0, 2.0, 2.0, 1.0))

    def test_realsense_input_contract_is_strict(self):
        with self.assertRaisesRegex(DepthProcessingError, "shape"):
            prepare_realsense_depth(np.zeros((480, 640), dtype=np.uint16), 0.001)
        with self.assertRaisesRegex(DepthProcessingError, "uint16"):
            prepare_realsense_depth(np.zeros((240, 424), dtype=np.float32), 0.001)
        with self.assertRaisesRegex(DepthProcessingError, "scale"):
            prepare_realsense_depth(np.zeros((240, 424), dtype=np.uint16), 0.0)

    def test_only_zero_to_two_meter_config_is_supported(self):
        with self.assertRaisesRegex(DepthProcessingError, "0-to-2-meter"):
            DepthProcessingConfig(far_clip=3.0)

    def test_constant_depth_has_expected_normalization(self):
        for meters, expected in ((0.5, -0.25), (1.0, 0.0), (2.0, 0.5)):
            result = preprocess_depth(
                np.full((60, 106), meters, dtype=np.float32)
            )

            self.assertEqual(tuple(result.shape), (1, 58, 87))
            torch.testing.assert_close(result, torch.full_like(result, expected))

    def test_bicubic_output_is_not_clipped_after_resize(self):
        depth = np.full((60, 106), 2.0, dtype=np.float32)
        depth[:, 50:] = 0.1

        result = preprocess_depth(depth)

        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(float(result.min()), -0.5)

    def test_wrong_metric_source_shape_is_rejected(self):
        with self.assertRaisesRegex(DepthProcessingError, "shape"):
            preprocess_depth(np.zeros((58, 87), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
