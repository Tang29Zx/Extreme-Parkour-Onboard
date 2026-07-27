"""Publish training-equivalent D435i depth for Extreme Parkour inference."""

from collections import OrderedDict
import json
import os.path as osp
import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch

from depth_processing import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    REALSENSE_DEPTH_HEIGHT,
    REALSENSE_DEPTH_WIDTH,
    DepthProcessingConfig,
    DepthProcessingError,
    prepare_realsense_depth,
    preprocess_depth,
)


class VisualHandlerNode(Node):
    """Own the D435i stream and publish normalized 58x87 depth."""

    def __init__(
        self,
        cfg: dict,
        rs_resolution=(REALSENSE_DEPTH_WIDTH, REALSENSE_DEPTH_HEIGHT),
        rs_fps: int = 30,
        depth_input_topic="/camera/forward_depth",
        forward_depth_image_topic="/forward_depth_image",
    ):
        super().__init__("depth_image")
        if tuple(rs_resolution) != (
            REALSENSE_DEPTH_WIDTH,
            REALSENSE_DEPTH_HEIGHT,
        ):
            raise ValueError("RealSense depth resolution must be 424x240.")
        if rs_fps <= 0:
            raise ValueError("RealSense FPS must be positive.")

        self.depth_config = DepthProcessingConfig.from_mapping(cfg)
        self.rs_resolution = tuple(rs_resolution)
        self.rs_fps = int(rs_fps)
        self.depth_input_topic = depth_input_topic
        self.forward_depth_image_topic = forward_depth_image_topic
        self.rs_pipeline = None
        self.depth_scale = None
        self.last_frame_number = None
        self.last_camera_timestamp_ms = None

        self.start_pipeline()
        self.start_ros_handlers()

    def start_pipeline(self) -> None:
        """Open the exact D435i stream used by the accepted deployment path."""
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_fps,
        )
        started = False
        try:
            profile = pipeline.start(stream_config)
            started = True
            depth_scale = float(
                profile.get_device().first_depth_sensor().get_depth_scale()
            )
            if not np.isfinite(depth_scale) or depth_scale <= 0.0:
                raise RuntimeError("RealSense returned an invalid depth scale.")
            self.rs_pipeline = pipeline
            self.depth_scale = depth_scale
        except Exception:
            if started:
                pipeline.stop()
            raise

    def stop_pipeline(self) -> None:
        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
        self.rs_pipeline = None
        self.depth_scale = None
        self.last_frame_number = None
        self.last_camera_timestamp_ms = None

    def start_ros_handlers(self) -> None:
        self.depth_input_pub = self.create_publisher(
            Image,
            self.depth_input_topic,
            1,
        )
        self.forward_depth_image_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_image_topic,
            1,
        )
        self.get_logger().info(
            "D435i depth publisher started: 424x240 -> 60x106 -> 58x87"
        )

    def _publish_metric_preview(self, normalized_depth: torch.Tensor) -> None:
        """Publish the exact resized network field in meters for inspection."""
        metric = (
            (normalized_depth[0] + 0.5)
            * (self.depth_config.far_clip - self.depth_config.near_clip)
            + self.depth_config.near_clip
        )
        metric_data = np.ascontiguousarray(metric.cpu().numpy(), dtype=np.float32)
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "d435_sim_depth_link"
        message.height = DEPTH_HEIGHT
        message.width = DEPTH_WIDTH
        message.encoding = "32FC1"
        message.is_bigendian = 0
        message.step = DEPTH_WIDTH * np.dtype(np.float32).itemsize
        message.data = metric_data.tobytes()
        self.depth_input_pub.publish(message)

    @torch.inference_mode()
    def get_depth_frame(self):
        """Read one frame and return normalized ``(1, 58, 87)`` depth."""
        if self.rs_pipeline is None or self.depth_scale is None:
            raise RuntimeError("RealSense pipeline is not started.")
        try:
            frames = self.rs_pipeline.wait_for_frames(1000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                self.get_logger().error("RealSense returned no depth frame.")
                return None
            frame_number = int(depth_frame.get_frame_number())
            camera_timestamp_ms = float(depth_frame.get_timestamp())
            if not np.isfinite(camera_timestamp_ms):
                raise RuntimeError("RealSense returned a non-finite timestamp.")
            if (
                self.last_frame_number is not None
                and frame_number <= self.last_frame_number
            ):
                raise RuntimeError("RealSense frame number did not advance.")
            if (
                self.last_camera_timestamp_ms is not None
                and camera_timestamp_ms <= self.last_camera_timestamp_ms
            ):
                raise RuntimeError("RealSense camera timestamp did not advance.")
            raw_z16 = np.asanyarray(depth_frame.get_data()).copy()
            metric_source = prepare_realsense_depth(
                raw_z16,
                self.depth_scale,
                self.depth_config,
            )
            normalized = preprocess_depth(
                metric_source,
                self.depth_config,
                device="cpu",
            )
            self.last_frame_number = frame_number
            self.last_camera_timestamp_ms = camera_timestamp_ms
        except (DepthProcessingError, RuntimeError) as error:
            self.get_logger().error(f"Rejected D435i depth frame: {error}")
            return None
        except Exception as error:
            self.get_logger().error(f"Failed to acquire D435i depth frame: {error}")
            return None

        self._publish_metric_preview(normalized)
        return normalized

    def publish_depth_data(self, depth_data: torch.Tensor) -> None:
        if tuple(depth_data.shape) != (1, DEPTH_HEIGHT, DEPTH_WIDTH):
            raise DepthProcessingError(
                "Published depth must have shape (1, 58, 87), "
                f"got {tuple(depth_data.shape)}."
            )
        message = Float32MultiArray()
        message.data = depth_data.flatten().cpu().numpy().tolist()
        self.forward_depth_image_pub.publish(message)

    def start_main_loop_timer(self, duration: float) -> None:
        self.create_timer(duration, self.main_loop)

    def main_loop(self) -> None:
        depth_image = self.get_depth_frame()
        if depth_image is not None:
            self.publish_depth_data(depth_image)
        else:
            self.get_logger().warning("One D435i depth frame was not published.")


@torch.inference_mode()
def main(args):
    rclpy.init()
    visual_node = None
    try:
        if args.logdir is None:
            raise ValueError("Please provide --logdir.")
        with open(osp.join(args.logdir, "config.json"), "r") as config_file:
            config = json.load(config_file, object_pairs_hook=OrderedDict)

        visual_node = VisualHandlerNode(
            cfg=config,
            rs_resolution=(args.width, args.height),
            rs_fps=args.fps,
        )
        duration = 1.0 / args.fps
        if args.loop_mode == "while":
            rclpy.spin_once(visual_node, timeout_sec=0.0)
            while rclpy.ok():
                started = time.monotonic()
                visual_node.main_loop()
                rclpy.spin_once(visual_node, timeout_sec=0.0)
                time.sleep(max(0.0, duration - (time.monotonic() - started)))
        else:
            visual_node.start_main_loop_timer(duration)
            rclpy.spin(visual_node)
    except KeyboardInterrupt:
        pass
    finally:
        if visual_node is not None:
            visual_node.stop_pipeline()
            visual_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory containing config.json and exported model files.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=REALSENSE_DEPTH_HEIGHT,
        help="D435i depth height; only 240 is supported.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=REALSENSE_DEPTH_WIDTH,
        help="D435i depth width; only 424 is supported.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--loop_mode",
        type=str,
        default="timer",
        choices=["while", "timer"],
    )
    main(parser.parse_args())
