#!/usr/bin/env python3
"""Read the controller's live diagnostic topic without publishing commands."""

import argparse
from datetime import datetime
import json
import sys

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from runtime_status import RUNTIME_STATUS_TOPIC


def _format_lost(counters):
    nonzero = [
        f"{index}:{int(value)}"
        for index, value in enumerate(counters)
        if int(value) != 0
    ]
    return ",".join(nonzero) if nonzero else "0"


def format_runtime_status(status):
    """Format one schema-v1 payload as a single terminal line."""
    output = status["output"]
    if output["dryrun"]:
        output_mode = "dryrun"
    elif output["real_lowcmd_authorized"]:
        output_mode = "real"
    else:
        output_mode = "blocked"

    if output["engagement_active"]:
        engagement = "ramp"
    elif status["phase"] == "policy":
        engagement = "steady"
    else:
        engagement = "off"

    timestamp = datetime.fromtimestamp(status["timestamp_unix_s"]).strftime(
        "%H:%M:%S.%f"
    )[:-3]
    ages = status["input_age_ms"]
    body = status["body"]
    feet = status["feet"]
    joint = status["joint"]
    motor = status["motor"]
    loop = status["loop_ms"]
    contacts = "".join("1" if value else "0" for value in feet["contact"])
    max_velocity = max(abs(value) for value in joint["measured_dq_rad_s"])
    return (
        f"{timestamp} phase={status['phase']} output={output_mode} "
        f"engage={engagement} "
        f"age_ms(low/remote/depth)="
        f"{ages['low_state']:.1f}/{ages['remote']:.1f}/{ages['depth']:.1f} "
        f"rpy_deg={body['roll_deg']:+.1f}/{body['pitch_deg']:+.1f} "
        f"contact={contacts} qerr={joint['max_tracking_error_rad']:.3f} "
        f"dq={max_velocity:.2f} temp={motor['max_temperature_c']:.1f}C "
        f"lost={_format_lost(motor['lost'])} "
        f"tau_ratio(actual/pd)={motor['max_abs_tau_ratio']:.2f}/"
        f"{motor['max_abs_pd_tau_ratio']:.2f} "
        f"loop_ms(p95/max)={loop['p95']:.1f}/{loop['max']:.1f}"
    )


class RuntimeStatusReader(Node):
    def __init__(self, topic, print_json, logger_name):
        super().__init__("extreme_parkour_runtime_status_reader")
        self.print_json = print_json
        self.logger_name = logger_name.lstrip("/")
        self.create_subscription(String, topic, self._on_status, 1)
        rosout_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Log, "/rosout", self._on_log, rosout_qos)

    def _on_status(self, message):
        try:
            status = json.loads(message.data)
            if self.print_json:
                rendered = json.dumps(
                    status,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            else:
                rendered = format_runtime_status(status)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Invalid runtime status: {error}", file=sys.stderr, flush=True)
            return
        print(rendered, flush=True)

    def _on_log(self, message):
        if message.name.lstrip("/") != self.logger_name:
            return
        level = {
            10: "DEBUG",
            20: "INFO",
            30: "WARN",
            40: "ERROR",
            50: "FATAL",
        }.get(int(message.level), str(message.level))
        timestamp = datetime.fromtimestamp(
            message.stamp.sec + message.stamp.nanosec / 1e9
        ).strftime("%H:%M:%S.%f")[:-3]
        stream = sys.stderr if self.print_json else sys.stdout
        print(
            f"{timestamp} [{level}] [{message.name}]: {message.msg}",
            file=stream,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Read live Extreme Parkour controller diagnostics.",
    )
    parser.add_argument(
        "--topic",
        default=RUNTIME_STATUS_TOPIC,
        help="Runtime status topic to subscribe to.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete schema-v1 JSON payload.",
    )
    parser.add_argument(
        "--logger-name",
        default="unitree_ros2_real",
        help="ROS logger name whose live /rosout messages are shown.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = RuntimeStatusReader(args.topic, args.json, args.logger_name)
    print(
        f"Listening to {args.topic} and /rosout[{args.logger_name}]; "
        "press Ctrl+C to stop.",
        flush=True,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
