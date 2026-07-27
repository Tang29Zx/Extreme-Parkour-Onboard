"""Low-overhead in-memory flight recorder for policy diagnostics."""

import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence

import numpy as np


CONTROL_CAPACITY = 250
VISUAL_CAPACITY = 50


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result.copy()


class FlightRecorder:
    """Retain recent control and visual samples, then atomically save NPZ."""

    def __init__(
        self,
        output_dir: str,
        control_capacity: int = CONTROL_CAPACITY,
        visual_capacity: int = VISUAL_CAPACITY,
    ) -> None:
        if int(control_capacity) <= 0 or int(visual_capacity) <= 0:
            raise ValueError("flight recorder capacities must be positive")
        self.output_dir = Path(output_dir).expanduser()
        self.control_records = deque(maxlen=int(control_capacity))  # type: Deque[Dict]
        self.visual_records = deque(maxlen=int(visual_capacity))  # type: Deque[Dict]

    def record_control(
        self,
        *,
        timestamp: float,
        phase: str,
        engagement_active: bool,
        raw_action: Sequence[float],
        executed_action: Sequence[float],
        requested_q: Sequence[float],
        commanded_q: Sequence[float],
        measured_q: Sequence[float],
        measured_dq: Sequence[float],
        imu_quaternion: Sequence[float],
        foot_force: Sequence[float],
        contact_state: Sequence[float],
        motor_temperature: Sequence[float],
        motor_lost: Sequence[float],
        motor_tau_est: Sequence[float],
        input_ages: Sequence[float],
        loop_s: float,
    ) -> None:
        scalar_values = np.asarray([timestamp, loop_s], dtype=np.float64)
        if not np.isfinite(scalar_values).all() or loop_s < 0.0:
            raise ValueError("flight control timing must be finite and non-negative")
        self.control_records.append(
            {
                "timestamp": float(timestamp),
                "phase": str(phase),
                "engagement_active": bool(engagement_active),
                "raw_action": _finite_vector(raw_action, 12, "raw action"),
                "executed_action": _finite_vector(
                    executed_action, 12, "executed action"
                ),
                "requested_q": _finite_vector(requested_q, 12, "requested q"),
                "commanded_q": _finite_vector(commanded_q, 12, "commanded q"),
                "measured_q": _finite_vector(measured_q, 12, "measured q"),
                "measured_dq": _finite_vector(measured_dq, 12, "measured dq"),
                "imu_quaternion": _finite_vector(
                    imu_quaternion, 4, "IMU quaternion"
                ),
                "foot_force": _finite_vector(foot_force, 4, "foot force"),
                "contact_state": _finite_vector(
                    contact_state, 4, "contact state"
                ),
                "motor_temperature": _finite_vector(
                    motor_temperature, 12, "motor temperature"
                ),
                "motor_lost": _finite_vector(motor_lost, 12, "motor lost"),
                "motor_tau_est": _finite_vector(
                    motor_tau_est, 12, "estimated motor torque"
                ),
                "input_ages": _finite_vector(input_ages, 3, "input ages"),
                "loop_s": float(loop_s),
            }
        )

    def record_visual(
        self,
        *,
        timestamp: float,
        depth_stats: Sequence[float],
        visual_output: Sequence[float],
    ) -> None:
        if not np.isfinite(float(timestamp)):
            raise ValueError("visual timestamp must be finite")
        self.visual_records.append(
            {
                "timestamp": float(timestamp),
                "depth_stats": _finite_vector(
                    depth_stats, 3, "depth min/max/mean"
                ),
                "visual_output": _finite_vector(
                    visual_output, 34, "visual output"
                ),
            }
        )

    def flush(self, reason: str) -> Optional[Path]:
        """Save buffered samples once and clear them after an atomic rename."""
        if not self.control_records and not self.visual_records:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        final_path = self.output_dir / f"extreme-flight-{timestamp}.npz"
        temporary_path = self.output_dir / f".{final_path.stem}.tmp.npz"

        controls = list(self.control_records)
        visuals = list(self.visual_records)
        payload = {
            "format_version": np.asarray([2], dtype=np.int32),
            "reason": np.asarray([str(reason)]),
            "control_timestamp": np.asarray(
                [record["timestamp"] for record in controls], dtype=np.float64
            ),
            "control_phase": np.asarray(
                [record["phase"] for record in controls]
            ),
            "engagement_active": np.asarray(
                [record["engagement_active"] for record in controls],
                dtype=np.bool_,
            ),
            "loop_s": np.asarray(
                [record["loop_s"] for record in controls], dtype=np.float32
            ),
        }
        control_vectors = {
            "raw_action": 12,
            "executed_action": 12,
            "requested_q": 12,
            "commanded_q": 12,
            "measured_q": 12,
            "measured_dq": 12,
            "imu_quaternion": 4,
            "foot_force": 4,
            "contact_state": 4,
            "motor_temperature": 12,
            "motor_lost": 12,
            "motor_tau_est": 12,
            "input_ages": 3,
        }
        for name, width in control_vectors.items():
            payload[name] = np.stack(
                [record[name] for record in controls], axis=0
            ) if controls else np.empty((0, width), dtype=np.float32)

        payload["visual_timestamp"] = np.asarray(
            [record["timestamp"] for record in visuals], dtype=np.float64
        )
        payload["depth_stats"] = np.stack(
            [record["depth_stats"] for record in visuals], axis=0
        ) if visuals else np.empty((0, 3), dtype=np.float32)
        payload["visual_output"] = np.stack(
            [record["visual_output"] for record in visuals], axis=0
        ) if visuals else np.empty((0, 34), dtype=np.float32)

        try:
            np.savez_compressed(str(temporary_path), **payload)
            os.replace(str(temporary_path), str(final_path))
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.control_records.clear()
        self.visual_records.clear()
        return final_path
