"""Low-overhead flight recorder with reset-resistant NPZ checkpoints."""

import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence

import numpy as np


CONTROL_CAPACITY = 500  # 10 seconds at 50 Hz.
VISUAL_CAPACITY = 100  # 10 seconds at 10 Hz.
DEPTH_INPUT_SHAPE = (58, 87)
FORMAT_VERSION = 4
CHECKPOINT_INTERVAL_S = 0.5
CHECKPOINT_REASON = "periodic_checkpoint"


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result.copy()


def _finite_depth_input(values: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != DEPTH_INPUT_SHAPE or not np.isfinite(result).all():
        raise ValueError(
            "depth input must have shape (58, 87) and contain finite values"
        )
    return result.copy()


class FlightRecorder:
    """Retain recent samples and durably checkpoint them outside control."""

    def __init__(
        self,
        output_dir: str,
        control_capacity: int = CONTROL_CAPACITY,
        visual_capacity: int = VISUAL_CAPACITY,
        checkpoint_interval_s: float = CHECKPOINT_INTERVAL_S,
    ) -> None:
        if int(control_capacity) <= 0 or int(visual_capacity) <= 0:
            raise ValueError("flight recorder capacities must be positive")
        if (
            not np.isfinite(float(checkpoint_interval_s))
            or float(checkpoint_interval_s) <= 0.0
        ):
            raise ValueError("checkpoint interval must be finite and positive")
        self.output_dir = Path(output_dir).expanduser()
        self.control_capacity = int(control_capacity)
        self.visual_capacity = int(visual_capacity)
        self.control_records = deque(maxlen=self.control_capacity)  # type: Deque[Dict]
        self.visual_records = deque(maxlen=self.visual_capacity)  # type: Deque[Dict]
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        started_at = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.checkpoint_path = (
            self.output_dir / f"extreme-flight-{started_at}-checkpoint.npz"
        )
        self._records_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._checkpoint_thread = None  # type: Optional[threading.Thread]
        self._checkpoint_in_progress = False
        self._checkpoint_error = None  # type: Optional[str]
        self._output_dir_metadata_synced = False
        self._next_checkpoint_time = time.monotonic()

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
        record = {
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
        with self._records_lock:
            self.control_records.append(record)
        self._start_checkpoint_if_due()

    def record_visual(
        self,
        *,
        timestamp: float,
        depth_input: Sequence[Sequence[float]],
        visual_output: Sequence[float],
    ) -> None:
        if not np.isfinite(float(timestamp)):
            raise ValueError("visual timestamp must be finite")
        depth = _finite_depth_input(depth_input)
        record = {
            "timestamp": float(timestamp),
            "depth_input": depth,
            "depth_stats": np.asarray(
                [depth.min(), depth.max(), depth.mean()],
                dtype=np.float32,
            ),
            "visual_output": _finite_vector(
                visual_output, 34, "visual output"
            ),
        }
        with self._records_lock:
            self.visual_records.append(record)

    @staticmethod
    def _build_payload(
        controls,
        visuals,
        reason: str,
        detail: str = "",
    ) -> Dict[str, np.ndarray]:
        payload = {
            "format_version": np.asarray([FORMAT_VERSION], dtype=np.int32),
            "reason": np.asarray([str(reason)]),
            "detail": np.asarray([str(detail)]),
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
        payload["depth_input"] = np.stack(
            [record["depth_input"] for record in visuals], axis=0
        ) if visuals else np.empty((0,) + DEPTH_INPUT_SHAPE, dtype=np.float32)
        payload["depth_stats"] = np.stack(
            [record["depth_stats"] for record in visuals], axis=0
        ) if visuals else np.empty((0, 3), dtype=np.float32)
        payload["visual_output"] = np.stack(
            [record["visual_output"] for record in visuals], axis=0
        ) if visuals else np.empty((0, 34), dtype=np.float32)
        return payload

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(str(directory), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._output_dir_metadata_synced:
            return
        directories = (self.output_dir.parent, self.output_dir.parent.parent)
        synced = set()
        for directory in directories:
            resolved = str(directory.resolve())
            if resolved not in synced:
                self._sync_directory(directory)
                synced.add(resolved)
        self._output_dir_metadata_synced = True

    def _write_payload_durable(
        self,
        path: Path,
        payload: Dict[str, np.ndarray],
    ) -> None:
        """Atomically replace an NPZ after syncing its data and directory."""
        with self._write_lock:
            self._ensure_output_dir()
            temporary_path = path.with_name(f".{path.name}.tmp")
            try:
                with open(str(temporary_path), "wb") as output:
                    np.savez_compressed(output, **payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(str(temporary_path), str(path))
                self._sync_directory(self.output_dir)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    def _snapshot(self):
        with self._records_lock:
            return list(self.control_records), list(self.visual_records)

    def _write_checkpoint(self, controls, visuals) -> None:
        try:
            payload = self._build_payload(
                controls,
                visuals,
                CHECKPOINT_REASON,
            )
            self._write_payload_durable(self.checkpoint_path, payload)
        except Exception as error:
            with self._records_lock:
                self._checkpoint_error = (
                    f"{type(error).__name__}: {error}"
                )
        finally:
            with self._records_lock:
                self._checkpoint_in_progress = False
                self._checkpoint_thread = None

    def _start_checkpoint_if_due(self) -> None:
        now = time.monotonic()
        with self._records_lock:
            if (
                now < self._next_checkpoint_time
                or self._checkpoint_in_progress
                or (not self.control_records and not self.visual_records)
            ):
                return
            controls = list(self.control_records)
            visuals = list(self.visual_records)
            self._checkpoint_in_progress = True
            self._next_checkpoint_time = now + self.checkpoint_interval_s
            thread = threading.Thread(
                target=self._write_checkpoint,
                args=(controls, visuals),
                name="flight-checkpoint",
                daemon=True,
            )
            self._checkpoint_thread = thread
        try:
            thread.start()
        except RuntimeError as error:
            with self._records_lock:
                self._checkpoint_in_progress = False
                self._checkpoint_thread = None
                self._checkpoint_error = (
                    f"{type(error).__name__}: {error}"
                )

    def _wait_for_pending_checkpoint(self) -> None:
        with self._records_lock:
            thread = self._checkpoint_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def checkpoint(self) -> Optional[Path]:
        """Synchronously persist the current ring buffer for diagnostics."""
        self._wait_for_pending_checkpoint()
        controls, visuals = self._snapshot()
        if not controls and not visuals:
            return None
        payload = self._build_payload(
            controls,
            visuals,
            CHECKPOINT_REASON,
        )
        self._write_payload_durable(self.checkpoint_path, payload)
        return self.checkpoint_path

    def pop_checkpoint_error(self) -> Optional[str]:
        """Return and clear the latest background persistence error."""
        with self._records_lock:
            error = self._checkpoint_error
            self._checkpoint_error = None
        return error

    def _restore_records(self, controls, visuals) -> None:
        with self._records_lock:
            current_controls = list(self.control_records)
            current_visuals = list(self.visual_records)
            self.control_records = deque(
                controls + current_controls,
                maxlen=self.control_capacity,
            )
            self.visual_records = deque(
                visuals + current_visuals,
                maxlen=self.visual_capacity,
            )

    def _remove_checkpoint_after_flush(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except FileNotFoundError:
            return
        self._sync_directory(self.output_dir)

    def flush(self, reason: str, detail: str = "") -> Optional[Path]:
        """Durably save buffered samples before clearing their ring buffers."""
        self._wait_for_pending_checkpoint()
        with self._records_lock:
            if not self.control_records and not self.visual_records:
                return None
            controls = list(self.control_records)
            visuals = list(self.visual_records)
            self.control_records = deque(maxlen=self.control_capacity)
            self.visual_records = deque(maxlen=self.visual_capacity)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        final_path = self.output_dir / f"extreme-flight-{timestamp}.npz"
        try:
            payload = self._build_payload(controls, visuals, reason, detail)
            self._write_payload_durable(final_path, payload)
        except Exception:
            self._restore_records(controls, visuals)
            raise

        try:
            self._remove_checkpoint_after_flush()
        except OSError as error:
            with self._records_lock:
                self._checkpoint_error = (
                    f"{type(error).__name__}: {error}"
                )
        return final_path
