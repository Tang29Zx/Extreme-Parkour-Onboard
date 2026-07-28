"""Dependency-light geometry shared by replay code and unit tests."""

import math

import numpy as np
import torch


BOX_FRONT_M = 2.0
BOX_LENGTH_M = 1.2
BOX_WIDTH_M = 1.2
BOX_HEIGHT_M = 0.2
BOX_PASS_MARGIN_M = 0.2
TRACK_CENTER_Y_M = 2.0
TERRAIN_SPAWN_X_M = 1.0


def deterministic_ground_noise(
    shape,
    vertical_scale: float,
    horizontal_scale: float,
    amplitude_m: float,
    patch_size_m: float,
    seed: int,
) -> np.ndarray:
    """Return a reproducible blockwise height field in terrain sample units."""
    rows, columns = (int(shape[0]), int(shape[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("ground noise shape must be positive")
    vertical = float(vertical_scale)
    horizontal = float(horizontal_scale)
    amplitude = float(amplitude_m)
    patch_size = float(patch_size_m)
    if not all(
        math.isfinite(value)
        for value in (vertical, horizontal, amplitude, patch_size)
    ):
        raise ValueError("ground noise parameters must be finite")
    if vertical <= 0.0 or horizontal <= 0.0 or patch_size <= 0.0:
        raise ValueError("ground noise scales must be positive")
    if amplitude < 0.0:
        raise ValueError("ground noise amplitude must be non-negative")
    levels = int(math.floor(amplitude / vertical + 1e-12))
    if levels == 0:
        return np.zeros((rows, columns), dtype=np.int16)
    patch_pixels = max(1, int(round(patch_size / horizontal)))
    coarse_rows = int(math.ceil(rows / patch_pixels))
    coarse_columns = int(math.ceil(columns / patch_pixels))
    generator = np.random.RandomState(int(seed))
    coarse = generator.randint(
        -levels,
        levels + 1,
        size=(coarse_rows, coarse_columns),
    )
    expanded = np.repeat(
        np.repeat(coarse, patch_pixels, axis=0),
        patch_pixels,
        axis=1,
    )
    return expanded[:rows, :columns].astype(np.int16, copy=False)


def single_box_world_bounds(env_origins: torch.Tensor):
    """Return the replay box bounds in each selected terrain's world frame."""
    box_front_x = env_origins[:, 0] + (BOX_FRONT_M - TERRAIN_SPAWN_X_M)
    box_center_y = env_origins[:, 1]
    zeros = torch.zeros_like(box_front_x)
    box_min = torch.stack(
        (
            box_front_x,
            box_center_y - BOX_WIDTH_M / 2.0,
            zeros,
        ),
        dim=-1,
    )[:, None, :]
    box_max = torch.stack(
        (
            box_front_x + BOX_LENGTH_M,
            box_center_y + BOX_WIDTH_M / 2.0,
            zeros + BOX_HEIGHT_M,
        ),
        dim=-1,
    )[:, None, :]
    return box_min, box_max


def single_box_rear_x(env_origins: torch.Tensor) -> torch.Tensor:
    """Return the world-x threshold used to declare a box traversal."""
    _, box_max = single_box_world_bounds(env_origins)
    return box_max[:, 0, 0] + BOX_PASS_MARGIN_M
