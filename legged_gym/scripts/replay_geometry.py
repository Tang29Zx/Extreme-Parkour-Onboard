"""Dependency-light geometry shared by replay code and unit tests."""

import torch


BOX_FRONT_M = 2.0
BOX_LENGTH_M = 1.2
BOX_WIDTH_M = 1.2
BOX_HEIGHT_M = 0.2
BOX_PASS_MARGIN_M = 0.2
TRACK_CENTER_Y_M = 2.0
TERRAIN_SPAWN_X_M = 1.0


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
