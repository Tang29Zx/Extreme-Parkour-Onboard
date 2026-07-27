"""Tensor helpers for rebuilding policy context before real engagement."""

from typing import Callable

import torch


def reset_policy_context(
    actions: torch.Tensor,
    proprio_history: torch.Tensor,
    episode_length: torch.Tensor,
    reset_depth_hidden: Callable[[], None],
) -> None:
    """Clear temporal policy state and reset the visual recurrent state once."""
    if not callable(reset_depth_hidden):
        raise TypeError("reset_depth_hidden must be callable")
    actions.zero_()
    proprio_history.zero_()
    episode_length.zero_()
    reset_depth_hidden()


def update_proprio_history(
    history: torch.Tensor,
    proprio: torch.Tensor,
    episode_length: torch.Tensor,
) -> torch.Tensor:
    """Append proprioception, replacing every stale frame at episode start."""
    if history.ndim != 3 or proprio.ndim != 2:
        raise ValueError("history and proprioception dimensions are invalid")
    if history.shape[0] != proprio.shape[0] or history.shape[2] != proprio.shape[1]:
        raise ValueError("history and proprioception shapes do not match")
    if tuple(episode_length.shape) != (history.shape[0],):
        raise ValueError("episode length shape does not match history batch")
    repeated = proprio.unsqueeze(1).repeat(1, history.shape[1], 1)
    shifted = torch.cat((history[:, 1:], proprio.unsqueeze(1)), dim=1)
    return torch.where(
        (episode_length <= 1)[:, None, None],
        repeated,
        shifted,
    )
