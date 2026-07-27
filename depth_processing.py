"""Training-equivalent D435i depth preprocessing for onboard inference."""

from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np
import torch
import torch.nn.functional as functional


DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
RAW_DEPTH_HEIGHT = 60
RAW_DEPTH_WIDTH = 106
REALSENSE_DEPTH_HEIGHT = 240
REALSENSE_DEPTH_WIDTH = 424
REALSENSE_DOWNSAMPLE_FACTOR = 4


class DepthProcessingError(ValueError):
    """Raised when depth data violates the exported model contract."""


@dataclass(frozen=True)
class DepthProcessingConfig:
    """Deterministic inference subset of the exported depth configuration."""

    original_width: int = RAW_DEPTH_WIDTH
    original_height: int = RAW_DEPTH_HEIGHT
    output_width: int = DEPTH_WIDTH
    output_height: int = DEPTH_HEIGHT
    near_clip: float = 0.0
    far_clip: float = 2.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DepthProcessingConfig":
        depth = config.get("depth", config)
        try:
            original_width, original_height = depth["original"]
            output_width, output_height = depth["resized"]
            return cls(
                original_width=int(original_width),
                original_height=int(original_height),
                output_width=int(output_width),
                output_height=int(output_height),
                near_clip=float(depth["near_clip"]),
                far_clip=float(depth["far_clip"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DepthProcessingError("Invalid depth configuration.") from error

    def __post_init__(self) -> None:
        if (self.original_width, self.original_height) != (106, 60):
            raise DepthProcessingError("Only 106x60 source depth is supported.")
        if (self.output_width, self.output_height) != (87, 58):
            raise DepthProcessingError("Only 87x58 model depth is supported.")
        if (self.near_clip, self.far_clip) != (0.0, 2.0):
            raise DepthProcessingError("Only the exported 0-to-2-meter range is supported.")


def prepare_realsense_depth(
    raw_depth: np.ndarray,
    depth_scale: float,
    config: DepthProcessingConfig = DepthProcessingConfig(),
) -> np.ndarray:
    """Convert upright D435i Z16 depth into metric ``(60, 106)`` depth."""
    value = np.asarray(raw_depth)
    expected_shape = (REALSENSE_DEPTH_HEIGHT, REALSENSE_DEPTH_WIDTH)
    if value.shape != expected_shape:
        raise DepthProcessingError(
            f"RealSense depth must have shape {expected_shape}, got {value.shape}."
        )
    if value.dtype != np.uint16:
        raise DepthProcessingError(
            f"RealSense Z16 depth must use uint16, got {value.dtype}."
        )
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise DepthProcessingError(
            "RealSense depth scale must be positive and finite."
        )

    depth_m = value.astype(np.float32) * np.float32(depth_scale)
    invalid = (
        (value == np.uint16(0))
        | (value == np.iinfo(np.uint16).max)
        | ~np.isfinite(depth_m)
        | (depth_m <= 0.0)
        | (depth_m > np.float32(config.far_clip))
    )
    depth_m = np.where(invalid, np.float32(config.far_clip), depth_m)
    depth_m = np.clip(
        depth_m,
        np.float32(config.near_clip),
        np.float32(config.far_clip),
    )
    depth_m = depth_m.reshape(
        config.original_height,
        REALSENSE_DOWNSAMPLE_FACTOR,
        config.original_width,
        REALSENSE_DOWNSAMPLE_FACTOR,
    ).mean(axis=(1, 3), dtype=np.float32)
    return np.ascontiguousarray(depth_m, dtype=np.float32)


def preprocess_depth(
    depth_m: Union[np.ndarray, torch.Tensor],
    config: DepthProcessingConfig = DepthProcessingConfig(),
    *,
    device: Union[str, torch.device, None] = None,
) -> torch.Tensor:
    """Convert positive metric source depth into normalized ``(B, 58, 87)``."""
    depth = torch.as_tensor(depth_m, dtype=torch.float32, device=device)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    expected_tail = (config.original_height, config.original_width)
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != expected_tail:
        raise DepthProcessingError(
            "depth must have shape (60, 106) or (B, 60, 106), "
            f"got {tuple(depth.shape)}."
        )

    far = torch.full_like(depth, config.far_clip)
    invalid = ~torch.isfinite(depth) | (depth <= 0.0)
    depth = torch.where(invalid, far, depth)
    depth = torch.clamp(depth, config.near_clip, config.far_clip)

    # Match LeggedRobot.crop_depth_image and torchvision bicubic Resize.
    depth = depth[..., :-2, 4:-4]
    depth = functional.interpolate(
        depth.unsqueeze(1),
        size=(config.output_height, config.output_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
    depth = (depth - config.near_clip) / (
        config.far_clip - config.near_clip
    ) - 0.5
    if not torch.isfinite(depth).all():
        raise DepthProcessingError("Processed depth contains NaN or Inf.")
    return depth
