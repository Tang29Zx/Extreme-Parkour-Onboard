"""Replay exported Extreme Parkour models in Isaac Gym without robot I/O."""

import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch.
import numpy as np
import torch
from torch import nn
from isaacgym import terrain_utils
from isaacgym.torch_utils import quat_apply

# Task registration imports training-only packages that replay never initializes.
if "wandb" not in sys.modules:
    wandb_module = types.ModuleType("wandb")
    wandb_module.__spec__ = ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = wandb_module
if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__spec__ = ModuleSpec("tqdm", loader=None)
    tqdm_module.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module
if "pydelatin" not in sys.modules:
    pydelatin_module = types.ModuleType("pydelatin")
    pydelatin_module.__spec__ = ModuleSpec("pydelatin", loader=None)

    class _UnavailableDelatin:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pydelatin is unavailable; use grid meshing")

    pydelatin_module.Delatin = _UnavailableDelatin
    sys.modules["pydelatin"] = pydelatin_module
if "pyfqmr" not in sys.modules:
    pyfqmr_module = types.ModuleType("pyfqmr")
    pyfqmr_module.__spec__ = ModuleSpec("pyfqmr", loader=None)

    class _UnavailableSimplify:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pyfqmr is unavailable; disable grid simplification")

    pyfqmr_module.Simplify = _UnavailableSimplify
    sys.modules["pyfqmr"] = pyfqmr_module

from legged_gym.envs import *  # noqa: F401,F403 -- registers Isaac Gym tasks.
from legged_gym.scripts.replay_geometry import (
    BOX_FRONT_M,
    BOX_HEIGHT_M,
    BOX_LENGTH_M,
    BOX_WIDTH_M,
    TRACK_CENTER_Y_M,
    single_box_world_bounds,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain
from rsl_rl.modules import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


FORWARD_COMMAND_MPS = 0.5
VISUAL_UPDATE_INTERVAL = 5


def install_single_box_terrain() -> None:
    """Limit this replay process to one exact 0.2 m box."""

    def make_single_box(self, choice, difficulty):
        del choice, difficulty
        terrain = terrain_utils.SubTerrain(
            "single_box_replay",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        x0 = round(BOX_FRONT_M / terrain.horizontal_scale)
        x1 = round((BOX_FRONT_M + BOX_LENGTH_M) / terrain.horizontal_scale)
        y0 = round(
            (TRACK_CENTER_Y_M - BOX_WIDTH_M / 2.0) / terrain.horizontal_scale
        )
        y1 = round(
            (TRACK_CENTER_Y_M + BOX_WIDTH_M / 2.0) / terrain.horizontal_scale
        )
        height = round(BOX_HEIGHT_M / terrain.vertical_scale)
        terrain.height_field_raw[x0:x1, y0:y1] = height
        terrain.goals = np.asarray(
            [
                [BOX_FRONT_M - 0.3, TRACK_CENTER_Y_M],
                [BOX_FRONT_M + BOX_LENGTH_M / 2.0, TRACK_CENTER_Y_M],
                [BOX_FRONT_M + BOX_LENGTH_M + 0.5, TRACK_CENTER_Y_M],
            ],
            dtype=np.float64,
        )
        terrain.idx = 17
        return terrain

    Terrain.make_terrain = make_single_box


def render_single_box_depth(env) -> torch.Tensor:
    """Raycast the flat ground and replay box into a D435-like depth image."""
    device = env.root_states.device
    height, width = 58, 87
    horizontal_fov = np.deg2rad(88.0)
    vertical_fov = 2.0 * np.arctan(
        np.tan(horizontal_fov / 2.0) * height / width
    )
    column_angle = (
        0.5
        - (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
        / width
    ) * horizontal_fov
    row_angle = (
        0.5
        - (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
        / height
    ) * vertical_fov
    try:
        row_grid, column_grid = torch.meshgrid(
            row_angle,
            column_angle,
            indexing="ij",
        )
    except TypeError:
        # Torch 1.9 always uses matrix (ij) indexing and has no keyword.
        row_grid, column_grid = torch.meshgrid(row_angle, column_angle)

    rays = torch.stack(
        (
            torch.ones_like(row_grid),
            torch.tan(column_grid),
            torch.tan(row_grid),
        ),
        dim=-1,
    ).reshape(-1, 3)
    camera_pitch = np.deg2rad(22.5)
    cosine = float(np.cos(camera_pitch))
    sine = float(np.sin(camera_pitch))
    pitched_x = cosine * rays[:, 0] + sine * rays[:, 2]
    pitched_z = -sine * rays[:, 0] + cosine * rays[:, 2]
    rays = torch.stack((pitched_x, rays[:, 1], pitched_z), dim=-1)

    base_quaternion = env.root_states[:, 3:7]
    camera_offset = torch.tensor(
        [0.355, 0.0, 0.065],
        dtype=torch.float32,
        device=device,
    ).repeat(env.num_envs, 1)
    camera_origin = env.root_states[:, :3] + quat_apply(
        base_quaternion,
        camera_offset,
    )
    ray_count = rays.shape[0]
    ray_world = quat_apply(
        base_quaternion[:, None, :].expand(-1, ray_count, -1).reshape(-1, 4),
        rays[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3),
    ).reshape(env.num_envs, ray_count, 3)
    origin = camera_origin[:, None, :]

    far = torch.full(
        (env.num_envs, ray_count),
        2.0,
        dtype=torch.float32,
        device=device,
    )
    ground_depth = torch.where(
        ray_world[:, :, 2] < -1e-6,
        -origin[:, :, 2] / ray_world[:, :, 2],
        far,
    )
    ground_depth = torch.where(ground_depth >= 0.0, ground_depth, far)

    box_min, box_max = single_box_world_bounds(env.env_origins)
    safe_direction = torch.where(
        torch.abs(ray_world) < 1e-6,
        torch.full_like(ray_world, 1e-6),
        ray_world,
    )
    first = (box_min - origin) / safe_direction
    second = (box_max - origin) / safe_direction
    near = torch.minimum(first, second).amax(dim=-1)
    distant = torch.maximum(first, second).amin(dim=-1)
    box_hit = (distant >= torch.maximum(near, torch.zeros_like(near))) & (
        near >= 0.0
    )
    box_depth = torch.where(box_hit, near, far)

    metric_depth = torch.minimum(torch.minimum(ground_depth, box_depth), far)
    metric_depth = torch.clamp(metric_depth, 0.0, 2.0)
    return metric_depth.reshape(env.num_envs, height, width) / 2.0 - 0.5


def configure_replay_environment(
    env_cfg,
    num_envs: int = 1,
    num_terrain_columns: int = 1,
) -> None:
    """Create one deterministic flat parkour lane for exported-policy review."""
    resources_dir = Path(
        os.environ.get("PARKOUR_RESOURCES_DIR", "/workspace/parkour-resources")
    )
    go2_urdf = resources_dir / "robots/go2/urdf/go2.urdf"
    if not go2_urdf.is_file():
        raise FileNotFoundError(f"Go2 URDF was not found at {go2_urdf}")
    env_cfg.asset.file = str(go2_urdf)

    env_cfg.env.num_envs = int(num_envs)
    env_cfg.env.episode_length_s = 60
    env_cfg.env.randomize_start_pos = False
    env_cfg.env.randomize_start_y = False
    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.randomize_start_pitch = False
    env_cfg.env.randomize_start_vel = False
    env_cfg.env.dof_pos_reset_range = [0.0, 0.0]

    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 60
    for ranges_name in ("ranges", "max_ranges"):
        ranges = getattr(env_cfg.commands, ranges_name)
        ranges.lin_vel_x = [FORWARD_COMMAND_MPS, FORWARD_COMMAND_MPS]
        ranges.lin_vel_y = [0.0, 0.0]
        ranges.ang_vel_yaw = [0.0, 0.0]
        ranges.heading = [0.0, 0.0]

    env_cfg.depth.use_camera = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.action_delay_view = 0

    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = int(num_terrain_columns)
    env_cfg.terrain.num_goals = 3
    env_cfg.terrain.height = [0.0, 0.0]
    env_cfg.terrain.horizontal_scale = 0.1
    env_cfg.terrain.hf2mesh_method = "grid"
    env_cfg.terrain.simplify_grid = False
    env_cfg.terrain.no_flat = False
    env_cfg.terrain.terrain_dict = {
        name: 0.0 for name in env_cfg.terrain.terrain_dict
    }
    env_cfg.terrain.terrain_dict["parkour_flat"] = 1.0
    env_cfg.terrain.terrain_proportions = list(
        env_cfg.terrain.terrain_dict.values()
    )


def load_exported_models(traced_dir: Path, device: torch.device):
    base_model = torch.jit.load(
        str(traced_dir / "base_jit.pt"),
        map_location=device,
    )
    base_model.eval()

    try:
        vision_state = torch.load(
            str(traced_dir / "vision_weight.pt"),
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # The Jetson-compatible Python 3.8 environment uses Torch 1.9, which
        # predates weights_only. The file is a trusted local traced asset.
        vision_state = torch.load(
            str(traced_dir / "vision_weight.pt"),
            map_location=device,
        )
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_state["depth_encoder_state_dict"])
    depth_encoder.eval()
    return base_model, depth_encoder


def fixed_command_observation(observation: torch.Tensor) -> torch.Tensor:
    result = observation.clone()
    result[:, 6:8] = 0.0
    result[:, 8:10] = 0.0
    result[:, 10] = FORWARD_COMMAND_MPS
    return result


@torch.inference_mode()
def replay(args) -> None:
    traced_dir = Path(
        os.environ.get(
            "EXTREME_TRACED_DIR",
            Path(__file__).resolve().parents[2] / "traced",
        )
    ).expanduser().resolve()
    required = ("base_jit.pt", "vision_weight.pt", "config.json")
    missing = [name for name in required if not (traced_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing traced model files in {traced_dir}: {', '.join(missing)}"
        )

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    configure_replay_environment(env_cfg)
    install_single_box_terrain()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    device = torch.device(env.device)

    base_model, depth_encoder = load_exported_models(traced_dir, device)
    estimator = base_model.estimator.estimator
    history_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    activation = nn.ELU()

    observation = env.get_observations()
    visual_output = None
    max_steps = int(os.environ.get("EXTREME_REPLAY_STEPS", "0"))

    print(f"Loaded exported models from: {traced_dir}")
    print("Replay command: 0.5 m/s forward")
    print("Scene: one 1.2 m x 1.2 m x 0.20 m box, 1.0 m ahead of spawn")
    print("Depth source: CPU raycast of the flat ground and single box")
    if not args.headless:
        print("Isaac Gym viewer is running. Press Esc in the viewer to exit.")

    step = 0
    while max_steps <= 0 or step < max_steps:
        env.commands.zero_()
        env.commands[:, 0] = FORWARD_COMMAND_MPS

        proprio = fixed_command_observation(
            observation[:, : env.cfg.env.n_proprio]
        )
        history = observation[
            :, -env.cfg.env.history_len * env.cfg.env.n_proprio :
        ].view(-1, env.cfg.env.history_len, env.cfg.env.n_proprio).clone()
        history[:, :, 6:10] = 0.0
        history[:, :, 10] = FORWARD_COMMAND_MPS

        if visual_output is None or step % VISUAL_UPDATE_INTERVAL == 0:
            depth_image = render_single_box_depth(env)
            visual_output = depth_encoder(depth_image, proprio)
            if not torch.isfinite(visual_output).all():
                raise RuntimeError("vision encoder produced NaN or Inf")

        proprio[:, 6:8] = visual_output[:, -2:] * 1.5
        depth_latent = visual_output[:, :-2]
        linear_velocity_latent = estimator(proprio)
        history_latent = history_encoder(activation, history)
        actor_observation = torch.cat(
            (
                proprio,
                depth_latent,
                linear_velocity_latent,
                history_latent,
            ),
            dim=-1,
        )
        action = actor(actor_observation)
        if action.shape != (env.num_envs, 12) or not torch.isfinite(action).all():
            raise RuntimeError("actor produced an invalid action")

        observation, _, _, done, _ = env.step(action)
        if bool(done.any()):
            depth_encoder.hidden_states = None

        step += 1
        if step % 250 == 0:
            position = env.root_states[0, :3].cpu().numpy()
            print(
                "step={} position=({:.2f}, {:.2f}, {:.2f}) "
                "max_abs_action={:.3f}".format(
                    step,
                    position[0],
                    position[1],
                    position[2],
                    float(torch.max(torch.abs(action))),
                )
            )


if __name__ == "__main__":
    replay(get_args())
