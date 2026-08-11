"""Shared visual style for Section Vf reinforcement-learning rollouts."""

from __future__ import annotations

import numpy as np
from matplotlib import colors

from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import BackboneColorConfig, RendererColorConfig

COLORS = {
    "pre_opt_1": "#006BA6",
    "pre_opt_2": "#0496FF",
    "post_opt_1": "#f1552e",
    "post_opt_2": "#D81159",
    "post_opt_3": "#8F2D56",
    "obstacle": "#7B2CBF",
    "target": "#0ead69",
    "ground_truth": "#FFBC42",
    "x_t": "#2a9d8f",
    "y_t": "#e9c46a",
    "z_t": "#D81159",
}

RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
BACKBONE_NUM_POINTS = 80
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
TARGET_SPHERE_RADIUS = 0.015
TARGET_TRAIL_RADIUS = 0.003
TARGET_TRAIL_STRIDE = 2

TARGET_COLOR = tuple(colors.to_rgb(COLORS["target"]))
_target_cmap = colors.LinearSegmentedColormap.from_list(
    "secVf_target_gradient", ["#FFFFFF", COLORS["target"]]
)
TARGET_TRAIL_COLOR = tuple(_target_cmap(np.array([0.65]))[0, :3])


def make_rl_camera_config(
    arm_length: float,
    *,
    fov: float = 60.0,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CameraConfig:
    """Return the camera used for the Section Vf paper renderings."""
    radius = 1.7
    angle = np.pi * 1.15
    x = float(radius * np.cos(angle))
    y = float(radius * np.sin(angle))
    return CameraConfig(
        position=(x, y, float(2.0 * arm_length)),
        look_at=(0.0, 0.0, float(0.35 * arm_length)),
        up=up,
        fov=fov,
    )


def make_rl_color_config(color_label: str = "post_opt_1") -> RendererColorConfig:
    """Return the shared arm, actuator, base, and ground-plane colors."""
    if color_label not in COLORS:
        raise KeyError(f"Unknown Section Vf color label: {color_label!r}")
    return RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.array(
                [colors.to_rgb(COLORS[color_label])], dtype=np.float64
            )
        ),
        base_plate_color=(0.2, 0.2, 0.2),
    )


def make_target_trail_spheres(
    ball_positions: np.ndarray,
    *,
    stride: int = TARGET_TRAIL_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one or more target trajectories into the shared dotted trail."""
    positions = np.asarray(ball_positions, dtype=np.float64)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if positions.ndim == 2 and positions.shape[-1] == 3:
        trail_points = positions[::stride]
    elif positions.ndim == 3 and positions.shape[-1] == 3:
        trail_points = positions[:, ::stride, :].reshape(-1, 3)
    else:
        raise ValueError(
            f"ball_positions must have shape (T, 3) or (N, T, 3); got {positions.shape}"
        )

    num_points = trail_points.shape[0]
    radii = np.full((num_points,), TARGET_TRAIL_RADIUS, dtype=np.float64)
    trail_colors = np.tile(
        np.asarray(TARGET_TRAIL_COLOR, dtype=np.float64)[None, :],
        (num_points, 1),
    )
    return trail_points, radii, trail_colors
