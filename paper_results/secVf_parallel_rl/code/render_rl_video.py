#!/usr/bin/env python3
"""Offline video renderer for RL rollout trajectories."""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import colors

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering import CameraConfig, Open3DRenderer, RendererColorConfig
from soromox.rendering.color_config import BackboneColorConfig
from soromox.systems import (
    PCS,
    PCSParams,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

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
cmap_target = colors.LinearSegmentedColormap.from_list(
    "grad_target", ["#FFFFFF", COLORS["target"]]
)
gradient = cmap_target(np.array([0.65, 1.0]))[:, :3]

jax.config.update("jax_enable_x64", True)


def build_rl_robot(arm_length: float = 0.25, arm_radius: float = 0.025) -> PCS:
    """Reconstruct the static robot parameters to match the 1-segment RL environment."""
    num_segments = 1
    rho = 1070 * jnp.ones((num_segments,))
    segment_length = jnp.array([arm_length])
    backbone_radius = jnp.array([arm_radius])

    # Standard internal damping proxy
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]) * segment_length[:, None]
        ).flatten()
    )

    # 4-tendon routing mapped at 90-degree intervals to match the 4D action space
    r0 = float(backbone_radius[0]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([0, 90, 180, 270]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)

    tendon_routing_params = {
        "ry": ry0,
        "rz": rz0,
        "my": jnp.zeros(4),
        "mz": jnp.zeros(4),
        "idx_seg_att": jnp.zeros(4, dtype=jnp.int32),
    }

    body_params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_length,
        radius=backbone_radius,
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=20e3 * jnp.ones((num_segments,)),
        shear_modulus=20e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )

    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.stack(
            (
                jnp.zeros_like(tendon_routing_params["ry"]),
                tendon_routing_params["ry"],
                tendon_routing_params["rz"],
            ),
            axis=-1,
        ),
        slope=jnp.stack(
            (
                jnp.zeros_like(tendon_routing_params["my"]),
                tendon_routing_params["my"],
                tendon_routing_params["mz"],
            ),
            axis=-1,
        ),
        start_segment_index=0,
        end_segment_index=tuple(
            int(index) for index in tendon_routing_params["idx_seg_att"]
        ),
    )

    return PCS(
        params=body_params,
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for offline rollout rendering."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_DIR / "traj" / "open3d_rollout_data.npz",
        help="Path to the .npz file containing rollout arrays",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR / "rl_rollout.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--show-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the target ball's trajectory as a trail of small red spheres",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Could not find trajectory file: {args.data}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    ts = data["ts"]
    q_ts = data["q_ts"]
    ball_ts = data["ball_ts"]

    # 2. Format the Dynamic Sphere (Target Ball)
    # Open3DRenderer expects dynamic sphere trajectories to be of shape (K_balls, T, 3)
    dynamic_spheres_positions = ball_ts.reshape(1, -1, 3)
    dynamic_spheres_radii = np.array([0.015], dtype=np.float64)
    dynamics_spheres_colors = np.array(
        [colors.to_rgb(COLORS["target"])], dtype=np.float64
    )

    # --- ADD THIS NEW BLOCK: 2.5 Format the Trajectory Trail (Static Spheres) ---
    static_spheres_positions = None
    static_spheres_radii = None
    static_spheres_colors = None

    if args.show_trajectory:
        trail_radius = 0.003
        trail_pts = ball_ts[::2]
        num_pts = trail_pts.shape[0]

        static_spheres_positions = trail_pts
        static_spheres_radii = np.full((num_pts,), trail_radius, dtype=np.float64)
        static_spheres_colors = np.tile(
            np.array([gradient[0]], dtype=np.float64), (num_pts, 1)
        )

    # 3. Build Robot Geometry
    print("Building robot geometry...")
    robot = build_rl_robot()

    # Manual camera settings
    radius = 1.7
    angle = np.pi * 1.15
    x, y = float(radius * np.cos(angle)), float(radius * np.sin(angle))
    camera_config = CameraConfig(
        position=(x, y, float(2 * np.sum(robot.L))),
        look_at=(0.0, 0.0, float(0.7 * np.sum(robot.L) / 2)),
        fov=60.0,
    )

    if "random" in str(args.data):
        color_label = "pre_opt_1"
    else:
        color_label = "post_opt_1"
    # 4. Configure Color Styling
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.array(
                [colors.to_rgb(COLORS[color_label])], dtype=np.float64
            )
        ),
        base_plate_color=(0.2, 0.2, 0.2),
    )

    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=color_config,
        width=1920,
        height=1080,
        backbone_style="discrete",
        background_color=(1.0, 1.0, 1.0),
        # base_plate_radius_scale=1.5,
        # base_plate_thickness=0.04,
    )

    # Calculate exact FPS based on timestamps
    dt_seq = np.diff(ts)
    fps = 1.0 / max(1e-6, float(np.median(dt_seq)))

    print(f"Rendering MP4 to {args.output} (Target FPS: {fps:.2f})...")

    # 5. Render to FFmpeg
    renderer.render_sequence(
        ts=ts,
        q_ts=q_ts,
        playback_speed=1.0,
        autoplay=True,
        loop=False,
        record_path=args.output,
        close_when_recording_done=True,
        render_actuators=True,
        camera_config=camera_config,
        static_spheres_positions=static_spheres_positions,
        static_spheres_radii=static_spheres_radii,
        static_spheres_colors=static_spheres_colors,
        dynamic_spheres_positions=dynamic_spheres_positions,
        dynamic_spheres_radii=dynamic_spheres_radii,
        dynamics_spheres_colors=dynamics_spheres_colors,
        window_name="SoRoMoX RL Rollout Offline",
    )


if __name__ == "__main__":
    main()
