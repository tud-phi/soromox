#!/usr/bin/env python3
"""Offline video renderer for RL rollout trajectories."""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering import Open3DRenderer
from soromox.systems import (
    PCS,
    PCSParams,
)

if __package__:
    from .rl_render_style import (
        BACKBONE_NUM_POINTS,
        BACKGROUND_COLOR,
        RENDER_HEIGHT,
        RENDER_WIDTH,
        TARGET_COLOR,
        TARGET_SPHERE_RADIUS,
        make_rl_camera_config,
        make_rl_color_config,
        make_target_trail_spheres,
    )
else:
    from rl_render_style import (
        BACKBONE_NUM_POINTS,
        BACKGROUND_COLOR,
        RENDER_HEIGHT,
        RENDER_WIDTH,
        TARGET_COLOR,
        TARGET_SPHERE_RADIUS,
        make_rl_camera_config,
        make_rl_color_config,
        make_target_trail_spheres,
    )

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

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
        help="Plot the target ball's trajectory as a dotted gradient trail",
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
    dynamic_spheres_radii = np.array([TARGET_SPHERE_RADIUS], dtype=np.float64)
    dynamics_spheres_colors = np.array([TARGET_COLOR], dtype=np.float64)

    static_spheres_positions = None
    static_spheres_radii = None
    static_spheres_colors = None

    if args.show_trajectory:
        (
            static_spheres_positions,
            static_spheres_radii,
            static_spheres_colors,
        ) = make_target_trail_spheres(ball_ts)

    # 3. Build Robot Geometry
    print("Building robot geometry...")
    robot = build_rl_robot()

    camera_config = make_rl_camera_config(float(np.sum(robot.L)))

    if "random" in str(args.data):
        color_label = "pre_opt_1"
    else:
        color_label = "post_opt_1"
    color_config = make_rl_color_config(color_label)

    renderer = Open3DRenderer(
        robot,
        num_points=BACKBONE_NUM_POINTS,
        color_config=color_config,
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        backbone_style="discrete",
        background_color=BACKGROUND_COLOR,
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
