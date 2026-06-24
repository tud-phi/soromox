#!/usr/bin/env python3
"""Offline video renderer for RL rollout trajectories."""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import colors

from soromox.rendering import Open3DRenderer, RendererColorConfig
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import BackboneColorConfig
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

COLORS = {
    "pre_opt_1": "#006BA6",
    "pre_opt_2": "#0496FF",
    "post_opt_1": "#D81159",
    "post_opt_2": "#8F2D56",
    "post_opt_3": "#FFBC42",
    "backbone_opt": "#7B2CBF",
    "backbone_init": "#0ead69",
    "ground_truth": "#FFBC42",
    "x_t": "#2a9d8f",
    "y_t": "#e9c46a",
    "z_t": "#e76f51",
}

jax.config.update("jax_enable_x64", True)


def build_rl_robot(
    arm_length: float = 0.25, arm_radius: float = 0.025
) -> TendonActuatedPCS:
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

    active_tendon_routing = LinearTendonRoutingParams(
        y_intercept=tendon_routing_params["ry"],
        z_intercept=tendon_routing_params["rz"],
        y_slope=tendon_routing_params["my"],
        z_slope=tendon_routing_params["mz"],
        attachment_segment_index=tendon_routing_params["idx_seg_att"],
    )

    return TendonActuatedPCS(
        params=TendonActuatedPCSParams(
            body=body_params,
            active_tendon_routing=active_tendon_routing,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        # required=True,
        default=DATA_DIR / "traj" / "open3d_rollout_data_random.npz",
        help="Path to the .npz file containing rollout arrays",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR / "rl_rollout.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--ball-radius", type=float, default=0.025, help="Radius of the target ball"
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        default=False,
        help="Plot the target ball's trajectory as a trail of small red spheres",
    )
    args = parser.parse_args()

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
    dynamic_spheres_radii = np.array([args.ball_radius], dtype=np.float64)
    dynamics_spheres_colors = np.array(
        [colors.to_rgb(COLORS["ground_truth"])], dtype=np.float64
    )  # [1.0, 170 / 255, 0.0]

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
            np.array([[0.8, 0.1, 0.1]], dtype=np.float64), (num_pts, 1)
        )

    # 3. Build Robot Geometry
    print("Building robot geometry...")
    robot = build_rl_robot()

    # 4. Configure Color Styling
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.array(
                [colors.to_rgb(COLORS["backbone_init"])], dtype=np.float64
            )  # [0.0039, 0.6510, 0.8392, 1.0]
        ),
        base_plate_color=(0.2, 0.2, 0.2),
    )

    renderer = Open3DRenderer(
        robot,
        num_points=40,
        color_config=color_config,
        width=1920,
        height=1920,
        backbone_style="discrete",
        background_color=(1.0, 1.0, 1.0),
        base_plate_radius_scale=1.5,
        base_plate_thickness=0.04,
    )

    # Calculate exact FPS based on timestamps
    dt_seq = np.diff(ts)
    fps = 1.0 / max(1e-6, float(np.median(dt_seq)))

    print(f"Rendering MP4 to {args.output} (Target FPS: {fps:.2f})...")

    # Define a custom camera config to zoom in
    # Lower distance_factor = closer camera (Default is usually around 10.0)
    my_camera_config = CameraConfig(
        distance_factor=6.0,
        fov=60.0,  # Optional: Lower FOV also creates a zoom effect
        position_offset=(0.0, -1.0, 0.3),
        up=(0.0, 0.0, 1.0),
    )

    # 5. Render to FFmpeg
    renderer.render_sequence(
        ts=ts,
        q_ts=q_ts,
        playback_speed=1.0,
        autoplay=True,
        loop=False,
        record_path=args.output,
        render_actuators=True,
        static_spheres_positions=static_spheres_positions,
        static_spheres_radii=static_spheres_radii,
        static_spheres_colors=static_spheres_colors,
        dynamic_spheres_positions=dynamic_spheres_positions,
        dynamic_spheres_radii=dynamic_spheres_radii,
        dynamics_spheres_colors=dynamics_spheres_colors,
        camera_config=my_camera_config,
        window_name="SoRoMoX RL Rollout Offline",
    )


if __name__ == "__main__":
    main()
