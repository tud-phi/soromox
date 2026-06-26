#!/usr/bin/env python3
"""Offline video renderer for SoRoMoX trajectories."""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import colors

from soromox.rendering import BackboneColorConfig, Open3DRenderer, RendererColorConfig
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

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


def build_robot(
    segment_length, backbone_radius, tendon_routing_params
) -> TendonActuatedPCS:
    """Reconstruct the static robot parameters for the visualizer."""
    num_segments = len(segment_length)
    rho = 1070 * jnp.ones((num_segments,))

    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_length[:, None]
        ).flatten()
    )

    body_params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_length,
        radius=backbone_radius,
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=20e3 * jnp.ones((num_segments,)),
        shear_modulus=20e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    active_tendon_routing = LinearTendonRoutingParams(
        y_intercept=tendon_routing_params["tendon_y_intercept"],
        z_intercept=tendon_routing_params["tendon_z_intercept"],
        y_slope=tendon_routing_params["tendon_y_slope"],
        z_slope=tendon_routing_params["tendon_z_slope"],
        attachment_segment_index=tendon_routing_params[
            "tendon_attachment_segment_index"
        ],
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
        "--data", type=str, default=DATA_DIR / "robot_trajectory_data_withCBF.npz"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=SCRIPT_DIR / "outputs" / "clf_cbf_rollout_with_cbf.mp4",
    )
    parser.add_argument(
        "--visible", action="store_true", help="Force window visibility during render"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Could not find trajectory file: {args.data}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    ts = data["ts"]
    q_ts = data["q_ts"]
    goal_position = data["goal_position"]
    obs_centers = data["obs_centers"]
    obs_radii = data["obs_radii"]
    backbone_radius = data["backbone_radius"]
    # safety_margin = data["safety_margin"]
    segment_length = data["segment_length"]
    backbone_radius = data["backbone_radius"]
    tendon_routing_params = {
        k: data[k]
        for k in (
            "tendon_y_intercept",
            "tendon_z_intercept",
            "tendon_y_slope",
            "tendon_z_slope",
            "tendon_attachment_segment_index",
        )
    }

    # Format obstacle & target spheres
    obs_colors = np.tile(
        np.array([colors.to_rgb(COLORS["post_opt_1"])], dtype=np.float64),
        (obs_radii.shape[0], 1),
    )
    target_position = goal_position[np.newaxis, :]
    target_radius = np.array([0.025])  # obs_radii[0, np.newaxis]
    target_color = np.array([colors.to_rgb(COLORS["ground_truth"])], dtype=np.float64)

    static_spheres_positions = np.concatenate([obs_centers, target_position], axis=0)
    static_spheres_radii = np.concatenate([obs_radii, target_radius], axis=0)
    static_spheres_colors = np.concatenate([obs_colors, target_color], axis=0)

    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.array(
                [colors.to_rgb(COLORS["backbone_opt"])], dtype=np.float64
            )
        ),
        base_plate_color=(0.2, 0.2, 0.2),
    )

    print("Building robot geometry...")
    robot = build_robot(segment_length, backbone_radius, tendon_routing_params)

    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=color_config,
        width=1920,
        height=1080,
        backbone_style="discrete",
        actuator_line_width=2.0,
    )

    # Use the same frame stride as the original script
    render_stride = max(1, len(ts) // 300)

    print(f"Rendering MP4 to {args.output}...")
    renderer.render_sequence(
        ts=ts[::render_stride],
        q_ts=q_ts[::render_stride],
        playback_speed=1.0,
        loop=False,
        record_path=args.output,
        render_actuators=True,
        static_spheres_positions=static_spheres_positions,
        static_spheres_radii=static_spheres_radii,
        static_spheres_colors=static_spheres_colors,
    )


if __name__ == "__main__":
    main()
