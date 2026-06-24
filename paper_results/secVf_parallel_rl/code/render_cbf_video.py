#!/usr/bin/env python3
"""Offline video renderer for SoRoMoX trajectories."""

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from soromox.rendering import Open3DRenderer, RendererColorConfig
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def build_robot() -> TendonActuatedPCS:
    """Reconstruct the static robot parameters for the visualizer."""
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    segment_length = 15e-2 * 2 / num_segments * jnp.ones((num_segments,))
    backbone_radius = 3.6e-2 * jnp.ones((num_segments,))

    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_length[:, None]
        ).flatten()
    )

    r0, r1 = float(backbone_radius[0]) - 0.005, float(backbone_radius[1]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([120, 240, 0]))
    theta1 = jnp.deg2rad(jnp.array([180, 300, 60]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    ry1, rz1 = r1 * jnp.cos(theta1), r1 * jnp.sin(theta1)

    tendon_routing_params = {
        "ry": jnp.concatenate([ry0, ry1]),
        "rz": jnp.concatenate([rz0, rz1]),
        "my": jnp.zeros(6),
        "mz": jnp.zeros(6),
        "idx_seg_att": jnp.array([0, 0, 0, 1, 1, 1]),
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
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
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
        "--data", type=str, default=SCRIPT_DIR / "data" / "clf_cbf_rollout_with_cbf.npz"
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
    obs_centers = data["obs_centers"]
    obs_radii = data["obs_radii"]
    target_center = data["target_center"].reshape(1, 3)

    # Format obstacle & target spheres
    obs_colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (obs_radii.shape[0], 1))
    target_radius = np.array([0.01])
    target_color = np.array([[1.0, 0.1, 0.1]])

    static_spheres_positions = np.concatenate([obs_centers, target_center], axis=0)
    static_spheres_radii = np.concatenate([obs_radii, target_radius], axis=0)
    static_spheres_colors = np.concatenate([obs_colors, target_color], axis=0)

    print("Building robot geometry...")
    robot = build_robot()

    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=RendererColorConfig(),
        width=1280,
        height=800,
        backbone_style="discrete",
        actuator_line_width=2.0,
    )

    # Use the same frame stride as the original script
    render_stride = max(1, len(ts) // 300)

    print(f"Rendering MP4 to {args.output}...")
    # Open3D's interactive backend will be used; if running on local windows,
    # the window will momentarily pop up to capture frames if we use standard execution
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
    print("Done!")


if __name__ == "__main__":
    main()
