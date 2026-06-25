#!/usr/bin/env python3
"""Render one saved HOCLF/HOCBF trajectory at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pcs_cf_cbf_clf_common import (
    CONTROL_STRATEGIES,
    DATA_DIR,
    OUTPUTS_DIR,
    build_simulation_setup,
    load_trajectory,
    trajectory_path,
)

from soromox.rendering import Open3DRenderer, RendererColorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a saved tendon-actuated PCS rollout to video.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--controller",
        choices=("with-cbf", "without-cbf"),
        default="with-cbf",
        help="Control strategy trajectory to render.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing saved trajectory .npz files.",
    )
    parser.add_argument(
        "--data",
        "--trajectory",
        dest="trajectory",
        type=Path,
        default=None,
        help="Explicit .npz trajectory path. Overrides --controller/--data-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 path. Defaults to outputs/<data filename>.mp4.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback in the interactive renderer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.trajectory or trajectory_path(args.controller, args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find trajectory file: {data_path}. Run control_pcs_with_cf_cbf_clf.py first."
        )

    result = load_trajectory(data_path)
    label = str(result["label"].item()) if "label" in result else args.controller

    output_path = args.output
    if output_path is None:
        output_path = OUTPUTS_DIR / f"{data_path.stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory from {data_path}")
    ts = result["ts"]
    q_ts = result["q_ts"]
    obs_centers = result["obs_centers"]
    obs_radii = result["obs_radii"]
    target_center = result["target_center"].reshape(1, 3)

    obs_colors = np.tile(
        np.array([[0.5, 0.5, 0.5]], dtype=np.float64), (obs_radii.shape[0], 1)
    )
    target_radius = np.array([0.01], dtype=np.float64)
    target_color = np.array([[1.0, 0.1, 0.1]], dtype=np.float64)
    static_spheres_positions = np.concatenate([obs_centers, target_center], axis=0)
    static_spheres_radii = np.concatenate([obs_radii, target_radius], axis=0)
    static_spheres_colors = np.concatenate([obs_colors, target_color], axis=0)

    print("Building robot geometry...")
    robot = build_simulation_setup().robot
    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=RendererColorConfig(),
        width=1280,
        height=800,
        backbone_style="discrete",
        actuator_line_width=2.0,
    )

    render_stride = max(1, len(ts) // 300)
    print(
        f"Rendering {CONTROL_STRATEGIES.get(args.controller, {}).get('label', label)} to {output_path}"
    )
    renderer.render_sequence(
        ts=ts[::render_stride],
        q_ts=q_ts[::render_stride],
        playback_speed=1.0,
        loop=args.loop,
        record_path=str(output_path),
        render_actuators=True,
        static_spheres_positions=static_spheres_positions,
        static_spheres_radii=static_spheres_radii,
        static_spheres_colors=static_spheres_colors,
    )
    print("Done")


if __name__ == "__main__":
    main()
