#!/usr/bin/env python3
"""Render one saved HOCLF/HOCBF trajectory at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import colors
from pcs_cf_cbf_clf_common import (
    CONTROL_STRATEGIES,
    DATA_DIR,
    OUTPUTS_DIR,
    build_simulation_setup,
    load_trajectory,
    trajectory_path,
)

from soromox.rendering import (
    BackboneColorConfig,
    CameraConfig,
    Open3DRenderer,
    RendererColorConfig,
)

PAPER_RESULTS_DIR = Path(__file__).resolve().parents[2]
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS as COLORS

cmap_pre_opt_1 = colors.LinearSegmentedColormap.from_list(
    "grad_pre_opt_1", ["#FFFFFF", COLORS["pre_opt_1"]]
)
cmap_post_opt_1 = colors.LinearSegmentedColormap.from_list(
    "grad_post_opt_1", ["#FFFFFF", COLORS["post_opt_1"]]
)


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


def _trajectory_controller_key(result: dict[str, np.ndarray], fallback: str) -> str:
    """Read controller semantics from trajectory metadata when available."""
    if "controller_key" in result:
        controller_key = str(np.asarray(result["controller_key"]).item())
    elif "use_cbf" in result:
        controller_key = (
            "with-cbf" if bool(np.asarray(result["use_cbf"]).item()) else "without-cbf"
        )
    else:
        controller_key = fallback

    if controller_key not in CONTROL_STRATEGIES:
        valid_keys = ", ".join(sorted(CONTROL_STRATEGIES))
        raise ValueError(
            f"Unknown controller_key {controller_key!r} in trajectory; expected one of {valid_keys}."
        )
    return controller_key


def main() -> None:
    args = parse_args()
    data_path = args.trajectory or trajectory_path(args.controller, args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find trajectory file: {data_path}. Run control_pcs_with_cf_cbf_clf.py first."
        )

    result = load_trajectory(data_path)
    controller_key = _trajectory_controller_key(result, args.controller)
    label = (
        str(np.asarray(result["label"]).item())
        if "label" in result
        else str(CONTROL_STRATEGIES[controller_key]["label"])
    )

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
        np.array([colors.to_rgb(COLORS["obstacle"])], dtype=np.float64),
        (obs_radii.shape[0], 1),
    )
    target_radius = np.array([0.015], dtype=np.float64)
    if controller_key == "with-cbf":
        gradient = cmap_post_opt_1(np.array([0.75, 1.0]))[:, :3]
    else:
        gradient = cmap_pre_opt_1(np.array([0.75, 1.0]))[:, :3]
    target_color = np.array([colors.to_rgb(COLORS["target"])], dtype=np.float64)
    static_spheres_positions = np.concatenate([obs_centers, target_center], axis=0)
    static_spheres_radii = np.concatenate([obs_radii, target_radius], axis=0)
    static_spheres_colors = np.concatenate([obs_colors, target_color], axis=0)

    # Manual camera settings
    radius = 2.4
    angle = -np.pi / 4
    x, y = float(radius * np.cos(angle)), float(radius * np.sin(angle))
    camera_config = CameraConfig(
        position=(x, y, 0.325),
        look_at=(0.0, 0.0, 0.125),
        fov=60.0,
    )

    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.asarray(gradient, dtype=np.float64)
        ),
        base_plate_color=(0.2, 0.2, 0.2),
    )

    print("Building robot geometry...")
    robot = build_simulation_setup().robot
    print("L_cum =", robot.L_cum)
    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=color_config,
        width=1920,
        height=1080,
        backbone_style="discrete",
        actuator_line_width=2.0,
    )

    render_stride = max(1, len(ts) // 300)
    print(f"Rendering {label} to {output_path}")
    renderer.render_sequence(
        ts=ts[::render_stride],
        q_ts=q_ts[::render_stride],
        playback_speed=1.0,
        loop=args.loop,
        record_path=str(output_path),
        close_when_recording_done=not args.loop,
        render_actuators=True,
        camera_config=camera_config,
        static_spheres_positions=static_spheres_positions,
        static_spheres_radii=static_spheres_radii,
        static_spheres_colors=static_spheres_colors,
    )
    print("Done")


if __name__ == "__main__":
    main()
