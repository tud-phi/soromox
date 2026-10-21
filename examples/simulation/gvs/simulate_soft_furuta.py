"""Simulate and render the first-joint-actuated Soft Furuta Pendulum."""

from __future__ import annotations

import argparse
from pathlib import Path

from soft_pendulums import (
    SoftSwingUpPendulumConfig,
    furuta_upright_configuration,
    make_soft_furuta,
    save_summary_figure,
    simulate_open_loop,
)
from soft_pendulums_viser_renderer import render_motion

from soromox.rendering import CameraConfig

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
VIDEOS_DIR = Path(__file__).resolve().parent / "videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque", type=float, default=0.0)
    parser.add_argument("--initial-angle-offset", type=float, default=0.15)
    parser.add_argument("--t1", type=float, default=3.0)
    parser.add_argument("--solver-dt", type=float, default=1e-3)
    parser.add_argument("--save-dt", type=float, default=2e-2)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--record-viser", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8084)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = make_soft_furuta(SoftSwingUpPendulumConfig())
    q0 = furuta_upright_configuration().at[1].add(args.initial_angle_offset)
    trajectory = simulate_open_loop(
        robot,
        initial_configuration=q0,
        control=args.torque,
        t1=args.t1,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    figure_path = args.figures_dir / "simulate_soft_furuta.png"
    save_summary_figure(
        robot,
        trajectory,
        figure_path,
        coordinate_labels=("theta1", "theta2", "a0", "a1"),
        show=not args.no_show,
    )
    print(f"Summary figure saved to {figure_path}")
    if not args.no_viser:
        record_path = (
            args.videos_dir / "simulate_soft_furuta_viser.mp4"
            if args.record_viser
            else None
        )
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        render_motion(
            robot,
            trajectory,
            cart=False,
            robot_name="Soft Furuta Pendulum",
            port=args.viser_port,
            record_path=record_path,
            camera_config=CameraConfig(
                fov=45.0,
                distance_factor=1.5,
                position_offset=(0.9, -0.9, 0.6),
                up=(0.0, 0.0, 1.0),
            ),
        )


if __name__ == "__main__":
    main()
