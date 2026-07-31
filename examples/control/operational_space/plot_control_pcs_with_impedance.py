"""
Plot and render a saved PCS operational-space impedance control rollout.

The trajectory ``.npz`` is produced by ``control_pcs_with_impedance.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from pcs_impedance_example import (
    DEFAULT_TRAJECTORY_OUTPUT,
    build_problem,
    load_run_npz,
    plot_run,
    print_problem_summary,
    print_tracking_summary,
    render_run,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Saved .npz trajectory file.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for generated PDF plots. Defaults to the trajectory path stem.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show plots.")
    parser.add_argument(
        "--no-render", action="store_true", help="Skip optional Viser rendering."
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Start Viser without opening a browser tab.",
    )
    parser.add_argument(
        "--robot-opacity",
        type=float,
        default=0.35,
        help="Soft robot opacity for Viser rendering.",
    )
    parser.add_argument(
        "--video-output",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save one Viser rendering pass as an MP4. If no path is provided, "
            "defaults to '<output-prefix>-render.mp4'."
        ),
    )
    parser.add_argument(
        "--record-every-n",
        type=int,
        default=1,
        help="Record every Nth frame when --video-output is used.",
    )
    parser.add_argument(
        "--record-client-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a Viser browser client when recording video.",
    )
    parser.add_argument(
        "--record-frame-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each browser-rendered video frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_run_npz(args.trajectory)
    problem = build_problem(run.trajectory_config)
    print_problem_summary(problem)
    output_prefix = args.output_prefix or str(Path(args.trajectory).with_suffix(""))
    tracking_output, config_output, tracking_error_pos, pose_error = plot_run(
        run,
        problem,
        output_prefix=output_prefix,
        no_show=args.no_show,
    )
    print_tracking_summary(problem, tracking_error_pos, pose_error)
    print("\nPlots saved to:")
    print(f"  - {tracking_output}")
    print(f"  - {config_output}")

    if args.no_render:
        return

    video_output = None
    if args.video_output is not None:
        video_output = args.video_output or f"{output_prefix}-render.mp4"
        video_output_path = Path(video_output)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        video_output = str(video_output_path)
        print(f"\nVideo will be saved to: {video_output}")

    render_run(
        run,
        problem,
        viser_port=args.viser_port,
        open_browser=not args.no_open_browser,
        robot_opacity=args.robot_opacity,
        video_output=video_output,
        record_every_n=args.record_every_n,
        record_client_timeout=args.record_client_timeout,
        record_frame_timeout=args.record_frame_timeout,
    )


if __name__ == "__main__":
    main()
