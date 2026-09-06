"""Simulate the fixed-base affine-curvature soft inverted pendulum.

This open-loop example uses a distributed-mass GVS link while retaining the
paper's affine-curvature basis, generalized stiffness and damping, tip-torque
map, physical parameters, and initial condition.

References:
    Della Santina, C. (2020). The soft inverted pendulum with affine
    curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
    4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soft_pendulums import (
    SoftInvertedPendulumConfig,
    make_fixed_pendulum,
    save_summary_figure,
    simulate_open_loop,
    sip_initial_configuration,
    sip_tip_torque_generalized_force,
)
from soft_pendulums_viser_renderer import render_motion

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
VIDEOS_DIR = Path(__file__).resolve().parent / "videos"


def parse_args() -> argparse.Namespace:
    """Parse command-line simulation and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stiffness",
        type=float,
        choices=(1.0, 4.0),
        default=1.0,
        help="Paper stiffness coefficient k (default: 1).",
    )
    parser.add_argument(
        "--tip-torque",
        type=float,
        default=0.0,
        help="Constant tip torque in N m (default: autonomous motion).",
    )
    parser.add_argument("--t1", type=float, default=10.0)
    parser.add_argument("--solver-dt", type=float, default=1e-3)
    parser.add_argument("--save-dt", type=float, default=2e-2)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the summary figure without opening Matplotlib.",
    )
    parser.add_argument(
        "--no-viser",
        action="store_true",
        help="Skip the interactive Viser motion rendering.",
    )
    parser.add_argument(
        "--record-viser",
        action="store_true",
        help="Record one Viser playback to the example videos directory.",
    )
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    return parser.parse_args()


def main() -> None:
    """Construct, simulate, plot, and render the fixed-base example."""
    args = parse_args()
    config = SoftInvertedPendulumConfig(stiffness_coefficient=args.stiffness)
    robot = make_fixed_pendulum(config)
    trajectory = simulate_open_loop(
        robot,
        initial_configuration=sip_initial_configuration(),
        external_force=sip_tip_torque_generalized_force(args.tip_torque),
        t1=args.t1,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )

    figure_path = args.figures_dir / (
        f"simulate_soft_inverted_pendulum_k{args.stiffness:g}.png"
    )
    save_summary_figure(
        robot,
        trajectory,
        figure_path,
        coordinate_labels=("a0", "a1"),
        show=not args.no_show,
    )
    print(f"Summary figure saved to {figure_path}")

    if not args.no_viser:
        record_path = (
            args.videos_dir
            / f"simulate_soft_inverted_pendulum_k{args.stiffness:g}_viser.mp4"
            if args.record_viser
            else None
        )
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        render_motion(
            robot,
            trajectory,
            cart=False,
            robot_name="Soft Inverted Pendulum",
            port=args.viser_port,
            record_path=record_path,
        )


if __name__ == "__main__":
    main()
