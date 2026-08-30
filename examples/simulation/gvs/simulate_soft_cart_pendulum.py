"""Simulate an affine-curvature soft inverted pendulum on a passive cart.

The soft link is clamped directly to a horizontal prismatic joint, so the
model has no revolute joint between the cart and pendulum. Its soft-link
parameters, basis, constitutive matrices, tip-torque map, and initial soft
configuration are identical to the fixed-base example.

References:
    Della Santina, C. (2020). The soft inverted pendulum with affine
    curvature. 2020 59th IEEE Conference on Decision and Control (CDC),
    4135-4142. https://doi.org/10.1109/CDC42340.2020.9303976

    Ajithkumar, A. (2023). Control and stabilization of soft inverted
    pendulum on a cart. Master's thesis, University of Maryland, College Park.
    https://doi.org/10.13016/dspace/7jir-df1p
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soft_inverted_pendulum import (
    SoftInvertedPendulumConfig,
    make_cart_pendulum,
    save_summary_figure,
    simulate_open_loop,
)
from soft_inverted_pendulum_viser_renderer import render_motion

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
        "--cart-mass",
        type=float,
        default=1.0,
        help="Passive cart mass in kg (default: 1).",
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
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    return parser.parse_args()


def main() -> None:
    """Construct, simulate, plot, and render the passive-cart example."""
    args = parse_args()
    config = SoftInvertedPendulumConfig(
        stiffness_coefficient=args.stiffness,
        cart_mass=args.cart_mass,
    )
    robot = make_cart_pendulum(config)
    trajectory = simulate_open_loop(
        robot,
        cart=True,
        tip_torque=args.tip_torque,
        t1=args.t1,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )

    figure_path = args.figures_dir / (
        f"simulate_soft_cart_pendulum_k{args.stiffness:g}.png"
    )
    save_summary_figure(
        robot,
        trajectory,
        figure_path,
        cart=True,
        show=not args.no_show,
    )
    print(f"Summary figure saved to {figure_path}")

    if not args.no_viser:
        record_path = (
            args.videos_dir
            / f"simulate_soft_cart_pendulum_k{args.stiffness:g}_viser.mp4"
            if args.record_viser
            else None
        )
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        render_motion(
            robot,
            trajectory,
            cart=True,
            port=args.viser_port,
            record_path=record_path,
        )


if __name__ == "__main__":
    main()
