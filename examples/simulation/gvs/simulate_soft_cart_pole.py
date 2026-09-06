"""Simulate and render the force-actuated, passively hinged Soft Cart-Pole."""

from __future__ import annotations

import argparse
from pathlib import Path

from soft_pendulums import (
    SoftInvertedPendulumConfig,
    SoftLinkStiffnessMode,
    cart_pole_upright_configuration,
    make_soft_cart_pole,
    save_summary_figure,
    simulate_open_loop,
)
from soft_pendulums_viser_renderer import render_motion

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
VIDEOS_DIR = Path(__file__).resolve().parent / "videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stiffness-mode",
        choices=tuple(mode.value for mode in SoftLinkStiffnessMode),
        default=SoftLinkStiffnessMode.EXPLICIT_AFFINE.value,
        help="Construct stiffness explicitly or derive it from E, radius, and L.",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=1.0,
        help="Coefficient k used by the explicit_affine mode.",
    )
    parser.add_argument(
        "--material-young-modulus",
        type=float,
        default=1.0e6,
        help="Young's modulus used by the material_derived mode in Pa.",
    )
    parser.add_argument(
        "--material-radius",
        type=float,
        default=0.03,
        help="Circular bending radius used by the material_derived mode in m.",
    )
    parser.add_argument("--cart-mass", type=float, default=1.0)
    parser.add_argument("--force", type=float, default=0.0)
    parser.add_argument("--initial-angle", type=float, default=0.15)
    parser.add_argument("--t1", type=float, default=5.0)
    parser.add_argument("--solver-dt", type=float, default=1e-3)
    parser.add_argument("--save-dt", type=float, default=2e-2)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--record-viser", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8082)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SoftInvertedPendulumConfig(
        stiffness_mode=SoftLinkStiffnessMode(args.stiffness_mode),
        stiffness_coefficient=args.stiffness,
        material_young_modulus=args.material_young_modulus,
        material_bending_radius=args.material_radius,
        cart_mass=args.cart_mass,
    )
    robot = make_soft_cart_pole(config)
    q0 = cart_pole_upright_configuration().at[1].set(args.initial_angle)
    trajectory = simulate_open_loop(
        robot,
        initial_configuration=q0,
        control=args.force,
        t1=args.t1,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    figure_path = args.figures_dir / "simulate_soft_cart_pole.png"
    save_summary_figure(
        robot,
        trajectory,
        figure_path,
        coordinate_labels=("d", "theta", "a0", "a1"),
        cart=True,
        show=not args.no_show,
    )
    print(f"Summary figure saved to {figure_path}")
    if not args.no_viser:
        record_path = (
            args.videos_dir / "simulate_soft_cart_pole_viser.mp4"
            if args.record_viser
            else None
        )
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
        render_motion(
            robot,
            trajectory,
            cart=True,
            robot_name="Soft Cart-Pole",
            port=args.viser_port,
            record_path=record_path,
        )


if __name__ == "__main__":
    main()
