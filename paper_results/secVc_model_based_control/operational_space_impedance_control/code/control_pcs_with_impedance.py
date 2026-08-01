"""
Operational-Space Impedance Control Example for PCS (Piecewise Constant Strain) Soft Robot.

This example demonstrates how to use the OperationalSpaceImpedanceControlTracker
to control a two-segment PCS soft robot in operational space. It supports both
end-effector position tracking and full end-effector pose tracking, with
trajectory primitives shared by the operational-space examples in this folder.

The operational-space impedance controller uses partial feedback linearization
and moving-reference feedforward to shape the task dynamics with a desired
impedance characterized by stiffness K_x and damping D_x gains.

This script runs the simulation and saves the rollout as an ``.npz`` file. Use
``plot_control_pcs_with_impedance.py`` to generate plots and Viser rendering
from the saved trajectory file.

References:
    Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
    Model-based dynamic feedback control of a planar soft robot: trajectory
    tracking and interaction with the environment. The International Journal
    of Robotics Research, 39(4-5), 490-513.

    Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
    into Learned Models for Control. Dissertation, Delft University of Technology.
    https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from operational_space_impedance_common import (
    DEFAULT_SAVE_DT,
    DEFAULT_SOLVER_DT,
    DEFAULT_TRAJECTORY_OUTPUT,
    build_problem,
    print_problem_summary,
    run_impedance_simulation,
    save_run_npz,
)
from trajectory_primitives import (
    ORIENTATION_PRIMITIVES,
    POSITION_PRIMITIVES,
    SURFACES,
    TRACKING_MODES,
    TaskSpaceTrajectoryConfig,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", choices=TRACKING_MODES, default="pose")
    parser.add_argument(
        "--position-primitive",
        choices=POSITION_PRIMITIVES,
        default="figure-eight",
    )
    parser.add_argument(
        "--orientation-primitive",
        choices=ORIENTATION_PRIMITIVES,
        default="surface-following",
    )
    parser.add_argument("--surface", choices=SURFACES, default="sphere")
    parser.add_argument("--t1", type=float, default=10.0, help="Final time [s].")
    parser.add_argument(
        "--period", type=float, default=4.0, help="Trajectory period [s]."
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.0,
        help="Minimum-jerk startup ramp duration [s].",
    )
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=DEFAULT_SOLVER_DT,
        help="Internal integration step [s].",
    )
    parser.add_argument(
        "--save-dt",
        type=float,
        default=DEFAULT_SAVE_DT,
        help="Saved trajectory sample period [s].",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Output .npz trajectory file.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing trajectory file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_config = TaskSpaceTrajectoryConfig(
        tracking=args.tracking,
        position_primitive=args.position_primitive,
        orientation_primitive=args.orientation_primitive,
        surface=args.surface,
        period=args.period,
        ramp_time=args.ramp_time,
    )
    problem = build_problem(trajectory_config)
    print_problem_summary(problem)
    run = run_impedance_simulation(
        problem,
        t1=args.t1,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    output = save_run_npz(args.trajectory_output, run, force=args.force)
    print(f"\nTrajectory saved to: {output}")
    print(
        "Plot/render with: "
        "uv run python paper_results/secVc_model_based_control/"
        "operational_space_impedance_control/code/"
        f"plot_control_pcs_with_impedance.py {output}"
    )


if __name__ == "__main__":
    main()
