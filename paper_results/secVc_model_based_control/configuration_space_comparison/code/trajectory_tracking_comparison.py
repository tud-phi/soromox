"""Run the PCS configuration-space trajectory tracking comparison.

This simulation script saves a reusable ``.npz`` rollout. Use
``plot_configuration_space_comparison.py`` for Matplotlib figures and
``render_configuration_space_comparison.py`` for Viser rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from configuration_space_comparison_simulation import (
    DEFAULT_LINEAR_INTEGRAL_ERROR_SCALE,
    DEFAULT_ROTATIONAL_INTEGRAL_ERROR_SCALE,
    DEFAULT_SAVE_DT,
    DEFAULT_SOLVER_DT,
    DEFAULT_TRAJECTORY_OUTPUT,
    run_trajectory_tracking_comparison,
    save_comparison_npz,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t1-slow",
        type=float,
        default=30.0,
        help="Final time for the slow trajectory [s].",
    )
    parser.add_argument(
        "--t1-fast",
        type=float,
        default=15.0,
        help="Final time for the fast trajectory [s].",
    )
    parser.add_argument(
        "--slow-frequency-scale",
        type=float,
        default=1.0,
        help="Frequency scale for the slow trajectory.",
    )
    parser.add_argument(
        "--fast-frequency-scale",
        type=float,
        default=3.0,
        help="Frequency scale for the fast trajectory.",
    )
    parser.add_argument(
        "--rotational-integral-error-scale",
        type=float,
        default=DEFAULT_ROTATIONAL_INTEGRAL_ERROR_SCALE,
        help="Rotational integral-error saturation scale [1/m].",
    )
    parser.add_argument(
        "--linear-integral-error-scale",
        type=float,
        default=DEFAULT_LINEAR_INTEGRAL_ERROR_SCALE,
        help="Linear integral-error saturation scale.",
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
        "--output",
        type=Path,
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Output .npz trajectory file.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_trajectory_tracking_comparison(
        t1_slow=args.t1_slow,
        t1_fast=args.t1_fast,
        slow_frequency_scale=args.slow_frequency_scale,
        fast_frequency_scale=args.fast_frequency_scale,
        rotational_integral_error_scale=args.rotational_integral_error_scale,
        linear_integral_error_scale=args.linear_integral_error_scale,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    output = save_comparison_npz(args.output, run, force=args.force)

    print(f"\nTrajectory saved to: {output}")
    print(
        "Plot with: "
        "uv run python paper_results/secVc_model_based_control/"
        "configuration_space_comparison/code/"
        f"plot_configuration_space_comparison.py {output}"
    )
    print(
        "Render slow trajectory with: "
        "uv run python paper_results/secVc_model_based_control/"
        "configuration_space_comparison/code/"
        f"render_configuration_space_comparison.py {output} --target slow"
    )
    print(
        "Render fast trajectory with: "
        "uv run python paper_results/secVc_model_based_control/"
        "configuration_space_comparison/code/"
        f"render_configuration_space_comparison.py {output} --target fast"
    )


if __name__ == "__main__":
    main()
