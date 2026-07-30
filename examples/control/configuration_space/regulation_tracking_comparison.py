"""Compare controllers on regulation followed by trajectory tracking.

Every controller runs over the same uninterrupted task: a sequence of constant
configuration-space setpoints followed by a smooth dynamic trajectory. The
simulation saves a reusable ``.npz`` rollout for the shared plotting and Viser
rendering scripts.
"""

from __future__ import annotations

import argparse

import jax
from configuration_space_comparison_simulation import (
    DEFAULT_FEEDBACK_DAMPING_RATIO,
    DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY,
    DEFAULT_FEEDBACK_NATURAL_FREQUENCY,
    DEFAULT_REGULATION_TRACKING_FREQUENCY_SCALE,
    DEFAULT_REGULATION_TRACKING_OUTPUT,
    DEFAULT_SAVE_DT,
    DEFAULT_SOLVER_DT,
    run_regulation_tracking_comparison,
    save_comparison_npz,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-start",
        type=float,
        default=15.0,
        help="Time at which the task changes from regulation to tracking [s].",
    )
    parser.add_argument("--t1", type=float, default=30.0, help="Final time [s].")
    parser.add_argument(
        "--tracking-frequency-scale",
        type=float,
        default=DEFAULT_REGULATION_TRACKING_FREQUENCY_SCALE,
        help="Frequency scale for the dynamic tracking trajectory.",
    )
    parser.add_argument(
        "--feedback-natural-frequency",
        type=float,
        default=DEFAULT_FEEDBACK_NATURAL_FREQUENCY,
        help="Acceleration-domain feedback natural frequency [rad/s].",
    )
    parser.add_argument(
        "--feedback-damping-ratio",
        type=float,
        default=DEFAULT_FEEDBACK_DAMPING_RATIO,
        help="Acceleration-domain feedback damping ratio.",
    )
    parser.add_argument(
        "--feedback-integral-frequency",
        type=float,
        default=DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY,
        help="Integral-to-proportional gain ratio [rad/s].",
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
        default=DEFAULT_REGULATION_TRACKING_OUTPUT,
        help="Output .npz trajectory file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_regulation_tracking_comparison(
        tracking_start=args.tracking_start,
        t1=args.t1,
        tracking_frequency_scale=args.tracking_frequency_scale,
        feedback_natural_frequency=args.feedback_natural_frequency,
        feedback_damping_ratio=args.feedback_damping_ratio,
        feedback_integral_frequency=args.feedback_integral_frequency,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
    )
    output = save_comparison_npz(args.output, run)

    print(f"\nTrajectory saved to: {output}")
    print(
        "Plot with: "
        "uv run python examples/control/configuration_space/"
        f"plot_configuration_space_comparison.py {output}"
    )
    print(
        "Render with: "
        "uv run python examples/control/configuration_space/"
        f"render_configuration_space_comparison.py {output}"
    )


if __name__ == "__main__":
    main()
