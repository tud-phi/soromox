"""Run the PCS configuration-space setpoint regulation comparison.

This simulation script saves a reusable ``.npz`` rollout. Use
``plot_configuration_space_comparison.py`` for Matplotlib figures and
``render_configuration_space_comparison.py`` for Viser rendering.
"""

from __future__ import annotations

import argparse

import jax
from configuration_space_comparison_simulation import (
    DEFAULT_SAVE_DT,
    DEFAULT_SETPOINT_OUTPUT,
    DEFAULT_SOLVER_DT,
    run_setpoint_comparison,
    save_comparison_npz,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=float, default=15.0, help="Final time [s].")
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
        default=DEFAULT_SETPOINT_OUTPUT,
        help="Output .npz trajectory file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_setpoint_comparison(
        t1=args.t1,
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
