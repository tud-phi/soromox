"""
Plot a saved PCS operational-space impedance control rollout.

The trajectory ``.npz`` is produced by ``control_pcs_with_impedance.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from operational_space_impedance_common import (
    DEFAULT_TRAJECTORY_OUTPUT,
    OUTPUTS_DIR,
    build_problem,
    load_run_npz,
    plot_run,
    print_problem_summary,
    print_tracking_summary,
)

jax.config.update("jax_enable_x64", True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Saved .npz trajectory file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for generated PDF plots. Defaults to the trajectory path stem.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show plots.")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing plot outputs."
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run = load_run_npz(args.trajectory)
    problem = build_problem(run.trajectory_config)
    print_problem_summary(problem)
    output_prefix = args.output_prefix or OUTPUTS_DIR / Path(args.trajectory).stem
    tracking_output, config_output, _, _ = plot_run(
        run,
        problem,
        output_prefix=str(output_prefix),
        no_show=args.no_show,
        force=args.force,
    )
    print_tracking_summary(run)
    print("\nPlots saved to:")
    print(f"  - {tracking_output}")
    print(f"  - {config_output}")


if __name__ == "__main__":
    main()
