"""Create the canonical Section Vc operational-space impedance figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from operational_space_impedance_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    PCSImpedanceProblem,
    PCSImpedanceRun,
    build_problem,
    cm_to_inches,
    configure_matplotlib,
    load_run_npz,
    plot_operational_tracking_axes,
    tracking_legend_handles,
)

DEFAULT_TRAJECTORY_INPUT = DATA_DIR / "control_pcs_with_impedance_trajectory.npz"
DEFAULT_OUTPUT = OUTPUTS_DIR / "operational_space_impedance_control.pdf"


def build_operational_space_paper_figure(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
) -> plt.Figure:
    """Build the canonical partial-feedback-linearization trajectory figure."""
    if run.trajectory_config.tracking != "pose":
        raise ValueError("The canonical operational figure requires pose tracking.")
    configure_matplotlib()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=cm_to_inches((16.5, 8.5)),
        constrained_layout=True,
    )
    plot_operational_tracking_axes(
        run,
        problem,
        np.asarray((axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])),
    )
    for ax in axes[1, :]:
        ax.set_xlabel(r"Time $\mathrm{[s]}$")
    handles, labels = tracking_legend_handles()
    fig.legend(handles, labels, loc="outside upper center", ncol=5)
    return fig


def save_operational_space_paper_figure(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    output: Path,
    *,
    show: bool,
    force: bool,
) -> Path:
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}. Pass --force.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_operational_space_paper_figure(run, problem)
    fig.savefig(output, bbox_inches=None)
    if show:
        plt.show()
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-input", type=Path, default=DEFAULT_TRAJECTORY_INPUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_run_npz(args.trajectory_input)
    problem = build_problem(run.trajectory_config)
    output = save_operational_space_paper_figure(
        run, problem, args.output, show=args.show, force=args.force
    )
    print(f"Figure written to {output}")


if __name__ == "__main__":
    main()
