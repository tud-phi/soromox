#!/usr/bin/env python3
"""Load exported HOCLF/HOCBF rollout data and create the comparison plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from pcs_cf_cbf_clf_common import DATA_DIR, OUTPUTS_DIR, load_results

plt.rcParams["pdf.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot force and goal-distance traces from saved rollout data."
    )
    parser.add_argument(
        "--controller",
        choices=("both", "with-cbf", "without-cbf"),
        default="both",
        help="Controller data to include in the plot.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing saved .npz trajectory files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory where plot files are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Matplotlib window after saving the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.controller, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    force_limit = 5.0
    force_axis_max = 1.08 * max(
        force_limit, *(float(result["force"].max()) for result in results)
    )
    fig, ax1 = plt.subplots(figsize=(8, 4))

    force_lines = []
    goal_distance_lines = []
    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        force_suffix = "+HOCBF" if result["label"] == "with CBF" else ""
        force_lines.append(
            ax1.plot(
                result["ts"],
                result["force"],
                color="tab:red",
                linewidth=2.5,
                linestyle=linestyle,
                label=f"Force for HOCLF{force_suffix} controller",
            )[0]
        )

    ax1.axhline(force_limit, color="tab:red", linestyle=":", linewidth=2.0)
    ax1.fill_between(
        results[0]["ts"],
        force_limit,
        force_axis_max,
        color="tab:red",
        alpha=0.12,
    )
    ax1.set_xlabel("Time [s]", fontsize=18)
    ax1.set_ylabel("Max Pairwise Normal Force [N]", color="tab:red", fontsize=16)
    ax1.tick_params(axis="x", labelsize=13)
    ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=13)
    ax1.set_ylim(0.0, force_axis_max)
    ax1.grid(True, alpha=0.25)
    ax1.spines["left"].set_color("tab:red")

    ax2 = ax1.twinx()
    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        goal_suffix = "+HOCBF" if result["label"] == "with CBF" else ""
        goal_distance_lines.append(
            ax2.plot(
                result["ts"],
                result["distance"],
                color="tab:blue",
                linewidth=2.5,
                linestyle=linestyle,
                label=f"Goal distance for HOCLF{goal_suffix} controller",
            )[0]
        )

    ax2.set_ylabel("Goal Distance [m]", color="tab:blue", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=13)
    ax2.spines["right"].set_color("tab:blue")

    if len(results) == 1:
        legend_lines = [force_lines[0], goal_distance_lines[0]]
    else:
        legend_lines = [
            force_lines[1],
            force_lines[0],
            goal_distance_lines[1],
            goal_distance_lines[0],
        ]
    ax1.legend(
        legend_lines,
        [line.get_label() for line in legend_lines],
        loc="center",
        fontsize=10,
    )

    fig.tight_layout()
    png_path = args.output_dir / "force_goal_distance_plot.png"
    pdf_path = args.output_dir / "force_goal_distance_plot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to {png_path}")
    print(f"Saved plot to {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
