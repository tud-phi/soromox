#!/usr/bin/env python3
"""Load exported HOCLF/HOCBF rollout data and create the comparison plot."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pcs_cf_cbf_clf_common import DATA_DIR, OUTPUTS_DIR, load_results

COLORS = {
    "pre_opt_1": "#006BA6",
    "pre_opt_2": "#0496FF",
    "post_opt_1": "#D81159",
    "post_opt_2": "#8F2D56",
    "post_opt_3": "#FFBC42",
    "backbone_opt": "#7B2CBF",
    "backbone_init": "#0ead69",
    "ground_truth": "#FFBC42",
    "x_t": "#2a9d8f",
    "y_t": "#e9c46a",
    "z_t": "#e76f51",
}


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            # --- Grid Configuration ---
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.color": "#B0B0B0",
            # --- Font Configuration ---
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 14,
            "axes.labelsize": 14,
            # --- Legend Configuration ---
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.8,
            "legend.edgecolor": "#A0A0A0",
            "legend.fancybox": False,
            "patch.linewidth": 0.7,
            # --- Axes and Spines ---
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "figure.dpi": 300,
            "lines.linewidth": 1.8,
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


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
    configure_matplotlib()

    args = parse_args()
    results = load_results(args.controller, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    force_limit = 5.0
    force_axis_max = 1.08 * max(
        force_limit, *(float(result["force"].max()) for result in results)
    )
    fig, ax1 = plt.subplots(figsize=(7, 4))

    force_lines = []
    goal_distance_lines = []
    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        force_lines.append(
            ax1.plot(
                result["ts"],
                result["force"],
                color=COLORS["post_opt_1"],
                linewidth=2.5,
                linestyle=linestyle,
            )[0]
        )

    ax1.axhline(force_limit, color=COLORS["post_opt_1"], linestyle=":", linewidth=2.0)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Max Pairwise Normal Force [N]", color=COLORS["post_opt_1"])
    ax1.tick_params(axis="x")
    ax1.tick_params(axis="y", labelcolor=COLORS["post_opt_1"])
    ax1.set_ylim(0.0, force_axis_max)
    ax1.spines["left"].set_color(COLORS["post_opt_1"])

    ax2 = ax1.twinx()
    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        goal_distance_lines.append(
            ax2.plot(
                result["ts"],
                result["distance"],
                color=COLORS["pre_opt_1"],
                linewidth=2.5,
                linestyle=linestyle,
            )[0]
        )

    ax2.set_ylabel("Goal Distance [m]", color=COLORS["pre_opt_1"])
    ax2.tick_params(axis="y", labelcolor=COLORS["pre_opt_1"])
    ax2.spines["right"].set_color(COLORS["pre_opt_1"])

    custom_handles = [
        Line2D([0], [0], color="black", linestyle="--"),
        Line2D([0], [0], color="black", linestyle="-"),
    ]
    ax1.legend(
        custom_handles,
        ["HOCLF controller", "HOCLF+HOCBF controller"],
        loc="center right",
    )
    ax1.set_ylim(0.0, force_axis_max)
    distance_max = max(float(result["distance"].max()) for result in results)
    ax2.set_ylim(0.0, distance_max * 1.08)

    ax1.yaxis.set_major_locator(ticker.LinearLocator(6))
    ax2.yaxis.set_major_locator(ticker.LinearLocator(6))

    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax1.grid(True)
    ax2.grid(False)

    ax1.set_xlim(min(result["ts"]), max(result["ts"]))
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
