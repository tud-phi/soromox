#!/usr/bin/env python3
"""Standalone visualization for HOCLF/HOCBF controller metrics."""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Directory Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
# os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

# --- Configuration Constants ---
COLORS = {
    "pre_opt_1": "#006BA6",  # Blue
    "post_opt_1": "#D81159",  # Red
}

data = np.load(DATA_DIR / "clf_cbf_rollout_with_cbf.npz")
for k in data:
    print(k)
    print(data[k].shape)
    print("-----")


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            # Adjust white padding around image
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
            "font.size": 11,
            "axes.labelsize": 11,
            # --- Legend Configuration ---
            "legend.fontsize": 11,
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
            "figure.dpi": 150,
            "lines.linewidth": 1.8,
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def cm2inch(*tupl: float) -> tuple[float, ...]:
    """Convert centimeters to inches for Matplotlib figsize."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


def plot_metrics(data_path, output_path, show) -> None:
    if not data_path:
        print("[!] No .npz files provided.")
        return

    data = np.load(data_path)
    ts = data["ts"]
    q_ts = data["q_ts"]
    obs_centers = data["obs_centers"]
    obs_radii = data["obs_radii"]
    target_center = data["target_center"]

    fig, ax1 = plt.subplots(figsize=cm2inch(18.0, 10.0), constrained_layout=True)
    ax2 = ax1.twinx()

    force_limit = 5.0
    global_force_max = force_limit
    ts_ref = None

    force_lines = []
    goal_lines = []

    # for path in npz_paths:
    #     try:
    #         data = np.load(path)
    #         ts = data["ts"]
    #         force = data["force"]
    #         distance = data["distance"]
    #     except KeyError as e:
    #         print(
    #             f"[!] File {path.name} is missing key {e}. Ensure the simulation script exports it."
    #         )
    #         continue

    #     if ts_ref is None:
    #         ts_ref = ts

    #     global_force_max = max(global_force_max, force.max())

    #     # Determine styling based on filename
    #     is_cbf = "without_cbf" not in path.name.lower()
    #     linestyle = "-" if is_cbf else "--"
    #     label_suffix = "+HOCBF" if is_cbf else ""

    #     (line_f,) = ax1.plot(
    #         ts,
    #         force,
    #         color=COLORS["post_opt_1"],
    #         linestyle=linestyle,
    #         label=f"Force for HOCLF{label_suffix} controller",
    #     )
    #     force_lines.append(line_f)

    #     (line_d,) = ax2.plot(
    #         ts,
    #         distance,
    #         color=COLORS["pre_opt_1"],
    #         linestyle=linestyle,
    #         label=f"Goal distance for HOCLF{label_suffix} controller",
    #     )
    #     goal_lines.append(line_d)

    if not force_lines:
        plt.close(fig)
        return

    # Render Force Limit Boundaries
    force_axis_max = 1.08 * global_force_max
    ax1.axhline(
        force_limit, color=COLORS["post_opt_1"], linestyle=":", linewidth=1.5, alpha=0.8
    )

    if ts_ref is not None:
        ax1.fill_between(
            ts_ref,
            force_limit,
            force_axis_max,
            color=COLORS["post_opt_1"],
            alpha=0.12,
            linewidth=0.0,
        )

    # Apply Standardized Formatting
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(
        "Max Pairwise Normal Force [N]", color=COLORS["post_opt_1"], fontweight="bold"
    )
    ax1.tick_params(axis="y", labelcolor=COLORS["post_opt_1"])
    ax1.set_ylim(0.0, force_axis_max)
    ax1.set_xlim(ts_ref.min(), ts_ref.max())

    ax2.set_ylabel("Goal Distance [m]", color=COLORS["pre_opt_1"], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLORS["pre_opt_1"])
    ax2.set_ylim(bottom=0.0)

    # Enforce Spine Colors
    ax1.spines["left"].set_color(COLORS["post_opt_1"])
    ax1.spines["left"].set_linewidth(1.2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["pre_opt_1"])
    ax2.spines["right"].set_linewidth(1.2)
    ax1.spines["right"].set_visible(
        False
    )  # Hide primary right spine to prevent black line overlap

    # Construct Legend
    all_lines = force_lines + goal_lines
    if len(npz_paths) > 1:
        # Reorder for visual clarity if comparing two trajectories
        all_lines = [force_lines[0], force_lines[1], goal_lines[0], goal_lines[1]]

    ax1.legend(
        all_lines,
        [line.get_label() for line in all_lines],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved figure to {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HOCLF/HOCBF controller metrics from .npz files."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        # required=True,
        default=DATA_DIR / "clf_cbf_rollout_with_cbf.npz",
        help="Path(s) to the data rollout files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "force_goal_distance_plot.pdf",
        help="Path to save the PDF plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_matplotlib()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plot_metrics(args.data, args.output, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # """
