"""Plot control gain optimization results with Matplotlib."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILENAME = "optimization_results.mat"

FIGURE_SIZE_CM = (28.0, 16.0)
CM_TO_INCH = 1.0 / 2.54
CM_TO_POINT = 72.0 / 2.54
FONT_SIZE_CM = 0.4
FONT_SIZE_PT = FONT_SIZE_CM * CM_TO_POINT

# MATLAB R2025a "gem" palette values. The first and third colors are swapped
# below to match the MATLAB script's orderedcolors("gem") post-processing.
GEM_COLORS = np.array(
    [
        [0.0660, 0.4430, 0.7450],
        [0.8660, 0.3290, 0.0000],
        [0.9290, 0.6940, 0.1250],
        [0.5210, 0.0860, 0.8190],
        [0.2310, 0.6660, 0.1960],
        [0.1840, 0.7450, 0.9370],
        [0.8190, 0.0150, 0.5450],
    ]
)

# MATLAB "glow" palette swatch values, used only for translucent fills.
GLOW_COLORS = np.array(
    [
        [0.1490, 0.5490, 0.8667],
        [0.9608, 0.4667, 0.1608],
        [1.0000, 0.9098, 0.3922],
        [0.7529, 0.3608, 0.9843],
        [0.2863, 0.8588, 0.2510],
        [0.4235, 0.9569, 1.0000],
        [0.9490, 0.4039, 0.7725],
    ]
)


@dataclass(frozen=True)
class BatchStats:
    """Per-timestep statistics for a batch of trajectories."""

    initial_median: np.ndarray
    initial_min: np.ndarray
    initial_max: np.ndarray
    best: np.ndarray
    best_min: np.ndarray
    best_max: np.ndarray


def cm_to_inches(size_cm: tuple[float, float]) -> tuple[float, float]:
    """Convert a Matplotlib figure size from centimeters to inches."""
    return (size_cm[0] * CM_TO_INCH, size_cm[1] * CM_TO_INCH)


def load_results(directory: Path) -> dict[str, np.ndarray]:
    """Load an optimization_results.mat file from a data directory."""
    path = directory / RESULTS_FILENAME
    return {
        key: np.asarray(value)
        for key, value in sio.loadmat(path, squeeze_me=False).items()
        if not key.startswith("__")
    }


def swapped_first_and_third(colors: np.ndarray) -> np.ndarray:
    """Mirror the MATLAB script's color row swap."""
    swapped = colors.copy()
    swapped[[0, 2]] = swapped[[2, 0]]
    return swapped


def best_batch_from_loss(history_loss: np.ndarray) -> int:
    """Return the batch index with the lowest loss attained over all iterations."""
    min_loss_per_batch = np.min(history_loss, axis=0)
    return int(np.argmin(min_loss_per_batch))


def summarize_batch(
    raw_initial: np.ndarray,
    raw_best: np.ndarray,
    best_batch: int,
) -> BatchStats:
    """Compute the median/min/max trajectories used by the original MATLAB script."""
    return BatchStats(
        initial_median=np.median(raw_initial, axis=0),
        initial_min=np.min(raw_initial, axis=0),
        initial_max=np.max(raw_initial, axis=0),
        best=raw_best[best_batch],
        best_min=np.min(raw_best, axis=0),
        best_max=np.max(raw_best, axis=0),
    )


def plot_range(
    ax: plt.Axes,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> None:
    """Plot a translucent min/max band."""
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)


def format_axes(ax: plt.Axes) -> None:
    """Apply the grid and boxed axes style used by MATLAB."""
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_visible(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a bold normalized panel label."""
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def plot_target(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
) -> None:
    """Plot a scalar target as yline, or a trajectory target against time."""
    target = np.asarray(y).squeeze()
    if target.ndim == 0 or target.size == 1:
        ax.axhline(float(target), linestyle="--", linewidth=2, color=color)
    else:
        ax.plot(x, target, linestyle="--", linewidth=2, color=color)


def plot_loss(
    ax: plt.Axes,
    history_loss: np.ndarray,
    panel_label: str,
    show_legend: bool,
) -> None:
    """Plot loss min/max range and best loss line."""
    xdata = np.arange(1, history_loss.shape[0] + 1)
    min_y = np.min(history_loss, axis=1)
    max_y = np.max(history_loss, axis=1)

    format_axes(ax)
    ax.fill_between(
        xdata,
        min_y,
        max_y,
        color=(0.6, 0.8, 1.0),
        alpha=0.5,
        linewidth=0,
        label="Min-Max Range",
    )
    ax.plot(
        xdata,
        min_y,
        "-o",
        color="b",
        markerfacecolor="b",
        linewidth=1.5,
        markersize=2,
        label="Best",
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, xdata[-1])
    if show_legend:
        ax.legend(loc="upper right")
    add_panel_label(ax, panel_label)


def plot_end_effector_position(
    ax: plt.Axes,
    data_s: dict[str, np.ndarray],
    best_batch: int,
    linecolors: np.ndarray,
    fillcolors: np.ndarray,
) -> None:
    """Plot synergistic end-effector position tracking."""
    ts = np.asarray(data_s["t_ts"])[0]
    stats = summarize_batch(data_s["x_ts_init"], data_s["x_ts_best"], best_batch)
    x_des_ts = np.asarray(data_s["x_des_ts"])

    format_axes(ax)
    for j in range(3):
        line_color = linecolors[j]
        fill_color = fillcolors[j]

        plot_target(ax, ts, x_des_ts[:, j], line_color)
        plot_range(
            ax,
            ts,
            stats.initial_min[:, j],
            stats.initial_max[:, j],
            fill_color,
            alpha=0.35,
        )
        ax.plot(ts, stats.initial_median[:, j], "-.", linewidth=1.5, color=line_color)
        plot_range(
            ax,
            ts,
            stats.best_min[:, j],
            stats.best_max[:, j],
            fill_color,
            alpha=0.5,
        )
        ax.plot(ts, stats.best[:, j], "-", linewidth=1.25, color=line_color)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("End-effector position [m]")
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6, 1))
    add_panel_label(ax, "(d)")


def plot_angular_strains(
    ax: plt.Axes,
    data_c: dict[str, np.ndarray],
    best_batch: int,
    linecolors: np.ndarray,
    fillcolors: np.ndarray,
) -> None:
    """Plot collocated angular strain tracking."""
    ts = np.asarray(data_c["t_ts"])[0]
    stats = summarize_batch(data_c["q_ts_init"], data_c["q_ts_best"], best_batch)
    q_des_ts = np.asarray(data_c["q_des_ts"])

    format_axes(ax)
    segment_index = 0
    for j in range(3):
        idx = 6 * segment_index + j
        line_color = linecolors[j]
        fill_color = fillcolors[j]

        plot_target(ax, ts, q_des_ts[:, idx], line_color)
        plot_range(
            ax,
            ts,
            stats.initial_min[:, idx],
            stats.initial_max[:, idx],
            fill_color,
            alpha=0.35,
        )
        ax.plot(ts, stats.initial_median[:, idx], "-.", linewidth=1.5, color=line_color)
        plot_range(
            ax,
            ts,
            stats.best_min[:, idx],
            stats.best_max[:, idx],
            fill_color,
            alpha=0.5,
        )
        ax.plot(ts, stats.best[:, idx], "-", linewidth=1.25, color=line_color)

    legend_handles = [
        Line2D([], [], color="k", linestyle="--", linewidth=2),
        Line2D([], [], color="k", linestyle="-.", linewidth=1.5),
        Line2D([], [], color="k", linestyle="-", linewidth=1.25),
    ]
    ax.legend(
        legend_handles,
        [
            "Target",
            r"Initial (median $\pm$ range)",
            r"Optimized (best $\pm$ range)",
        ],
        loc="best",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular strains [rad/m]")
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6, 1))
    add_panel_label(ax, "(b)")


def build_figure() -> plt.Figure:
    """Load result data and build the four-panel summary figure."""
    data_c = load_results(SCRIPT_DIR / "data" / "collocated")
    data_s = load_results(SCRIPT_DIR / "data" / "synergistic")

    linecolors = swapped_first_and_third(GEM_COLORS)
    fillcolors = swapped_first_and_third(GLOW_COLORS)

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cm_to_inches(FIGURE_SIZE_CM),
        constrained_layout=True,
    )

    plot_loss(axes[0, 0], data_c["history_loss"] / 100.0, "(a)", show_legend=True)
    plot_loss(axes[1, 0], data_s["history_loss"], "(c)", show_legend=False)

    best_batch = best_batch_from_loss(data_c["history_loss"])
    plot_angular_strains(axes[0, 1], data_c, best_batch, linecolors, fillcolors)
    plot_end_effector_position(axes[1, 1], data_s, best_batch, linecolors, fillcolors)

    return fig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot control gain optimization results from .mat files."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the figure as figures/plots.pdf.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and optionally save/show the optimization summary figure."""
    args = parse_args()
    fig = build_figure()

    if args.save:
        output_dir = SCRIPT_DIR / "figures"
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "plots.pdf", dpi=300)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
