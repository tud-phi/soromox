#!/usr/bin/env python3
"""Plot soft tentacle parameter identification results with standard template aesthetics."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Configuration & Constants ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Standardized color palette to match the template's visual language
COLORS = {
    "loss": "#0072B2",  # Blue
    "before": "#0072B2",  # Blue
    "after": "#D55E00",  # Orange
    "marker_orig": "#E69F00",  # Light Orange
    "marker_hat": "#D55E00",  # Dark Orange/Red
    "marker_meas": "#0072B2",  # Blue
    "backbone": "#333333",  # Dark Grey
}


@dataclass
class ShapeComparisonData:
    """Standardized container for the 3D shape comparison data."""

    curves_orig: np.ndarray
    curves_hat: np.ndarray
    markers_orig: np.ndarray
    markers_hat: np.ndarray
    measured_pts: np.ndarray


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "font.size": 10,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )


# --- Plotting Functions ---


def plot_loss_history(loss_history: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)

    ax.plot(loss_history, "o-", color=COLORS["loss"], markersize=4)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.margins(x=0.01)

    return fig


def plot_rmse_comparison(rmse_before: np.ndarray, rmse_after: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)

    marker_names = [f"M{i + 1}" for i in range(len(rmse_before))]
    x = np.arange(len(marker_names))
    width = 0.35

    ax.bar(
        x - width / 2,
        rmse_before,
        width,
        label="Initial param.",
        color=COLORS["before"],
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        rmse_after,
        width,
        label="Optimized param.",
        color=COLORS["after"],
        alpha=0.8,
    )

    ax.set_xticks(x, marker_names)
    ax.set_ylabel(r"RMS Position Error $\mathrm{(m)}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    return fig


def plot_configuration_errors(
    errors_before: np.ndarray, errors_after: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)

    norm_before = np.sqrt(np.sum(errors_before**2, axis=1))
    norm_after = np.sqrt(np.sum(errors_after**2, axis=1))

    ax.plot(norm_before, "o-", color=COLORS["before"], label="Before")
    ax.plot(norm_after, "s-", color=COLORS["after"], label="After")

    ax.set_xlabel("Sample index")
    ax.set_ylabel(r"Total position error norm $\mathrm{(m)}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    return fig


def plot_3d_shapes(data: ShapeComparisonData) -> plt.Figure:
    # 3D plots require overriding some rcParams like grid display in 3D projection
    fig = plt.figure(figsize=(6.5, 5.0), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")

    M = len(data.curves_orig)
    alpha_orig = 0.3
    alpha_hat = 1.0

    # Calculate bounds to enforce equal aspect ratio
    all_pts = np.vstack(
        [
            data.curves_orig,
            data.curves_hat,
            data.markers_orig,
            data.markers_hat,
            data.measured_pts,
        ]
    )
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (maxs + mins)
    max_range = (maxs - mins).max()

    for m in range(M):
        curve0 = data.curves_orig[m]
        curve_hat = data.curves_hat[m]

        ax.plot(
            curve0[:, 0],
            curve0[:, 1],
            curve0[:, 2],
            lw=2,
            color=COLORS["backbone"],
            alpha=alpha_orig,
        )
        ax.plot(
            curve_hat[:, 0],
            curve_hat[:, 1],
            curve_hat[:, 2],
            lw=2,
            color=COLORS["backbone"],
            alpha=alpha_hat,
        )

        m0 = data.markers_orig[m]
        mh = data.markers_hat[m]
        mm = data.measured_pts[m]

        grad_alphas = np.linspace(0.25, 1.0, 4)

        for i in range(4):
            a = float(grad_alphas[i])
            ax.scatter(
                m0[i, 0],
                m0[i, 1],
                m0[i, 2],
                marker="x",
                color=COLORS["marker_orig"],
                s=30,
                alpha=a,
            )
            ax.scatter(
                mh[i, 0],
                mh[i, 1],
                mh[i, 2],
                marker="x",
                color=COLORS["marker_hat"],
                s=60,
                alpha=a,
            )
            ax.scatter(
                mm[i, 0],
                mm[i, 1],
                mm[i, 2],
                marker="o",
                facecolors="none",
                edgecolors=COLORS["marker_meas"],
                s=60,
                alpha=a,
            )

    ax.set_xlabel(r"X $\mathrm{(m)}$")
    ax.set_ylabel(r"Y $\mathrm{(m)}$")
    ax.set_zlabel(r"Z $\mathrm{(m)}$")
    ax.view_init(elev=20, azim=15)

    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    # Standardized Legend
    legend_elems = [
        Line2D(
            [0],
            [0],
            color=COLORS["backbone"],
            lw=2.0,
            alpha=alpha_orig,
            label="Original Backbone",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["backbone"],
            lw=2.0,
            alpha=alpha_hat,
            label="Optimized Backbone",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor=COLORS["marker_orig"],
            markeredgecolor=COLORS["marker_orig"],
            markersize=7,
            label="Pred. markers (Orig)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor=COLORS["marker_hat"],
            markeredgecolor=COLORS["marker_hat"],
            markersize=7,
            label="Pred. markers (Opt)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor=COLORS["marker_meas"],
            markerfacecolor="none",
            markersize=7,
            label="Measured markers",
        ),
    ]

    # 3D legends need careful placement to avoid clipping the box
    ax.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.05, 0.5))
    return fig


# --- Main Execution / I-O ---


def save_figure(
    fig: plt.Figure, output_dir: Path, filename: str, output_format: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.{output_format}"
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Saved {filepath.resolve()}")


def main() -> int:
    # Configuration visualization options
    configure_matplotlib()

    # Load data
    loss_history = np.load(DEFAULT_DATA_DIR / "loss_history.npy")
    rmse_before = np.load(DEFAULT_DATA_DIR / "rmse_before.npy")
    rmse_after = np.load(DEFAULT_DATA_DIR / "rmse_after.npy")
    errors_before = np.load(DEFAULT_DATA_DIR / "errors_before.npy")
    errors_after = np.load(DEFAULT_DATA_DIR / "errors_after.npy")
    shape_data = ShapeComparisonData(
        curves_orig=np.load(DEFAULT_DATA_DIR / "curves_orig.npy"),
        curves_hat=np.load(DEFAULT_DATA_DIR / "curves_hat.npy"),
        markers_orig=np.load(DEFAULT_DATA_DIR / "markers_orig.npy"),
        markers_hat=np.load(DEFAULT_DATA_DIR / "markers_hat.npy"),
        measured_pts=np.load(DEFAULT_DATA_DIR / "measured_pts.npy"),
    )

    # Create figures
    fig_loss = plot_loss_history(loss_history)
    fig_rmse = plot_rmse_comparison(rmse_before, rmse_after)
    fig_errors = plot_configuration_errors(errors_before, errors_after)
    fig_shape = plot_3d_shapes(shape_data)

    # Save
    save_figure(fig_loss, DEFAULT_OUTPUT_DIR, "loss_over_time", "pdf")
    save_figure(fig_rmse, DEFAULT_OUTPUT_DIR, "rmse", "pdf")
    save_figure(fig_errors, DEFAULT_OUTPUT_DIR, "errors", "pdf")
    save_figure(fig_shape, DEFAULT_OUTPUT_DIR, "shape", "pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
