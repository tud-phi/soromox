#!/usr/bin/env python3
"""Plot soft tentacle residual identification results with standard template aesthetics."""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# --- Configuration & Constants ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
PAPER_RESULTS_DIR = SCRIPT_DIR.parent
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE

AXIS = {
    "x_axis_label": r"$x$ $\mathrm{[m]}$",
    "y_axis_label": r"$y$ $\mathrm{[m]}$",
    "z_axis_label": r"$z$ $\mathrm{[m]}$",
}


@dataclass
class ShapeComparisonData:
    curves_orig: np.ndarray
    curves_tau_star: np.ndarray
    markers_orig: np.ndarray
    markers_tau_star: np.ndarray
    measured_pts: np.ndarray


@dataclass
class RmseData:
    initial: np.ndarray
    optimal: np.ndarray
    residual: np.ndarray
    nn_pred: np.ndarray


def cm2inch(*tupl: float) -> tuple[float, ...]:
    """Convert centimeters to inches for Matplotlib figsize."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


"""LEGEND OF LEGEND_LOC = 0: 'best', 1: 'upper right', 2: 'upper left', 3: 'lower left', 4: 'lower right',
5: 'right', 6: 'center left', 7: 'center right', 8: 'lower center', 9: 'upper center', 10: 'center' """


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "savefig.pad_inches": 0.1,
            "font.size": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "figure.dpi": 150,
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


# --- Plotting Functions ---


def plot_radar_rmse(
    data: RmseData,
    figsize_cm: tuple[float, float] = (16.5, 12.7),
) -> plt.Figure:
    """Note: Radar plots can distort perception of independent categorical data."""

    # Data preparation (closing the loop)
    def _close_loop(arr: np.ndarray) -> list:
        val = arr.tolist()
        return val + val[:1]

    d_init = _close_loop(data.initial)
    d_opt = _close_loop(data.optimal)
    d_resid = _close_loop(data.residual)
    d_nn = _close_loop(data.nn_pred)

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]
    labels = [rf"$M_{{{i + 1}}}$" for i in range(4)]

    # 3D/Polar plots require overriding standard constrained layouts safely
    fig, ax = plt.subplots(
        figsize=cm2inch(figsize_cm), subplot_kw={"projection": "polar"}
    )

    # Plotting lines and fills
    ax.plot(
        angles,
        d_init,
        "o-",
        label="Initial parameters",
        linewidth=0.7,
        markersize=2,
        color=COLORS["pre_opt_1"],
    )
    ax.fill(angles, d_init, alpha=0.2, color=COLORS["pre_opt_1"])

    ax.plot(
        angles,
        d_opt,
        "o-",
        label="Optimal parameters",
        linewidth=0.7,
        markersize=2,
        color=COLORS["post_opt_1"],
    )
    ax.fill(angles, d_opt, alpha=0.2, color=COLORS["post_opt_1"])

    ax.plot(
        angles,
        d_resid,
        "o-",
        label="Optimal residual",
        linewidth=0.7,
        markersize=2,
        color=COLORS["post_opt_2"],
    )
    ax.fill(angles, d_resid, alpha=0.2, color=COLORS["post_opt_2"])

    ax.plot(
        angles,
        d_nn,
        # "^--",
        "o-",
        label="NN-predicted residual",
        linewidth=0.5,
        markersize=0.75,
        color=COLORS["post_opt_3"],
    )
    ax.fill(angles, d_nn, alpha=0.2, color=COLORS["post_opt_3"])

    ax.tick_params(axis="x", pad=-5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(45)

    rmax = ax.get_rmax()
    ax.text(
        np.pi / 7,
        rmax * 0.64,
        r"Position RMSE $\mathrm{[m]}$",
        rotation=45,
        ha="center",
        va="center",
    )

    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    leg = ax.legend(loc="upper center")

    for handle in leg.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(0.7)
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(2)

    return fig


def plot_bar_overall_rmse(
    data: dict[str, float],
    figsize_cm: tuple[float, float] = (14.7, 8.9),
    legend_loc: str = "best",
    box: bool = False,
) -> plt.Figure:
    # Apply the cm-to-inch conversion here
    fig, ax = plt.subplots(figsize=cm2inch(figsize_cm), constrained_layout=True)
    ax.set_axisbelow(True)

    # labels = list(data.keys())
    labels = [
        "Initial\nparameters",
        "Optimal\nparameters",
        "Optimal\nresidual",
        "NN-predicted\nresidual",
    ]
    values = list(data.values())
    colors = [
        COLORS["pre_opt_1"],
        COLORS["post_opt_1"],
        COLORS["post_opt_2"],
        COLORS["post_opt_3"],
    ]

    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_ylabel(r"Overall RMSE $\mathrm{[m]}$")
    ax.set_yticks(np.arange(0.0, 0.045, 0.01))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0)

    # --- Box Toggle Logic ---
    ax.spines["top"].set_visible(box)
    ax.spines["right"].set_visible(box)

    ax.grid(axis="x")

    return fig


def plot_violin_error_distribution(
    df: pd.DataFrame, figsize_cm=(15.0, 10.0), legend_loc="upper right", box=True
) -> plt.Figure:
    """Expects DataFrame with columns: ['Marker', 'Tau Condition', 'Error']"""
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)

    # Ensure palette maps correctly to tau conditions in your dataframe
    palette = [COLORS["post_opt_1"], COLORS["post_opt_2"], COLORS["post_opt_3"]]

    sns.violinplot(
        data=df,
        x="Marker",
        y="Error",
        hue="Tau Condition",
        cut=0,
        ax=ax,
        palette=palette,
        split=False,
        inner="quartile",
        linewidth=1.0,
    )

    ax.set_ylabel(r"Position Error Norm $\mathrm{(m)}$")
    ax.set_xlabel("Markers")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    return fig


def plot_3d_comparison(
    data: ShapeComparisonData,
    figsize_cm: tuple[float, float] = (16.5, 12.7),
    legend_loc: str = "upper left",
) -> plt.Figure:
    fig = plt.figure(figsize=cm2inch(figsize_cm), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")

    M = len(data.curves_orig)
    # alpha_orig = 0.0
    alpha_tau_star = 1.0

    all_pts = np.vstack(
        [
            data.curves_orig.reshape(-1, 3),
            data.curves_tau_star.reshape(-1, 3),
            data.markers_orig.reshape(-1, 3),
            data.markers_tau_star.reshape(-1, 3),
            data.measured_pts.reshape(-1, 3),
        ]
    )
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (maxs + mins)
    max_range = (maxs - mins).max()

    # Iterate through configurations
    for m in range(M):
        curve_origin = data.curves_orig[m]
        curve_tau_star = data.curves_tau_star[m]

        ax.plot(
            curve_origin[:, 0],
            curve_origin[:, 1],
            curve_origin[:, 2],
            lw=0.8,
            color=COLORS["pre_opt_1"],
            # alpha=alpha_orig,
        )
        ax.plot(
            curve_tau_star[:, 0],
            curve_tau_star[:, 1],
            curve_tau_star[:, 2],
            lw=0.8,
            color=COLORS["post_opt_1"],
            alpha=alpha_tau_star,
        )

        m0 = data.markers_orig[m, [1, 3], :]
        mt = data.markers_tau_star[m, [1, 3], :]
        mm = data.measured_pts[m, [1, 3], :]

        grad_alphas = np.array([0.65, 1.0])  # np.linspace(0.25, 1.0, 4)

        for i in range(2):  # 4
            a = float(grad_alphas[i])
            ax.scatter(
                m0[i, 0],
                m0[i, 1],
                m0[i, 2],
                marker="*",
                color=COLORS["pre_opt_1"],
                s=6,
                alpha=a,
            )
            ax.scatter(
                mt[i, 0],
                mt[i, 1],
                mt[i, 2],
                marker="*",
                color=COLORS["post_opt_1"],
                s=6,
                alpha=a,
            )
            ax.scatter(
                mm[i, 0],
                mm[i, 1],
                mm[i, 2],
                marker="o",
                facecolors="none",
                edgecolors=COLORS["ground_truth"],
                s=6,
                alpha=a,
            )

    ax.set_xlabel(AXIS["x_axis_label"], labelpad=-3)
    ax.set_ylabel(AXIS["y_axis_label"], labelpad=-5)
    ax.set_zlabel(AXIS["z_axis_label"], labelpad=-3)
    ax.view_init(elev=20, azim=15)

    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    legend_elems = [
        Line2D(
            [0],
            [0],
            color=COLORS["pre_opt_1"],
            lw=0.8,
            # alpha=alpha_orig,
            label="Backbone (Init)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["post_opt_1"],
            lw=0.8,
            # alpha=alpha_tau_star,
            label="Backbone (Opt)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["pre_opt_1"],
            markeredgecolor=COLORS["pre_opt_1"],
            markersize=5,
            label="Pred. marks (Init)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["post_opt_1"],
            markeredgecolor=COLORS["post_opt_1"],
            markersize=5,
            label="Pred. marks (Opt)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor=COLORS["ground_truth"],
            markerfacecolor="none",
            markersize=5,
            label="Measured marks",
        ),
    ]

    legend_elems = [
        Line2D(
            [0],
            [0],
            color=COLORS["pre_opt_1"],
            lw=2.0,
            # alpha=0.25,
            label="Initial backbones",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["post_opt_1"],
            lw=2.0,
            # alpha=1.0,
            label="Optimized backbones",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["pre_opt_1"],
            markeredgecolor=COLORS["pre_opt_1"],
            markersize=7,
            label="Initial predicted markers",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["post_opt_1"],
            markeredgecolor=COLORS["post_opt_1"],
            markersize=7,
            label="Optimized predicted markers",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor=COLORS["ground_truth"],
            markerfacecolor="none",
            markersize=7,
            label="Measured markers",
        ),
    ]

    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=-2)
    ax.tick_params(axis="z", pad=-1)

    # ax.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.legend(handles=legend_elems, loc=legend_loc)
    return fig


# --- Main Execution / I-O ---


def save_figure(
    fig: plt.Figure, output_dir: Path, filename: str, output_format: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.{output_format}"
    fig.savefig(filepath, dpi=300, pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {filepath.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    # 1. Load Radar Data
    try:
        rmse_data = RmseData(
            initial=np.load(args.data_dir / "errnorm_before_id.npy"),
            optimal=np.load(args.data_dir / "errnorm_zero.npy"),
            residual=np.load(args.data_dir / "errnorm_star.npy"),
            nn_pred=np.load(args.data_dir / "errnorm_pred.npy"),
        )
    except FileNotFoundError as e:
        print(f"Error loading RMSE radar data: {e}")
        return 1

    # 2. Load Bar Chart Data (Overall RMSE)
    try:
        overall_rmse_arr = np.load(args.data_dir / "rmse_overall.npy")
        overall_rmse_dict = {
            "Initial": float(overall_rmse_arr[0]),
            "Opt. Param": float(overall_rmse_arr[1]),
            "Opt. Resid": float(overall_rmse_arr[2]),
            "NN Pred": float(overall_rmse_arr[3]),
        }
    except FileNotFoundError as e:
        print(f"Error loading overall RMSE data: {e}")
        return 1

    # 3. Load Violin Plot Data
    try:
        df_violin = pd.read_csv(args.data_dir / "violin_data.csv")
    except FileNotFoundError as e:
        print(f"Error loading violin dataframe: {e}")
        return 1

    # 4. Load 3D Shape Data
    try:
        shape_data = ShapeComparisonData(
            curves_orig=np.load(args.data_dir / "curves_orig.npy")[::2],
            curves_tau_star=np.load(args.data_dir / "curves_tau_star.npy")[::2],
            markers_orig=np.load(args.data_dir / "markers_orig.npy")[::2],
            markers_tau_star=np.load(args.data_dir / "markers_tau_star.npy")[::2],
            measured_pts=np.load(args.data_dir / "measured_pts.npy")[::2],
        )
    except FileNotFoundError as e:
        print(f"Error loading 3D shape arrays: {e}")
        return 1

    # Generate and save all figures
    # Radar Plot
    fig_radar = plot_radar_rmse(rmse_data, (8, 8))
    save_figure(fig_radar, args.output_dir, "radar_rmse_per_marker", "pdf")

    # Bar Plot
    fig_bar = plot_bar_overall_rmse(overall_rmse_dict, figsize_cm=(6.5, 5.0), box=True)
    save_figure(fig_bar, args.output_dir, "bar_overall_rmse", "pdf")

    # Violin Plot
    fig_violin = plot_violin_error_distribution(
        df_violin, figsize_cm=(15.0, 10.0), legend_loc="upper right", box=True
    )
    save_figure(fig_violin, args.output_dir, "violin_error_distribution", "pdf")

    # 3D Comparison
    fig_3d = plot_3d_comparison(shape_data, (8, 8), legend_loc="upper left")
    save_figure(fig_3d, args.output_dir, "3d_shape_comparison", "pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
