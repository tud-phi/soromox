"""Create the canonical Section Vd comparison figure from schema-v3 archives."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
PAPER_RESULTS_DIR = CASE_DIR.parent
for path in (CODE_DIR, PAPER_RESULTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_style import PAPER_COLORS as COLORS  # noqa: E402
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE  # noqa: E402
from secvd_results import improvement_over_initial_median, load_results  # noqa: E402

FIGURE_SIZE_CM = (28.0, 16.0)

# Seconds of the rollout the trajectory panels display
TRAJECTORY_WINDOW_S = 2.0

# Opacity of the across-start range bands
BAND_ALPHA = 0.3

# Multiplicative padding on the shared loss limits
LOSS_YLIM_PAD = 1.1


def configure_matplotlib() -> None:
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "figure.dpi": 300,
        }
    )
    if _latex_is_usable():
        plt.rcParams.update({"text.usetex": True})


def _latex_is_usable() -> bool:
    latex, kpsewhich = shutil.which("latex"), shutil.which("kpsewhich")
    if latex is None or kpsewhich is None:
        return False
    result = subprocess.run(
        [kpsewhich, "type1ec.sty"], capture_output=True, check=False, text=True
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _format_axes(ax: plt.Axes, panel: str) -> None:
    """Grid the axes and label the panel outside its top-left corner."""
    ax.grid(True)
    ax.annotate(
        rf"$\mathbf{{{panel}}}$",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-30, -15),
        textcoords="offset points",
        fontsize=18,
        va="bottom",
        ha="left",
        annotation_clip=False,
    )


def loss_extent(history_loss: np.ndarray, finite_mask: np.ndarray):
    """Return the smallest and largest loss actually drawn, or ``None``."""
    loss = np.where(np.asarray(finite_mask, dtype=bool), history_loss, np.nan)
    if not np.any(np.isfinite(loss)):
        return None
    return float(np.nanmin(loss)), float(np.nanmax(loss))


def shared_loss_limits(*extents) -> tuple[float, float] | None:
    """Combine per-panel extents into one padded limit for both loss panels."""
    present = [e for e in extents if e is not None]
    if not present:
        return None
    low = min(lo for lo, _ in present) / LOSS_YLIM_PAD
    high = max(hi for _, hi in present) * LOSS_YLIM_PAD
    if not low < high:
        # A degenerate span, which an archive of constant loss reaches whenever
        # LOSS_YLIM_PAD is 1.0. Matplotlib would expand it itself and warn; let
        # it autoscale deliberately instead.
        return None
    return low, high


def plot_loss(
    ax: plt.Axes,
    history_loss: np.ndarray,
    finite_mask: np.ndarray,
    panel: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot the spread across starts with the best loss per iteration on top.

    Args:
        ylim: Shared limits for both loss panels; ``None`` autoscales.
    """
    loss = np.where(np.asarray(finite_mask, dtype=bool), history_loss, np.nan)
    iterations = np.arange(1, loss.shape[0] + 1)
    if loss.shape[1] > 1:
        ax.fill_between(
            iterations,
            np.nanmin(loss, axis=1),
            np.nanmax(loss, axis=1),
            color=COLORS["pre_opt_1"],
            alpha=BAND_ALPHA,
            linewidth=0,
        )
    ax.plot(
        iterations,
        np.nanmin(loss, axis=1),
        "-o",
        color=COLORS["pre_opt_1"],
        markersize=3,
    )
    ax.set_yscale("log")
    ax.set(xlabel="Iterations", ylabel="Loss", xlim=(1, max(1, len(iterations))))
    if ylim is not None:
        ax.set_ylim(*ylim)
    _format_axes(ax, panel)


def _plot_three_channels(
    ax: plt.Axes,
    t: np.ndarray,
    initial: np.ndarray,
    best: np.ndarray,
    target: np.ndarray,
    *,
    ylabel: str,
    panel: str,
    legend: bool,
) -> None:
    """Draw target, the initial spread across starts, and the best start.

    Args:
        ax: Axes to draw on.
        t: Shared time grid, shape ``(timesteps,)``.
        initial: Every start's initial rollout, shape ``(B, timesteps, 3)``.
        best: The best start's optimized rollout, shape ``(timesteps, 3)``.
        target: The reference, shape ``(timesteps, 3)``.
        ylabel: Axis label.
        panel: Panel letter.
        legend: Whether to draw the line-style legend.
    """
    colors = [COLORS["x_t"], COLORS["y_t"], COLORS["z_t"]]
    for channel, color in enumerate(colors):
        ax.plot(t, target[:, channel], "--", linewidth=2, color=color)
        if initial.shape[0] > 1:
            ax.fill_between(
                t,
                initial[:, :, channel].min(axis=0),
                initial[:, :, channel].max(axis=0),
                color=color,
                alpha=BAND_ALPHA,
                linewidth=0,
            )
        ax.plot(
            t,
            np.median(initial[:, :, channel], axis=0),
            "-.",
            linewidth=1.5,
            color=color,
        )
        ax.plot(t, best[:, channel], "-", linewidth=1.25, color=color)
    if legend:
        ax.legend(
            [
                Line2D([], [], color="k", linestyle="--", linewidth=2),
                Line2D([], [], color="k", linestyle="-.", linewidth=1.5),
                # Patch(facecolor="k", alpha=BAND_ALPHA, linewidth=0),
                Line2D([], [], color="k", linestyle="-", linewidth=1.25),
            ],
            [
                "Target",
                "Initial (median)",
                # "Initial (range over starts)",
                "Optimized (best)",
            ],
            loc="best",
        )
    ax.set(
        xlabel="Time [s]",
        ylabel=ylabel,
        xlim=(float(t[0]), min(float(t[-1]), TRAJECTORY_WINDOW_S)),
    )
    _format_axes(ax, panel)


def build_figure(data_dir: Path) -> plt.Figure:
    """Assemble the four-panel Section Vd comparison figure."""
    collocated = load_results(data_dir / "collocated", expected_method="collocated")
    synergistic = load_results(data_dir / "synergistic", expected_method="synergistic")
    best_batch_c = int(collocated["best_batch"])
    best_batch_s = int(synergistic["best_batch"])
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_SIZE_CM[0] / 2.54, FIGURE_SIZE_CM[1] / 2.54),
        constrained_layout=True,
    )
    loss_ylim = shared_loss_limits(
        loss_extent(collocated["history_loss"], collocated["history_finite_mask"]),
        loss_extent(synergistic["history_loss"], synergistic["history_finite_mask"]),
    )
    plot_loss(
        axes[0, 0],
        collocated["history_loss"],
        collocated["history_finite_mask"],
        "A",
        ylim=loss_ylim,
    )
    _plot_three_channels(
        axes[0, 1],
        collocated["t_ts"],
        collocated["q_ts_init"][:, :, :3],
        collocated["q_ts_best"][best_batch_c, :, :3],
        collocated["q_des_ts"][:, :3],
        ylabel="Angular strains [rad/m]",
        panel="B",
        legend=True,
    )
    plot_loss(
        axes[1, 0],
        synergistic["history_loss"],
        synergistic["history_finite_mask"],
        "C",
        ylim=loss_ylim,
    )
    _plot_three_channels(
        axes[1, 1],
        synergistic["t_ts"],
        synergistic["x_ts_init"][:, :, 3:6],
        synergistic["x_ts_best"][best_batch_s, :, 3:6],
        synergistic["x_des_ts"][:, 3:6],
        ylabel="End-effector position [m]",
        panel="D",
        legend=False,
    )
    print("Section Vd summary:")
    for name, data in (("collocated", collocated), ("synergistic", synergistic)):
        improvement = improvement_over_initial_median(
            data["history_loss"], data["history_finite_mask"]
        )
        print(
            f"  {name:12s} improvement over the initial median: "
            f"{100 * improvement:5.2f} %"
        )
    print(f"  best start: collocated {best_batch_c}, synergistic {best_batch_s}")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CASE_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "outputs")
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing outputs (default). --no-force refuses instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [
        args.output_dir / "plot_control_gain_opt.pdf",
        args.output_dir / "plot_control_gain_opt.png",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {existing}; drop --no-force to replace them"
        )
    configure_matplotlib()
    figure = build_figure(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputs[0], dpi=300)
    figure.savefig(outputs[1], dpi=300)
    plt.close(figure)
    print("Generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
