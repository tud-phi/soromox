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
    ax.grid(True)
    ax.text(
        0.02,
        0.95,
        rf"$\mathbf{{{panel}}}$",
        transform=ax.transAxes,
        fontsize=18,
        va="top",
    )


def plot_loss(
    ax: plt.Axes, history_loss: np.ndarray, finite_mask: np.ndarray, panel: str
) -> None:
    """Plot the spread across starts with the best loss per iteration on top.

    Starts frozen by ``history_finite_mask`` are dropped from the band rather
    than drawn as gaps. A single-start archive gets no band, since there is no
    spread to shade.
    """
    loss = np.where(np.asarray(finite_mask, dtype=bool), history_loss, np.nan)
    iterations = np.arange(1, loss.shape[0] + 1)
    if loss.shape[1] > 1:
        ax.fill_between(
            iterations,
            np.nanmin(loss, axis=1),
            np.nanmax(loss, axis=1),
            color=COLORS["pre_opt_1"],
            alpha=0.2,
            linewidth=0,
        )
    ax.plot(
        iterations,
        np.nanmin(loss, axis=1),
        "-o",
        color=COLORS["pre_opt_1"],
        markersize=3,
    )
    ax.set(xlabel="Iterations", ylabel="Loss", xlim=(1, max(1, len(iterations))))
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
                alpha=0.15,
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
                Line2D([], [], color="k", linestyle="-", linewidth=1.25),
            ],
            ["Target", "Initial (median)", "Optimized (best start)"],
            loc="best",
        )
    ax.set(xlabel="Time [s]", ylabel=ylabel, xlim=(float(t[0]), float(t[-1])))
    _format_axes(ax, panel)


def build_figure(data_dir: Path) -> plt.Figure:
    """Assemble the four-panel Section Vd comparison figure.

    Each panel takes its best start from **its own** archive. Issue #128 was one
    index derived from the collocated losses and reused for the synergistic
    panel; it went unnoticed only because both methods happened to peak at start
    3 in the legacy data.
    """
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
    # No controller-specific divisor: both losses are plotted as optimized.
    plot_loss(
        axes[0, 0], collocated["history_loss"], collocated["history_finite_mask"], "A"
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
        axes[1, 0], synergistic["history_loss"], synergistic["history_finite_mask"], "C"
    )
    # Pose layout is [rotation-vector xyz, Cartesian position xyz].
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [
        args.output_dir / "plot_control_gain_opt.pdf",
        args.output_dir / "plot_control_gain_opt.png",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")
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
