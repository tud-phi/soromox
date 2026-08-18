"""Create the canonical Section Vd comparison figure from schema-v1 archives."""

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
from secvd_results import load_results  # noqa: E402

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


def plot_loss(ax: plt.Axes, history_loss: np.ndarray, panel: str) -> None:
    """Plot the one recorded run; B=1 intentionally has no artificial band."""
    if history_loss.ndim != 2 or history_loss.shape[1] != 1:
        raise ValueError(
            f"Expected single-start loss with shape (iterations, 1), got {history_loss.shape}"
        )
    iterations = np.arange(1, history_loss.shape[0] + 1)
    ax.plot(
        iterations, history_loss[:, 0], "-o", color=COLORS["pre_opt_1"], markersize=3
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
    colors = [COLORS["x_t"], COLORS["y_t"], COLORS["z_t"]]
    for channel, color in enumerate(colors):
        ax.plot(t, target[:, channel], "--", linewidth=2, color=color)
        ax.plot(t, initial[:, channel], "-.", linewidth=1.5, color=color)
        ax.plot(t, best[:, channel], "-", linewidth=1.25, color=color)
    if legend:
        ax.legend(
            [
                Line2D([], [], color="k", linestyle="--", linewidth=2),
                Line2D([], [], color="k", linestyle="-.", linewidth=1.5),
                Line2D([], [], color="k", linestyle="-", linewidth=1.25),
            ],
            ["Target", "Initial", "Optimized"],
            loc="best",
        )
    ax.set(xlabel="Time [s]", ylabel=ylabel, xlim=(float(t[0]), float(t[-1])))
    _format_axes(ax, panel)


def build_figure(data_dir: Path) -> plt.Figure:
    collocated = load_results(data_dir / "collocated", expected_method="collocated")
    synergistic = load_results(data_dir / "synergistic", expected_method="synergistic")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_SIZE_CM[0] / 2.54, FIGURE_SIZE_CM[1] / 2.54),
        constrained_layout=True,
    )
    plot_loss(axes[0, 0], collocated["history_loss"] / 100.0, "A")
    _plot_three_channels(
        axes[0, 1],
        collocated["t_ts"][0],
        collocated["q_ts_init"][0, :, :3],
        collocated["q_ts_best"][0, :, :3],
        collocated["q_des_ts"][0, :, :3],
        ylabel="Angular strains [rad/m]",
        panel="B",
        legend=True,
    )
    plot_loss(axes[1, 0], synergistic["history_loss"], "C")
    # Pose layout is [rotation-vector xyz, Cartesian position xyz].
    _plot_three_channels(
        axes[1, 1],
        synergistic["t_ts"][0],
        synergistic["x_ts_init"][0, :, 3:6],
        synergistic["x_ts_best"][0, :, 3:6],
        synergistic["x_des_ts"][0, :, 3:6],
        ylabel="End-effector position [m]",
        panel="D",
        legend=False,
    )
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
