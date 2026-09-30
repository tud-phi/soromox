#!/usr/bin/env python3
"""Plot the sequential CPU computation times reported in paper Table IV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR.parent))
from paper_style import PAPER_COLORS, PAPER_STYLE_PATH


def plot_benchmark_cpu_runtime(data: dict) -> plt.Figure:
    """Show reported runtimes on a zero-based linear axis, with speedup ratios."""
    cases = data["cases"]
    values = np.array(
        [[case["sorosim_s"], case["soromox_s"]] for case in cases], dtype=float
    )
    if not len(cases) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Runtime values must be finite and positive.")

    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "text.usetex": True,
            "svg.fonttype": "path",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.3), layout="constrained")
    x = np.arange(len(cases))
    width = 0.32
    styles = (
        ("SoRoSim", PAPER_COLORS["pre_opt_1"]),
        ("SoRoMoX", PAPER_COLORS["post_opt_1"]),
    )
    for index, (label, color) in enumerate(styles):
        bars = ax.bar(
            x + (index - 0.5) * (width + 0.035),
            values[:, index],
            width,
            label=label,
            color=color,
            edgecolor="white",
            zorder=3,
        )
        ax.bar_label(bars, fmt=r"$%.2f$", padding=4, fontsize=9)

    max_runtime = float(values.max())
    for xpos, speedup in zip(x, values[:, 0] / values[:, 1]):
        ax.text(
            xpos,
            max_runtime * 1.19,
            rf"${speedup:.1f}\times$ faster",
            ha="center",
            va="center",
            fontsize=10,
        )

    labels = [
        case["label"].replace(" PCS", "\nPCS").replace(" GVS", "\nGVS")
        for case in cases
    ]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel(r"Wall-clock time $\mathrm{[s]}$")
    ax.set_ylim(0, max_runtime * 1.30)
    ax.set_xlim(-0.55, len(cases) - 0.45)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=CASE_DIR / "data" / "benchmark_cpu_runtime.json"
    )
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "outputs")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing figures."
    )
    args = parser.parse_args()
    outputs = [
        args.output_dir / f"benchmark_cpu_runtime.{ext}"
        for ext in ("pdf", "svg", "png")
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        parser.error("Outputs already exist; pass --force to replace them.")
    data = json.loads(args.data.read_text())
    fig = plot_benchmark_cpu_runtime(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
