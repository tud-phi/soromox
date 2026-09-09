#!/usr/bin/env python3
"""Assemble Figure 13 from saved safety/RL data and the current rollout videos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SAFETY = ROOT / "secVe_safety_constrained_control"
RL = ROOT / "secVf_parallel_rl"
for directory in (SAFETY / "code", RL / "code"):
    sys.path.insert(0, str(directory))

import plot_pcs_with_cf_cbf_clf as safety_plot  # noqa: E402
import plot_rl as rl_plot  # noqa: E402
from pcs_cf_cbf_clf_common import load_results  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT / "final_outputs" / "control_barrier_function_reinforcement_learning"
)


def extract_frame(video: Path, time: float) -> np.ndarray:
    """Decode one frame without creating intermediate files."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(time),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"No frame at {time:g} s in {video}")
    return np.asarray(Image.open(io.BytesIO(result.stdout)).convert("RGB"))


def shared_crop(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Keep a common scale and crop across both controllers and all times."""
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError(
            "Videos in each comparison must have matching frame dimensions"
        )
    bounds = []
    for frame in frames:
        rgb = frame.astype(float) / 255.0
        # Saturated robot/marker colors distinguish geometry from the grey cove.
        mask = (rgb.max(axis=2) - rgb.min(axis=2)) > 0.18
        y, x = np.where(mask)
        if not len(x):
            raise ValueError("No colored robot or markers found in a snapshot")
        bounds.append((x.min(), y.min(), x.max() + 1, y.max() + 1))
    bounds = np.asarray(bounds)
    left, top = bounds[:, :2].min(axis=0)
    right, bottom = bounds[:, 2:].max(axis=0)
    # Include the dark mounting plate and keep breathing room around all poses.
    margin = int(0.24 * (bottom - top))
    horizontal_margin = int(0.08 * (bottom - top))
    height, width = frames[0].shape[:2]
    # Fill the snapshot slot without turning narrow safety poses into thin strips.
    target_width = (2.0 / 3.0) * (min(height, bottom + margin) - max(0, top - margin))
    horizontal_margin = max(
        horizontal_margin, int(np.ceil((target_width - (right - left)) / 2))
    )
    return (
        max(0, left - horizontal_margin),
        max(0, top - margin),
        min(width, right + horizontal_margin),
        min(height, bottom + margin),
    )


def add_snapshots(fig, videos, times, row_bottoms, titles):
    """Draw two equally scaled rows of four snapshots and explicit timestamps."""
    rows = [[extract_frame(video, time) for time in times] for video in videos]
    left, top, right, bottom = shared_crop([frame for row in rows for frame in row])
    for row, y, title in zip(rows, row_bottoms, titles, strict=True):
        fig.text(0.235, y + 0.187, title, ha="center", fontsize=8)
        for index, (frame, time) in enumerate(zip(row, times, strict=True)):
            ax = fig.add_axes([0.02 + index * 0.108, y, 0.103, 0.177])
            ax.imshow(frame[top:bottom, left:right])
            ax.set_anchor("N")
            ax.set_axis_off()
            ax.text(
                0.5,
                -0.06,
                f"{time:g} s",
                transform=ax.transAxes,
                ha="center",
                fontsize=6,
                color="0.4",
            )


def build_figure(safety_dir: Path, rl_dir: Path, safety_times, rl_times):
    """Combine raster snapshots with vector plots using the standalone plotters."""
    fig = plt.figure(figsize=(18.1 / 2.54, 16.0 / 2.54))
    safety_videos = [
        safety_dir / "outputs" / f"rollout_{name}.mp4"
        for name in ("without_cbf", "with_cbf")
    ]
    rl_videos = [
        rl_dir / "outputs" / f"rl_rollout_{name}_1_env.mp4"
        for name in ("initialized", "trained")
    ]
    add_snapshots(
        fig,
        safety_videos,
        safety_times,
        (0.765, 0.545),
        ("HOCLF controller", "HOCLF+HOCBF controller"),
    )
    add_snapshots(
        fig,
        rl_videos,
        rl_times,
        (0.285, 0.065),
        ("Initialized RL controller", "Trained RL controller"),
    )
    force_ax = fig.add_axes([0.565, 0.635, 0.365, 0.315])
    safety_plot.build_figure(load_results("both", safety_dir / "data"), ax=force_ax)
    reward_ax = fig.add_axes([0.565, 0.085, 0.365, 0.415])
    groups = rl_plot.discover_csv_groups(rl_dir / "data" / "reward_logs")
    rl_plot.build_figure(
        groups, argparse.Namespace(points=500, smooth_scale=0.5), ax=reward_ax
    )
    for ax in fig.axes:
        if not ax.axison:
            continue
        for line in ax.lines:
            line.set_linewidth(1.1)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(6.5)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.1)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(4)
    for label, x, y in (
        ("A", 0.015, 0.983),
        ("B", 0.49, 0.983),
        ("C", 0.015, 0.503),
        ("D", 0.49, 0.533),
    ):
        fig.text(x, y, label, fontsize=13, fontweight="bold", va="top")
    return fig, safety_videos + rl_videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-dir", type=Path, default=SAFETY)
    parser.add_argument("--rl-dir", type=Path, default=RL)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--safety-times", type=float, nargs=4, default=(0.0, 0.75, 2.0, 7.9)
    )
    parser.add_argument(
        "--rl-times", type=float, nargs=4, default=(0.0, 5.0, 10.0, 15.0)
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    outputs = [
        args.output_base.with_suffix(ext) for ext in (".pdf", ".svg", ".png", ".json")
    ]
    if any(path.exists() for path in outputs) and not args.force:
        parser.error("Outputs already exist; pass --force to replace them")
    if any(
        time < 0 or not np.isfinite(time)
        for time in (*args.safety_times, *args.rl_times)
    ):
        parser.error("Snapshot times must be finite and nonnegative")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    with (
        plt.style.context(ROOT / "paper.mplstyle"),
        plt.rc_context(
            {
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "text.usetex": False,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
                "savefig.bbox": None,
                "svg.hashsalt": "soromox-safety-rl",
            }
        ),
    ):
        fig, videos = build_figure(
            args.safety_dir, args.rl_dir, args.safety_times, args.rl_times
        )
        try:
            for output in outputs[:3]:
                fig.savefig(output, dpi=300, facecolor="white", bbox_inches=None)
        finally:
            plt.close(fig)
    sources = (
        videos
        + sorted((args.safety_dir / "data").glob("rollout_*.npz"))
        + sorted((args.rl_dir / "data" / "reward_logs").glob("*.csv"))
    )
    svg = outputs[1]
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    outputs[3].write_text(
        json.dumps(
            {
                "safety_times_s": args.safety_times,
                "rl_times_s": args.rl_times,
                "reward_interpolation_points": 500,
                "reward_smoothing_scale": 0.5,
                "sources": [
                    {
                        "path": str(path.relative_to(ROOT))
                        if path.is_relative_to(ROOT)
                        else str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in sources
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
