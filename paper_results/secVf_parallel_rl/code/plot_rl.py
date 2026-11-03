#!/usr/bin/env python3
"""Plot reward curves from local CSV exports."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PAPER_RESULTS_DIR = SCRIPT_DIR.parent
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))
PAPER_STYLE = PAPER_STYLE_PATH

# --- Configuration Constants ---
REWARD_KEY = "episode_reward_mean"
TIME_KEY = "wall_time"
FIGURE_SIZE_CM = (12.0, 12.0)  # Matches original 3.8 x 3.8 inches

# Subfolders of --csv-dir plotted by default. The device tag in the legend is
# taken from the folder name: the GPU model for GPU folders, otherwise CPU.
REWARD_LOG_SUBDIRS = ("PyElastica-CPU", "SoRoMoX-CPU", "SoRoMoX-GPU-RTX4070")
# Legend names of the GPU models in GPU folder names (SoRoMoX-GPU-<model>).
GPU_NAMES = {"RTX4070": "RTX 4070"}
# PyElastica and SoRoMoX-CPU file names carry no env count; those runs used 2 envs.
CPU_NUM_ENVS = 2
# Completed episodes averaged in one CSV record. SoRoMoX train.py writes one
# record per --reward-batch-episodes (default 8) episodes; the PyElastica runs
# logged one record per vector step, i.e. the 2 episodes that ended together.
EPISODES_PER_RECORD = {"PyElastica-CPU": 2}
DEFAULT_EPISODES_PER_RECORD = 8
# W&B-style plot: "Random sampling" point aggregation keeps at most this many
# records per run (fixed seed), then Gaussian smoothing over neighbouring
# records with the W&B smoothing value.
WANDB_SAMPLE_POINTS = 1500
WANDB_SAMPLE_SEED = 0
WANDB_SMOOTHING = 10.0

COLORS_RL = {
    "PyElastica (CPU, 2 envs)": COLORS["pre_opt_1"],
    "SoRoMoX (CPU, 2 envs)": COLORS["target"],
    "SoRoMoX (RTX 4070, 64 envs)": COLORS["pre_opt_2"],
    "SoRoMoX (RTX 4070, 128 envs)": COLORS["post_opt_1"],
    "SoRoMoX (RTX 4070, 256 envs)": COLORS["post_opt_2"],
    "SoRoMoX (RTX 4070, 512 envs)": COLORS["post_opt_3"],
}


@dataclass(frozen=True)
class Curve:
    source: Path
    time_seconds: np.ndarray
    reward: np.ndarray
    episodes: np.ndarray  # Completed episodes averaged in each point.


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
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


def parse_note(note_path: Path, csv_dir: Path) -> dict[str, list[Path]]:
    if not note_path.exists():
        return discover_csv_groups(csv_dir)

    groups: dict[str, list[Path]] = {}
    current_label: str | None = None
    label_re = re.compile(r"^\s*(.+?)\s+record\s+\d+\s*:\s*$", re.IGNORECASE)
    csv_re = re.compile(r"^\s*([\w./-]+\.csv)\s*$")
    run_re = re.compile(r"^\s*[\w.-]+/[\w.-]+/([A-Za-z0-9_-]+)\s*$")

    for line in note_path.read_text(encoding="utf-8").splitlines():
        label_match = label_re.match(line)
        if label_match:
            current_label = normalize_label(label_match.group(1))
            groups.setdefault(current_label, [])
            continue

        csv_match = csv_re.match(line)
        if csv_match and current_label:
            path = Path(csv_match.group(1))
            groups[current_label].append(path if path.is_absolute() else csv_dir / path)
            continue

        run_match = run_re.match(line)
        if run_match and current_label:
            matches = sorted(csv_dir.glob(f"*_{run_match.group(1)}.csv"))
            groups[current_label].extend(matches)

    groups = {label: paths for label, paths in groups.items() if paths}
    return groups or discover_csv_groups(csv_dir)


def normalize_label(raw: str) -> str:
    text = " ".join(raw.strip().split())
    lower = text.lower()
    if lower == "elastica":
        return "PyElastica"
    if lower.startswith("soromox"):
        number = re.search(r"\d+", text)
        return f"SoRoMoX {number.group(0)} envs" if number else "SoRoMoX"
    return text


def device_from_dir(path: Path) -> str | None:
    folder = path.parent.name.upper()
    if "GPU" in folder:
        model = path.parent.name.partition("GPU-")[2]
        return GPU_NAMES.get(model, model) or "GPU"
    if "CPU" in folder:
        return "CPU"
    return None


def label_from_csv(path: Path) -> str:
    stem = path.stem
    device = device_from_dir(path)
    if stem.startswith("PyElastica"):
        if device == "CPU":
            return f"PyElastica ({device}, {CPU_NUM_ENVS} envs)"
        return f"PyElastica ({device})" if device else "PyElastica"
    if stem.startswith("SoRoMoX"):
        match = re.match(r"^SoRoMoX_(\d+)_envs(?:_.+)?$", stem)
        if match:
            num_envs = match.group(1)
        elif device == "CPU":
            num_envs = str(CPU_NUM_ENVS)
        else:
            num_envs = None
        details = [part for part in (device, num_envs and f"{num_envs} envs") if part]
        return f"SoRoMoX ({', '.join(details)})" if details else "SoRoMoX"
    return " ".join(stem.split("_"))


def discover_csv_groups(csv_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    subdirs = [csv_dir / name for name in REWARD_LOG_SUBDIRS if (csv_dir / name).is_dir()]
    for folder in subdirs or [csv_dir]:
        for path in sorted(folder.glob("*.csv")):
            groups.setdefault(label_from_csv(path), []).append(path)
    if not groups:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    return groups


def episodes_per_record(path: Path) -> int:
    return EPISODES_PER_RECORD.get(path.parent.name, DEFAULT_EPISODES_PER_RECORD)


def read_reward_csv(path: Path) -> Curve | None:
    if not path.exists():
        return None

    times: list[float] = []
    rewards: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row[TIME_KEY])
                r = float(row[REWARD_KEY])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(r):
                times.append(t)
                rewards.append(r)

    return clean_curve(path, times, rewards, episodes_per_record(path)) if times else None


def clean_curve(
    source: Path,
    times: Iterable[float],
    rewards: Iterable[float],
    record_episodes: int = 1,
) -> Curve:
    data = sorted(
        (float(t), float(r))
        for t, r in zip(times, rewards)
        if np.isfinite(t) and np.isfinite(r)
    )
    if not data:
        raise ValueError(f"No valid data points for {source}")

    by_time: dict[float, list[float]] = {}
    for t, r in data:
        by_time.setdefault(t, []).append(r)

    unique_times = np.array(sorted(by_time), dtype=float)
    unique_rewards = np.array([np.mean(by_time[t]) for t in unique_times], dtype=float)
    episodes = np.array([len(by_time[t]) * record_episodes for t in unique_times])
    return Curve(
        source=source,
        time_seconds=unique_times,
        reward=unique_rewards,
        episodes=episodes,
    )


def bin_by_episodes(curve: Curve, episodes_per_point: int) -> Curve:
    """Merge consecutive points until each averages ``episodes_per_point`` episodes.

    Time and reward of a merged point are episode-weighted means. A trailing
    partial bin is dropped so that every point averages the same number of
    episodes; a single point that already holds more episodes is kept as is.
    """
    if episodes_per_point <= 0:
        return curve

    times: list[float] = []
    rewards: list[float] = []
    episodes: list[int] = []
    t_sum = r_sum = 0.0
    n = 0
    for t, r, k in zip(curve.time_seconds, curve.reward, curve.episodes):
        t_sum += t * k
        r_sum += r * k
        n += k
        if n >= episodes_per_point:
            times.append(t_sum / n)
            rewards.append(r_sum / n)
            episodes.append(n)
            t_sum = r_sum = 0.0
            n = 0
    return Curve(
        source=curve.source,
        time_seconds=np.array(times, dtype=float),
        reward=np.array(rewards, dtype=float),
        episodes=np.array(episodes),
    )


def load_wandb_style_curve(
    path: Path, rng: np.random.Generator
) -> Curve:
    """Mimic a W&B line plot with random sampling and Gaussian smoothing.

    Records are kept in logged order, randomly subsampled to at most
    ``WANDB_SAMPLE_POINTS`` like W&B's "Random sampling" point aggregation, and then
    smoothed over neighbouring records. Records sharing a wall time (GPU runs)
    are merged only after smoothing so that time interpolation stays defined.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            (float(row[TIME_KEY]), float(row[REWARD_KEY]))
            for row in csv.DictReader(f)
            if row.get(TIME_KEY) and row.get(REWARD_KEY)
        ]
    if WANDB_SAMPLE_POINTS < len(rows):
        keep = np.sort(rng.choice(len(rows), size=WANDB_SAMPLE_POINTS, replace=False))
        rows = [rows[i] for i in keep]
    times = np.array([t for t, _ in rows], dtype=float)
    rewards = gaussian_smooth(np.array([r for _, r in rows], dtype=float), WANDB_SMOOTHING)
    curve = clean_curve(path, times, rewards)
    print(
        f"Loaded {path.name}: records={len(rows)}, points={len(curve.time_seconds)}, "
        f"last_wall_time={curve.time_seconds[-1]:.3f}s, "
        f"last_reward={curve.reward[-1]:.6g}"
    )
    return curve


def load_curve(path: Path, episodes_per_point: int = 0) -> Curve:
    curve = read_reward_csv(path)
    if curve is not None:
        curve = bin_by_episodes(curve, episodes_per_point)
    if curve is None or len(curve.time_seconds) < 2:
        raise RuntimeError(f"Could not load at least two valid records from {path}")
    print(
        f"Loaded {path.name}: points={len(curve.time_seconds)}, "
        f"episodes/point={int(np.median(curve.episodes))}, "
        f"last_wall_time={curve.time_seconds[-1]:.3f}s, "
        f"last_reward={curve.reward[-1]:.6g}"
    )
    return curve


def gaussian_smooth(values: np.ndarray, scale: float) -> np.ndarray:
    """Apply a W&B-like Gaussian smoothing kernel to one reward curve."""
    if scale <= 0:
        return values

    radius = max(1, int(np.ceil(3.0 * scale)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / scale) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def aggregate_curves(
    curves: list[Curve],
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(curve.time_seconds[0] for curve in curves)
    end = min(curve.time_seconds[-1] for curve in curves)
    if end <= start:
        raise ValueError("Runs do not have overlapping wall-time ranges.")

    grid = np.linspace(start, end, num_points)
    interpolated = np.vstack(
        [
            np.interp(grid, curve.time_seconds, curve.reward)
            for curve in curves
        ]
    )
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return grid / 60.0, mean, std


def build_figure(
    groups: dict[str, list[Path]], args: argparse.Namespace, *, ax=None
) -> plt.Figure:
    """Construct the figure and axes, applying all data plots."""
    if ax is None:
        fig, ax = plt.subplots(figsize=cm2inch(FIGURE_SIZE_CM), constrained_layout=True)
    else:
        fig = ax.figure

    global_max_x = 0.0
    rng = np.random.default_rng(WANDB_SAMPLE_SEED)

    ordered_labels = sorted(
        groups.keys(),
        key=lambda k: list(COLORS_RL.keys()).index(k) if k in COLORS_RL else 999,
    )

    for label in ordered_labels:
        csv_paths = groups[label]
        if args.episodes_per_point > 0:
            curves = [load_curve(path, args.episodes_per_point) for path in csv_paths]
        else:
            curves = [
                load_wandb_style_curve(path, rng) for path in csv_paths
            ]
        x_min, mean, std = aggregate_curves(curves, args.points)

        # Fallback to dark gray if a group label is missing from the global COLORS_RL dict
        color = COLORS_RL.get(label, "#333333")

        ax.plot(x_min, mean, label=label, color=color)
        ax.fill_between(
            x_min,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        # Update the maximum x-value found so far
        if x_min[-1] > global_max_x:
            global_max_x = x_min[-1]

    ax.set_xlim(0, global_max_x)
    ax.set_ylim(-60.0, 150.0)
    ax.set_xlabel("Training time [min]")
    ax.set_ylabel("Reward")
    ax.legend(loc="lower right")

    return fig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note",
        type=Path,
        default=None,
        help="Optional note.txt listing CSV records or W&B run ids.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "reward_logs",
        help=(
            "Directory containing reward CSV exports; its PyElastica-CPU, SoRoMoX-CPU "
            "and SoRoMoX-GPU-RTX4070 subfolders are used when present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "reward_curve.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=500,
        help="Interpolation points per averaged curve.",
    )
    parser.add_argument(
        "--episodes-per-point",
        type=int,
        default=0,
        help=(
            "Instead of the default W&B-style plot, merge consecutive reward records "
            "so that every plotted point averages this many completed episodes, "
            "giving all curves the same noise level (e.g. 512, which matches the "
            "512-env runs whose records end together)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively after saving.",
    )
    return parser


def main() -> int:
    configure_matplotlib()
    args = build_argparser().parse_args()

    groups = (
        parse_note(args.note, args.csv_dir)
        if args.note is not None
        else discover_csv_groups(args.csv_dir)
    )

    fig = build_figure(groups, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved {args.output.resolve()}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
