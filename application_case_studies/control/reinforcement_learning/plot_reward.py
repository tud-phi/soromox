#!/usr/bin/env python3
"""Plot reward curves from local CSV exports.

By default this script reads all ``*.csv`` files from ``reward_logs``.
Each CSV is expected to contain ``wall_time`` and ``episode_reward_mean`` columns.
Files are grouped by their names, for example
``SoRoMoX_64_envs_1bqkkr1d.csv`` is grouped as ``SoRoMoX 64 envs``.
"""

import argparse
import csv
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REWARD_KEY = "episode_reward_mean"
TIME_KEY = "wall_time"


@dataclass(frozen=True)
class Curve:
    source: Path
    time_seconds: np.ndarray
    reward: np.ndarray


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


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


def label_from_csv(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^SoRoMoX_(\d+)_envs(?:_.+)?$", stem)
    if match:
        return f"SoRoMoX {match.group(1)} envs"
    if stem.startswith("PyElastica"):
        return "PyElastica"
    return " ".join(stem.split("_"))


def discover_csv_groups(csv_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        groups.setdefault(label_from_csv(path), []).append(path)
    if not groups:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    return groups


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

    return clean_curve(path, times, rewards) if times else None


def clean_curve(
    source: Path, times: Iterable[float], rewards: Iterable[float]
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
    return Curve(source=source, time_seconds=unique_times, reward=unique_rewards)


def load_curve(path: Path) -> Curve:
    curve = read_reward_csv(path)
    if curve is None or len(curve.time_seconds) < 2:
        raise RuntimeError(f"Could not load at least two valid records from {path}")
    print(
        "record "
        f"{path.name}: points={len(curve.time_seconds)}, "
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
    smooth_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(curve.time_seconds[0] for curve in curves)
    end = min(curve.time_seconds[-1] for curve in curves)
    if end <= start:
        raise ValueError("Runs do not have overlapping wall-time ranges.")

    grid = np.linspace(start, end, num_points)
    interpolated = np.vstack(
        [
            np.interp(
                grid,
                curve.time_seconds,
                gaussian_smooth(curve.reward, smooth_scale),
            )
            for curve in curves
        ]
    )
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return grid / 60.0, mean, std


def plot(groups: dict[str, list[Path]], args: argparse.Namespace) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.8, 3.8), constrained_layout=True)

    colors = {
        "PyElastica": "#0072B2",
        "SoRoMoX 64 envs": "#D55E00",
        "SoRoMoX 128 envs": "#009E73",
        "SoRoMoX 256 envs": "#E69F00",
        "SoRoMoX 512 envs": "#CC79A7",
    }

    for label, csv_paths in groups.items():
        curves = [load_curve(path) for path in csv_paths]
        x_min, mean, std = aggregate_curves(curves, args.points, args.smooth_scale)
        color = colors.get(label)
        ax.plot(x_min, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(
            x_min,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Training time (min)")
    ax.set_ylabel("Reward")
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


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
        default=SCRIPT_DIR / "reward_logs",
        help="Directory containing reward CSV exports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "visualizations" / "reward_curve.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=500,
        help="Interpolation points per averaged curve.",
    )
    parser.add_argument(
        "--smooth-scale",
        type=float,
        default=0.5,
        help="Gaussian smoothing scale, similar to W&B smoothing. Use 0 for no smoothing.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    groups = (
        parse_note(args.note, args.csv_dir)
        if args.note is not None
        else discover_csv_groups(args.csv_dir)
    )
    plot(groups, args)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
