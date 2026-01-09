#!/usr/bin/env python3
"""Visualise batch-scaling benchmark results stored in CSV format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

if shutil.which("latex"):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Romand"],
        }
    )


def _coerce(value: str) -> Any:
    """Best-effort numeric conversion for CSV entries."""

    try:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
    except AttributeError:
        return value

    for cast in (int, float):
        try:
            return cast(value)
        except (ValueError, TypeError):
            continue
    return value


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [{key: _coerce(val) for key, val in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _filter_rows(
    rows: Iterable[Mapping[str, Any]],
    systems: Sequence[str] | None,
    segment_counts: Sequence[int] | None,
) -> List[Mapping[str, Any]]:
    filtered: List[Mapping[str, Any]] = []
    for row in rows:
        if systems and row["system"] not in systems:
            continue
        if segment_counts and int(row["segment_count"]) not in segment_counts:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No rows remain after applying the requested filters.")
    return filtered


def _plot(
    rows: List[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    systems = sorted({row["system"] for row in rows})
    if sns is not None:
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        2,
        len(systems),
        figsize=(6 * len(systems), 8),
        squeeze=False,
        sharex="col",
    )

    for col, system in enumerate(systems):
        ax_per_env = axes[0, col]
        ax_total = axes[1, col]
        subset = [row for row in rows if row["system"] == system]
        size_label = subset[0]["size_label"]
        if size_label == "num_segments":
            size_label = r"Number of Segments $N$"
        segment_values = sorted({row["segment_count"] for row in subset})
        for seg in segment_values:
            seg_rows = sorted(
                (row for row in subset if row["segment_count"] == seg),
                key=lambda r: r["batch_size"],
            )
            x = [row["batch_size"] for row in seg_rows]
            y_per_env = [row.get("per_env_speed_ratio", row.get("speed_ratio")) for row in seg_rows]
            y_total = [row.get("total_speed_ratio") for row in seg_rows]
            label = f"{size_label}={seg}"
            ax_per_env.plot(x, y_per_env, marker="o", label=label)
            ax_total.plot(x, y_total, marker="o", label=label)

        ax_per_env.set_title(f"{system} – per-env speed")
        ax_total.set_title(f"{system} – total throughput")
        ax_total.set_xlabel(r"Number of environments $n_\mathrm{envs}$ (parallel simulations)")
        for axis in (ax_per_env, ax_total):
            axis.grid(True, linestyle=":", linewidth=0.6)
            if log_x:
                axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
        ax_per_env.legend()

    axes[0, 0].set_ylabel(r"Simulated time / wall time (per env)")
    axes[1, 0].set_ylabel(r"Total simulated time / wall time ($r_\mathrm{s/w}$)")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"[+] Saved figure to {output}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_total_throughput(
    rows: List[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        print("[!] No results available for plotting.")
        return

    systems = sorted({row["system"] for row in rows})
    if sns is not None:
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        1,
        len(systems),
        figsize=(6 * len(systems), 4),
        squeeze=False,
        sharex="col",
    )

    for col, system in enumerate(systems):
        ax = axes[0, col]
        subset = [row for row in rows if row["system"] == system]
        size_label = subset[0]["size_label"]
        if size_label in ["num_segments", "num_links"]:
            size_label = r"$N$"
        segment_values = sorted({row["segment_count"] for row in subset})
        for seg in segment_values:
            seg_rows = sorted(
                (row for row in subset if row["segment_count"] == seg),
                key=lambda r: r["batch_size"],
            )
            x = [row["batch_size"] for row in seg_rows]
            y_total = [row.get("total_speed_ratio") for row in seg_rows]
            label = f"{size_label}={seg}"
            ax.plot(x, y_total, marker="o", label=label)

        ax.set_xlabel(r"Number of environments $n_\mathrm{envs}$")
        ax.grid(True, linestyle=":", linewidth=0.6)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    axes[0, 0].set_ylabel(r"Total simulated time / wall time ($r_\mathrm{s/w}$)")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"[+] Saved total-throughput figure to {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise batch-scaling benchmark CSV results."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to the CSV generated by benchmark_simulation_batch_scaling.py",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        help="Optional subset of systems to plot",
    )
    parser.add_argument(
        "--segment-counts",
        nargs="*",
        type=int,
        help="Optional subset of segment counts to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot (defaults to CSV path with .pdf suffix)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic scale for the number-of-environments axis",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic scale for the speed ratio axes",
    )
    args = parser.parse_args(argv)
    if args.output is None and args.csv.suffix.lower() == ".csv":
        args.output = args.csv.with_suffix(".pdf")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = _load_rows(args.csv)
    rows = _filter_rows(rows, args.systems, args.segment_counts)
    throughput_output: Path | None = None
    if args.output is not None:
        throughput_output = args.output.with_name(
            f"{args.output.stem}_total_throughput{args.output.suffix}"
        )
    _plot(rows, args.output, args.show, args.log_x, args.log_y)
    if throughput_output is not None:
        _plot_total_throughput(rows, throughput_output, args.show, args.log_x, args.log_y)
    if not args.output and not args.show:
        print("[!] Neither --output nor --show was supplied; nothing was rendered.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
