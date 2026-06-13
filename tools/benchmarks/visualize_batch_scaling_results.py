#!/usr/bin/env python3
"""Visualise batch-scaling benchmark CSV results.

The script reads one or more outputs from
``benchmark_simulation_batch_scaling.py`` and renders per-system scaling plots,
total-throughput summaries, and a consolidated model comparison.

Examples:
    Plot the GVS segment-count sweep::

        uv run python tools/benchmarks/visualize_batch_scaling_results.py \
          benchmarks/batch_scaling/articulated_soft_robot_1s.csv \
          benchmarks/batch_scaling/planar_pcs_1s.csv \
          benchmarks/batch_scaling/pcs_1s.csv \
          benchmarks/batch_scaling/gvs_segments_1s.csv \
          --output benchmarks/batch_scaling/batch_scaling_segments.pdf \
          --log-x \
          --log-y

    Plot the GVS strain-basis-order sweep::

        uv run python tools/benchmarks/visualize_batch_scaling_results.py \
          benchmarks/batch_scaling/articulated_soft_robot_1s.csv \
          benchmarks/batch_scaling/planar_pcs_1s.csv \
          benchmarks/batch_scaling/pcs_1s.csv \
          benchmarks/batch_scaling/gvs_basis_order_1s.csv \
          --output benchmarks/batch_scaling/batch_scaling_strain_basis_order.pdf \
          --log-x \
          --log-y
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

LATEX_AVAILABLE = shutil.which("latex") is not None
LATEX_FONT_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
}


def _theme_rc(rc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    theme_rc = dict(rc or {})
    if LATEX_AVAILABLE:
        theme_rc.update(LATEX_FONT_RC)
    return theme_rc


if LATEX_AVAILABLE:
    plt.rcParams.update(LATEX_FONT_RC)


MODEL_ORDER = ["articulated_soft_robot", "planar_pcs", "pcs", "gvs"]
MODEL_LABELS = {
    "articulated_soft_robot": "Articulated Soft Robot",
    "planar_pcs": "Planar PCS",
    "pcs": "Spatial PCS",
    "gvs": "Spatial GVS",
}
MODEL_COLORS = {
    "articulated_soft_robot": "#0072B2",
    "planar_pcs": "#009E73",
    "pcs": "#D55E00",
    "gvs": "#CC79A7",
}
SIZE_STYLES = {
    1: ("-", "o"),
    2: ("--", "s"),
    4: ("-.", "^"),
    8: (":", "D"),
    16: ((0, (5, 2)), "p"),
    32: ((0, (1, 1)), "x"),
}
FALLBACK_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1))]
FALLBACK_MARKERS = ["o", "s", "^", "D", "p", "x", "v", "*"]
SEGMENT_SIZE_LABELS = {"num_segments", "num_links"}


def _system_key(system: Any) -> str:
    return str(system).strip().replace("-", "_").replace(" ", "_").lower()


def _compact_system_key(system: Any) -> str:
    return _system_key(system).replace("_", "")


def _canonical_system(system: Any) -> str:
    key = _system_key(system)
    if key in MODEL_LABELS:
        return key
    compact = key.replace("_", "")
    for canonical in MODEL_LABELS:
        if compact == canonical.replace("_", ""):
            return canonical
    return key


def _system_sort_key(system: Any) -> tuple[int, str]:
    canonical = _canonical_system(system)
    try:
        return MODEL_ORDER.index(canonical), canonical
    except ValueError:
        return len(MODEL_ORDER), canonical


def _model_label(system: Any) -> str:
    canonical = _canonical_system(system)
    return MODEL_LABELS.get(canonical, str(system))


def _model_color(system: Any, index: int) -> str:
    canonical = _canonical_system(system)
    if canonical in MODEL_COLORS:
        return MODEL_COLORS[canonical]
    palette = sns.color_palette("colorblind", n_colors=8).as_hex()
    return palette[index % len(palette)]


def _size_value(row: Mapping[str, Any]) -> int:
    return int(row.get("size_value", row["segment_count"]))


def _size_label(size_label: str, *, compact: bool = False) -> str:
    if size_label in {"num_segments", "num_links"}:
        return r"$N$" if compact else r"Number of segments/links $N$"
    if size_label == "strain_basis_order":
        return r"$p$" if compact else r"Strain-basis order $p$"
    return size_label


def _speed_value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "") and key == "total_speed_ratio":
        value = row.get("throughput_ratio")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _style_for_size(size_value: int) -> tuple[Any, str]:
    if size_value in SIZE_STYLES:
        return SIZE_STYLES[size_value]
    idx = abs(int(size_value)) % len(FALLBACK_LINESTYLES)
    marker_idx = abs(int(size_value)) % len(FALLBACK_MARKERS)
    return FALLBACK_LINESTYLES[idx], FALLBACK_MARKERS[marker_idx]


def _coerce(value: str) -> Any:
    """Best-effort numeric conversion for CSV entries."""

    try:
        if value.lower() in {"", "none", "null"}:
            return None
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


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"none", "null"}


def _load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows.extend(
                {key: _coerce(val) for key, val in row.items()} for row in reader
            )
    if not rows:
        raise ValueError(f"No rows found in {', '.join(str(path) for path in paths)}")
    return rows


def _filter_rows(
    rows: Iterable[Mapping[str, Any]],
    systems: Sequence[str] | None,
    segment_counts: Sequence[int] | None,
    gauss_points: Sequence[int] | None,
) -> list[Mapping[str, Any]]:
    filtered: list[Mapping[str, Any]] = []
    system_keys = (
        {_compact_system_key(system) for system in systems} if systems else None
    )
    for row in rows:
        if system_keys and _compact_system_key(row["system"]) not in system_keys:
            continue
        if segment_counts and _size_value(row) not in segment_counts:
            continue
        if (
            gauss_points
            and not _is_missing(row.get("gauss_points"))
            and int(row["gauss_points"]) not in gauss_points
        ):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No rows remain after applying the requested filters.")
    return filtered


def _filter_to_shared_segment_range(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    size_labels = {row["size_label"] for row in rows}
    if not size_labels <= SEGMENT_SIZE_LABELS:
        return list(rows)

    sizes_by_system: dict[Any, list[int]] = {}
    for row in rows:
        sizes_by_system.setdefault(row["system"], []).append(_size_value(row))

    if len(sizes_by_system) < 2:
        return list(rows)

    common_min = max(min(sizes) for sizes in sizes_by_system.values())
    common_max = min(max(sizes) for sizes in sizes_by_system.values())
    filtered = [row for row in rows if common_min <= _size_value(row) <= common_max]
    if not filtered:
        raise ValueError(
            "No rows remain after limiting to the shared segment-count range."
        )
    if len(filtered) < len(rows):
        print(
            "[i] Limited segment/link plots to the shared size range "
            f"{common_min}..{common_max}."
        )
    return filtered


def _gauss_groups(rows: Sequence[Mapping[str, Any]]) -> Sequence[int | None]:
    gauss_values = sorted(
        {
            int(row["gauss_points"])
            for row in rows
            if not _is_missing(row.get("gauss_points"))
        }
    )
    return gauss_values if gauss_values else [None]


def _rows_for_group(
    rows: Sequence[Mapping[str, Any]],
    size_value: int,
    gauss_points: int | None,
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if _size_value(row) == size_value
            and (
                gauss_points is None
                or (
                    not _is_missing(row.get("gauss_points"))
                    and int(row["gauss_points"]) == gauss_points
                )
            )
        ),
        key=lambda row: row["batch_size"],
    )


def _group_label(
    size_label: str,
    size_value: int,
    gauss_points: int | None,
    show_gauss: bool,
) -> str:
    label = f"{_size_label(size_label, compact=True)}={size_value}"
    if show_gauss:
        label = f"{label}, gauss={gauss_points}"
    return label


def _plot(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    sns.set_theme(style="whitegrid", rc=_theme_rc())

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
        segment_values = sorted({_size_value(row) for row in subset})
        gauss_groups = _gauss_groups(subset)
        show_gauss = len(gauss_groups) > 1
        for size in segment_values:
            for gauss_points in gauss_groups:
                seg_rows = _rows_for_group(subset, size, gauss_points)
                if not seg_rows:
                    continue
                x = [row["batch_size"] for row in seg_rows]
                y_per_env = [
                    _speed_value(row, "per_env_speed_ratio")
                    if row.get("per_env_speed_ratio") not in (None, "")
                    else _speed_value(row, "speed_ratio")
                    for row in seg_rows
                ]
                y_total = [_speed_value(row, "total_speed_ratio") for row in seg_rows]
                label = _group_label(size_label, size, gauss_points, show_gauss)
                ax_per_env.plot(x, y_per_env, marker="o", label=label)
                ax_total.plot(x, y_total, marker="o", label=label)

        ax_per_env.set_title(f"{_model_label(system)} - per-env speed")
        ax_total.set_title(f"{_model_label(system)} - total throughput")
        ax_total.set_xlabel(
            r"Number of environments $n_\mathrm{envs}$ (parallel simulations)"
        )
        for axis in (ax_per_env, ax_total):
            axis.grid(True, linestyle=":", linewidth=0.6)
            if log_x:
                axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
        ax_per_env.legend()

    axes[0, 0].set_ylabel(r"Simulated time / wall time (per env)")
    axes[1, 0].set_ylabel(
        r"Total simulated time / wall time ($r_{\mathrm{s}/\mathrm{w}}$)"
    )
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"[+] Saved figure to {output}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_total_throughput(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        print("[!] No results available for plotting.")
        return

    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    sns.set_theme(style="whitegrid", rc=_theme_rc())

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
        segment_values = sorted({_size_value(row) for row in subset})
        gauss_groups = _gauss_groups(subset)
        show_gauss = len(gauss_groups) > 1
        for size in segment_values:
            for gauss_points in gauss_groups:
                seg_rows = _rows_for_group(subset, size, gauss_points)
                if not seg_rows:
                    continue
                x = [row["batch_size"] for row in seg_rows]
                y_total = [_speed_value(row, "total_speed_ratio") for row in seg_rows]
                label = _group_label(size_label, size, gauss_points, show_gauss)
                ax.plot(x, y_total, marker="o", label=label)

        ax.set_title(_model_label(system))

        ax.set_xlabel(r"Number of environments $n_\mathrm{envs}$")
        ax.grid(True, linestyle=":", linewidth=0.6)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    axes[0, 0].set_ylabel(
        r"Total simulated time / wall time ($r_{\mathrm{s}/\mathrm{w}}$)"
    )
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"[+] Saved total-throughput figure to {output}")
    if show:
        plt.show()
    plt.close(fig)


def _curve_xy(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values_by_batch: dict[float, list[float]] = {}
    for row in rows:
        try:
            x = float(row["batch_size"])
        except (TypeError, ValueError):
            continue
        y = _speed_value(row, "total_speed_ratio")
        if x <= 0 or y is None or y <= 0:
            continue
        values_by_batch.setdefault(x, []).append(y)

    x_values = np.array(sorted(values_by_batch), dtype=float)
    y_values = np.array(
        [np.mean(values_by_batch[x]) for x in x_values],
        dtype=float,
    )
    return x_values, y_values


def _plot_combined_scaling(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
) -> None:
    if not rows:
        print("[!] No results available for the combined scaling plot.")
        return

    sns.set_theme(
        context="paper",
        style="ticks",
        rc=_theme_rc(
            {
                "axes.linewidth": 0.8,
                "axes.labelsize": 9,
                "axes.titlesize": 10,
                "font.size": 9,
                "legend.fontsize": 8,
                "legend.title_fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        ),
    )

    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    all_x: list[float] = []
    all_y: list[float] = []

    for system_index, system in enumerate(systems):
        subset = [row for row in rows if row["system"] == system]
        if not subset:
            continue

        color = _model_color(system, system_index)
        size_values = sorted({_size_value(row) for row in subset})
        curves: list[tuple[int, np.ndarray, np.ndarray]] = []
        for size in size_values:
            curve_rows = [row for row in subset if _size_value(row) == size]
            x_values, y_values = _curve_xy(curve_rows)
            if x_values.size == 0:
                continue
            curves.append((size, x_values, y_values))
            all_x.extend(x_values.tolist())
            all_y.extend(y_values.tolist())

        if not curves:
            continue

        envelope_x = np.array(sorted({x for _, xs, _ in curves for x in xs}))
        lower: list[float] = []
        upper: list[float] = []
        fill_x: list[float] = []
        for x_val in envelope_x:
            y_at_x: list[float] = []
            for _, xs, ys in curves:
                matches = np.where(xs == x_val)[0]
                if matches.size:
                    y_at_x.append(float(ys[matches[0]]))
            if y_at_x:
                fill_x.append(float(x_val))
                lower.append(min(y_at_x))
                upper.append(max(y_at_x))

        if len(fill_x) >= 2:
            ax.fill_between(
                fill_x,
                lower,
                upper,
                color=color,
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )

        outer_sizes = {
            min(size for size, _, _ in curves),
            max(size for size, _, _ in curves),
        }
        for size, x_values, y_values in curves:
            linestyle, marker = _style_for_size(size)
            is_outer = size in outer_sizes
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.2 if is_outer else 3.4,
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=1.9 if is_outer else 1.15,
                alpha=0.95 if is_outer else 0.52,
                zorder=3 if is_outer else 2,
            )

    if not all_x or not all_y:
        print("[!] No positive total-throughput values available for plotting.")
        plt.close(fig)
        return

    x_min = min(all_x)
    y_min, y_max = min(all_y), max(all_y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of parallel environments $n_\mathrm{envs}$")
    ax.set_ylabel(r"Simulation throughput $r_{\mathrm{s}/\mathrm{w}}$")
    ax.set_title("GPU Batch-Scaling Throughput")

    if y_min <= 1.0 <= y_max:
        ax.axhline(1.0, color="0.45", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.text(
            x_min,
            1.0,
            "real time",
            color="0.35",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.grid(True, which="major", color="0.82", linewidth=0.65)
    ax.grid(True, which="minor", color="0.92", linewidth=0.45, alpha=0.8)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=_model_color(system, idx),
            linewidth=2.2,
            label=_model_label(system),
        )
        for idx, system in enumerate(systems)
    ]
    size_values = sorted({_size_value(row) for row in rows})
    size_labels = {row["size_label"] for row in rows}
    if size_labels <= {"num_segments", "num_links"}:
        size_label = r"$N$"
        size_legend_title = "Number of Segments"
    elif size_labels == {"strain_basis_order"}:
        size_label = r"$p$"
        size_legend_title = "Basis Order"
    elif size_labels <= {"num_segments", "num_links", "strain_basis_order"}:
        size_label = r"$N$ or $p$"
        size_legend_title = "Resolution"
    else:
        size_label = _size_label(rows[0]["size_label"], compact=True)
        size_legend_title = "Resolution"
    size_handles = []
    for size in size_values:
        linestyle, marker = _style_for_size(size)
        size_handles.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=1.3,
                label=rf"{size_label}={size}",
            )
        )

    first_legend = ax.legend(
        handles=model_handles,
        title="Robot model",
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        borderpad=0.6,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=size_handles,
        title=size_legend_title,
        loc="lower right",
        ncol=2 if len(size_handles) > 4 else 1,
        frameon=True,
        framealpha=0.94,
        borderpad=0.6,
        columnspacing=1.2,
        handlelength=2.2,
    )

    sns.despine(fig=fig, ax=ax)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"[+] Saved combined scaling figure to {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise batch-scaling benchmark CSV results."
    )
    parser.add_argument(
        "csv",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to CSV files generated by benchmark_simulation_batch_scaling.py"
        ),
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
        "--gauss-points",
        nargs="*",
        type=int,
        help="Optional subset of Gauss-Legendre point counts to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot (defaults to CSV path with .pdf suffix)",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help=(
            "Optional path for the consolidated all-model log-log scaling plot "
            "(defaults to <output>_combined_scaling.<suffix>)"
        ),
    )
    parser.add_argument(
        "--no-combined-plot",
        action="store_true",
        help="Do not generate the consolidated all-model scaling plot",
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
    if (
        args.output is None
        and len(args.csv) == 1
        and args.csv[0].suffix.lower() == ".csv"
    ):
        args.output = args.csv[0].with_suffix(".pdf")
    if (
        args.combined_output is None
        and args.output is not None
        and not args.no_combined_plot
    ):
        args.combined_output = args.output.with_name(
            f"{args.output.stem}_combined_scaling{args.output.suffix}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = _load_rows(args.csv)
    rows = _filter_rows(rows, args.systems, args.segment_counts, args.gauss_points)
    rows = _filter_to_shared_segment_range(rows)
    throughput_output: Path | None = None
    if args.output is not None:
        throughput_output = args.output.with_name(
            f"{args.output.stem}_total_throughput{args.output.suffix}"
        )
    _plot(rows, args.output, args.show, args.log_x, args.log_y)
    if throughput_output is not None:
        _plot_total_throughput(
            rows, throughput_output, args.show, args.log_x, args.log_y
        )
    if not args.no_combined_plot and (args.combined_output is not None or args.show):
        _plot_combined_scaling(rows, args.combined_output, args.show)
    if not args.output and not args.combined_output and not args.show:
        print("[!] Neither --output nor --show was supplied; nothing was rendered.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
